"""Model-checkpoint resume with a metadata-only sampler cursor, never trajectories."""

import copy
import hashlib
import json
import logging
import os
from pathlib import Path

from agentic_rollout_contract.canonical import canonical_digest
from dressage.rollout.data_source import DressageDataSource

from .execution import write_artifact


def write_cursor(path, state):
    write_artifact(
        path,
        json.dumps({"payload": state, "digest": canonical_digest(state)}).encode(),
    )


def checkpoint_from_env():
    value = os.environ.get("TRAIN_RESUME_FROM")
    if not value:
        return None
    root = Path(value).absolute()
    expected = Path(os.environ["GPU_SOURCE_ROOT"]) / "outputs/checkpoints"
    if (
        root.parent != expected
        or root.resolve() != root
        or root.name == os.environ["DRESSAGE_DISTROLL_RUN_ID"]
    ):
        raise ValueError("resume requires another run's nonsymlink checkpoint root")
    iteration = int((root / "latest_checkpointed_iteration.txt").read_text())
    if iteration < 0:
        raise ValueError("negative resume iteration")
    directory = root / f"iter_{iteration:07d}"
    for name in (".metadata", "metadata.json", "common.pt"):
        if not (directory / name).is_file() or not (directory / name).stat().st_size:
            raise ValueError(f"incomplete resume checkpoint: {name}")
    return root, iteration


class TrainingDataSource(DressageDataSource):
    """Use native samples; persist only frozen-input order and cursor counters."""

    def cursor(self, iteration, source_run):
        if self._samples is None or self._mixture_samples is not None:
            raise ValueError("training cursor requires the frozen text dataset")
        return {
            "version": 1,
            "run_id": source_run,
            "iteration": iteration,
            "prompt_sha256": hashlib.sha256(
                Path(self.args.prompt_data).read_bytes()
            ).hexdigest(),
            "width": self.args.n_samples_per_prompt,
            "seed": self.args.rollout_seed,
            "order": [s.metadata["harbor_task_ref"] for s in self._samples],
            "counters": {
                k: getattr(self, k)
                for k in (
                    "epoch_id",
                    "sample_offset",
                    "sample_group_index",
                    "sample_index",
                )
            },
            "buffer": [
                [
                    {
                        "task_ref": s.metadata["harbor_task_ref"],
                        "index": s.index,
                        "group_index": s.group_index,
                        "session_id": getattr(s, "session_id", None),
                        "deterministic_session": s.metadata.get(
                            "dressage_deterministic_session_id"
                        ),
                    }
                    for s in group
                ]
                for group in self.buffer
            ],
        }

    def save(self, rollout_id):
        root = Path(self.args.save)
        if int((root / "latest_checkpointed_iteration.txt").read_text()) != rollout_id:
            raise ValueError("save sampler only after matching model checkpoint")
        path = root / f"iter_{rollout_id:07d}/source_cursor.json"
        write_cursor(path, self.cursor(rollout_id, root.name))
        logging.getLogger(__name__).info(
            "sampler checkpoint saved iteration=%s offset=%s",
            rollout_id,
            self.sample_offset,
        )

    def load(self, rollout_id=None):
        checkpoint = checkpoint_from_env()
        if checkpoint is None:
            return
        root, iteration = checkpoint
        if rollout_id != iteration:
            raise ValueError("loaded model iteration disagrees with sampler resume")
        path = Path(
            os.environ.get("TRAIN_RESUME_CURSOR")
            or root / f"iter_{iteration:07d}/source_cursor.json"
        )
        document = json.loads(path.read_text())
        state = document["payload"]
        if document["digest"] != canonical_digest(state):
            raise ValueError("sampler cursor digest mismatch")
        current = self.cursor(iteration, root.name)
        for key in ("version", "run_id", "iteration", "prompt_sha256", "width", "seed"):
            if state[key] != current[key]:
                raise ValueError(f"sampler resume mismatch: {key}")
        samples = {s.metadata["harbor_task_ref"]: s for s in self._samples}
        if (
            len(samples) != len(self._samples)
            or len(state["order"]) != len(samples)
            or set(state["order"]) != set(samples)
        ):
            raise ValueError("sampler resume task order is incomplete or duplicated")
        counters = state["counters"]
        if set(counters) != set(current["counters"]) or any(
            type(v) is not int or v < 0 for v in counters.values()
        ):
            raise ValueError("invalid sampler counters")
        if counters["sample_offset"] > len(samples):
            raise ValueError("sampler offset exceeds dataset")
        buffered = []
        for group in state["buffer"]:
            if len(group) != self.args.n_samples_per_prompt:
                raise ValueError("incomplete sampler buffer group")
            restored = []
            for item in group:
                s = copy.deepcopy(samples[item["task_ref"]])
                s.index, s.group_index = item["index"], item["group_index"]
                s.session_id = item["session_id"]
                if item["deterministic_session"] is not None:
                    s.metadata["dressage_deterministic_session_id"] = item[
                        "deterministic_session"
                    ]
                restored.append(s)
            buffered.append(restored)
        self._samples = [samples[key] for key in state["order"]]
        for key, value in counters.items():
            setattr(self, key, value)
        self.buffer = buffered
        logging.getLogger(__name__).info(
            "sampler checkpoint restored iteration=%s epoch=%s offset=%s groups=%s buffered=%s",
            iteration,
            self.epoch_id,
            self.sample_offset,
            self.sample_group_index,
            len(self.buffer),
        )


if __name__ == "__main__":
    checkpoint = checkpoint_from_env()
    if checkpoint is None:
        raise ValueError("TRAIN_RESUME_FROM is required")
    print(checkpoint[1])
