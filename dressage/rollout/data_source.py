"""DressageDataSource — text-first data source extending slime's base.

Supports plain string prompts in JSONL without requiring a multimodal processor.
All fields beyond prompt and label are stored in Sample.metadata for downstream dispatch.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import random
from typing import Any

logger = logging.getLogger(__name__)

try:
    from slime.rollout.data_source import RolloutDataSourceWithBuffer
    from slime.utils.types import Sample
except ImportError:
    logger.warning(
        "slime not importable — DressageDataSource will use standalone types. "
        "For production use, ensure slime is installed."
    )

    from dataclasses import dataclass, field
    from enum import Enum

    @dataclass
    class Sample:
        group_index: int | None = None
        index: int | None = None
        prompt: str | list[dict[str, str]] = ""
        tokens: list[int] = field(default_factory=list)
        response: str = ""
        response_length: int = 0
        label: str | None = None
        reward: float | dict[str, Any] | None = None
        loss_mask: list[int] | None = None
        rollout_log_probs: list[float] | None = None
        remove_sample: bool = False
        metadata: dict = field(default_factory=dict)
        generate_function_path: str | None = None
        train_metadata: dict | None = None
        teacher_log_probs: list[float] | None = None

        class Status(Enum):
            PENDING = "pending"
            COMPLETED = "completed"
            TRUNCATED = "truncated"
            ABORTED = "aborted"
            FAILED = "failed"

        status: Status = Status.PENDING

    class RolloutDataSourceWithBuffer:
        def __init__(self, args):
            self.args = args
            self.buffer = []
            self.metadata = {}

        def get_samples(self, num_samples):
            raise NotImplementedError

        def add_samples(self, samples):
            if not samples:
                return
            for group in samples:
                self.buffer.append(group)

        def save(self, rollout_id):
            pass

        def load(self, rollout_id=None):
            pass

        def __len__(self):
            return 0


def _read_jsonl(path: str):
    """Read a JSONL file and yield dicts."""
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("JSON decode error at line %d: %s", line_num, e)
                continue


class DressageDataSource(RolloutDataSourceWithBuffer):
    """Text-first data source for Dressage.

    Extends slime's data source with:
    - Plain string prompts work without a multimodal processor.
    - Per-sample metadata passthrough: agent_mode, env_type, tool_set, reward_fn
      are preserved in Sample.metadata for downstream dispatch.
    """

    def __init__(self, args: Any) -> None:
        self.args = args
        self.buffer: list[list[Sample]] = []
        self.metadata: dict = {}

        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0

        prompt_key = getattr(args, "input_key", None) or "prompt"
        label_key = getattr(args, "label_key", None)
        metadata_key = getattr(args, "metadata_key", "metadata")
        multimodal_keys = getattr(args, "multimodal_keys", None)
        apply_chat_template = getattr(args, "apply_chat_template", False)

        prompt_data = getattr(args, "prompt_data", None) or os.environ.get(
            "PROMPT_DATA"
        )
        self._use_text_first = multimodal_keys is None

        # A MOPD config may own a weighted collection of datasets.  Keep the
        # legacy single --prompt-data path unchanged when no datasets are
        # declared in that config.
        self._mixture_samples: list[list[Sample]] | None = None
        self._mixture_weights: list[float] = []
        self._mixture_current: list[float] = []
        self._mixture_offsets: list[int] = []
        self._mixture_epochs: list[int] = []
        mopd_config_path = getattr(args, "mopd_teacher_config", None) or os.environ.get(
            "DRESSAGE_MOPD_TEACHER_CONFIG"
        )
        if self._use_text_first and mopd_config_path:
            from dressage.rollout.mopd import load_mopd_config

            mopd_config = load_mopd_config(str(mopd_config_path))
            if mopd_config.datasets:
                self._mixture_samples = []
                for dataset in mopd_config.datasets:
                    dataset_samples = self._load_text_first(
                        dataset.path,
                        prompt_key,
                        label_key,
                        metadata_key,
                        metadata_overrides=dataset.metadata,
                        generate_function_path=dataset.generate_function_path,
                    )
                    if not dataset_samples:
                        raise ValueError(
                            f"MOPD dataset {dataset.name!r} has no valid samples: {dataset.path}"
                        )
                    self._mixture_samples.append(dataset_samples)
                    self._mixture_weights.append(dataset.weight)
                self._mixture_current = [0.0] * len(self._mixture_samples)
                self._mixture_offsets = [0] * len(self._mixture_samples)
                self._mixture_epochs = [0] * len(self._mixture_samples)
                self._samples = None
                logger.info(
                    "DressageDataSource: loaded MOPD mixture datasets=%s weights=%s",
                    [dataset.name for dataset in mopd_config.datasets],
                    self._mixture_weights,
                )

        if self._mixture_samples is not None:
            pass
        elif self._use_text_first and prompt_data:
            self._samples = self._load_text_first(
                prompt_data, prompt_key, label_key, metadata_key
            )
            if getattr(args, "rollout_shuffle", False):
                self._shuffle(self.epoch_id)
        elif prompt_data:
            try:
                super().__init__(args)
                self._samples = None
            except Exception:
                logger.warning("Failed slime parent init, falling back to text-first")
                self._samples = self._load_text_first(
                    prompt_data, prompt_key, label_key, metadata_key
                )
        else:
            self._samples = []

    def _load_text_first(
        self,
        path: str,
        prompt_key: str,
        label_key: str | None,
        metadata_key: str,
        metadata_overrides: dict[str, Any] | None = None,
        generate_function_path: str | None = None,
    ) -> list[Sample]:
        """Load JSONL as plain-text samples with metadata passthrough."""
        samples = []
        for data in _read_jsonl(path):
            prompt = data.get(prompt_key, "")
            label = data.get(label_key) if label_key else data.get("label")

            meta = dict(data.get(metadata_key) or {})
            for key in (
                "agent_mode",
                "env_type",
                "env_args",
                "tool_set",
                "agent_id",
                "reward_fn",
            ):
                if key in data and key not in meta:
                    meta[key] = data[key]

            for key, val in data.items():
                if (
                    key not in (prompt_key, "label", metadata_key, label_key)
                    and key not in meta
                ):
                    meta[key] = val

            # Dataset-level routing is authoritative. In particular, it must
            # not collide with task-local fields such as ALFWorld's task_type.
            if metadata_overrides:
                meta.update(copy.deepcopy(metadata_overrides))

            # Slime natively dispatches on this per-sample field. Dataset
            # routing is resolved once here; no MOPD generate wrapper is needed.
            sample_generate_path = (
                generate_function_path
                or data.get("generate_function_path")
                or meta.get("generate_function_path")
            )
            samples.append(
                Sample(
                    prompt=prompt,
                    label=str(label) if label is not None else None,
                    metadata=meta,
                    generate_function_path=(
                        str(sample_generate_path).strip()
                        if sample_generate_path
                        else None
                    ),
                    train_metadata=(
                        {"teacher_id": str(meta["teacher_id"])}
                        if meta.get("teacher_id") not in (None, "")
                        else None
                    ),
                )
            )

        logger.info("DressageDataSource: loaded %d samples from %s", len(samples), path)
        return samples

    def _shuffle(self, epoch_id: int) -> None:
        if self._samples is None:
            return
        seed = getattr(self.args, "rollout_seed", 42)
        random.seed(seed + epoch_id)
        random.shuffle(self._samples)
        self.epoch_id = epoch_id

    def _next_mixture_sample(self) -> Sample:
        """Select a source by smooth weighted round-robin and return one row."""
        assert self._mixture_samples is not None
        total_weight = sum(self._mixture_weights)
        for index, weight in enumerate(self._mixture_weights):
            self._mixture_current[index] += weight
        dataset_index = max(
            range(len(self._mixture_current)),
            key=lambda index: (self._mixture_current[index], -index),
        )
        self._mixture_current[dataset_index] -= total_weight

        dataset = self._mixture_samples[dataset_index]
        offset = self._mixture_offsets[dataset_index]
        if offset >= len(dataset):
            self._mixture_epochs[dataset_index] += 1
            if getattr(self.args, "rollout_shuffle", False):
                seed = int(getattr(self.args, "rollout_seed", 42))
                rng = random.Random(
                    seed + self._mixture_epochs[dataset_index] * 1009 + dataset_index
                )
                rng.shuffle(dataset)
            offset = 0
        sample = dataset[offset]
        self._mixture_offsets[dataset_index] = offset + 1
        return sample

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """Return num_samples prompt groups. Each group has n_samples_per_prompt clones."""
        buffer_samples = self._get_samples_from_buffer(num_samples)
        num_samples -= len(buffer_samples)

        if num_samples == 0:
            return buffer_samples

        if self._mixture_samples is not None:
            base_samples = [self._next_mixture_sample() for _ in range(num_samples)]
        elif self._samples is None:
            return buffer_samples + super().get_samples(num_samples)
        else:
            base_samples = []
            for _ in range(num_samples):
                if self.sample_offset >= len(self._samples):
                    self.epoch_id += 1
                    if getattr(self.args, "rollout_shuffle", False):
                        self._shuffle(self.epoch_id)
                    self.sample_offset = 0

                if self.sample_offset >= len(self._samples):
                    break

                base_samples.append(self._samples[self.sample_offset])
                self.sample_offset += 1

        n_per = getattr(self.args, "n_samples_per_prompt", 1)
        groups = []

        for base_sample in base_samples:

            group = []
            for _ in range(n_per):
                s = copy.deepcopy(base_sample)
                s.group_index = self.sample_group_index
                s.index = self.sample_index
                self.sample_index += 1
                group.append(s)

            self.sample_group_index += 1
            groups.append(group)

        return buffer_samples + groups

    def _get_samples_from_buffer(self, num_samples: int) -> list[list[Sample]]:
        if not self.buffer or num_samples == 0:
            return []
        num_to_pop = min(len(self.buffer), num_samples)
        samples = self.buffer[:num_to_pop]
        del self.buffer[:num_to_pop]
        return samples

    def add_samples(self, samples: list[list[Sample]]) -> None:
        """Add sample groups back to buffer (for retry on failure)."""
        if not samples:
            return
        for group in samples:
            self.buffer.append(group)

    def save(self, rollout_id: int) -> None:
        pass

    def load(self, rollout_id: int | None = None) -> None:
        pass

    def __len__(self) -> int:
        if self._mixture_samples is not None:
            return sum(len(dataset) for dataset in self._mixture_samples)
        if self._samples is not None:
            return len(self._samples)
        return 0
