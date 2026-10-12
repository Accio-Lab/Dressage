"""Run-local checkpoint rotation: four complete copies plus one writer."""

import fcntl
import logging
import re
import shutil
from pathlib import Path


class CheckpointWindow:
    KEEP = 5

    def __init__(self, save, project_root, run_id):
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", run_id) or run_id in (
            ".",
            "..",
        ):
            raise ValueError("invalid checkpoint run ID")
        expected = Path(project_root).resolve() / "outputs" / "checkpoints" / run_id
        self.root = Path(save).absolute()
        if self.root != expected or self.root.resolve() != expected:
            raise ValueError(
                "checkpoint save must be the nonsymlink current run directory"
            )
        self.completed = []
        self.lock = None
        self.failed = False

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        lock = self.root / ".checkpoint-writer.lock"
        if lock.is_symlink():
            raise ValueError("checkpoint lock must not be a symlink")
        self.lock = lock.open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Resume uses a new run directory. Never adopt/delete unknown outputs.
            if any(p.name != lock.name for p in self.root.iterdir()):
                raise ValueError(
                    "checkpoint run directory is not empty; use a new run ID"
                )
        except BaseException:
            self.lock.close()
            self.lock = None
            raise
        return self

    def __exit__(self, *_):
        if self.lock is not None:
            self.lock.close()
            self.lock = None

    def save(self, actor_model, step):
        if self.lock is None or self.failed:
            raise RuntimeError("checkpoint writer is closed or a previous save failed")
        if step < 0 or (self.completed and step <= self.completed[-1]):
            raise ValueError("checkpoint steps must increase")
        try:
            expected = {f"iter_{n:07d}" for n in self.completed}
            actual = {p.name for p in self.root.iterdir() if p.name.startswith("iter_")}
            if actual != expected:
                raise RuntimeError(
                    "unexpected checkpoint directories; refusing rotation"
                )
            for name in expected:
                path = self.root / name
                if path.is_symlink() or not path.is_dir():
                    raise RuntimeError(
                        "checkpoint directory replaced; refusing rotation"
                    )
            tracker = self.root / "latest_checkpointed_iteration.txt"
            if tracker.is_symlink():
                raise RuntimeError("checkpoint tracker must not be a symlink")
            if self.completed and tracker.read_text().strip() != str(
                self.completed[-1]
            ):
                raise RuntimeError(
                    "checkpoint tracker disagrees with last completed save"
                )
            if len(self.completed) == self.KEEP:
                oldest = self.completed[0]
                path = self.root / f"iter_{oldest:07d}"
                shutil.rmtree(path)
                self.completed.pop(0)
                logging.getLogger(__name__).info("checkpoint retired: %s", path)
            # Actor group waits for all ranks; force_sync finalizes async writers.
            actor_model.save_model(step, force_sync=True)
            current = self.root / f"iter_{step:07d}"
            actual = {p.name for p in self.root.iterdir() if p.name.startswith("iter_")}
            expected = {f"iter_{n:07d}" for n in [*self.completed, step]}
            if (
                actual != expected
                or current.is_symlink()
                or not current.is_dir()
                or tracker.is_symlink()
                or tracker.read_text().strip() != str(step)
            ):
                raise RuntimeError(
                    "checkpoint save did not publish the expected iteration"
                )
            self.completed.append(step)
        except BaseException:
            # Preserve partial evidence; never prune/retry after an ambiguous save.
            self.failed = True
            raise
