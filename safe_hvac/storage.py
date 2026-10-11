"""Streaming scientific records outside simulator episode directories."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import numpy as np


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(jsonable(payload), stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class RunStore:
    """Exclusive creation prevents accidental merging or overwriting of runs."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.streams = {}
        self.counts = {}

    def write(self, name, record):
        if name not in self.streams:
            self.streams[name] = gzip.open(self.root / f"{name}.jsonl.gz", "wt", compresslevel=6)
            self.counts[name] = 0
        stream = self.streams[name]
        stream.write(json.dumps(jsonable(record), allow_nan=False, separators=(",", ":")) + "\n")
        self.counts[name] += 1
        if self.counts[name] % 64 == 0:
            stream.flush()

    def close(self):
        for stream in self.streams.values():
            stream.close()
        self.streams.clear()
