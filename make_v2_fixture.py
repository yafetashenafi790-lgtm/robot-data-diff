"""Build testdata/v2toy: a tiny synthetic LeRobot v2.1-format dataset.

Layout follows the documented v2.1 format:
  meta/info.json, meta/episodes.jsonl, meta/episodes_stats.jsonl,
  meta/tasks.jsonl, data/chunk-000/episode_000000.parquet ...
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path("testdata/v2toy")
rng = np.random.default_rng(7)

(ROOT / "meta").mkdir(parents=True, exist_ok=True)
(ROOT / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

N_EPS, T, D = 4, 60, 2
info = {
    "codebase_version": "v2.1",
    "robot_type": "toy",
    "total_episodes": N_EPS,
    "total_frames": N_EPS * T,
    "total_tasks": 1,
    "fps": 10,
    "features": {
        "observation.state": {"dtype": "float32", "shape": [D]},
        "action": {"dtype": "float32", "shape": [D]},
    },
    "splits": {"train": f"0:{N_EPS}"},
}
(ROOT / "meta" / "info.json").write_text(json.dumps(info, indent=2))

with open(ROOT / "meta" / "episodes.jsonl", "w") as f, \
     open(ROOT / "meta" / "episodes_stats.jsonl", "w") as fs, \
     open(ROOT / "meta" / "tasks.jsonl", "w") as ft:
    ft.write(json.dumps({"task_index": 0, "task": "toy reaching task"}) + "\n")
    for ei in range(N_EPS):
        acts = np.cumsum(rng.normal(0, 1, (T, D)), axis=0)
        states = np.cumsum(rng.normal(0, 0.5, (T, D)), axis=0)
        df = pd.DataFrame({
            "observation.state": [s.tolist() for s in states],
            "action": [a.tolist() for a in acts],
            "episode_index": ei,
            "frame_index": np.arange(T),
            "timestamp": np.arange(T) / 10.0,
        })
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                       ROOT / "data" / "chunk-000" / f"episode_{ei:06d}.parquet")
        f.write(json.dumps({"episode_index": ei, "tasks": ["toy reaching task"],
                            "length": T}) + "\n")
        fs.write(json.dumps({
            "episode_index": ei,
            "action": {"min": acts.min(0).tolist(), "max": acts.max(0).tolist(),
                       "mean": acts.mean(0).tolist(), "std": acts.std(0).tolist()},
            "observation.state": {"min": states.min(0).tolist(),
                                  "max": states.max(0).tolist()},
        }) + "\n")

print(f"v2 fixture ready: {N_EPS} episodes x {T} frames")
