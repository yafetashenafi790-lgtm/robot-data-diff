"""Build testdata/pusht_v2: a modified copy of testdata/pusht.

Modifications (all deliberate, to exercise the differ):
  - episodes 200-205 REMOVED (6 episodes)
  - episodes 0-4 CHANGED (actions scaled x1.5 -> stat deltas)
  - episode 206 ADDED (copy of episode 10's data, new index)
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SRC = Path("testdata/pusht")
DST = Path("testdata/pusht_v2")

REMOVED = set(range(200, 206))
CHANGED = set(range(0, 5))
ADDED_SRC = 10
ADDED_NEW = 206

if DST.exists():
    shutil.rmtree(DST)
shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns(".cache"))

# ---- data parquet ----
df = pq.read_table(DST / "data/chunk-000/file-000.parquet").to_pandas()

# 1. perturb actions for CHANGED episodes
mask = df["episode_index"].isin(CHANGED)
df.loc[mask, "action"] = df.loc[mask, "action"].apply(
    lambda a: (np.asarray(a, dtype=np.float64) * 1.5).tolist()
)

# 2. drop REMOVED episodes
df = df[~df["episode_index"].isin(REMOVED)].copy()

# 3. add episode 206 as a copy of episode 10
add = df[df["episode_index"] == ADDED_SRC].copy()
add["episode_index"] = ADDED_NEW
max_idx = int(df["index"].max())
add["index"] = np.arange(max_idx + 1, max_idx + 1 + len(add))
df = pd.concat([df, add], ignore_index=True)

pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
               DST / "data/chunk-000/file-000.parquet")

# ---- episodes parquet ----
ep = pq.read_table(DST / "meta/episodes/chunk-000/file-000.parquet").to_pandas()

# recompute action stats for CHANGED episodes from the modified data
for ei in CHANGED:
    acts = np.stack(df[df["episode_index"] == ei]["action"]
                    .apply(lambda a: np.asarray(a, dtype=np.float64)))
    row = ep.index[ep["episode_index"] == ei][0]
    ep.at[row, "stats/action/min"] = acts.min(axis=0)
    ep.at[row, "stats/action/max"] = acts.max(axis=0)
    ep.at[row, "stats/action/mean"] = acts.mean(axis=0)

# drop REMOVED, append ADDED (copy of episode 10's row, new index/stats)
ep = ep[~ep["episode_index"].isin(REMOVED)].copy()
new_row = ep[ep["episode_index"] == ADDED_SRC].iloc[0].copy()
new_row["episode_index"] = ADDED_NEW
acts = np.stack(df[df["episode_index"] == ADDED_NEW]["action"]
                .apply(lambda a: np.asarray(a, dtype=np.float64)))
new_row["stats/action/min"] = acts.min(axis=0)
new_row["stats/action/max"] = acts.max(axis=0)
new_row["stats/action/mean"] = acts.mean(axis=0)
new_row["dataset_from_index"] = int(df[df["episode_index"] == ADDED_NEW]["index"].min())
new_row["dataset_to_index"] = int(df[df["episode_index"] == ADDED_NEW]["index"].max()) + 1
ep = pd.concat([ep, new_row.to_frame().T], ignore_index=True)

pq.write_table(pa.Table.from_pandas(ep, preserve_index=False),
               DST / "meta/episodes/chunk-000/file-000.parquet")

# ---- info.json ----
info = json.loads((DST / "meta/info.json").read_text())
info["total_episodes"] = len(ep)
info["total_frames"] = int(len(df))
(DST / "meta/info.json").write_text(json.dumps(info, indent=2))

print(f"fixture ready: {len(ep)} episodes, {len(df)} frames "
      f"(removed={sorted(REMOVED)}, changed={sorted(CHANGED)}, added=[{ADDED_NEW}])")
