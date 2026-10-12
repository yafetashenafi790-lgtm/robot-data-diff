"""Load LeRobot datasets (v2.1 and v3.0) from a local path or a Hugging Face repo id.

v3.0 layout (current):
    meta/info.json                  codebase_version "v3.0", fps, features, totals
    meta/episodes/chunk-*/file-*.parquet   one row per episode; includes
        episode_index, tasks, length, and per-episode stats/action/min|max|mean
    meta/tasks.parquet              task_index -> task string (in the index)
    data/chunk-*/file-*.parquet     frames: episode_index, frame_index,
        timestamp, action, observation.state, ...

v2.1 layout (legacy, still common on the Hub):
    meta/info.json                  codebase_version "v2.1", fps, features
    meta/episodes.jsonl             episode_index, tasks, length (one JSON/line)
    meta/episodes_stats.jsonl       per-episode min/max/mean/std per feature
    meta/tasks.jsonl                task_index -> task
    data/chunk-*/episode_*.parquet  one parquet per episode with the frames

Videos are never read — everything here is metadata + data parquet only.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .stats import action_quality


@dataclass
class EpisodeInfo:
    index: int
    length: int
    tasks: list = field(default_factory=list)
    action_min: list | None = None
    action_max: list | None = None
    action_mean: list | None = None
    state_min: list | None = None
    state_max: list | None = None
    idle_ratio: float | None = None
    spike_count: int | None = None


@dataclass
class DatasetView:
    name: str
    version: str
    fps: float | None
    features: dict  # feature name -> {"dtype": str, "shape": list}
    episodes: dict  # episode_index -> EpisodeInfo
    root: Path


# ---------------------------------------------------------------------------
# reference resolution: local path or Hugging Face id
# ---------------------------------------------------------------------------

def _strip_no_proxy():
    # The sandbox proxy env breaks httpx2's no_proxy parsing (IPv6 entries);
    # the egress proxy handles everything, so drop no_proxy for HF calls.
    for k in ("no_proxy", "NO_PROXY"):
        os.environ.pop(k, None)


def resolve_ref(ref: str, cache_dir: str | Path | None = None) -> Path:
    """Turn a dataset reference into a local directory Path.

    Accepts a local path, or a Hugging Face dataset id like
    "lerobot/pusht" or "lerobot/pusht@<revision>".  HF downloads fetch
    meta/** and data/** only — videos are skipped.
    """
    p = Path(ref)
    if p.exists():
        return p
    _strip_no_proxy()
    from huggingface_hub import snapshot_download

    repo_id, _, revision = ref.removeprefix("hf://").partition("@")
    if "/" not in repo_id:
        raise ValueError(
            f"'{ref}' is not a local path and not a Hugging Face id "
            "(expected like 'lerobot/pusht' or 'lerobot/pusht@<rev>')."
        )
    local = snapshot_download(
        repo_id,
        repo_type="dataset",
        revision=revision or None,
        allow_patterns=["meta/**", "data/**"],
        local_dir=cache_dir,
    )
    return Path(local)


def _as_list(v):
    if v is None:
        return None
    return [float(x) for x in np.asarray(v).ravel()]


# ---------------------------------------------------------------------------
# v3 loader
# ---------------------------------------------------------------------------

def _load_v3(root: Path, name: str) -> DatasetView:
    info = json.loads((root / "meta" / "info.json").read_text())
    features = {
        k: {"dtype": str(v.get("dtype")), "shape": list(v.get("shape") or [])}
        for k, v in (info.get("features") or {}).items()
    }
    fps = info.get("fps")

    ep_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not ep_files:
        raise ValueError(f"no episode metadata found under {root}/meta/episodes")
    episodes: dict[int, EpisodeInfo] = {}
    for f in ep_files:
        table = pq.read_table(f).to_pandas()
        for _, row in table.iterrows():
            idx = int(row["episode_index"])
            tasks = [str(t) for t in (row.get("tasks") or [])]
            episodes[idx] = EpisodeInfo(
                index=idx,
                length=int(row["length"]),
                tasks=tasks,
                action_min=_as_list(row.get("stats/action/min")),
                action_max=_as_list(row.get("stats/action/max")),
                action_mean=_as_list(row.get("stats/action/mean")),
                state_min=_as_list(row.get("stats/observation.state/min")),
                state_max=_as_list(row.get("stats/observation.state/max")),
            )
    ds = DatasetView(name=name, version="v3.0", fps=fps, features=features,
                     episodes=episodes, root=root)
    _add_quality_v3(ds)
    return ds


def _add_quality_v3(ds: DatasetView):
    """Compute idle-ratio / spike-count per episode from v3 data chunks."""
    data_files = sorted((ds.root / "data").rglob("*.parquet"))
    if not data_files:
        return
    for f in data_files:
        try:
            table = pq.read_table(f, columns=["episode_index", "action"])
        except Exception:
            continue  # chunk without action column; skip
        df = table.to_pandas()
        for ep_idx, grp in df.groupby("episode_index"):
            ep = ds.episodes.get(int(ep_idx))
            if ep is None:
                continue
            acts = np.stack([np.asarray(a, dtype=np.float64).ravel()
                             for a in grp["action"].to_numpy()])
            ep.idle_ratio, ep.spike_count = action_quality(acts)


# ---------------------------------------------------------------------------
# v2 loader (documented v2.1 layout)
# ---------------------------------------------------------------------------

def _load_v2(root: Path, name: str) -> DatasetView:
    info = json.loads((root / "meta" / "info.json").read_text())
    features = {
        k: {"dtype": str(v.get("dtype")), "shape": list(v.get("shape") or [])}
        for k, v in (info.get("features") or {}).items()
    }
    fps = info.get("fps")

    episodes: dict[int, EpisodeInfo] = {}
    ep_path = root / "meta" / "episodes.jsonl"
    for line in ep_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        idx = int(rec["episode_index"])
        episodes[idx] = EpisodeInfo(
            index=idx,
            length=int(rec.get("length", 0)),
            tasks=[str(t) for t in (rec.get("tasks") or [])],
        )

    stats_path = root / "meta" / "episodes_stats.jsonl"
    if stats_path.exists():
        for line in stats_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ep = episodes.get(int(rec["episode_index"]))
            if ep is None:
                continue
            feats = rec.get("stats") or rec.get("features") or rec
            for feat_key, target, attr in (
                ("action", "action", None),
                ("observation.state", "state", None),
                ("observation_state", "state", None),
            ):
                f = feats.get(feat_key)
                if isinstance(f, dict):
                    if target == "action":
                        ep.action_min = _as_list(f.get("min"))
                        ep.action_max = _as_list(f.get("max"))
                        ep.action_mean = _as_list(f.get("mean"))
                    else:
                        ep.state_min = _as_list(f.get("min"))
                        ep.state_max = _as_list(f.get("max"))

    ds = DatasetView(name=name, version="v2.1", fps=fps, features=features,
                     episodes=episodes, root=root)
    _add_quality_v2(ds)
    return ds


def _add_quality_v2(ds: DatasetView):
    """Per-episode quality from v2 per-episode data files."""
    for f in sorted((ds.root / "data").rglob("episode_*.parquet")):
        try:
            table = pq.read_table(f, columns=["action"])
        except Exception:
            continue
        acts = np.stack([np.asarray(a, dtype=np.float64).ravel()
                         for a in table.column("action").to_pylist()])
        # episode index from filename episode_000123.parquet
        try:
            ep_idx = int(f.stem.split("_")[-1])
        except ValueError:
            continue
        ep = ds.episodes.get(ep_idx)
        if ep is not None:
            ep.idle_ratio, ep.spike_count = action_quality(acts)


# ---------------------------------------------------------------------------

def load_dataset(ref: str, cache_dir: str | Path | None = None,
                 with_quality: bool = True) -> DatasetView:
    """Load a dataset reference into a DatasetView (v2.1 or v3.0)."""
    root = resolve_ref(ref, cache_dir)
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise ValueError(f"{root} does not look like a LeRobot dataset "
                         "(missing meta/info.json).")
    version = str(json.loads(info_path.read_text()).get("codebase_version", ""))
    name = ref if isinstance(ref, str) else str(ref)
    if version.startswith("v3"):
        ds = _load_v3(root, name)
    elif version.startswith("v2"):
        ds = _load_v2(root, name)
    else:
        raise ValueError(f"unsupported LeRobot codebase_version: {version!r}")
    if not with_quality:
        for ep in ds.episodes.values():
            ep.idle_ratio, ep.spike_count = None, None
    return ds
