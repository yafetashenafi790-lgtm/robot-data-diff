"""Episode-aware diff between two DatasetViews."""
from __future__ import annotations

from dataclasses import dataclass, field

from .loader import DatasetView
from .stats import fingerprint


@dataclass
class EpisodeDelta:
    index: int
    old_length: int
    new_length: int
    d_length: int
    tasks_changed: bool
    old_tasks: list = field(default_factory=list)
    new_tasks: list = field(default_factory=list)
    d_action_min: list | None = None   # per-dim new - old
    d_action_max: list | None = None
    d_action_mean: list | None = None
    d_idle_ratio: float | None = None
    d_spike_count: int | None = None


@dataclass
class DiffResult:
    a_name: str
    b_name: str
    a_version: str
    b_version: str
    a_fps: float | None
    b_fps: float | None
    a_episodes: int
    b_episodes: int
    added: list[int] = field(default_factory=list)
    removed: list[int] = field(default_factory=list)
    changed: list[EpisodeDelta] = field(default_factory=list)
    unchanged: int = 0
    task_counts: dict = field(default_factory=dict)  # task -> (count_a, count_b)
    features_added: list = field(default_factory=list)
    features_removed: list = field(default_factory=list)
    features_changed: list = field(default_factory=list)
    mean_idle_a: float | None = None
    mean_idle_b: float | None = None
    spikes_a: int | None = None
    spikes_b: int | None = None
    checkpoint: str | None = None


def _sub(a, b):
    if a is None or b is None or len(a) != len(b):
        return None
    return [round(float(x) - float(y), 6) for x, y in zip(b, a)]


def diff_datasets(a: DatasetView, b: DatasetView,
                  checkpoint: str | None = None) -> DiffResult:
    res = DiffResult(
        a_name=a.name, b_name=b.name,
        a_version=a.version, b_version=b.version,
        a_fps=a.fps, b_fps=b.fps,
        a_episodes=len(a.episodes), b_episodes=len(b.episodes),
        checkpoint=checkpoint,
    )
    ai, bi = set(a.episodes), set(b.episodes)
    res.added = sorted(bi - ai)
    res.removed = sorted(ai - bi)

    for idx in sorted(ai & bi):
        ea, eb = a.episodes[idx], b.episodes[idx]
        if fingerprint(ea) == fingerprint(eb):
            res.unchanged += 1
            continue
        res.changed.append(EpisodeDelta(
            index=idx,
            old_length=ea.length, new_length=eb.length,
            d_length=eb.length - ea.length,
            tasks_changed=sorted(ea.tasks) != sorted(eb.tasks),
            old_tasks=ea.tasks, new_tasks=eb.tasks,
            d_action_min=_sub(ea.action_min, eb.action_min),
            d_action_max=_sub(ea.action_max, eb.action_max),
            d_action_mean=_sub(ea.action_mean, eb.action_mean),
            d_idle_ratio=(round(eb.idle_ratio - ea.idle_ratio, 4)
                          if ea.idle_ratio is not None and eb.idle_ratio is not None
                          else None),
            d_spike_count=(eb.spike_count - ea.spike_count
                           if ea.spike_count is not None and eb.spike_count is not None
                           else None),
        ))

    # task inventory: task string -> (count in A, count in B)
    def task_hist(ds: DatasetView) -> dict:
        h: dict[str, int] = {}
        for ep in ds.episodes.values():
            for t in ep.tasks or ["<no task>"]:
                h[t] = h.get(t, 0) + 1
        return h

    ha, hb = task_hist(a), task_hist(b)
    for t in sorted(set(ha) | set(hb)):
        res.task_counts[t] = (ha.get(t, 0), hb.get(t, 0))

    # schema drift
    fa, fb = set(a.features), set(b.features)
    res.features_added = sorted(fb - fa)
    res.features_removed = sorted(fa - fb)
    for f in sorted(fa & fb):
        if a.features[f] != b.features[f]:
            res.features_changed.append(
                f"{f}: {a.features[f]} -> {b.features[f]}")

    # quality summary
    def qmean(ds: DatasetView):
        vals = [e.idle_ratio for e in ds.episodes.values()
                if e.idle_ratio is not None]
        sp = [e.spike_count for e in ds.episodes.values()
              if e.spike_count is not None]
        return (round(sum(vals) / len(vals), 4) if vals else None,
                sum(sp) if sp else None)

    res.mean_idle_a, res.spikes_a = qmean(a)
    res.mean_idle_b, res.spikes_b = qmean(b)
    return res
