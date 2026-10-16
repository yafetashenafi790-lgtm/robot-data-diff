# robot-data-diff

**Episode-aware diff for LeRobot robot-training datasets.**

You retrained the policy and it got worse. Was it the code — or the data?
`robot-data-diff` answers *"what changed in the data since the last good run?"*
by diffing two revisions of a dataset, episode by episode.

Free, CPU-only, pure Python. No GPU, no uploads, no accounts needed.

## Install

```bash
pip install robot-data-diff
```

From source (for contributors):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Dependencies (all free): `pyarrow`, `pandas`, `numpy`, `huggingface_hub`.

> Note: this sandbox's egress proxy breaks `httpx`'s `no_proxy` parsing
> (IPv6 entries). The tool strips `no_proxy`/`NO_PROXY` for its own
> Hugging Face calls; your shell is untouched.

## Usage

```bash
# two local dataset directories
robot-data-diff diff ./my_dataset_v1 ./my_dataset_v2

# Hugging Face ids (videos are never downloaded — meta + data parquet only)
robot-data-diff diff lerobot/pusht ./my_local_copy

# same repo, two revisions
robot-data-diff diff lerobot/pusht@main lerobot/pusht@<commit-sha>

# link the diff to a policy checkpoint, and write an HTML report
robot-data-diff diff v1 v2 --checkpoint ckpt-42 --html report.html

# metadata-only mode (skips per-episode quality stats; fastest)
robot-data-diff diff v1 v2 --no-quality
```

Each side can be a local path or a Hugging Face dataset id
(`org/repo`, optionally `org/repo@revision` or `hf://org/repo@revision`).

## What the report shows

- **Episode inventory:** total counts, episodes added / removed / changed
  (same index, different data fingerprint)
- **Per-episode deltas** for changed episodes: frame count, action
  min/max/mean per dimension, idle-frame-ratio delta, action-spike delta
- **Task inventory:** episodes per task, old → new
- **Schema drift:** features added / removed / changed
- **Quality summary:** mean idle-frame ratio and total action spikes,
  old → new (lightweight CPU heuristics, not a full validator)

## Supported formats

- **LeRobot v3.0** — verified against the real `lerobot/pusht` dataset
  (parquet episode metadata + data chunks + `meta/info.json`).
- **LeRobot v2.1** — implemented against the documented v2.1 layout
  (`meta/episodes.jsonl`, `meta/episodes_stats.jsonl`, per-episode data
  files). Tested with a synthetic v2 fixture (`tests/make_v2_fixture.py`);
  no public v2 dataset was reachable at build time, so treat v2 support
  as beta and report issues.

## Example

`demo/diff-demo-output.txt` is the real output of diffing `lerobot/pusht`
(206 episodes) against a modified copy: episodes 200–205 removed, episodes
0–4 had their actions scaled ×1.5, episode 206 added. The tool reports
exactly that:

```
Episodes: 206 -> 201 (+1 added, -6 removed, ~5 changed, =195 unchanged)
  added:   [206]
  removed: [200, 201, 202, 203, 204, 205]

CHANGED EPISODES (5):
  [ep 0] length 161 -> 161 (+0 frames)
           d(action_min)  [+46.5000, +35.5000]
           d(action_max)  [+187.5000, +224.5000]
           d(action_mean) [+104.5031, +144.1553]
           d(idle_ratio)  +0.0000
           d(spikes)      +0
```

See `demo/diff-demo-report.html` for the HTML version.

## Reproduce the demo

```bash
source .venv/bin/activate
python tests/make_modified_fixture.py      # builds testdata/pusht_v2
robot-data-diff diff testdata/pusht testdata/pusht_v2 \
    --checkpoint ckpt-42 --html demo/diff-demo-report.html
```

## Honest limits

- This is a **diff**, not a validator: it tells you *what changed* between
  revisions, not whether your data is good. (The validator lane already has
  6+ tools — that's why this exists.)
- "Changed" is detected via a stats fingerprint (length, action
  min/max/mean, idle ratio, spike count). Bit-identical data with reordered
  episodes shows as added+removed, not moved — episode identity is the index.
- Remote datasets download `meta/**` + `data/**`; very large datasets will
  take a while on first run (cached afterwards via `--cache-dir`).
- RLDS / Zarr / HDF5 / rosbag support is out of scope for v0.1 (LeRobot only).

## Related work

- [openbot-data](https://github.com/openbotai/openbot-data) (MIT) — a broader robot-data toolkit whose `openbot-data diff` compares two *snapshot files* (`snapshot` first, then `diff`, with `--fail-on breaking` for CI-style breaking-change detection). robot-data-diff is narrower on purpose: it diffs two dataset revisions *directly* — local dirs or Hugging Face ids, no snapshot step — episode by episode, with per-episode stat deltas.

## License

Apache-2.0
