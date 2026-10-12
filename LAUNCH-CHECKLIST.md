# robot-data-diff — launch checklist (draft for Yafet's review)

Status: built and tested 2026-10-08 on the real `lerobot/pusht` dataset (206 episodes).
Everything below needs Yafet's accounts — nothing is published yet.

## What's in the package (verified 2026-10-08)
- `robot_data_diff/` — 662 lines, pure Python, CPU-only: loader (LeRobot v2.1 + v3.0,
  local path or HF repo id), differ (added/removed/modified episodes + per-episode
  stat deltas), CLI (`robot-data-diff diff v1 v2`), HTML report output
- `pyproject.toml` — name `robot-data-diff`, version 0.1.0, Apache-2.0,
  console entry point `robot-data-diff`, deps: pyarrow, pandas, numpy, huggingface_hub
- `tests/` + `testdata/` (pusht, pusht_v2, toy fixtures), `demo/` (diff-demo-output.txt,
  diff-demo-report.html)

## Steps only Yafet can do
1. **GitHub repo** — create `robot-data-diff` on your account, push this directory.
2. **PyPI** — build verified 2026-10-10: `python -m build` builds cleanly (sdist + wheel) and `twine check dist/*` passes 2/2, no warnings. Publishing is one command with his PyPI account: `twine upload dist/*`. (PyPI name `robot-data-diff` verified free 2026-10-09.)
3. **Announce (optional, your call)** — e.g. Hugging Face LeRobot forum. Draft below,
   nothing posted without your approval.

## Draft announcement (review first — NOT posted)
> robot-data-diff: an episode-aware diff for LeRobot datasets.
>
> Retrain a policy and it gets worse — was it the code or the data? This CLI diffs two revisions of a dataset directly — local dirs or HF repo ids (v2.1 and v3.0), no snapshot step in between — and reports episodes added/removed/modified with per-episode stat deltas. CPU-only, no GPU, videos never downloaded. Free, Apache-2.0.
>
> Happy to get feedback from anyone iterating on datasets.

Note for launch: `openbot-data` (MIT, on PyPI as `openbot-data`) has a snapshot-based diff for breaking-change CI — worth one line in the post pre-empting confusion ("diffs revisions directly, not snapshot files").

New adjacent tools found 2026-10-10 (not competitors — no change needed, just know they exist):
- `acrosson/lerobot-dataset-editor` (GitHub, web UI for reviewing/trimming v3.0 episodes) — an *editor*, complementary: editors create new revisions; robot-data-diff shows what those revisions changed.
- `mulkakhileshmj/robotruth` (GitHub) — eval contracts / A-B eval comparison / failure taxonomy — compares *policy eval results*, not dataset revisions.

**CORRECTION 2026-10-12 — EpisodeVault IS a direct functional competitor (known since
June 2026, missed by earlier lane checks; verified this morning).** Repo:
https://github.com/Rohan-Prabhakar/EpisodeVault (created 2026-06-10, 28 commits,
MIT, `pip install episodevault`), blog post 2026-06-12 (digitado.com.br), demoed on
ROS Discourse. Does episode-level diffing, task-distribution shifts, quality-metric
deltas, anomaly detection, HF Hub diffing, HTML reports. **BUT: 0 stars, 0 forks —
one-person project, zero adoption.** The lane is validated but effectively open; the
"no episode-aware diff exists" claim is dead, use "nobody's using one" instead.
Framing for the announce post: EpisodeVault needs you to adopt its track/commit
workflow (`.episodevault/` store); robot-data-diff diffs any two dataset dirs
directly, no workflow, no setup — lead with that. Expect "this already exists"
replies; they're fair game and also free validation of the idea.

## Decisions needed from Yafet
- Repo name / org on GitHub (default: `robot-data-diff` on his account)
- PyPI now or later
- Announce publicly or keep it quiet while iterating
