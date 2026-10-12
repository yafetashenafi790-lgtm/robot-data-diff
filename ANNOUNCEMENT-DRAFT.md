# robot-data-diff — announcement draft (REVIEW FIRST — nothing posted)

Status: revised 2026-10-12 (Mon) with EpisodeVault-aware framing. This is a draft for
Yafet's review. Nothing here has been posted anywhere; he approves the exact text
before anything goes live.

## Post (e.g. Hugging Face LeRobot forum)

> **robot-data-diff: episode-aware diffs for LeRobot datasets — no setup, no workflow**
>
> Retrain a policy and it gets worse — was it the code or the data? robot-data-diff is
> a CLI that diffs any two revisions of a LeRobot dataset directly: local dirs or HF
> repo ids (v2.1 + v3.0), no tracking setup, no commit workflow — just point it at two
> dirs. Reports episodes added / removed / modified with per-episode stat deltas, plus
> an HTML report. CPU-only, no GPU, videos never downloaded. Free, Apache-2.0.
>
> Similar ideas exist (EpisodeVault), but nobody's using them, and they make you adopt
> their whole track/commit workflow first. This one meets you where your data already is.
>
> Repo + pip package links below. Would love feedback — especially: what's the first
> diff you ran, and did it surprise you?

## Why this framing (from the 2026-10-12 lane correction)
- EpisodeVault (github.com/Rohan-Prabhakar/EpisodeVault, since June 2026) is a real
  functional competitor: episode-level diffs, task-distribution shifts, anomaly
  detection. BUT: 0 stars, 0 forks — zero adoption. The lane is validated, effectively open.
- Do NOT claim "no episode-aware diff exists" — that claim is dead. Claim "nobody's
  using one" instead.
- The differentiator to lead with: EpisodeVault needs you to adopt its track/commit
  workflow (`.episodevault/` store); robot-data-diff diffs any two dataset dirs
  directly, no workflow, no setup.
- Expect "this already exists" replies — they're fair and double as free validation
  of the idea. The pre-emptive paragraph above defuses them.
- Do not over-promise v0.2 integrity checks in this post — v0.2 ships later, and its
  overlap with EpisodeVault's statistical anomaly detection is already accounted for
  (v0.2 keeps only the structural checks nobody owns — see ../robot-data-diff-v0.2-build-spec.md).

## Before posting, fill in
- GitHub repo URL (repo is `robot-data-diff`, currently private — flip to public first)
- PyPI package URL (publish with `twine upload dist/*` first)
