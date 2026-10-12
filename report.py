"""Human-readable text and HTML reports for a DiffResult."""
from __future__ import annotations

import html

from .differ import DiffResult, EpisodeDelta


def _fmt_vec(v, nd=4):
    if v is None:
        return "n/a"
    return "[" + ", ".join(f"{x:+.{nd}f}" for x in v) + "]"


def render_text(r: DiffResult) -> str:
    L: list[str] = []
    L.append("=" * 72)
    L.append("robot-data-diff  —  episode-aware dataset diff")
    L.append("=" * 72)
    L.append(f"A: {r.a_name}   ({r.a_version}, {r.a_episodes} episodes"
             + (f", {r.a_fps} fps" if r.a_fps else "") + ")")
    L.append(f"B: {r.b_name}   ({r.b_version}, {r.b_episodes} episodes"
             + (f", {r.b_fps} fps" if r.b_fps else "") + ")")
    if r.checkpoint:
        L.append(f"policy checkpoint: {r.checkpoint}")
    L.append("")

    L.append(f"Episodes: {r.a_episodes} -> {r.b_episodes} "
             f"(+{len(r.added)} added, -{len(r.removed)} removed, "
             f"~{len(r.changed)} changed, ={r.unchanged} unchanged)")
    if r.added:
        L.append(f"  added:   {r.added}")
    if r.removed:
        L.append(f"  removed: {r.removed}")
    L.append("")

    if r.changed:
        L.append(f"CHANGED EPISODES ({len(r.changed)}):")
        for d in r.changed:
            L.append(f"  [ep {d.index}] length {d.old_length} -> {d.new_length} "
                     f"({d.d_length:+d} frames)")
            if d.tasks_changed:
                L.append(f"           tasks: {d.old_tasks} -> {d.new_tasks}")
            if d.d_action_min is not None:
                L.append(f"           d(action_min)  {_fmt_vec(d.d_action_min)}")
            if d.d_action_max is not None:
                L.append(f"           d(action_max)  {_fmt_vec(d.d_action_max)}")
            if d.d_action_mean is not None:
                L.append(f"           d(action_mean) {_fmt_vec(d.d_action_mean)}")
            if d.d_idle_ratio is not None:
                L.append(f"           d(idle_ratio)  {d.d_idle_ratio:+.4f}")
            if d.d_spike_count is not None:
                L.append(f"           d(spikes)      {d.d_spike_count:+d}")
        L.append("")
    else:
        L.append("No changed episodes (same index + identical stats).")
        L.append("")

    if r.task_counts:
        L.append("TASK INVENTORY (episodes per task: A -> B):")
        for t, (ca, cb) in r.task_counts.items():
            mark = "  *" if ca != cb else "   "
            L.append(f"{mark} {t[:60]:62s} {ca:5d} -> {cb:5d}")
        L.append("")

    if r.features_added or r.features_removed or r.features_changed:
        L.append("SCHEMA DRIFT:")
        for f in r.features_added:
            L.append(f"  + {f}")
        for f in r.features_removed:
            L.append(f"  - {f}")
        for f in r.features_changed:
            L.append(f"  ~ {f}")
        L.append("")
    else:
        L.append("Schema: identical feature sets.")
        L.append("")

    if r.mean_idle_a is not None:
        L.append("QUALITY SUMMARY (heuristic, CPU-only):")
        L.append(f"  mean idle-frame ratio: {r.mean_idle_a:.4f} -> {r.mean_idle_b:.4f} "
                 f"({r.mean_idle_b - r.mean_idle_a:+.4f})")
        L.append(f"  total action spikes:   {r.spikes_a} -> {r.spikes_b} "
                 f"({r.spikes_b - r.spikes_a:+d})")
        L.append("")

    L.append("=" * 72)
    return "\n".join(L)


def render_html(r: DiffResult) -> str:
    e = html.escape

    def changed_rows():
        rows = []
        for d in r.changed:
            rows.append(
                f"<tr><td>{d.index}</td><td>{d.old_length} &rarr; {d.new_length} "
                f"({d.d_length:+d})</td>"
                f"<td>{_fmt_vec(d.d_action_min)}</td>"
                f"<td>{_fmt_vec(d.d_action_max)}</td>"
                f"<td>{d.d_idle_ratio if d.d_idle_ratio is not None else 'n/a'}</td>"
                f"<td>{d.d_spike_count if d.d_spike_count is not None else 'n/a'}</td></tr>"
            )
        return "\n".join(rows) or '<tr><td colspan="6">none</td></tr>'

    def task_rows():
        rows = []
        for t, (ca, cb) in r.task_counts.items():
            cls = ' class="hl"' if ca != cb else ""
            rows.append(f'<tr{cls}><td>{e(t[:80])}</td><td>{ca}</td><td>{cb}</td></tr>')
        return "\n".join(rows)

    ckpt = f"<p><b>Policy checkpoint:</b> {e(r.checkpoint)}</p>" if r.checkpoint else ""
    qual = ""
    if r.mean_idle_a is not None:
        qual = (f"<p>Mean idle-frame ratio: {r.mean_idle_a:.4f} &rarr; "
                f"{r.mean_idle_b:.4f} ({r.mean_idle_b - r.mean_idle_a:+.4f})<br>"
                f"Total action spikes: {r.spikes_a} &rarr; {r.spikes_b} "
                f"({r.spikes_b - r.spikes_a:+d})</p>")

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>robot-data-diff report</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 1000px; margin: 2em auto; padding: 0 1em; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; font-size: 14px; }}
th {{ background: #f4f4f4; }} .hl {{ background: #fff8e1; }}
code {{ background: #f4f4f4; padding: 1px 5px; border-radius: 3px; }}
h1 {{ font-size: 1.4em; }} h2 {{ font-size: 1.1em; margin-top: 1.6em; }}
</style></head><body>
<h1>robot-data-diff report</h1>
<p><b>A:</b> {e(r.a_name)} <code>{e(r.a_version)}</code> ({r.a_episodes} episodes)<br>
<b>B:</b> {e(r.b_name)} <code>{e(r.b_version)}</code> ({r.b_episodes} episodes)</p>
{ckpt}
<h2>Episode inventory</h2>
<p>{r.a_episodes} &rarr; {r.b_episodes} episodes &mdash;
+{len(r.added)} added, &minus;{len(r.removed)} removed,
~{len(r.changed)} changed, ={r.unchanged} unchanged</p>
<p>Added: <code>{e(str(r.added))}</code><br>Removed: <code>{e(str(r.removed))}</code></p>
<h2>Changed episodes</h2>
<table><tr><th>ep</th><th>length</th><th>&Delta;action_min</th>
<th>&Delta;action_max</th><th>&Delta;idle</th><th>&Delta;spikes</th></tr>
{changed_rows()}</table>
<h2>Task inventory</h2>
<table><tr><th>task</th><th>A</th><th>B</th></tr>{task_rows()}</table>
<h2>Schema drift</h2>
<p>Added: <code>{e(str(r.features_added))}</code><br>
Removed: <code>{e(str(r.features_removed))}</code><br>
Changed: <code>{e(str(r.features_changed))}</code></p>
<h2>Quality summary</h2>
{qual}
</body></html>
"""
