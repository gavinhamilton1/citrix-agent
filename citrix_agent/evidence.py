"""Per-run evidence: annotated screenshots, a JSONL step log, a summary and an HTML report."""
from __future__ import annotations

import html
import json
import time
from pathlib import Path
from typing import Any

from PIL import ImageDraw

from .actions import Action
from .screen import Frame


class RunLog:
    def __init__(self, runs_dir: str | Path, provider: str, model: str, goal: str, mode: str):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = Path(runs_dir) / f"{stamp}_{provider}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta = {"provider": provider, "model": model, "goal": goal, "mode": mode,
                     "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.t0 = time.time()
        self.steps: list[dict[str, Any]] = []
        self._shot = 0
        self.usage_totals: dict[str, int] = {}

    def save_frame(self, frame: Frame, label: str, actions: list[Action] | None = None) -> str:
        """Save the full-res capture, with markers where the agent is about to act."""
        self._shot += 1
        name = f"{self._shot:03d}_{label}.png"
        img = frame.full.copy()
        if actions:
            d = ImageDraw.Draw(img)
            r = frame.region
            for a in actions:
                for (x, y, colour) in ((a.start_x, a.start_y, "blue"), (a.x, a.y, "red")):
                    if x is None:
                        continue
                    px, py = x - r.left, y - r.top
                    d.ellipse((px - 14, py - 14, px + 14, py + 14), outline=colour, width=4)
                    d.line((px - 22, py, px + 22, py), fill=colour, width=2)
                    d.line((px, py - 22, px, py + 22), fill=colour, width=2)
        img.save(self.dir / name)
        return name

    def log_step(self, record: dict[str, Any]) -> None:
        record["t"] = round(time.time() - self.t0, 2)
        self.steps.append(record)
        for k, v in (record.get("usage") or {}).items():
            if isinstance(v, int):
                self.usage_totals[k] = self.usage_totals.get(k, 0) + v
        with open(self.dir / "steps.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def finish(self, outcome: str, message: str) -> Path:
        summary = {**self.meta, "outcome": outcome, "message": message,
                   "duration_s": round(time.time() - self.t0, 1),
                   "model_turns": sum(1 for s in self.steps if s.get("type") == "turn"),
                   "actions": sum(len(s.get("calls", [])) for s in self.steps if s.get("type") == "turn"),
                   "usage": self.usage_totals}
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2))
        self._write_html(summary)
        return self.dir

    def _write_html(self, summary: dict) -> None:
        e = html.escape
        rows = []
        for s in self.steps:
            if s.get("type") != "turn":
                continue
            calls = "".join(
                f"<li class='{'bad' if not c.get('ok', True) else ''}'>{e(c['description'])}"
                f"{' &rarr; ' + e(c.get('result', '')) if c.get('result') else ''}</li>"
                for c in s.get("calls", []))
            shots = "".join(f"<a href='{e(n)}'><img src='{e(n)}' loading='lazy'></a>" for n in s.get("screens", []))
            rows.append(f"<section><h3>Turn {s['step']} <small>{s['t']}s</small></h3>"
                        f"<pre>{e(s.get('text', ''))}</pre><ul>{calls}</ul><div class='shots'>{shots}</div></section>")
        doc = f"""<!doctype html><meta charset="utf-8"><title>Run {e(self.dir.name)}</title>
<style>body{{font:14px system-ui;margin:24px;max-width:1100px}}pre{{white-space:pre-wrap;background:#f4f4f4;padding:8px}}
.shots img{{max-width:48%;border:1px solid #ccc;margin:2px}}li.bad{{color:#b00}}section{{border-top:1px solid #ddd;margin-top:16px}}</style>
<h1>{e(summary['provider'])} / {e(summary['model'])}: {e(summary['outcome'])}</h1>
<p><b>Goal:</b> {e(summary['goal'])}</p><p><b>Result:</b> {e(summary['message'])}</p>
<p>{summary['model_turns']} model turns, {summary['actions']} actions, {summary['duration_s']}s, usage {e(json.dumps(summary['usage']))}</p>
{''.join(rows)}"""
        (self.dir / "report.html").write_text(doc, encoding="utf-8")
