"""Tabulate runs for a Claude vs Gemini comparison: python -m citrix_agent.compare runs/"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(runs_dir: str = "runs") -> None:
    rows = []
    for p in sorted(Path(runs_dir).glob("*/summary.json")):
        s = json.loads(p.read_text())
        u = s.get("usage", {})
        rows.append([p.parent.name, s["provider"], s["model"], s["outcome"], s["model_turns"],
                     s["actions"], s["duration_s"], u.get("input_tokens", ""), u.get("output_tokens", ""),
                     s["goal"][:40]])
    head = ["run", "provider", "model", "outcome", "turns", "actions", "secs", "in_tok", "out_tok", "goal"]
    widths = [max(len(str(r[i])) for r in rows + [head]) for i in range(len(head))]
    for r in [head] + rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


if __name__ == "__main__":
    main(*sys.argv[1:])
