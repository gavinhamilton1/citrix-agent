"""CLI: python -m citrix_agent --provider claude --goal "..." --window-title "Citrix"."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="citrix_agent", description=__doc__)
    ap.add_argument("--provider", choices=["claude", "gemini"], required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--goal", help="Plain-English task for the agent")
    g.add_argument("--goal-file", type=Path, help="File containing the task")
    ap.add_argument("--mode", choices=["step", "auto"], default="step",
                    help="step: approve each action (default). auto: run unattended up to --max-steps")
    ap.add_argument("--window-title", help="Substring of the browser window title hosting the Citrix "
                    "session (default: capture the whole primary monitor)")
    ap.add_argument("--max-steps", type=int, default=40, help="Maximum model turns")
    ap.add_argument("--max-long-edge", type=int, default=1568, help="Screenshot long edge sent to the model (px)")
    ap.add_argument("--model", help="Override the provider's default model ID")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"],
                    help="Claude effort level")
    ap.add_argument("--no-fallbacks", action="store_true", help="Claude: disable server-side refusal fallbacks")
    ap.add_argument("--gemini-environment", choices=["browser", "desktop"], default="browser")
    ap.add_argument("--kill-hotkey", default="<ctrl>+<alt>+k", help="pynput GlobalHotKeys syntax")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--dry-run", action="store_true",
                    help="Do not send any input; log and mark intended actions on screenshots")
    args = ap.parse_args(argv)

    goal = args.goal or args.goal_file.read_text(encoding="utf-8").strip()

    from .screen import Screen, make_dpi_aware
    make_dpi_aware()  # before any capture or input, so all coordinates are physical pixels

    from .agent import Agent
    from .evidence import RunLog
    from .executor import DryRunBackend, Executor, PyAutoGuiBackend
    from .providers import make_provider
    from .safety import KillSwitch, Operator

    provider = make_provider(args.provider, model=args.model, effort=args.effort,
                             use_fallbacks=not args.no_fallbacks, environment=args.gemini_environment)
    screen = Screen(window_title=args.window_title, max_long_edge=args.max_long_edge)
    backend = DryRunBackend() if args.dry_run else PyAutoGuiBackend()
    log = RunLog(args.runs_dir, provider.name, provider.model, goal, args.mode + (" dry-run" if args.dry_run else ""))
    kill = KillSwitch(args.kill_hotkey)
    kill.start()

    print(f"Provider: {provider.name} ({provider.model})   mode: {args.mode}   evidence: {log.dir}")
    print(f"STOP: press {args.kill_hotkey}, or slam the mouse into a screen corner, or Ctrl+C here.")
    if args.provider == "claude" and not args.no_fallbacks:
        print("Note: Claude server-side refusal fallbacks are on (--no-fallbacks to disable).")

    agent = Agent(provider, screen, Executor(backend), log, Operator(args.mode), kill, max_steps=args.max_steps)
    result = agent.run(goal)
    kill.stop()

    print(f"\nOutcome: {result.outcome}\n{result.message}\nEvidence: {result.run_dir}/report.html")
    return 0 if result.outcome == "model_reported_done" else 1


if __name__ == "__main__":
    sys.exit(main())
