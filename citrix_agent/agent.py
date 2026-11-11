"""The perceive -> decide -> act -> verify loop, shared by both providers."""
from __future__ import annotations

import time
from dataclasses import dataclass

from .actions import OBSERVE_KINDS
from .evidence import RunLog
from .executor import ActionRejected, Executor
from .providers.base import CallResult, Provider
from .safety import KillSwitch, Operator

NOT_EXECUTED = "Not executed: an earlier computer action in this turn failed."


class Abort(Exception):
    def __init__(self, outcome: str, message: str):
        super().__init__(message)
        self.outcome, self.message = outcome, message


@dataclass
class RunResult:
    outcome: str     # model_reported_done | stopped | declined | error | max_steps
    message: str
    run_dir: str


class Agent:
    def __init__(self, provider: Provider, screen, executor: Executor, log: RunLog,
                 operator: Operator, kill: KillSwitch, max_steps: int = 40,
                 settle_timeout: float = 10.0):
        self.p, self.screen, self.ex, self.log = provider, screen, executor, log
        self.op, self.kill = operator, kill
        self.max_steps, self.settle_timeout = max_steps, settle_timeout
        self._turn_screens: list[str] = []

    # --- helpers ------------------------------------------------------------
    def _check_kill(self):
        if self.kill.triggered:
            raise Abort("stopped", "Kill switch pressed by operator")

    def _fresh_frame(self, label: str):
        self.screen.wait_until_stable(timeout=self.settle_timeout)
        frame = self.screen.capture()
        self.p.observe(frame)
        self._turn_screens.append(self.log.save_frame(frame, label))
        return frame

    def _sleep(self, seconds: float):
        end = time.time() + min(seconds, 30.0)
        while time.time() < end:
            self._check_kill()
            time.sleep(0.1)

    def _run_call(self, call) -> CallResult:
        """Execute every Action of one tool call. Raises on failure."""
        image = None
        msg = "OK"
        input_actions = [a for a in call.actions if a.kind not in OBSERVE_KINDS]
        if input_actions:
            # Evidence: where the agent is about to click, drawn on what it saw.
            self._turn_screens.append(self.log.save_frame(self.p.frame, f"act_{call.name}", input_actions))
        for a in call.actions:
            self._check_kill()
            frame = self.p.frame
            if a.kind == "screenshot":
                image = self._fresh_frame("screenshot").model_png()
            elif a.kind == "zoom":
                image = frame.zoom_png(a.region, self.screen.max_long_edge)
            elif a.kind == "cursor_position":
                mx, my = frame.screen_to_model(*self.ex.backend.position())
                msg = f"X={mx},Y={my}"
            elif a.kind == "wait":
                self._sleep(a.duration)
            elif a.kind == "noop":
                msg = a.note or "OK"
            else:
                self.screen.focus()
                self.ex.execute(a, frame.region)
                time.sleep(0.15)
        return CallResult(call=call, ok=True, message=msg, image_png=image)

    # --- main loop ----------------------------------------------------------
    def run(self, goal: str) -> RunResult:
        try:
            outcome, message = self._loop(goal)
        except Abort as a:
            outcome, message = a.outcome, a.message
        except KeyboardInterrupt:
            outcome, message = "stopped", "Interrupted from the console"
        except Exception as e:
            if type(e).__name__ == "FailSafeException":
                outcome, message = "stopped", "pyautogui failsafe: mouse moved to a screen corner"
            else:
                outcome, message = "error", f"{type(e).__name__}: {e}"
        run_dir = self.log.finish(outcome, message)
        return RunResult(outcome, message, str(run_dir))

    def _loop(self, goal: str) -> tuple[str, str]:
        self.screen.focus()
        first = self._fresh_frame("start")
        self.p.start(goal, first)
        self.log.log_step({"type": "start", "goal": goal, "screens": self._turn_screens})

        for step in range(1, self.max_steps + 1):
            self._check_kill()
            self._turn_screens = []
            print(f"\n=== turn {step} ({self.p.name}) - asking model...")
            turn = self.p.next_turn()
            if turn.text:
                print("  " + turn.text.replace("\n", "\n  "))

            record = {"type": "turn", "step": step, "text": turn.text, "stop_reason": turn.stop_reason,
                      "usage": turn.usage, "calls": [], "screens": self._turn_screens}

            if turn.error:
                self.log.log_step(record)
                return "declined" if "declined" in turn.error else "error", turn.error
            if turn.done:
                self.log.log_step(record)
                return "model_reported_done", turn.text or "(no summary)"

            results: list[CallResult] = []
            failed = False
            for call in turn.calls:
                desc = call.describe()
                entry = {"id": call.id, "name": call.name, "raw": call.raw, "description": desc,
                         "actions": [a.to_dict() for a in call.actions]}
                if failed:
                    res = CallResult(call, ok=False, message=NOT_EXECUTED)
                elif call.error:
                    res = CallResult(call, ok=False, message=f"Error: {call.error}")
                    failed = True
                else:
                    observe_only = all(a.kind in OBSERVE_KINDS for a in call.actions)
                    decision = "approve" if (observe_only and not call.requires_confirmation) \
                        else self.op.review(desc, call.requires_confirmation)
                    if decision == "quit":
                        raise Abort("stopped", "Operator quit")
                    if decision == "skip":
                        if call.requires_confirmation:
                            raise Abort("stopped", "Operator declined an action the model flagged as sensitive")
                        res = CallResult(call, ok=False, message="The operator rejected this action. Choose another approach.")
                        failed = True
                    else:
                        print(f"  -> {desc}")
                        try:
                            res = self._run_call(call)
                            res.operator_approved = bool(call.requires_confirmation)
                        except ActionRejected as e:
                            print(f"     blocked: {e}")
                            res = CallResult(call, ok=False, message=f"Error: blocked by safety policy: {e}")
                            failed = True
                if self.p.screenshot_after_each_call:
                    res.image_png = self._fresh_frame(f"after_{call.name}").model_png()
                entry.update(ok=res.ok, result=res.message)
                record["calls"].append(entry)
                results.append(res)

            self.log.log_step(record)
            self.p.submit(results)

        return "max_steps", f"Stopped after {self.max_steps} model turns without finishing"
