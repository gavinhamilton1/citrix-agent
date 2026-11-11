"""Attended-run controls: a global kill hotkey and operator approval."""
from __future__ import annotations

import threading
from typing import Literal

Decision = Literal["approve", "skip", "quit", "auto"]


class KillSwitch:
    """Global hotkey (default ctrl+alt+k) that stops the agent at the next check.

    pyautogui's FAILSAFE (mouse slammed into a screen corner) is a second,
    independent stop that works even if this listener is not running.
    """

    def __init__(self, hotkey: str = "<ctrl>+<alt>+k"):
        self.hotkey = hotkey
        self.event = threading.Event()
        self._listener = None

    def start(self) -> None:
        try:
            from pynput import keyboard
        except Exception as e:  # pragma: no cover - depends on platform
            print(f"[safety] kill hotkey unavailable ({e}); use the mouse-corner failsafe")
            return
        self._listener = keyboard.GlobalHotKeys({self.hotkey: self.trigger})
        self._listener.daemon = True
        self._listener.start()

    def trigger(self) -> None:
        if not self.event.is_set():
            print("\n[safety] KILL SWITCH - stopping")
        self.event.set()

    @property
    def triggered(self) -> bool:
        return self.event.is_set()

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()


class Operator:
    """Console approval for each proposed action.

    In step mode every tool call is shown and must be approved. In auto mode
    only calls the model itself flags as needing confirmation are shown.
    """

    def __init__(self, mode: Literal["step", "auto"] = "step"):
        self.mode = mode

    def review(self, description: str, flagged: str | None = None) -> Decision:
        if self.mode == "auto" and not flagged:
            return "approve"
        if flagged:
            print(f"  !! model asks for confirmation: {flagged}")
        prompt = "  [Enter]=run  s=skip  a=auto from now  q=quit > "
        while True:
            ans = input(f"  next: {description}\n{prompt}").strip().lower()
            if ans in ("", "y", "yes"):
                return "approve"
            if ans == "s":
                return "skip"
            if ans == "q":
                return "quit"
            if ans == "a":
                if flagged:
                    print("  (flagged actions always need an explicit Enter)")
                    continue
                self.mode = "auto"
                return "auto"
