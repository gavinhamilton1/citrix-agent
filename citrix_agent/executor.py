"""The only component that sends input. Validates every action against the
safety policy, then drives pyautogui (synthetic Windows SendInput events,
which the browser and the Citrix HTML5 client treat as real user input).
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Protocol

from .actions import Action
from .keys import DEFAULT_BLOCKED, blocked_reason
from .screen import Region


class ActionRejected(Exception):
    """The action broke the safety policy and was not executed."""


class InputBackend(Protocol):
    def move(self, x: int, y: int) -> None: ...
    def click(self, x: int, y: int, button: str, clicks: int) -> None: ...
    def mouse_down(self, button: str) -> None: ...
    def mouse_up(self, button: str) -> None: ...
    def drag(self, x0: int, y0: int, x1: int, y1: int) -> None: ...
    def scroll(self, notches: int, horizontal: bool) -> None: ...
    def key_down(self, key: str) -> None: ...
    def key_up(self, key: str) -> None: ...
    def hotkey(self, keys: list[str]) -> None: ...
    def write(self, text: str, interval: float) -> None: ...
    def position(self) -> tuple[int, int]: ...


class PyAutoGuiBackend:
    def __init__(self, pause: float = 0.05):
        import pyautogui
        pyautogui.FAILSAFE = True   # slam the mouse into a screen corner to abort
        pyautogui.PAUSE = pause
        self.p = pyautogui
        # On Windows pyautogui passes the value straight to the wheel event,
        # where one notch is 120 units.
        self.wheel_unit = 120 if sys.platform == "win32" else 1

    def move(self, x, y): self.p.moveTo(x, y, duration=0.1)
    def click(self, x, y, button, clicks): self.p.click(x, y, clicks=clicks, interval=0.08, button=button)
    def mouse_down(self, button): self.p.mouseDown(button=button)
    def mouse_up(self, button): self.p.mouseUp(button=button)

    def drag(self, x0, y0, x1, y1):
        self.p.moveTo(x0, y0, duration=0.1)
        self.p.mouseDown()
        self.p.moveTo(x1, y1, duration=0.4)   # slow enough for Citrix to see the motion
        self.p.mouseUp()

    def scroll(self, notches, horizontal):
        if horizontal:
            if hasattr(self.p, "hscroll") and sys.platform.startswith("linux"):
                self.p.hscroll(notches * self.wheel_unit)
            else:  # Windows: shift+wheel scrolls horizontally in browsers
                self.p.keyDown("shift")
                try:
                    self.p.scroll(notches * self.wheel_unit)
                finally:
                    self.p.keyUp("shift")
        else:
            self.p.scroll(notches * self.wheel_unit)

    def key_down(self, key): self.p.keyDown(key)
    def key_up(self, key): self.p.keyUp(key)
    def hotkey(self, keys): self.p.hotkey(*keys, interval=0.05)
    def write(self, text, interval): self.p.write(text, interval=interval)
    def position(self): return tuple(self.p.position())


class DryRunBackend:
    """Logs instead of acting. Useful for checking coordinate mapping safely."""
    def __init__(self): self.calls: list[tuple] = []; self._pos = (0, 0)
    def __getattr__(self, name):
        def rec(*a, **k):
            self.calls.append((name, a))
            if name in ("move", "click"):
                self._pos = (a[0], a[1])
        return rec
    def position(self): return self._pos


@dataclass
class SafetyPolicy:
    blocked_combos: list[frozenset[str]] = field(default_factory=lambda: list(DEFAULT_BLOCKED))
    max_text_len: int = 2000
    max_scroll_notches: int = 25
    max_hold_seconds: float = 5.0
    type_interval: float = 0.04    # Citrix drops keystrokes sent too fast


class Executor:
    def __init__(self, backend: InputBackend, policy: SafetyPolicy | None = None):
        self.backend = backend
        self.policy = policy or SafetyPolicy()

    # --- validation ---------------------------------------------------------
    def validate(self, a: Action, region: Region) -> None:
        for (x, y) in ((a.x, a.y), (a.start_x, a.start_y)):
            if x is not None and not region.contains(x, y):
                raise ActionRejected(f"({x},{y}) is outside the Citrix window {region}")
        combo = list(a.keys)
        if a.kind in ("click", "scroll", "drag") and a.modifiers:
            if "win" in a.modifiers:
                raise ActionRejected("Windows-key modifiers are not allowed")
            if a.kind == "scroll" and "ctrl" in a.modifiers:
                raise ActionRejected("ctrl+scroll zooms the browser and rescales the Citrix canvas")
        if a.kind in ("key", "hold_key"):
            reason = blocked_reason(combo, self.policy.blocked_combos)
            if reason:
                raise ActionRejected(reason)
        if a.kind == "type":
            if not a.text:
                raise ActionRejected("Nothing to type")
            if len(a.text) > self.policy.max_text_len:
                raise ActionRejected("Text is longer than the policy limit")
        if a.kind == "scroll" and abs(a.amount) > self.policy.max_scroll_notches:
            raise ActionRejected("Scroll amount exceeds the policy limit")
        if a.kind == "hold_key" and a.duration > self.policy.max_hold_seconds:
            raise ActionRejected("hold_key duration exceeds the policy limit")

    # --- dispatch -----------------------------------------------------------
    def execute(self, a: Action, region: Region) -> str:
        self.validate(a, region)
        b = self.backend
        if a.kind == "move":
            b.move(a.x, a.y)
        elif a.kind == "click":
            with self._held(a.modifiers):
                if a.x is None:
                    x, y = b.position()
                else:
                    x, y = a.x, a.y
                b.click(x, y, a.button, a.clicks)
        elif a.kind == "drag":
            with self._held(a.modifiers):
                b.drag(a.start_x, a.start_y, a.x, a.y)
        elif a.kind == "mouse_down":
            if a.x is not None:
                b.move(a.x, a.y)
            b.mouse_down(a.button)
        elif a.kind == "mouse_up":
            if a.x is not None:
                b.move(a.x, a.y)
            b.mouse_up(a.button)
        elif a.kind == "scroll":
            if a.x is not None:
                b.move(a.x, a.y)
            sign = 1 if a.direction in ("up", "left") else -1
            with self._held(a.modifiers):
                b.scroll(sign * a.amount, horizontal=a.direction in ("left", "right"))
        elif a.kind == "type":
            b.write(a.text, self.policy.type_interval)
        elif a.kind == "key":
            for _ in range(max(1, a.repeat)):
                b.hotkey(a.keys)
                time.sleep(0.03)
        elif a.kind == "hold_key":
            for k in a.keys:
                b.key_down(k)
            try:
                time.sleep(a.duration)
            finally:
                for k in reversed(a.keys):
                    b.key_up(k)
        else:
            raise ActionRejected(f"Executor cannot run action kind {a.kind!r}")
        return "OK"

    class _Held:
        def __init__(self, backend, keys): self.b, self.keys = backend, keys
        def __enter__(self):
            for k in self.keys: self.b.key_down(k)
        def __exit__(self, *exc):
            for k in reversed(self.keys): self.b.key_up(k)

    def _held(self, keys):
        return Executor._Held(self.backend, keys)
