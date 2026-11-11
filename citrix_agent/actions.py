"""Provider-neutral action model.

Both model adapters (Claude and Gemini) translate their tool calls into these
`Action` objects, already mapped into *screen* pixel coordinates. The executor
only ever sees `Action`s, so the input layer is identical for both models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

ActionKind = Literal[
    "screenshot",      # capture and return the screen to the model
    "zoom",            # return a magnified crop (Claude only)
    "cursor_position", # report the pointer position (Claude only)
    "click",
    "move",
    "drag",
    "mouse_down",
    "mouse_up",
    "scroll",
    "type",
    "key",
    "hold_key",
    "wait",
    "noop",            # acknowledged but nothing to do (e.g. Gemini open_web_browser)
]

# Kinds that only read the screen or wait; they never send input to Citrix.
OBSERVE_KINDS = {"screenshot", "zoom", "cursor_position", "wait", "noop"}


@dataclass
class Action:
    kind: ActionKind
    # Screen-space coordinates (physical pixels, after model->screen mapping).
    x: Optional[int] = None
    y: Optional[int] = None
    start_x: Optional[int] = None   # drag start
    start_y: Optional[int] = None
    button: Literal["left", "right", "middle"] = "left"
    clicks: int = 1
    text: Optional[str] = None      # text to type
    keys: list[str] = field(default_factory=list)       # normalised key combo
    modifiers: list[str] = field(default_factory=list)  # held during click/scroll
    direction: Optional[Literal["up", "down", "left", "right"]] = None
    amount: int = 0                 # scroll wheel notches
    repeat: int = 1
    duration: float = 0.0           # seconds, for wait / hold_key
    region: Optional[tuple[int, int, int, int]] = None  # zoom box, screen coords
    note: str = ""                  # human-readable origin, for logs

    def describe(self) -> str:
        parts = [self.kind]
        if self.kind == "click":
            parts = [f"{self.button}_click" + (f" x{self.clicks}" if self.clicks > 1 else "")]
        if self.x is not None:
            parts.append(f"@({self.x},{self.y})")
        if self.start_x is not None:
            parts.append(f"from({self.start_x},{self.start_y})")
        if self.modifiers:
            parts.append("mods=" + "+".join(self.modifiers))
        if self.keys:
            parts.append("+".join(self.keys) + (f" x{self.repeat}" if self.repeat > 1 else ""))
        if self.text is not None:
            shown = self.text if len(self.text) <= 60 else self.text[:57] + "..."
            parts.append(repr(shown))
        if self.direction:
            parts.append(f"{self.direction} {self.amount}")
        if self.duration:
            parts.append(f"{self.duration:.1f}s")
        if self.region:
            parts.append(f"region={self.region}")
        return " ".join(parts)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v not in (None, [], "", 0, 0.0) or k == "kind"}
