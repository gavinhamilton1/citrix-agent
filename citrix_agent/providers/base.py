"""Common interface for the model adapters.

The agent loop is identical for every model:

    provider.start(goal, first_frame)
    loop:
        turn = provider.next_turn()           # one model API call
        for call in turn.calls: run call.actions, build a CallResult
        provider.submit(results)              # tool results (+ screenshots) go back

Each adapter owns its own conversation history and wire format, and maps
model coordinates to screen coordinates using the Frame the model last saw.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from ..actions import Action
from ..screen import Frame

SYSTEM_PROMPT = """\
You are operating a Windows PC on behalf of an operator who is watching.
The screenshots show a web browser tab running the Citrix Workspace HTML5 client;
the target web application runs inside that remote Citrix session.

How to work:
- Use only what you can see on screen. Prefer keyboard navigation (Tab, Enter,
  arrow keys, the application's own shortcuts) where it is reliable; click when
  it is clearer.
- The Citrix session is remote, so the screen updates with a delay. After each
  meaningful action, look at a fresh screenshot to confirm it worked before
  moving on. If the screen is still loading, wait and look again.
- Never use browser or operating-system shortcuts: they act on the local
  browser, not on the Citrix session, and can close or reload it. That includes
  ctrl+w, ctrl+t, ctrl+n, ctrl+l, ctrl+r, F5, F11, F12, alt+F4, alt+tab,
  alt+left/right, ctrl+plus/minus/0 and the Windows key. Such actions are
  blocked and will return an error.
- Do not type passwords or other credentials. If a login prompt, security
  warning, or anything unexpected appears, stop and describe it.
- Do not do anything irreversible (delete, submit payments, send messages)
  unless the task explicitly asks for it.
- When the task is finished, stop calling tools and reply with a short summary
  of what you did and what is now visible on screen that proves it. If you
  cannot complete it, stop and explain why.
"""


@dataclass
class ToolCall:
    id: str
    name: str
    raw: dict[str, Any]
    actions: list[Action] = field(default_factory=list)
    error: Optional[str] = None                 # translation failure -> returned to model
    requires_confirmation: Optional[str] = None # provider-flagged; operator must approve

    def describe(self) -> str:
        if self.error:
            return f"{self.name} -> REJECTED: {self.error}"
        return f"{self.name}: " + "; ".join(a.describe() for a in self.actions)


@dataclass
class ModelTurn:
    calls: list[ToolCall]
    text: str = ""            # narration / reasoning summary shown to the operator
    done: bool = False
    stop_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None   # e.g. a refusal; ends the run


@dataclass
class CallResult:
    call: ToolCall
    ok: bool
    message: str = "OK"
    image_png: Optional[bytes] = None   # screenshot / zoom output
    operator_approved: bool = False     # set when the operator confirmed a flagged call


class Provider(ABC):
    name: str = "base"
    #: True if every tool result must carry a fresh screenshot (Gemini).
    screenshot_after_each_call: bool = False

    def __init__(self, model: str):
        self.model = model
        self.frame: Optional[Frame] = None

    def observe(self, frame: Frame) -> None:
        """Record the latest frame the model will see (used for coordinate mapping)."""
        self.frame = frame

    @abstractmethod
    def start(self, goal: str, frame: Frame) -> None: ...

    @abstractmethod
    def next_turn(self) -> ModelTurn: ...

    @abstractmethod
    def submit(self, results: list[CallResult]) -> None: ...
