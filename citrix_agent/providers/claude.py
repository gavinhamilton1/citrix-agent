"""Claude adapter: Messages API with the computer use toolset.

Wire format (computer_toolset_20260801):
- one `tools` entry {"type": "computer_toolset_20260801"} - no name, no display size
- each action arrives as its own `tool_use` block whose `name` is the member
  (left_click, type, key, screenshot, zoom, ...) with toolset_name="computer";
  several can arrive in one turn (a batch)
- every result echoes toolset_name="computer"; only screenshot/zoom carry images
- coordinates are pixels of the screenshot we sent; we map them back to screen
- full history (including thinking blocks) is sent back unchanged every turn;
  old screenshots are NOT pruned client-side (that would invalidate the
  model's preserved thinking), so screenshots are kept small instead.
"""
from __future__ import annotations

import base64
from typing import Any

from ..actions import Action
from ..keys import KeyError_, parse_combo
from ..screen import Frame
from .base import SYSTEM_PROMPT, CallResult, ModelTurn, Provider, ToolCall

DEFAULT_MODEL = "claude-opus-5-5"
TOOLSET = {"type": "computer_toolset_20260801"}


def _xy(frame: Frame, coord) -> tuple[int, int]:
    if not isinstance(coord, (list, tuple)) or len(coord) != 2:
        raise ValueError(f"Bad coordinate {coord!r}")
    return frame.model_to_screen(float(coord[0]), float(coord[1]))


def _mods(text: str | None) -> list[str]:
    return parse_combo(text) if text else []


def translate(name: str, inp: dict[str, Any], frame: Frame) -> list[Action]:
    """Map one computer-toolset member call to neutral Actions (screen coords)."""
    note = f"claude:{name}"
    clicks = {"left_click": ("left", 1), "right_click": ("right", 1),
              "middle_click": ("middle", 1), "double_click": ("left", 2),
              "triple_click": ("left", 3)}
    if name == "screenshot":
        return [Action("screenshot", note=note)]
    if name == "zoom":
        x0, y0, x1, y1 = inp["region"]
        sx0, sy0 = frame.model_to_screen(x0, y0)
        sx1, sy1 = frame.model_to_screen(x1, y1)
        return [Action("zoom", region=(sx0, sy0, sx1 + 1, sy1 + 1), note=note)]
    if name in clicks:
        button, n = clicks[name]
        x = y = None
        if inp.get("coordinate") is not None:
            x, y = _xy(frame, inp["coordinate"])
        return [Action("click", x=x, y=y, button=button, clicks=n,
                       modifiers=_mods(inp.get("text")), note=note)]
    if name == "left_click_drag":
        sx, sy = _xy(frame, inp["start_coordinate"])
        x, y = _xy(frame, inp["coordinate"])
        return [Action("drag", start_x=sx, start_y=sy, x=x, y=y,
                       modifiers=_mods(inp.get("text")), note=note)]
    if name == "mouse_move":
        x, y = _xy(frame, inp["coordinate"])
        return [Action("move", x=x, y=y, note=note)]
    if name == "left_mouse_down":
        return [Action("mouse_down", note=note)]
    if name == "left_mouse_up":
        return [Action("mouse_up", note=note)]
    if name == "cursor_position":
        return [Action("cursor_position", note=note)]
    if name == "scroll":
        x = y = None
        if inp.get("coordinate") is not None:
            x, y = _xy(frame, inp["coordinate"])
        return [Action("scroll", x=x, y=y, direction=inp["scroll_direction"],
                       amount=int(inp.get("scroll_amount", 3)),
                       modifiers=_mods(inp.get("text")), note=note)]
    if name == "type":
        return [Action("type", text=inp["text"], note=note)]
    if name == "key":
        return [Action("key", keys=parse_combo(inp["text"]),
                       repeat=int(inp.get("repeat", 1)), note=note)]
    if name == "hold_key":
        return [Action("hold_key", keys=parse_combo(inp["text"]),
                       duration=float(inp.get("duration", 1)), note=note)]
    if name == "wait":
        return [Action("wait", duration=float(inp.get("duration", 1)), note=note)]
    raise ValueError(f"Unsupported computer action {name!r}")


def _image_block(png: bytes) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": base64.b64encode(png).decode()}}


class ClaudeProvider(Provider):
    name = "claude"
    screenshot_after_each_call = False   # Claude asks for screenshots itself

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "medium",
                 use_fallbacks: bool = True, max_tokens: int = 16000, client=None):
        super().__init__(model)
        if client is None:
            import anthropic
            client = anthropic.Anthropic()   # ANTHROPIC_API_KEY or `ant auth login`
        self.client = client
        self.effort = effort
        self.use_fallbacks = use_fallbacks
        self.max_tokens = max_tokens
        self.messages: list[dict] = []

    def start(self, goal: str, frame: Frame) -> None:
        self.observe(frame)
        self.messages = [{"role": "user", "content": [
            {"type": "text", "text": f"Task: {goal}\n\nThis is the current screen."},
            _image_block(frame.model_png()),
        ]}]

    def _create(self):
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            tools=[TOOLSET],
            messages=self.messages,
            thinking={"type": "adaptive", "display": "summarized"},  # readable reasoning for the operator
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},   # history grows every turn; cache the prefix
        )
        if self.use_fallbacks:
            # If a safety classifier declines, the API retries on a fallback model in the same call.
            return self.client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        return self.client.messages.create(**kwargs)

    def next_turn(self) -> ModelTurn:
        resp = self._create()
        self.messages.append({"role": "assistant", "content": resp.content})

        texts, calls = [], []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text" and block.text.strip():
                texts.append(block.text.strip())
            elif btype == "thinking" and getattr(block, "thinking", ""):
                texts.append("(thinking) " + block.thinking.strip())
            elif btype == "fallback":
                texts.append(f"(served by fallback model {block.to.model})")
            elif btype == "tool_use" and getattr(block, "toolset_name", None) == "computer":
                inp = dict(block.input or {})
                call = ToolCall(id=block.id, name=block.name, raw=inp)
                try:
                    call.actions = translate(block.name, inp, self.frame)
                except (KeyError, KeyError_, ValueError, TypeError) as e:
                    call.error = str(e)
                calls.append(call)

        u = resp.usage
        usage = {k: getattr(u, k, None) for k in
                 ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")}
        turn = ModelTurn(calls=calls, text="\n".join(texts), stop_reason=resp.stop_reason or "", usage=usage)
        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            turn.error = f"Model declined (category: {getattr(details, 'category', None)})"
        elif resp.stop_reason == "max_tokens" and not calls:
            turn.error = "Response hit max_tokens before choosing an action"
        elif not calls:
            turn.done = True
        return turn

    def submit(self, results: list[CallResult]) -> None:
        content = []
        for r in results:
            block: dict[str, Any] = {"type": "tool_result", "tool_use_id": r.call.id,
                                     "toolset_name": "computer"}
            if r.ok and r.image_png is not None:
                block["content"] = [_image_block(r.image_png)]
            else:
                block["content"] = r.message
                if not r.ok:
                    block["is_error"] = True
            content.append(block)
        self.messages.append({"role": "user", "content": content})
