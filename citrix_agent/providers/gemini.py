"""Gemini adapter: Gemini API Computer Use tool (google-genai SDK).

Wire format:
- tool: types.Tool(computer_use=types.ComputerUse(environment=...))
- actions arrive as function_call parts with predefined names (click_at,
  type_text_at, key_combination, scroll_at, ...); coordinates are normalised
  0-1000 over the screenshot, independent of its pixel size
- every function_response must carry a fresh screenshot and a `url`
- a call may carry args["safety_decision"] = {"decision": "require_confirmation"};
  it must be confirmed by the human, and the response then includes
  "safety_acknowledgement": "true"
- Browser-navigation actions are excluded: the Citrix session lives in one tab
  and navigating away would disconnect it.

The computer-use model ID and action set are preview features that change;
verify against https://ai.google.dev/gemini-api/docs/computer-use before a run.
"""
from __future__ import annotations

from typing import Any

from ..actions import Action
from ..keys import KeyError_, parse_combo
from ..screen import Frame
from .base import SYSTEM_PROMPT, CallResult, ModelTurn, Provider, ToolCall

DEFAULT_MODEL = "gemini-2.5-computer-use-preview-10-2025"
EXCLUDED = ["open_web_browser", "navigate", "search", "go_back", "go_forward"]
KEEP_SCREENSHOTS = 3   # older screenshots are stripped from history to bound cost


def translate(name: str, args: dict[str, Any], frame: Frame) -> list[Action]:
    note = f"gemini:{name}"

    def xy(kx="x", ky="y"):
        return frame.normalised_to_screen(float(args[kx]), float(args[ky]))

    if name == "open_web_browser":
        return [Action("noop", note="browser is already open (Citrix tab)")]
    if name == "wait_5_seconds":
        return [Action("wait", duration=5.0, note=note)]
    if name == "click_at":
        x, y = xy()
        return [Action("click", x=x, y=y, note=note)]
    if name == "hover_at":
        x, y = xy()
        return [Action("move", x=x, y=y, note=note)]
    if name == "type_text_at":
        x, y = xy()
        acts = [Action("click", x=x, y=y, note=note)]
        if args.get("clear_before_typing", True):
            acts += [Action("key", keys=["ctrl", "a"], note=note), Action("key", keys=["delete"], note=note)]
        acts.append(Action("type", text=str(args["text"]), note=note))
        if args.get("press_enter", True):
            acts.append(Action("key", keys=["enter"], note=note))
        return acts
    if name == "key_combination":
        return [Action("key", keys=parse_combo(str(args["keys"])), note=note)]
    if name == "scroll_document":
        cx, cy = frame.region.center
        return [Action("scroll", x=cx, y=cy, direction=args["direction"], amount=5, note=note)]
    if name == "scroll_at":
        x, y = xy()
        magnitude = float(args.get("magnitude", 800))
        # magnitude is in normalised units; ~100 screen px per wheel notch
        px = magnitude / 1000.0 * frame.region.height
        return [Action("scroll", x=x, y=y, direction=args["direction"],
                       amount=max(1, min(25, round(px / 100))), note=note)]
    if name == "drag_and_drop":
        sx, sy = xy()
        x, y = xy("destination_x", "destination_y")
        return [Action("drag", start_x=sx, start_y=sy, x=x, y=y, note=note)]
    if name in ("navigate", "search", "go_back", "go_forward"):
        raise ValueError(f"{name} is disabled: navigating the browser would disconnect the Citrix session")
    raise ValueError(f"Unsupported action {name!r}")


class GeminiProvider(Provider):
    name = "gemini"
    screenshot_after_each_call = True

    def __init__(self, model: str = DEFAULT_MODEL, environment: str = "browser",
                 session_url: str = "citrix-html5-session", client=None):
        super().__init__(model)
        from google import genai
        from google.genai import types
        self.types = types
        self.client = client or genai.Client()   # GEMINI_API_KEY / GOOGLE_API_KEY
        env = {"browser": types.Environment.ENVIRONMENT_BROWSER,
               "desktop": types.Environment.ENVIRONMENT_DESKTOP}[environment]
        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[types.Tool(computer_use=types.ComputerUse(
                environment=env, excluded_predefined_functions=EXCLUDED))],
        )
        self.session_url = session_url
        self.contents: list = []

    def start(self, goal: str, frame: Frame) -> None:
        t = self.types
        self.observe(frame)
        self.contents = [t.Content(role="user", parts=[
            t.Part(text=f"Task: {goal}\n\nThis is the current screen."),
            t.Part.from_bytes(data=frame.model_png(), mime_type="image/png"),
        ])]

    def next_turn(self) -> ModelTurn:
        resp = self.client.models.generate_content(
            model=self.model, contents=self.contents, config=self.config)
        if not resp.candidates or resp.candidates[0].content is None:
            fb = getattr(resp, "prompt_feedback", None)
            return ModelTurn(calls=[], error=f"No candidate returned ({fb})")
        cand = resp.candidates[0]
        self.contents.append(cand.content)

        texts, calls = [], []
        for i, part in enumerate(cand.content.parts or []):
            if part.text and part.text.strip():
                texts.append(part.text.strip())
            fc = part.function_call
            if fc is None:
                continue
            args = dict(fc.args or {})
            call = ToolCall(id=fc.id or f"{fc.name}-{i}", name=fc.name, raw=args)
            sd = args.pop("safety_decision", None)
            if isinstance(sd, dict) and sd.get("decision") == "require_confirmation":
                call.requires_confirmation = sd.get("explanation", "Model requested confirmation")
            try:
                call.actions = translate(fc.name, args, self.frame)
            except (KeyError, KeyError_, ValueError, TypeError) as e:
                call.error = str(e)
            calls.append(call)

        um = resp.usage_metadata
        usage = {"input_tokens": getattr(um, "prompt_token_count", None),
                 "output_tokens": getattr(um, "candidates_token_count", None)}
        finish = str(getattr(cand, "finish_reason", "") or "")
        return ModelTurn(calls=calls, text="\n".join(texts), done=not calls,
                         stop_reason=finish, usage=usage)

    def submit(self, results: list[CallResult]) -> None:
        t = self.types
        parts = []
        for r in results:
            response: dict[str, Any] = {"url": self.session_url}
            if not r.ok:
                response["error"] = r.message
            if r.call.requires_confirmation and r.operator_approved:
                response["safety_acknowledgement"] = "true"
            img_parts = None
            if r.image_png is not None:
                img_parts = [t.FunctionResponsePart(inline_data=t.FunctionResponseBlob(
                    mime_type="image/png", data=r.image_png))]
            parts.append(t.Part(function_response=t.FunctionResponse(
                id=r.call.id if not r.call.id.startswith(f"{r.call.name}-") else None,
                name=r.call.name, response=response, parts=img_parts)))
        self.contents.append(t.Content(role="user", parts=parts))
        self._prune_screenshots()

    def _prune_screenshots(self) -> None:
        seen = 0
        for content in reversed(self.contents):
            if content.role != "user":
                continue
            has_img = False
            for p in content.parts or []:
                fr = p.function_response
                if fr is not None and fr.parts:
                    has_img = True
                    if seen >= KEEP_SCREENSHOTS:
                        fr.parts = None
            if has_img:
                seen += 1
