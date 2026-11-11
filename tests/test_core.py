"""Offline tests: no screen, no input, no API calls. Run with `pytest`."""
import json
import math
from types import SimpleNamespace as NS

import pytest
from PIL import Image

from citrix_agent.actions import Action
from citrix_agent.agent import Agent
from citrix_agent.evidence import RunLog
from citrix_agent.executor import ActionRejected, DryRunBackend, Executor
from citrix_agent.keys import blocked_reason, parse_combo
from citrix_agent.providers import claude as claude_mod
from citrix_agent.providers import gemini as gemini_mod
from citrix_agent.providers.base import CallResult, ModelTurn, Provider, ToolCall
from citrix_agent.safety import KillSwitch, Operator
from citrix_agent.screen import Region, fit_scale, frame_from_image

# A 1920x1080 browser window sitting at (100, 50) on the desktop.
REGION = Region(100, 50, 1920, 1080)


def make_frame(max_long_edge=1568):
    return frame_from_image(Image.new("RGB", (REGION.width, REGION.height), "white"), REGION, max_long_edge)


# --- geometry -----------------------------------------------------------------
def test_fit_scale_respects_claude_limits():
    for w, h in [(1920, 1080), (2560, 1440), (3840, 2160), (1366, 768)]:
        s = fit_scale(w, h, 2576)
        sw, sh = int(w * s), int(h * s)
        assert max(sw, sh) <= 2576
        assert math.ceil(sw / 28) * math.ceil(sh / 28) <= 4784


def test_model_to_screen_roundtrip():
    f = make_frame()
    assert f.model_image.width == 1568
    # centre of the model image maps to centre of the window on screen
    x, y = f.model_to_screen(f.model_image.width / 2, f.model_image.height / 2)
    assert (x, y) == (100 + 960, 50 + 540)
    assert f.screen_to_model(x, y) == (784, round(f.model_image.height / 2))


def test_normalised_mapping_and_clamp():
    f = make_frame()
    assert f.normalised_to_screen(500, 500) == (1060, 590)
    assert f.normalised_to_screen(1000, 1000) == (100 + 1919, 50 + 1079)  # clamped inside


# --- keys ---------------------------------------------------------------------
@pytest.mark.parametrize("combo,expected", [
    ("Return", ["enter"]), ("ctrl+s", ["ctrl", "s"]), ("Control+A", ["ctrl", "a"]),
    ("Page_Down", ["pagedown"]), ("alt+F4", ["alt", "f4"]), ("ctrl++", ["ctrl", "+"]),
])
def test_parse_combo(combo, expected):
    assert parse_combo(combo) == expected


@pytest.mark.parametrize("combo", ["ctrl+w", "F5", "alt+F4", "ctrl+l", "alt+Left", "super+r", "ctrl+plus"])
def test_browser_shortcuts_blocked(combo):
    assert blocked_reason(parse_combo(combo)) is not None


@pytest.mark.parametrize("combo", ["ctrl+s", "Tab", "Return", "ctrl+a", "shift+Tab", "alt+n"])
def test_app_shortcuts_allowed(combo):
    assert blocked_reason(parse_combo(combo)) is None


# --- executor -----------------------------------------------------------------
def test_executor_rejects_outside_window_and_ctrl_scroll():
    ex = Executor(DryRunBackend())
    with pytest.raises(ActionRejected):
        ex.execute(Action("click", x=10, y=10), REGION)
    with pytest.raises(ActionRejected):
        ex.execute(Action("scroll", x=500, y=500, direction="down", amount=3, modifiers=["ctrl"]), REGION)
    with pytest.raises(ActionRejected):
        ex.execute(Action("key", keys=["ctrl", "w"]), REGION)


def test_executor_click_with_modifier_order():
    b = DryRunBackend()
    Executor(b).execute(Action("click", x=500, y=500, modifiers=["shift"]), REGION)
    assert [c[0] for c in b.calls] == ["key_down", "click", "key_up"]


# --- Claude translation -------------------------------------------------------
def test_claude_translate_members():
    f = make_frame()
    s = f.model_image.width / REGION.width
    a = claude_mod.translate("left_click", {"coordinate": [784, 441], "text": "ctrl"}, f)[0]
    assert (a.kind, a.button, a.modifiers) == ("click", "left", ["ctrl"])
    assert abs(a.x - (100 + 784 / s)) <= 1
    assert claude_mod.translate("double_click", {"coordinate": [10, 10]}, f)[0].clicks == 2
    assert claude_mod.translate("key", {"text": "ctrl+s", "repeat": 2}, f)[0].repeat == 2
    d = claude_mod.translate("left_click_drag", {"start_coordinate": [0, 0], "coordinate": [100, 100]}, f)[0]
    assert (d.start_x, d.start_y) == (100, 50)
    z = claude_mod.translate("zoom", {"region": [0, 0, 392, 220]}, f)[0]
    assert z.kind == "zoom" and z.region[0] == 100
    assert claude_mod.translate("scroll", {"scroll_direction": "down", "scroll_amount": 4}, f)[0].amount == 4
    with pytest.raises(ValueError):
        claude_mod.translate("teleport", {}, f)


def _claude_response(blocks, stop="tool_use"):
    return NS(content=blocks, stop_reason=stop, stop_details=None,
              usage=NS(input_tokens=100, output_tokens=20, cache_read_input_tokens=0, cache_creation_input_tokens=0))


def test_claude_provider_wire_format():
    sent = []

    class FakeMessages:
        def create(self, **kw):
            sent.append(kw)
            if len(sent) == 1:
                return _claude_response([
                    NS(type="thinking", thinking="I should open the menu."),
                    NS(type="tool_use", id="t1", name="left_click", input={"coordinate": [50, 50]}, toolset_name="computer"),
                    NS(type="tool_use", id="t2", name="screenshot", input={}, toolset_name="computer"),
                ])
            return _claude_response([NS(type="text", text="Done: record saved.")], stop="end_turn")

    client = NS(beta=NS(messages=FakeMessages()), messages=FakeMessages())
    p = claude_mod.ClaudeProvider(client=client)
    p.start("do it", make_frame())
    turn = p.next_turn()
    assert [c.name for c in turn.calls] == ["left_click", "screenshot"] and not turn.done
    kw = sent[0]
    assert kw["tools"] == [{"type": "computer_toolset_20260801"}]
    assert kw["model"] == "claude-opus-5-5" and kw["fallbacks"] == "default"
    p.submit([CallResult(turn.calls[0], ok=True), CallResult(turn.calls[1], ok=True, image_png=b"png")])
    results = p.messages[-1]["content"]
    assert all(r["toolset_name"] == "computer" for r in results)
    assert results[0]["content"] == "OK" and results[1]["content"][0]["type"] == "image"
    assert p.next_turn().done


# --- Gemini translation -------------------------------------------------------
def test_gemini_translate():
    f = make_frame()
    acts = gemini_mod.translate("type_text_at", {"x": 500, "y": 500, "text": "hello"}, f)
    assert [a.kind for a in acts] == ["click", "key", "key", "type", "key"]
    assert (acts[0].x, acts[0].y) == (1060, 590)
    acts = gemini_mod.translate("type_text_at", {"x": 1, "y": 1, "text": "x", "press_enter": False,
                                                 "clear_before_typing": False}, f)
    assert [a.kind for a in acts] == ["click", "type"]
    assert gemini_mod.translate("key_combination", {"keys": "Control+S"}, f)[0].keys == ["ctrl", "s"]
    with pytest.raises(ValueError):
        gemini_mod.translate("navigate", {"url": "http://x"}, f)


def test_gemini_provider_wire_format():
    from google.genai import types as t

    def resp(parts):
        return NS(candidates=[NS(content=t.Content(role="model", parts=parts), finish_reason="STOP")],
                  usage_metadata=NS(prompt_token_count=10, candidates_token_count=2))

    queue = [
        resp([t.Part(function_call=t.FunctionCall(name="click_at", args={
            "x": 100, "y": 100, "safety_decision": {"decision": "require_confirmation", "explanation": "submit?"}}))]),
        resp([t.Part(text="All done.")]),
    ]
    client = NS(models=NS(generate_content=lambda **kw: queue.pop(0)))
    p = gemini_mod.GeminiProvider(client=client)
    p.start("goal", make_frame())
    turn = p.next_turn()
    call = turn.calls[0]
    assert call.requires_confirmation == "submit?" and "safety_decision" not in call.raw
    p.submit([CallResult(call, ok=True, image_png=b"png", operator_approved=True)])
    fr = p.contents[-1].parts[0].function_response
    assert fr.response["safety_acknowledgement"] == "true" and fr.response["url"]
    assert fr.parts[0].inline_data.data == b"png"
    assert p.next_turn().done


# --- agent loop ---------------------------------------------------------------
class FakeScreen:
    max_long_edge = 1568
    def focus(self): pass
    def capture(self): return make_frame()
    def wait_until_stable(self, timeout=10): return True


class ScriptedProvider(Provider):
    name = "fake"

    def __init__(self, turns):
        super().__init__("fake-model")
        self.turns, self.submitted = turns, []

    def start(self, goal, frame): self.observe(frame)
    def next_turn(self): return self.turns.pop(0)
    def submit(self, results): self.submitted.append(results)


def test_agent_loop_batch_failure_skips_rest(tmp_path):
    f = make_frame()
    good = ToolCall("1", "left_click", {}, [Action("click", x=500, y=500)])
    blocked = ToolCall("2", "key", {}, [Action("key", keys=["ctrl", "w"])])
    after = ToolCall("3", "type", {}, [Action("type", text="never")])
    shot = ToolCall("4", "screenshot", {}, [Action("screenshot")])
    prov = ScriptedProvider([ModelTurn(calls=[good, blocked, after]), ModelTurn(calls=[shot]),
                             ModelTurn(calls=[], text="finished", done=True)])
    backend = DryRunBackend()
    log = RunLog(tmp_path, "fake", "fake-model", "goal", "auto")
    agent = Agent(prov, FakeScreen(), Executor(backend), log, Operator("auto"), KillSwitch())
    res = agent.run("goal")
    assert res.outcome == "model_reported_done"
    first = prov.submitted[0]
    assert [r.ok for r in first] == [True, False, False]
    assert "safety policy" in first[1].message and first[2].message.startswith("Not executed")
    assert prov.submitted[1][0].image_png  # screenshot returned
    assert [c[0] for c in backend.calls] == ["click"]  # nothing after the blocked key ran
    summary = json.loads((tmp_path.glob("*_fake/summary.json").__next__()).read_text())
    assert summary["model_turns"] == 3
    assert any(p.name == "report.html" for p in tmp_path.rglob("report.html"))


def test_kill_switch_stops_run(tmp_path):
    kill = KillSwitch()
    kill.trigger()
    prov = ScriptedProvider([ModelTurn(calls=[], done=True)])
    agent = Agent(prov, FakeScreen(), Executor(DryRunBackend()), RunLog(tmp_path, "fake", "m", "g", "auto"),
                  Operator("auto"), kill)
    assert agent.run("g").outcome == "stopped"
