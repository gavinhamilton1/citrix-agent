# citrix-agent: AI-driven control of a standard Citrix session (PoC)

The goal is to show that an **unmodified Citrix Workspace HTML5 session** running in a normal browser
can be driven end to end by an AI model, using only:

- **screenshots** of the browser window (`mss`)
- **synthetic keyboard and mouse input** (`pyautogui`, i.e. Windows `SendInput`)

Nothing is installed or changed on the Citrix side: no policy changes, no agent on the VDA, no app changes.
The same loop runs with **Claude** or **Gemini**, so the two models can be compared on the same task.

```
operator ─goal─▶ agent loop ──screenshot──▶ model (Claude | Gemini computer use)
                    ▲    │◀──── actions ─────┘
                    │    ▼
               mss capture   executor (safety policy) ─▶ pyautogui ─▶ browser tab ─▶ Citrix HTML5 ─▶ web app
                    │
               evidence: annotated screenshots, steps.jsonl, report.html
```

## Layout

| File | Role |
|---|---|
| `citrix_agent/agent.py` | Model-neutral loop: ask the model, run its actions, send results back. Stops when the model says it is done. |
| `citrix_agent/providers/claude.py` | Claude Messages API with the `computer_toolset_20260801` tool. Default model is `claude-opus-5-5`. |
| `citrix_agent/providers/gemini.py` | Gemini Computer Use tool through `google-genai`. Default model is `gemini-2.5-computer-use-preview-10-2025`. |
| `citrix_agent/screen.py` | DPI awareness, finding the browser window, capture, model-to-screen coordinate mapping, waiting for the screen to settle. |
| `citrix_agent/executor.py` | The only code that sends input. Checks every action against the safety policy first. |
| `citrix_agent/keys.py` | Normalises key names and blocks browser and OS shortcuts. |
| `citrix_agent/safety.py` | Kill hotkey and operator approval (step mode). |
| `citrix_agent/evidence.py` | Per-run folder with screenshots, `steps.jsonl`, `summary.json` and `report.html`. |
| `citrix_agent/compare.py` | Table of all runs, for comparing Claude and Gemini. |

## Setup (Windows client VM)

```bash
py -3.12 -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

API keys are read from the environment. Use whichever you need:

```bash
set ANTHROPIC_API_KEY=...
```

```bash
set GEMINI_API_KEY=...
```

If Python can't be installed on the client, the tool can be packaged as a single exe with PyInstaller.
That is still nothing installed on the Citrix side.

## Running

1. Open the Citrix session in the browser in the normal way and bring the web app to its starting screen.
   For coordinate accuracy, keep the browser on the **primary monitor**.
2. Note a word that appears in the browser window title (often the published app name, or "Citrix").
3. First do a dry run. No input is sent; intended clicks are marked in red on the saved screenshots.
   This checks capture and coordinate mapping safely:

```bash
python -m citrix_agent --provider claude --window-title "Citrix" --dry-run --max-steps 3 --goal "Describe the screen, then click the Search box"
```

4. Run for real in step mode. Every input action is shown, and you press Enter to run it,
   `s` to skip, `a` to switch to auto, or `q` to quit:

```bash
python -m citrix_agent --provider claude --window-title "Citrix" --goal "Open customer record 1001, change the phone number to 0123 456789, save, and confirm the saved value is shown"
```

```bash
python -m citrix_agent --provider gemini --window-title "Citrix" --goal "Open customer record 1001, change the phone number to 0123 456789, save, and confirm the saved value is shown"
```

5. Compare the runs:

```bash
python -m citrix_agent.compare runs
```

Each run writes `runs/<timestamp>_<provider>/report.html`. It contains the model's reasoning, every action
with its result, and screenshots with click markers. This is the PoC evidence.

### Stopping

- Press **Ctrl+Alt+K** (global hotkey; change it with `--kill-hotkey`).
- Or slam the mouse into any **screen corner** (pyautogui failsafe).
- Or press **Ctrl+C** in the console, or `q` at a step-mode prompt.

### Useful options

| Option | Default | Notes |
|---|---|---|
| `--mode step\|auto` | `step` | Auto still pauses for any action Gemini flags as needing confirmation. |
| `--max-steps` | 40 | Maximum number of model turns. |
| `--max-long-edge` | 1568 | Screenshot size sent to the model. Raise it (up to 2576) if small text is misread. |
| `--effort` | `medium` | Claude effort level. Try `high` if it struggles with the app. |
| `--no-fallbacks` | off | Claude has server-side refusal fallbacks on by default (`fallbacks="default"`). This turns them off. |
| `--gemini-environment` | `browser` | `desktop` is also offered by the SDK, but its action set has not been checked here (see below). |
| `--model` | provider default | For example a newer Gemini computer-use model. |

## How a standard session is handled without config changes

| Default-session issue | Handling |
|---|---|
| Display scaling (125%, 150%, ...) | The process is made per-monitor DPI-aware at start-up, so capture, window bounds and clicks all use physical pixels. |
| Window size and position | Bounds are read live each turn. Clicks outside the window are rejected. |
| Remote rendering lag | After each action the agent waits until frames stop changing (up to 10 s), never a fixed sleep. |
| Fast typing gets dropped | Keys are typed 40 ms apart. |
| Clipboard redirection may be off | Clipboard is never used; text is always typed. |
| Browser owns some shortcuts | Ctrl+W, Ctrl+T, Ctrl+L, Ctrl+R, F5, F11, F12, Alt+F4, Alt+Tab, Alt+Left/Right, browser zoom and the Windows key are blocked (`keys.py`). The model gets an error and picks another way. |
| Focus is lost to the console | The browser window is brought to the front before every input action. |

## Model-specific notes

**Claude.** Uses the `computer_toolset_20260801` tool (no beta header). Actions can arrive in batches. If one
fails, the rest of the batch is answered "Not executed". The full history, including thinking blocks, is sent
back every turn and old screenshots are deliberately *not* pruned, which keeps the model's preserved reasoning
valid. Cost is controlled by screenshot size and `--max-steps` instead. Prompt caching is on.

**Gemini.** Coordinates are 0–1000 normalised. Every function response carries a fresh screenshot, and only the
last 3 screenshots are kept in history. Browser-navigation functions (`navigate`, `search`, `go_back`,
`go_forward`, `open_web_browser`) are excluded, because leaving the tab would disconnect Citrix.
`type_text_at` becomes click, then select-all and delete, then type, then Enter. Safety-flagged calls always
need operator approval, and the acknowledgement is sent back as the API requires.

## Verify before the first live run

- **Gemini docs.** The Gemini computer-use API is a preview. The action names and response format here come
  from the `google-genai` SDK and its tests, but ai.google.dev could not be reached while this was built.
  Check the current model ID and action list at https://ai.google.dev/gemini-api/docs/computer-use. The
  translator returns an error to the model for any action it does not know, and logs it.
- **Scroll units.** One wheel notch is sent as 120 units on Windows. If scrolling is too fast or too slow in
  the Citrix HTML5 client, adjust `wheel_unit` in `executor.py`.
- **Non-ASCII text.** `pyautogui.write` only types characters that map to keys on the local layout.

## Tests

The tests run offline, with no screen, no input and no API calls:

```bash
pytest -q
```

They cover coordinate mapping, image-size limits, key normalisation and blocking, executor policy,
Claude and Gemini action translation, the request and response format of both providers (with fake
clients), batch-failure handling, evidence output and the kill switch.

## Known limitations (PoC)

- Single primary monitor only, and one Citrix tab.
- No on-screen "AI in control" overlay. The console shows each action instead.
- "Done" means the model *says* it is done. The operator confirms using the report's final screenshots.
