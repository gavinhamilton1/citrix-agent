"""Key-name normalisation and the blocked-shortcut policy.

Claude sends xdotool-style names ("Return", "ctrl+s", "Page_Down"); Gemini
sends names like "Control+A" or "Enter". Both are normalised to pyautogui key
names here.

The Citrix HTML5 client runs inside a browser tab, so the *browser* sees every
keystroke first. Shortcuts the browser owns (close tab, reload, back, address
bar, zoom, fullscreen) would kill or disturb the Citrix session, so they are
blocked by default.
"""
from __future__ import annotations

ALIASES = {
    "return": "enter", "enter": "enter", "kp_enter": "enter",
    "escape": "esc", "esc": "esc",
    "tab": "tab", "backspace": "backspace", "back_space": "backspace",
    "delete": "delete", "del": "delete", "insert": "insert",
    "space": "space", " ": "space",
    "page_up": "pageup", "pageup": "pageup", "prior": "pageup",
    "page_down": "pagedown", "pagedown": "pagedown", "next": "pagedown",
    "home": "home", "end": "end",
    "up": "up", "down": "down", "left": "left", "right": "right",
    "arrowup": "up", "arrowdown": "down", "arrowleft": "left", "arrowright": "right",
    "ctrl": "ctrl", "control": "ctrl", "control_l": "ctrl", "control_r": "ctrl",
    "alt": "alt", "alt_l": "alt", "alt_r": "alt", "option": "alt",
    "shift": "shift", "shift_l": "shift", "shift_r": "shift",
    "super": "win", "super_l": "win", "win": "win", "windows": "win",
    "meta": "win", "cmd": "win", "command": "win",
    "caps_lock": "capslock", "capslock": "capslock",
    "print": "printscreen", "printscreen": "printscreen",
    "menu": "apps", "apps": "apps",
    "plus": "+", "minus": "-", "equal": "=", "comma": ",", "period": ".",
    "slash": "/", "backslash": "\\", "semicolon": ";", "apostrophe": "'",
    "grave": "`", "bracketleft": "[", "bracketright": "]",
}
ALIASES.update({f"f{i}": f"f{i}" for i in range(1, 25)})

MODIFIERS = {"ctrl", "alt", "shift", "win"}

# Combos are compared as sets of normalised key names.
DEFAULT_BLOCKED: list[frozenset[str]] = [frozenset(c) for c in [
    {"ctrl", "w"}, {"ctrl", "f4"}, {"alt", "f4"},          # close tab / browser
    {"ctrl", "t"}, {"ctrl", "n"}, {"ctrl", "shift", "n"},  # new tab / window
    {"ctrl", "shift", "t"}, {"ctrl", "shift", "w"},
    {"ctrl", "r"}, {"f5"}, {"ctrl", "f5"}, {"shift", "f5"}, # reload = disconnect
    {"ctrl", "l"}, {"alt", "d"}, {"f6"},                    # address bar
    {"alt", "left"}, {"alt", "right"}, {"alt", "home"},     # back / forward / home
    {"f11"}, {"f12"}, {"ctrl", "shift", "i"},               # fullscreen / devtools
    {"ctrl", "="}, {"ctrl", "+"}, {"ctrl", "-"}, {"ctrl", "0"},  # browser zoom
    {"alt", "tab"}, {"ctrl", "tab"}, {"ctrl", "shift", "tab"},   # focus leaves Citrix
    {"ctrl", "alt", "delete"}, {"ctrl", "shift", "esc"},
]]


class KeyError_(ValueError):
    """Raised for a key name we cannot map."""


def normalise_key(name: str) -> str:
    raw = name.strip()
    if len(raw) == 1:
        return raw.lower()
    k = raw.lower().replace("-", "_")
    if k in ALIASES:
        return ALIASES[k]
    raise KeyError_(f"Unknown key name: {name!r}")


def parse_combo(combo: str) -> list[str]:
    """'ctrl+shift+s' / 'Control+A' / 'Return' -> ['ctrl', 'shift', 's']."""
    combo = combo.strip()
    if combo == "+":
        return ["+"]
    # Allow a trailing literal plus, e.g. "ctrl++".
    if combo.endswith("++"):
        head = combo[:-2]
        return [normalise_key(p) for p in head.split("+") if p] + ["+"]
    return [normalise_key(p) for p in combo.split("+") if p]


def blocked_reason(keys: list[str], blocked: list[frozenset[str]] | None = None) -> str | None:
    blocked = DEFAULT_BLOCKED if blocked is None else blocked
    s = frozenset(keys)
    if "win" in s:
        return "Windows-key shortcuts act on the local PC, not the Citrix session"
    if s in blocked:
        return f"'{'+'.join(keys)}' is a browser/OS shortcut that would disturb the Citrix tab"
    return None
