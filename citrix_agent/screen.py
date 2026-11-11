"""Screen capture, window geometry, coordinate mapping and settle detection.

All coordinates handled outside the providers are *physical screen pixels*.
`make_dpi_aware()` must run before anything touches the screen so that mss,
pygetwindow and pyautogui all agree on that coordinate space, whatever the
Windows display-scaling setting is (100%, 125%, 150%...).
"""
from __future__ import annotations

import io
import math
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional

from PIL import Image

# Claude image limits for the computer toolset models: long edge <= 2576 px and
# <= 4784 visual tokens (ceil(w/28) * ceil(h/28)). We default lower to keep
# per-step cost down; raise --max-long-edge if small UI text is misread.
CLAUDE_MAX_LONG_EDGE = 2576
CLAUDE_MAX_VISUAL_TOKENS = 4784


def make_dpi_aware() -> None:
    if sys.platform != "win32":
        return
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor aware
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


@dataclass(frozen=True)
class Region:
    left: int
    top: int
    width: int
    height: int

    def contains(self, x: int, y: int) -> bool:
        return self.left <= x < self.left + self.width and self.top <= y < self.top + self.height

    @property
    def center(self) -> tuple[int, int]:
        return self.left + self.width // 2, self.top + self.height // 2


def fit_scale(width: int, height: int, max_long_edge: int) -> float:
    """Largest scale <= 1 that keeps the image inside the long-edge and token limits."""
    long_edge = min(max_long_edge, CLAUDE_MAX_LONG_EDGE)
    scale = min(1.0, long_edge / max(width, height))
    while scale > 0.05:
        w, h = max(1, int(width * scale)), max(1, int(height * scale))
        if math.ceil(w / 28) * math.ceil(h / 28) <= CLAUDE_MAX_VISUAL_TOKENS:
            return scale
        scale *= 0.98
    return scale


@dataclass
class Frame:
    """One capture: the full-resolution image plus the downscaled copy the model sees."""
    region: Region
    full: Image.Image
    model_image: Image.Image
    captured_at: float

    @property
    def scale(self) -> float:
        return self.model_image.width / self.region.width

    def model_png(self) -> bytes:
        buf = io.BytesIO()
        self.model_image.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    # --- coordinate mapping -------------------------------------------------
    def model_to_screen(self, mx: float, my: float) -> tuple[int, int]:
        """Model-image pixels (Claude) -> physical screen pixels."""
        sx = self.region.left + mx * self.region.width / self.model_image.width
        sy = self.region.top + my * self.region.height / self.model_image.height
        return self._clamp(sx, sy)

    def normalised_to_screen(self, nx: float, ny: float, scale: float = 1000.0) -> tuple[int, int]:
        """0..1000 normalised coordinates (Gemini) -> physical screen pixels."""
        sx = self.region.left + nx / scale * self.region.width
        sy = self.region.top + ny / scale * self.region.height
        return self._clamp(sx, sy)

    def screen_to_model(self, sx: int, sy: int) -> tuple[int, int]:
        mx = (sx - self.region.left) * self.model_image.width / self.region.width
        my = (sy - self.region.top) * self.model_image.height / self.region.height
        return round(mx), round(my)

    def _clamp(self, sx: float, sy: float) -> tuple[int, int]:
        r = self.region
        x = min(max(round(sx), r.left), r.left + r.width - 1)
        y = min(max(round(sy), r.top), r.top + r.height - 1)
        return x, y

    def zoom_png(self, screen_box: tuple[int, int, int, int], max_long_edge: int) -> bytes:
        """Crop a screen-space box from the full-res capture, sized to the model limits."""
        x0, y0, x1, y1 = screen_box
        r = self.region
        box = (max(0, x0 - r.left), max(0, y0 - r.top),
               min(r.width, x1 - r.left), min(r.height, y1 - r.top))
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"Zoom region {screen_box} is empty or outside the captured area")
        crop = self.full.crop(box)
        # Upscale small crops so detail is legible, but stay within limits.
        target = min(max_long_edge, CLAUDE_MAX_LONG_EDGE)
        s = target / max(crop.size)
        w, h = max(1, int(crop.width * s)), max(1, int(crop.height * s))
        s2 = fit_scale(w, h, target)
        crop = crop.resize((max(1, int(w * s2)), max(1, int(h * s2))), Image.LANCZOS)
        buf = io.BytesIO()
        crop.save(buf, format="PNG", optimize=True)
        return buf.getvalue()


def frame_from_image(img: Image.Image, region: Region, max_long_edge: int) -> Frame:
    s = fit_scale(img.width, img.height, max_long_edge)
    model = img if s >= 0.999 else img.resize(
        (max(1, int(img.width * s)), max(1, int(img.height * s))), Image.LANCZOS)
    return Frame(region=region, full=img, model_image=model, captured_at=time.time())


class Screen:
    """Finds the browser window that hosts the Citrix HTML5 client and captures it.

    If `window_title` is None the whole primary monitor is used.
    """

    def __init__(self, window_title: Optional[str] = None, max_long_edge: int = 1568):
        self.window_title = window_title
        self.max_long_edge = max_long_edge
        self._sct = None

    # --- geometry -----------------------------------------------------------
    def _window(self):
        import pygetwindow as gw  # Windows-only in practice
        matches = [w for w in gw.getWindowsWithTitle(self.window_title) if w.width > 0 and w.height > 0]
        if not matches:
            raise RuntimeError(f"No window whose title contains {self.window_title!r}. "
                               "Open the Citrix session in the browser, or fix --window-title.")
        return matches[0]

    def region(self) -> Region:
        if self.window_title:
            w = self._window()
            if w.isMinimized:
                raise RuntimeError("The Citrix browser window is minimised")
            return Region(w.left, w.top, w.width, w.height)
        sct = self._mss()
        mon = sct.monitors[1]
        return Region(mon["left"], mon["top"], mon["width"], mon["height"])

    def focus(self) -> None:
        """Bring the Citrix browser window to the foreground before sending input."""
        if not self.window_title or sys.platform != "win32":
            return
        import ctypes
        user32 = ctypes.windll.user32
        hwnd = self._window()._hWnd
        if user32.GetForegroundWindow() == hwnd:
            return
        # Windows only lets the foreground process change focus. Tapping ALT
        # first is the standard workaround; it goes to whichever window is
        # currently focused (normally this console), not to Citrix.
        VK_MENU, KEYUP = 0x12, 0x2
        user32.keybd_event(VK_MENU, 0, 0, 0)
        user32.SetForegroundWindow(hwnd)
        user32.keybd_event(VK_MENU, 0, KEYUP, 0)
        time.sleep(0.15)
        if user32.GetForegroundWindow() != hwnd:
            raise RuntimeError("Could not focus the Citrix browser window; click on it and retry")

    # --- capture ------------------------------------------------------------
    def _mss(self):
        if self._sct is None:
            import mss
            self._sct = mss.mss()
        return self._sct

    def grab(self, region: Optional[Region] = None) -> tuple[Image.Image, Region]:
        region = region or self.region()
        shot = self._mss().grab({"left": region.left, "top": region.top,
                                 "width": region.width, "height": region.height})
        return Image.frombytes("RGB", shot.size, shot.rgb), region

    def capture(self) -> Frame:
        img, region = self.grab()
        return frame_from_image(img, region, self.max_long_edge)

    def wait_until_stable(self, timeout: float = 10.0, interval: float = 0.3,
                          threshold: float = 0.6, stable_needed: int = 2) -> bool:
        return wait_until_stable(lambda: self.grab()[0], timeout, interval, threshold, stable_needed)


def _signature(img: Image.Image):
    import numpy as np
    small = img.convert("L").resize((160, max(1, int(160 * img.height / img.width))))
    return np.asarray(small, dtype="float32")


def wait_until_stable(grab: Callable[[], Image.Image], timeout: float = 10.0,
                      interval: float = 0.3, threshold: float = 0.6, stable_needed: int = 2) -> bool:
    """Poll until consecutive frames stop changing (mean abs grey diff < threshold).

    Citrix renders asynchronously and over the network, so a fixed sleep is
    either too short or wastes time. Returns False on timeout (the caller
    carries on anyway; the model will see whatever is on screen).
    """
    import numpy as np
    deadline = time.time() + timeout
    prev = _signature(grab())
    stable = 0
    while time.time() < deadline:
        time.sleep(interval)
        cur = _signature(grab())
        if cur.shape == prev.shape and float(np.abs(cur - prev).mean()) < threshold:
            stable += 1
            if stable >= stable_needed:
                return True
        else:
            stable = 0
        prev = cur
    return False
