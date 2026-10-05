# Earthquake Simulator -- a physically grounded earthquake simulator.
# Copyright (C) 2026 Earthquake Simulator contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""UI font.

The interface is English only, but it uses a few typographic symbols
(·, ×, ², ₁) that Panda3D's built-in font does not have -- it draws them as
blanks and floods the console with warnings. So this picks up a system font
(Segoe UI on Windows, Helvetica on macOS, DejaVu Sans on Linux; any font
dropped into a fonts/ folder next to earthquake.py takes precedence) and records
whether one was found:

    level 1   a real font       text passes through untouched
    level 0   Panda's default   symbols replaced by ASCII equivalents

Fonts are opened directly as DynamicTextFont from a Filename. (Loader.loadFont
takes a string, not a Filename, and raises TypeError otherwise -- which is
exactly the failure an earlier version of this file silently swallowed.)
"""

from __future__ import annotations

import os
import sys

from panda3d.core import DynamicTextFont, Filename, TextNode

# Ordered best-first within each platform: the platform's own UI font, then
# anything else with full Latin and symbol coverage.
CANDIDATES = {
    "win32": [
        r"C:\Windows\Fonts\segoeui.ttf",       # Segoe UI
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\msyh.ttc",          # Microsoft YaHei (Latin is fine)
        r"C:\Windows\Fonts\Deng.ttf",
    ],
    "darwin": [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ],
    "linux": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ],
}

# Substitutions when the active font lacks the typographic symbols.
ASCII_FALLBACK = {
    "·": "|", "×": "x", "²": "^2", "₁": "1", "—": "-", "–": "-",
    "°": " deg", "≈": "~", "±": "+/-", "μ": "u", "σ": "s", "Δ": "d",
    "“": '"', "”": '"', "’": "'",
}

_LEVEL = 0
_FONT_NAME = ""


def _platform_key() -> str:
    if sys.platform.startswith("win"):
        return "win32"
    if sys.platform == "darwin":
        return "darwin"
    return "linux"


def _paths() -> list[str]:
    out = list(CANDIDATES[_platform_key()])
    # Anything the user drops into a fonts/ folder next to earthquake.py wins.
    here = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "fonts")
    if os.path.isdir(here):
        for name in sorted(os.listdir(here)):
            if name.lower().endswith((".ttf", ".ttc", ".otf")):
                out.insert(0, os.path.join(here, name))
    return out


def _load(path: str):
    try:
        font = DynamicTextFont(Filename.fromOsSpecific(path))
    except Exception:
        return None
    if font is None or not font.isValid():
        return None
    return font


def _install_shim() -> None:
    """Route every piece of on-screen text through `safe()`.

    One choke point rather than conversions sprinkled through the interface:
    text arrives from labels, buttons, dropdown items and the results panel,
    and any one that was missed would silently render as blanks.
    """
    from direct.gui.OnscreenText import OnscreenText

    if getattr(OnscreenText, "_quakesim_shimmed", False):
        return
    orig_init = OnscreenText.__init__
    orig_set = OnscreenText.setText

    def init(self, text="", *a, **kw):
        return orig_init(self, safe(text), *a, **kw)

    def set_text(self, text):
        return orig_set(self, safe(text))

    OnscreenText.__init__ = init
    OnscreenText.setText = set_text
    OnscreenText._quakesim_shimmed = True


def install_font(base=None):
    """Load the best available font and make it the default for all text."""
    global _LEVEL, _FONT_NAME
    for path in _paths():
        if not os.path.exists(path):
            continue
        font = _load(path)
        if font is None:
            continue
        font.setPixelsPerUnit(64)
        font.setSpaceAdvance(0.28)
        TextNode.setDefaultFont(font)
        _LEVEL = 1
        _FONT_NAME = os.path.basename(path)
        print(f"[quakesim] UI font: {_FONT_NAME}")
        return font

    print("[quakesim] no usable system font found; interface labels will use "
          "plain ASCII. Drop a .ttf into a 'fonts' folder next to earthquake.py to "
          "get the typographic symbols back.")
    _LEVEL = 0
    _install_shim()
    return None


def level() -> int:
    return _LEVEL


def have_font() -> bool:
    return _LEVEL >= 1


def safe(text: str) -> str:
    """Replace the symbols Panda's built-in font cannot draw (only used when
    no real font was found)."""
    if _LEVEL >= 1:
        return text
    out = []
    for ch in text:
        if ch in ASCII_FALLBACK:
            out.append(ASCII_FALLBACK[ch])
        elif ord(ch) < 128 or ch in "\n\t":
            out.append(ch)
    return "".join(out)
