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

"""Movable, dockable interface panels.

Each panel is a frame with a title bar you can grab. By default panels dock
to an edge -- left, right, bottom, top -- and the dock position is computed
from the window's actual aspect ratio, so a 21:9 monitor and a 4:3 one both
get a sane layout. Once you drag a panel it stays where you put it, across
window resizes and across runs (the layout is saved next to `earthquake.py`), until
you press the reset-layout key. A moved panel is remembered by its offset
from the nearest edges, so it keeps hugging them when the window is
resized or maximised.

Coordinates: the frame's origin is its top-left corner. Content is laid out
in a `body` node whose origin is the top-left corner of the area under the
title bar, with x increasing rightward and y increasing *downward* as
negative values -- the way you would sketch a form on paper.
"""

from __future__ import annotations

import json
import os

from direct.gui.DirectGui import DGG, DirectFrame
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.DirectObject import DirectObject
from panda3d.core import NodePath, TextNode

BAR = 0.062
MARGIN = 0.018

PANEL_BG = (0.06, 0.07, 0.09, 0.84)
BAR_BG = (0.11, 0.13, 0.17, 0.96)
BAR_HOVER = (0.16, 0.20, 0.26, 0.98)
ACCENT = (0.36, 0.72, 0.92, 1)
DIM = (0.62, 0.66, 0.70, 1)


def _layout_path() -> str:
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(here, "layout.json")


class Panel:
    def __init__(self, manager: "PanelManager", key: str, title: str,
                 width: float, height: float, dock: str,
                 dock_order: int = 0):
        self.m = manager
        self.key = key
        self.width = width
        self.height = height
        self.dock_side = dock
        self.dock_order = dock_order
        self.moved = False
        # Where a moved panel sits, as offsets from its nearest horizontal
        # and vertical screen edges, so it keeps hugging those edges when the
        # window is resized or maximised: ("left"|"right", dx, "top"|"bottom", dy).
        self.anchor = None
        self.visible = True

        base = manager.base
        self.frame = DirectFrame(parent=base.aspect2d, frameColor=PANEL_BG,
                                 frameSize=(0, width, -height, 0),
                                 state=DGG.NORMAL, suppressMouse=1)
        self.frame.setTransparency(1)

        self.bar = DirectFrame(parent=self.frame, frameColor=BAR_BG,
                               frameSize=(0, width, -BAR, 0), pos=(0, 0, 0),
                               state=DGG.NORMAL, suppressMouse=1)
        self.bar.bind(DGG.B1PRESS, self._press)
        self.bar.bind(DGG.B1RELEASE, self._release)
        self.bar.bind(DGG.ENTER, lambda e: self.bar.__setitem__("frameColor", BAR_HOVER))
        self.bar.bind(DGG.EXIT, lambda e: self.bar.__setitem__("frameColor", BAR_BG))
        self.title = OnscreenText(text=title, parent=self.bar,
                                  pos=(0.022, -BAR * 0.66), scale=0.034,
                                  fg=ACCENT, align=TextNode.ALeft, mayChange=True)
        self.grip = OnscreenText(text="::::", parent=self.bar,
                                 pos=(width - 0.022, -BAR * 0.66), scale=0.034,
                                 fg=DIM, align=TextNode.ARight)

        self.body = self.frame.attachNewNode(f"{key}-body")
        self.body.setPos(0, 0, -BAR)
        self._grab = None
        self._task = None

    # -- geometry ----------------------------------------------------------
    @property
    def body_height(self) -> float:
        return self.height - BAR

    def set_pos(self, x: float, y: float) -> None:
        self.frame.setPos(x, 0, y)

    def get_pos(self) -> tuple[float, float]:
        p = self.frame.getPos()
        return float(p.x), float(p.z)

    def set_anchor_from_pos(self, aspect: float) -> None:
        """Record the current position relative to the nearest edges."""
        x, y = self.get_pos()
        left, right = x + aspect, aspect - (x + self.width)
        top, bottom = 1.0 - y, (y - self.height) + 1.0
        hside, dx = ("left", left) if left <= right else ("right", right)
        vside, dy = ("top", top) if top <= bottom else ("bottom", bottom)
        self.anchor = (hside, dx, vside, dy)

    def apply_anchor(self, aspect: float) -> None:
        if not self.anchor:
            return
        hside, dx, vside, dy = self.anchor
        x = (-aspect + dx) if hside == "left" else (aspect - self.width - dx)
        y = (1.0 - dy) if vside == "top" else (-1.0 + dy + self.height)
        self.set_pos(x, y)

    def docked_pos(self, aspect: float, stack_offset: float = 0.0,
                   left_width: float = 0.0) -> tuple[float, float]:
        """Where this panel goes when docked."""
        top = 1.0 - MARGIN
        if self.dock_side == "left":
            return (-aspect + MARGIN, top - stack_offset)
        if self.dock_side == "right":
            return (aspect - self.width - MARGIN, top - stack_offset)
        if self.dock_side == "bottom":
            return (-self.width / 2.0, -1.0 + MARGIN + self.height)
        # top: beside the left-hand stack, or centred when there is none
        if left_width > 0.0:
            return (-aspect + MARGIN + left_width + MARGIN, top)
        return (-self.width / 2.0, top)

    def dock(self, aspect: float, stack_offset: float = 0.0,
             left_width: float = 0.0) -> None:
        """Place at the docked position, unless the user has moved it -- a
        moved panel keeps its offset from the edges nearest to it."""
        if self.moved:
            self.apply_anchor(aspect)
            self.clamp(aspect)
            return
        self.set_pos(*self.docked_pos(aspect, stack_offset, left_width))
        self.clamp(aspect)

    def clamp(self, aspect: float) -> None:
        """Keep the title bar reachable after a resize."""
        x, y = self.get_pos()
        x = max(-aspect, min(aspect - self.width, x))
        y = max(-1.0 + BAR, min(1.0, y))
        self.set_pos(x, y)

    def show(self) -> None:
        self.visible = True
        self.frame.show()

    def hide(self) -> None:
        self.visible = False
        self.frame.hide()

    def toggle(self) -> None:
        (self.hide if self.visible else self.show)()

    # -- dragging ----------------------------------------------------------
    def _mouse(self):
        mw = self.m.base.mouseWatcherNode
        if mw is None or not mw.hasMouse():
            return None
        m = mw.getMouse()
        return m.getX() * self.m.aspect(), m.getY()

    def _press(self, _event=None) -> None:
        mp = self._mouse()
        if mp is None:
            return
        x, y = self.get_pos()
        self._grab = (mp[0] - x, mp[1] - y)
        self._press_pos = (x, y)
        self.m.dragging = self
        self.frame.reparentTo(self.m.base.aspect2d)      # raise to the top
        if self._task is None:
            self._task = self.m.base.taskMgr.add(self._drag, f"drag-{self.key}")

    def _drag(self, task):
        mp = self._mouse()
        if mp is not None and self._grab is not None:
            self.set_pos(mp[0] - self._grab[0], mp[1] - self._grab[1])
            self.clamp(self.m.aspect())
        return task.cont

    def _release(self, _event=None) -> None:
        if self._task is not None:
            self.m.base.taskMgr.remove(self._task)
            self._task = None
        if self._grab is not None:
            self._grab = None
            x, y = self.get_pos()
            x0, y0 = getattr(self, "_press_pos", (x, y))
            # A click on the title bar is not a move: only a real drag
            # takes the panel out of the docked layout.
            if abs(x - x0) > 0.012 or abs(y - y0) > 0.012:
                self.moved = True
                self.set_anchor_from_pos(self.m.aspect())
                self.m.save()
            elif not self.moved:
                self.m.dock_all()
        self.m.dragging = None


class PanelManager(DirectObject):
    """Owns the panels, docks them, and remembers where you left them."""

    def __init__(self, base):
        DirectObject.__init__(self)
        self.base = base
        self.panels: dict[str, Panel] = {}
        self.dragging: Panel | None = None
        self._saved = self._load()
        self._last_aspect = None
        # Listen as our own object. `base.accept("window-event", ...)` would
        # REPLACE ShowBase's own handler for that event -- the one that
        # rescales aspect2d when the window changes size -- which is exactly
        # how maximising the window used to leave every panel off-screen.
        self.accept("window-event", self._on_window)

    def aspect(self) -> float:
        try:
            a = self.base.getAspectRatio()
        except Exception:
            a = 16.0 / 9.0
        return float(a) if a and a > 0 else 16.0 / 9.0

    def add(self, key, title, width, height, dock, dock_order=0) -> Panel:
        p = Panel(self, key, title, width, height, dock, dock_order)
        self.panels[key] = p
        saved = self._saved.get(key)
        if saved and saved.get("moved"):
            p.moved = True
            anchor = saved.get("anchor")
            if (isinstance(anchor, (list, tuple)) and len(anchor) == 4
                    and anchor[0] in ("left", "right") and anchor[2] in ("top", "bottom")):
                p.anchor = (anchor[0], float(anchor[1]), anchor[2], float(anchor[3]))
                p.apply_anchor(self.aspect())
            else:
                # An older layout file: absolute position; anchor it now.
                x, y = saved["pos"]
                p.set_pos(x, y)
                p.set_anchor_from_pos(self.aspect())
            x, y = p.get_pos()
            # A panel within a whisker of its docked place was never really
            # moved (a nudge, or a click that used to count as a drag):
            # let it dock, so it follows the layout again.
            left_width = max((q.width for q in self.panels.values()
                              if q.dock_side == "left" and q is not p), default=0.0)
            offset = sum(q.height + MARGIN for q in self.panels.values()
                         if q.dock_side == p.dock_side and q is not p
                         and q.dock_order < p.dock_order and not q.moved)
            dx_, dy_ = p.docked_pos(self.aspect(), offset, left_width)
            if abs(x - dx_) < 0.15 and abs(y - dy_) < 0.15:
                p.moved = False
                p.anchor = None
                return p
            # A panel that grew since the layout was saved would otherwise
            # come back with its bottom rows off the screen.
            if height < 2.0:
                y = max(-1.0 + height, min(1.0, y))
            p.set_pos(x, y)
        return p

    def dock_all(self) -> None:
        """Dock every un-moved panel; stack same-side panels top to bottom;
        top panels sit beside the left-hand stack."""
        aspect = self.aspect()
        left_width = max((q.width for q in self.panels.values()
                          if q.dock_side == "left" and not q.moved), default=0.0)
        for side in ("left", "right", "top", "bottom"):
            offset = 0.0
            for p in sorted((q for q in self.panels.values()
                             if q.dock_side == side),
                            key=lambda q: q.dock_order):
                p.dock(aspect, offset, left_width)
                if side in ("left", "right") and not p.moved:
                    offset += p.height + MARGIN
        self._last_aspect = aspect

    def reset_layout(self) -> None:
        for p in self.panels.values():
            p.moved = False
            p.anchor = None
            p.show()
        self.dock_all()
        self.save()

    def toggle_all(self) -> None:
        any_visible = any(p.visible for p in self.panels.values())
        for p in self.panels.values():
            (p.hide if any_visible else p.show)()

    def is_over_panel(self) -> bool:
        """True if the pointer is over any panel (so the camera must not grab it)."""
        mw = self.base.mouseWatcherNode
        if mw is None or not mw.hasMouse():
            return False
        m = mw.getMouse()
        mx, my = m.getX() * self.aspect(), m.getY()
        for p in self.panels.values():
            if not p.visible:
                continue
            x, y = p.get_pos()
            if x <= mx <= x + p.width and y - p.height <= my <= y:
                return True
        return False

    def _on_window(self, win=None) -> None:
        self.tick()

    def tick(self) -> None:
        """Re-dock if the aspect ratio changed. Called on window events and
        once a frame from the HUD as a belt-and-braces check, since the
        event can arrive before ShowBase has applied the new size."""
        a = self.aspect()
        if self._last_aspect is None or abs(a - self._last_aspect) > 1e-3:
            self.dock_all()

    # -- persistence -------------------------------------------------------
    def _load(self) -> dict:
        try:
            with open(_layout_path(), "r", encoding="utf-8") as fh:
                d = json.load(fh)
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}

    def save(self) -> None:
        data = {k: {"pos": list(p.get_pos()), "moved": p.moved,
                    "anchor": list(p.anchor) if p.anchor else None}
                for k, p in self.panels.items()}
        try:
            with open(_layout_path(), "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=1)
        except Exception:
            pass
