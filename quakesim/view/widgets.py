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

"""Styled widgets for the panels.

Panda's stock DirectOptionMenu opens a grey Windows-95 list on top of the
button, tall enough for every item. `Dropdown` is a replacement built from
the DirectGUI parts: a flat field with an accent triangle, and a list that
opens *below* it in the panels' colours, at most `max_rows` rows tall with a
real scroll bar (drag it, click its arrows, or roll the wheel over the list)
when there are more. Clicking anywhere else closes it.

`button()` makes the flat buttons the panels use, with hover and press
shading. `Overlay` is a centred modal card (the F1 help).
"""

from __future__ import annotations

from direct.gui.DirectGui import (DGG, DirectButton, DirectFrame,
                                  DirectScrolledFrame)
from direct.gui.OnscreenText import OnscreenText
from direct.showbase import ShowBaseGlobal
from panda3d.core import (Geom, GeomNode, GeomTriangles, GeomVertexData,
                          GeomVertexFormat, GeomVertexWriter, MouseButton,
                          NodePath, OmniBoundingVolume, PGButton, TextNode)

from .panels import ACCENT, DIM

TEXT = (0.90, 0.92, 0.94, 1)
FIELD = (0.13, 0.15, 0.19, 1)          # the closed control
FIELD_HOVER = (0.19, 0.22, 0.27, 1)
LIST = (0.10, 0.11, 0.14, 1)           # the open list
BORDER = (0.36, 0.72, 0.92, 0.60)      # accent hairline round the list
HIGHLIGHT = (0.20, 0.40, 0.58, 1)      # the row under the pointer
INK = (0.05, 0.06, 0.07, 1)
SCROLL_TRACK = (0.15, 0.17, 0.21, 1)
SCROLL_THUMB = (0.42, 0.48, 0.56, 1)
SCROLL_THUMB_HOVER = (0.56, 0.64, 0.74, 1)

WHEEL_UP = PGButton.getPressPrefix() + MouseButton.wheelUp().getName() + "-"
WHEEL_DOWN = PGButton.getPressPrefix() + MouseButton.wheelDown().getName() + "-"


def _shade(c, k):
    """Lighten (k > 0) or darken (k < 0) a colour, keeping its alpha."""
    return (min(1.0, c[0] + k), min(1.0, c[1] + k), min(1.0, c[2] + k), c[3])


def _triangle(colour, size: float = 0.30) -> NodePath:
    """A small downward-pointing triangle, the dropdown's marker."""
    fmt = GeomVertexFormat.getV3c4()
    vdata = GeomVertexData("tri", fmt, Geom.UHStatic)
    vw = GeomVertexWriter(vdata, "vertex")
    cw = GeomVertexWriter(vdata, "color")
    for x, z in ((-size, size * 0.55), (size, size * 0.55), (0.0, -size * 0.55)):
        vw.addData3(x, 0, z)
        cw.addData4(*colour)
    tri = GeomTriangles(Geom.UHStatic)
    tri.addVertices(0, 2, 1)
    geom = Geom(vdata)
    geom.addPrimitive(tri)
    node = GeomNode("marker")
    node.addGeom(geom)
    return NodePath(node)


class Dropdown:
    """A drop-down list in the panels' style.

    `pos` is the top-left corner of the control in the parent's units,
    `width` its width in the same units, `scale` the text scale. `items` are
    strings; `command(item)` is called when the user picks one (not when
    `set` is called from code, unless asked).
    """

    ROW = 1.45              # row height, in text units
    PAD = 0.40              # text inset from the left edge
    GAP = 0.10              # between the control and the list
    MARGIN = 0.08           # border round the list
    SCROLL_W = 0.95         # scroll bar width

    def __init__(self, parent, pos, scale, width, items=(), initialitem=0,
                 command=None, max_rows=10):
        self.scale = float(scale)
        self.width = float(width)
        self.command = command
        self.max_rows = int(max_rows)
        self.items: list[str] = []
        self.selected: int | None = None
        self.rows: list[DirectButton] = []
        self.open = False
        w = self.width / self.scale
        self._w = w
        self.control = DirectButton(
            parent=parent, pos=(pos[0], 0, pos[2]), scale=self.scale,
            frameSize=(0, w, -self.ROW, 0),
            frameColor=(FIELD, FIELD_HOVER, FIELD_HOVER, FIELD),
            relief=DGG.FLAT, text="", text_align=TextNode.ALeft,
            text_pos=(self.PAD, -self.ROW + 0.45), text_fg=TEXT,
            textMayChange=True, command=self.toggle, suppressMouse=1,
            pressEffect=0)
        self.marker = _triangle(ACCENT)
        self.marker.reparentTo(self.control)
        self.marker.setPos(w - 0.62, 0, -self.ROW / 2.0)
        # A full-screen invisible catcher for clicks outside the open list.
        # Created before the list so the list's rows win the click.
        self.cancel = DirectFrame(parent=self.control, frameSize=(-1, 1, -1, 1),
                                  relief=None, state=DGG.NORMAL, suppressMouse=1)
        self.cancel.setBin("gui-popup", 0)
        self.cancel.node().setBounds(OmniBoundingVolume())
        self.cancel.bind(DGG.B1PRESS, self._cancel)
        self.cancel.bind(DGG.B3PRESS, self._cancel)
        self.cancel.hide()
        # The list: a bordered frame holding a scrolled frame of rows.
        self.popup = DirectFrame(parent=self.control, frameColor=BORDER,
                                 relief=DGG.FLAT, frameSize=(0, 1, -1, 0),
                                 state=DGG.NORMAL, suppressMouse=1)
        self.popup.setBin("gui-popup", 0)
        sw = self.SCROLL_W
        self.sf = DirectScrolledFrame(
            parent=self.popup, frameSize=(0, w, -self.ROW, 0),
            canvasSize=(0, w, -self.ROW, 0), frameColor=LIST, relief=DGG.FLAT,
            scrollBarWidth=sw, borderWidth=(0, 0), state=DGG.NORMAL,
            suppressMouse=1, manageScrollBars=1, autoHideScrollBars=1,
            verticalScroll_relief=DGG.FLAT, verticalScroll_frameColor=SCROLL_TRACK,
            verticalScroll_borderWidth=(0, 0),
            verticalScroll_thumb_relief=DGG.FLAT,
            verticalScroll_thumb_frameColor=(SCROLL_THUMB, SCROLL_THUMB_HOVER,
                                             SCROLL_THUMB_HOVER, SCROLL_THUMB),
            verticalScroll_incButton_relief=DGG.FLAT,
            verticalScroll_incButton_frameSize=(-sw / 2, sw / 2, -sw / 2, sw / 2),
            verticalScroll_incButton_frameColor=(SCROLL_TRACK, SCROLL_THUMB,
                                                 _shade(SCROLL_TRACK, 0.08),
                                                 SCROLL_TRACK),
            verticalScroll_decButton_relief=DGG.FLAT,
            verticalScroll_decButton_frameSize=(-sw / 2, sw / 2, -sw / 2, sw / 2),
            verticalScroll_decButton_frameColor=(SCROLL_TRACK, SCROLL_THUMB,
                                                 _shade(SCROLL_TRACK, 0.08),
                                                 SCROLL_TRACK),
            horizontalScroll_relief=None)
        self.sf.horizontalScroll.hide()
        self.bar = self.sf.verticalScroll
        self._arrows = []
        for btn, flip in ((self.bar.decButton, True), (self.bar.incButton, False)):
            tri = _triangle(DIM, 0.30)
            tri.reparentTo(btn)
            if flip:
                tri.setR(180)
            self._arrows.append(tri)
        self._list_w = w
        for np_ in (self.sf, self.bar, self.bar.thumb, self.bar.incButton,
                    self.bar.decButton):
            np_.bind(WHEEL_UP, self._wheel, [-1])
            np_.bind(WHEEL_DOWN, self._wheel, [1])
        self.popup.hide()
        self.set_items(items)
        if items:
            self.set(0 if initialitem is None else initialitem, fCommand=0)

    # -- items ---------------------------------------------------------------
    def set_items(self, items) -> None:
        for r in self.rows:
            r.destroy()
        self.rows = []
        self.items = [str(s) for s in items]
        n = len(self.items)
        canvas = self.sf.getCanvas()
        # The list is as wide as the control, or as wide as its longest
        # item plus the scroll bar if that is more.
        tn = TextNode("measure")
        font = DGG.getDefaultFont()
        if font is not None:
            tn.setFont(font)
        longest = 0.0
        for name in self.items:
            tn.setText(name)
            longest = max(longest, tn.getWidth())
        scroll = self.SCROLL_W if n > self.max_rows else 0.0
        self._list_w = max(self._w, longest + 2 * self.PAD + scroll)
        wc = self._list_w - scroll
        for i, name in enumerate(self.items):
            r = DirectButton(
                parent=canvas, pos=(0, 0, -i * self.ROW),
                frameSize=(0, wc, -self.ROW, 0),
                frameColor=(LIST, HIGHLIGHT, HIGHLIGHT, LIST), relief=DGG.FLAT,
                text=name, text_align=TextNode.ALeft,
                text_pos=(self.PAD, -self.ROW + 0.45), text_fg=TEXT,
                command=self._choose, extraArgs=[i], suppressMouse=1,
                pressEffect=0)
            r.bind(WHEEL_UP, self._wheel, [-1])
            r.bind(WHEEL_DOWN, self._wheel, [1])
            self.rows.append(r)
        self.sf["canvasSize"] = (0, wc, -max(n, 1) * self.ROW, 0)
        if self.selected is not None and self.selected >= n:
            self.selected = None
        self._paint_selection()
        if self.open:
            self.close()

    def __setitem__(self, key, value):
        if key == "items":
            self.set_items(value)
        elif key == "text":
            self.control["text"] = value
        else:
            raise KeyError(key)

    def __getitem__(self, key):
        if key == "items":
            return list(self.items)
        if key == "text":
            return self.control["text"]
        raise KeyError(key)

    def index(self, item) -> int | None:
        if isinstance(item, int):
            return item if 0 <= item < len(self.items) else None
        return self.items.index(item) if item in self.items else None

    def set(self, item, fCommand=1) -> None:
        i = self.index(item) if item is not None else None
        self.selected = i
        self.control["text"] = self.items[i] if i is not None else ""
        self._paint_selection()
        if fCommand and i is not None and self.command:
            self.command(self.items[i])

    def get(self) -> str | None:
        return self.items[self.selected] if self.selected is not None else None

    def _paint_selection(self):
        for i, r in enumerate(self.rows):
            r["text_fg"] = ACCENT if i == self.selected else TEXT

    def _choose(self, i):
        self.close()
        self.set(i, fCommand=1)

    # -- open / close --------------------------------------------------------
    def toggle(self):
        (self.close if self.open else self.show)()

    def _cancel(self, _event=None):
        self.close()

    def close(self):
        self.open = False
        self.popup.hide()
        self.cancel.hide()

    def show(self, _event=None):
        n = len(self.items)
        if n == 0:
            return
        r2d = ShowBaseGlobal.render2d
        sz = self.control.getScale(r2d)[2]          # screen units per text unit
        top = self.control.getPos(r2d)[2]
        bottom = top - self.ROW * sz
        row = self.ROW * sz
        fixed = (self.GAP + 2 * self.MARGIN) * sz
        fit_below = int((bottom - (-1.0) - fixed) // row)
        fit_above = int((1.0 - top - fixed) // row)
        rows = min(n, self.max_rows)
        if rows <= fit_below:
            below = True
        elif rows <= fit_above:
            below = False
        else:
            below = fit_below >= fit_above
            rows = max(1, min(rows, fit_below if below else fit_above))
        m, w = self.MARGIN, self._list_w
        h = rows * self.ROW
        scrolling = n > rows
        wc = w - (self.SCROLL_W if scrolling else 0.0)
        for r in self.rows:
            fs = r["frameSize"]
            if abs(fs[1] - wc) > 1e-6:
                r["frameSize"] = (0, wc, fs[2], fs[3])
        self.sf["canvasSize"] = (0, wc, -n * self.ROW, 0)
        self.sf["frameSize"] = (0, w, -h, 0)
        self.popup["frameSize"] = (-m, w + m, -h - m, m)
        if below:
            self.popup.setPos(0, 0, -self.ROW - self.GAP - m)
        else:
            self.popup.setPos(0, 0, self.GAP + m + h)
        # Never past the right edge of the window.
        px = self.popup.getPos(r2d)[0]
        sx = self.popup.getScale(r2d)[0]
        over = px + (w + m) * sx - 1.0
        if over > 0:
            self.popup.setX(self.popup.getX() - over / sx)
        self.popup.show()
        self.cancel.show()
        self.cancel.setPos(r2d, 0, 0, 0)
        self.cancel.setScale(r2d, 1, 1, 1)
        self.open = True
        # Bring the chosen item into view.
        if scrolling and self.selected is not None:
            lo, hi = self.bar["range"]
            span = max(hi - lo, 1e-9)
            want = (self.selected - rows // 2) * self.ROW
            self.bar["value"] = lo + min(max(want, 0.0), n * self.ROW - h) * span / max(n * self.ROW - h, 1e-9)
        else:
            self.bar["value"] = self.bar["range"][0]

    def _wheel(self, direction, _event=None):
        if not self.open:
            return
        lo, hi = self.bar["range"]
        n = len(self.items)
        h = -self.sf["frameSize"][2]
        hidden = max(n * self.ROW - h, 1e-9)
        step = self.ROW * (hi - lo) / hidden          # one row per notch
        v = float(self.bar["value"]) + direction * step
        self.bar["value"] = min(max(v, lo), hi)

    # -- misc ----------------------------------------------------------------
    def destroy(self):
        self.close()
        self.control.destroy()


def button(parent, text, pos, scale, command, width, primary=False,
           colour=None, extra_args=None):
    """A flat button in the panel style, `width` in the parent's units."""
    fc = colour or (ACCENT if primary else (0.26, 0.29, 0.34, 1))
    half = width / scale / 2.0
    return DirectButton(parent=parent, pos=(pos[0], 0, pos[1]), scale=scale,
                        text=text, text_fg=INK if (primary or colour) else TEXT,
                        frameColor=(fc, _shade(fc, -0.10), _shade(fc, 0.06), fc),
                        relief=DGG.FLAT, frameSize=(-half, half, -0.55, 0.85),
                        text_align=TextNode.ACenter, command=command,
                        extraArgs=extra_args or [], suppressMouse=1,
                        pressEffect=0, textMayChange=True)


class Overlay:
    """A centred card above everything, closed by its button, F1 or a click
    outside. Rows are (key, what it does) pairs."""

    def __init__(self, base, title, rows, width=1.60, key_col=0.42,
                 line=0.052, scale=0.031, on_close=None):
        self.on_close = on_close
        n = len(rows)
        height = 0.14 + n * line + 0.13
        self.cancel = DirectFrame(parent=base.aspect2d, frameSize=(-1, 1, -1, 1),
                                  frameColor=(0, 0, 0, 0.45), relief=DGG.FLAT,
                                  state=DGG.NORMAL, suppressMouse=1)
        self.cancel.setBin("gui-popup", 0)
        self.cancel.node().setBounds(OmniBoundingVolume())
        self.cancel.setTransparency(1)
        self.cancel.bind(DGG.B1PRESS, lambda e: self.hide())
        self.frame = DirectFrame(parent=base.aspect2d,
                                 frameColor=(0.07, 0.08, 0.10, 0.97),
                                 frameSize=(0, width, -height, 0),
                                 pos=(-width / 2, 0, height / 2),
                                 state=DGG.NORMAL, suppressMouse=1)
        self.frame.setBin("gui-popup", 1)
        self.frame.setTransparency(1)
        bar = DirectFrame(parent=self.frame, frameColor=(0.11, 0.13, 0.17, 1),
                          frameSize=(0, width, -0.075, 0))
        OnscreenText(text=title, parent=bar, pos=(0.03, -0.05), scale=0.036,
                     fg=ACCENT, align=TextNode.ALeft)
        y = -0.125
        for key, what in rows:
            OnscreenText(text=key, parent=self.frame, pos=(0.04, y), scale=scale,
                         fg=ACCENT, align=TextNode.ALeft)
            OnscreenText(text=what, parent=self.frame, pos=(0.04 + key_col, y),
                         scale=scale, fg=TEXT, align=TextNode.ALeft)
            y -= line
        button(self.frame, "CLOSE  (F1)", (width / 2, -height + 0.055), 0.034,
               self.hide, 0.30, primary=True)
        self.hide()

    @property
    def visible(self) -> bool:
        return not self.frame.isHidden()

    def show(self):
        r2d = ShowBaseGlobal.render2d
        self.cancel.show()
        self.cancel.setPos(r2d, 0, 0, 0)
        self.cancel.setScale(r2d, 1, 1, 1)
        self.frame.show()

    def hide(self, _event=None):
        self.cancel.hide()
        self.frame.hide()
        if self.on_close:
            self.on_close()

    def toggle(self):
        (self.hide if self.visible else self.show)()
