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

"""The loading screen.

Building a scene, synthesising a record and seeking through one are all
blocking jobs of a few seconds. This draws a progress card over the window
while they run -- title, what is happening now, a bar and the percentage --
and renders a frame each time the job reports progress, so the window keeps
updating instead of freezing.

Three looks:

    splash   a scene is being built: the 3D view is hidden behind an opaque
             backdrop, Adobe-style, because the old scene is gone and the
             new one is half built
    card     the record is being synthesised: a dimmed scene with the card
             over it, since the scene itself is not changing
    freeze   a seek: the last frame is kept on screen as a still picture
             with the card over it, so the physics catching up with the new
             time is never shown as a fast-forward
"""

from __future__ import annotations

import time

from direct.gui.DirectGui import DGG, DirectFrame
from direct.gui.OnscreenImage import OnscreenImage
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import OmniBoundingVolume, TextNode

from .panels import ACCENT, DIM
from .widgets import button

TEXT = (0.90, 0.92, 0.94, 1)
CARD = (0.07, 0.08, 0.10, 0.97)
TRACK = (0.17, 0.19, 0.23, 1)
SPLASH = (0.035, 0.040, 0.050, 1)


class LoadingScreen:
    W, H = 1.20, 0.36
    RENDER_EVERY = 0.045            # s between progress frames

    def __init__(self, base, app_name: str = "Earthquake Simulator"):
        self.base = base
        self.active = False
        self._shown = False
        self._delay = 0.0
        self._t0 = 0.0
        self._last_render = 0.0
        self._mode = "card"
        self._render_hidden = False
        self._texture = None

        # The still picture of the last frame (seeks), below the panels;
        # made when the first one is taken.
        self.frozen: OnscreenImage | None = None
        # The backdrop: opaque for a splash, a dim veil otherwise. Above
        # every panel, and it swallows clicks. It lives under aspect2d like
        # the card, created first, so that the card is drawn after it (the
        # popup bin draws in scene-graph order); _show() stretches it over
        # the whole window in render2d units.
        self.dim = DirectFrame(parent=base.aspect2d, frameSize=(-1, 1, -1, 1),
                               frameColor=SPLASH, relief=DGG.FLAT,
                               state=DGG.NORMAL, suppressMouse=1)
        self.dim.setBin("gui-popup", 0)
        self.dim.node().setBounds(OmniBoundingVolume())
        self.dim.setTransparency(1)
        self.dim.hide()

        # The splash wordmark, above the card, for scene loads only.
        self.wordmark = OnscreenText(text=app_name.upper(), parent=base.aspect2d,
                                     pos=(0, 0.40), scale=0.085, fg=ACCENT,
                                     align=TextNode.ACenter, mayChange=False)
        self.wordmark.setBin("gui-popup", 1)
        self.tagline = OnscreenText(
            text="source  ·  path  ·  site  ·  building  ·  room  ·  object",
            parent=base.aspect2d, pos=(0, 0.31), scale=0.032, fg=DIM,
            align=TextNode.ACenter, mayChange=False)
        self.tagline.setBin("gui-popup", 1)
        self.wordmark.hide()
        self.tagline.hide()

        w, h = self.W, self.H
        self.card = DirectFrame(parent=base.aspect2d, frameColor=CARD,
                                frameSize=(0, w, -h, 0), pos=(-w / 2, 0, h / 2),
                                state=DGG.NORMAL, suppressMouse=1)
        self.card.setBin("gui-popup", 1)
        self.card.setTransparency(1)
        self.title = OnscreenText(text="", parent=self.card, pos=(0.05, -0.085),
                                  scale=0.048, fg=ACCENT, align=TextNode.ALeft,
                                  mayChange=True)
        self.status = OnscreenText(text="", parent=self.card, pos=(0.05, -0.160),
                                   scale=0.030, fg=TEXT, align=TextNode.ALeft,
                                   mayChange=True)
        self.percent = OnscreenText(text="0 %", parent=self.card,
                                    pos=(w - 0.05, -0.160), scale=0.034,
                                    fg=ACCENT, align=TextNode.ARight,
                                    mayChange=True)
        bx0, bx1, by = 0.05, w - 0.05, -0.235
        self._bar = (bx0, bx1, by)
        self.track = DirectFrame(parent=self.card, frameColor=TRACK, relief=DGG.FLAT,
                                 frameSize=(bx0, bx1, by - 0.012, by + 0.012))
        self.fill = DirectFrame(parent=self.card, frameColor=ACCENT, relief=DGG.FLAT,
                                frameSize=(bx0, bx0, by - 0.012, by + 0.012))
        self.footer = OnscreenText(text=app_name, parent=self.card,
                                   pos=(0.05, -h + 0.045), scale=0.026,
                                   fg=DIM, align=TextNode.ALeft, mayChange=True)
        self._cancel = None
        self.cancel_button = button(self.card, "CANCEL  (Esc)", (w - 0.19, -h + 0.052),
                                    0.030, self._on_cancel, 0.26)
        self.cancel_button.hide()
        self.card.hide()

    # -- lifecycle -----------------------------------------------------------
    def begin(self, title: str, status: str = "", mode: str = "card",
              delay: float = 0.0, cancel=None) -> None:
        """Start a job. `delay` holds the card back that long, so a job that
        finishes quickly never flashes one. `cancel`, if given, is what the
        card's CANCEL button calls."""
        if self.active:
            self.end()
        self.active = True
        self._mode = mode
        self._cancel = cancel
        if cancel is not None:
            self.cancel_button.show()
        else:
            self.cancel_button.hide()
        self._delay = delay
        self._t0 = time.perf_counter()
        self._shown = False
        self.title.setText(title)
        self.status.setText(status)
        self.percent.setText("0 %")
        self._set_fill(0.0)
        if mode == "freeze":
            self._freeze_frame()
        if delay <= 0.0:
            self._show()
            self._render(force=True)

    def set(self, fraction: float, status: str | None = None,
            render: bool = True) -> None:
        """Report progress. From a blocking job leave `render` on, so the
        window updates; from a per-frame job pass False (the frame is
        rendered anyway)."""
        if not self.active:
            return
        f = min(max(float(fraction), 0.0), 1.0)
        if status is not None:
            self.status.setText(status)
        self.percent.setText(f"{int(f * 100 + 0.5)} %")
        self._set_fill(f)
        if not self._shown:
            if time.perf_counter() - self._t0 < self._delay:
                return
            self._show()
        if render:
            self._render()

    def span(self, lo: float, hi: float):
        """A progress function for a sub-job that owns [lo, hi] of the bar."""
        def report(fraction, status=None, render=True):
            self.set(lo + (hi - lo) * min(max(fraction, 0.0), 1.0), status, render)
        return report

    def end(self) -> None:
        if not self.active:
            return
        self.active = False
        self.card.hide()
        self.dim.hide()
        self.wordmark.hide()
        self.tagline.hide()
        if self.frozen is not None:
            self.frozen.hide()
        if self._render_hidden:
            self.base.render.show()
            self._render_hidden = False
        self._texture = None

    # -- internals -----------------------------------------------------------
    def _on_cancel(self) -> None:
        cb = self._cancel
        if cb is not None and self.active:
            cb()

    def _set_fill(self, f: float) -> None:
        bx0, bx1, by = self._bar
        self.fill["frameSize"] = (bx0, bx0 + (bx1 - bx0) * f, by - 0.012, by + 0.012)

    def _show(self) -> None:
        self._shown = True
        if self._mode == "splash":
            self.dim["frameColor"] = SPLASH
            self.wordmark.show()
            self.tagline.show()
            if not self.base.render.isHidden():
                self.base.render.hide()
                self._render_hidden = True
        elif self._mode == "freeze":
            self.dim["frameColor"] = (0.0, 0.0, 0.0, 0.30)
            if self._texture is not None:
                self.frozen.show()
                if not self.base.render.isHidden():
                    self.base.render.hide()
                    self._render_hidden = True
        else:
            self.dim["frameColor"] = (0.0, 0.0, 0.0, 0.45)
        r2d = self.base.render2d
        self.dim.setPos(r2d, 0, 0, 0)
        self.dim.setScale(r2d, 1, 1, 1)
        self.dim.show()
        self.card.show()

    def _freeze_frame(self) -> None:
        """Keep the last frame as a picture while the physics catches up."""
        self._texture = None
        try:
            tex = self.base.win.getScreenshot()
            if tex is None:
                return
            if self.frozen is None:
                self.frozen = OnscreenImage(image=tex, parent=self.base.render2d,
                                            pos=(0, 0, 0), scale=(1, 1, 1))
            else:
                self.frozen.setImage(tex)
            self.frozen.setBin("background", 0)
            self.frozen.hide()
            self._texture = tex
        except Exception:
            self._texture = None

    def _render(self, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self._last_render < self.RENDER_EVERY:
            return
        self._last_render = now
        try:
            self.base.graphicsEngine.renderFrame()
        except Exception:
            pass
