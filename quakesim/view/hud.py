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

"""On-screen controls and readouts, as movable panels.

    EARTHQUAKE   docks left    -- what you set: magnitude, type, depth, distance...
    AT THIS SITE docks right   -- what that produces here: PGA, PGV, arrivals...
    INTENSITY    docks right   -- MMI / JMA / GB-T, and what breaks
    SEISMOGRAM   docks bottom  -- the three-component record with a time cursor
    STATUS       docks top     -- scene, camera, clock, frame rate, renderer

Grab any title bar to move a panel; L puts everything back; H hides them all.
"""

from __future__ import annotations

import math

import numpy as np
from collections import deque

from direct.gui.DirectGui import DGG, DirectButton, DirectFrame, DirectSlider
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import LineSegs, MouseButton, NodePath, TextNode

from ..seismo import constants as K
from ..seismo.intensity import china_intensity, shaking_colour
from ..seismo.site import SITES
from ..seismo.source import MW_MAX, TYPES, TYPE_ORDER
from .fonts import safe
from .panels import ACCENT, DIM, PanelManager
from .widgets import Dropdown, Overlay, button

TEXT = (0.90, 0.92, 0.94, 1)
WARN = (0.98, 0.72, 0.28, 1)
INK = (0.05, 0.06, 0.07, 1)

HELP_ROWS = [
    ("Mouse drag  /  Tab", "look around  (Tab captures the mouse; Tab again releases it)"),
    ("W A S D", "walk   ·   Space / Ctrl  up and down   ·   Shift  faster"),
    ("Mouse wheel", "zoom  (over a list it scrolls the list)"),
    ("1 - 9   or   [  ]", "viewpoints   ·   0  back to the current viewpoint"),
    ("V", "camera mode:  walk, look, orbit, top, fly"),
    ("N   /   Shift+N", "next  /  previous scene"),
    ("R", "run the earthquake"),
    ("K", "set this scene's collapse case, then R"),
    ("P", "play / pause   ·   the seismogram bar also seeks"),
    ("F", "reset the scene"),
    ("T", "time of day"),
    ("G", "show the collision shapes"),
    ("H   /   L", "hide the panels  /  put them back in place"),
    ("Drag a title bar", "move a panel  (the layout is remembered)"),
    ("F1   /   Esc", "this help  /  quit"),
]


def _text(parent, text, x, y, scale=0.030, colour=TEXT, align=TextNode.ALeft,
          wrap=None):
    return OnscreenText(text=text, parent=parent, pos=(x, y), scale=scale,
                        fg=colour, align=align, mayChange=True,
                        shadow=(0, 0, 0, 0.55), wordwrap=wrap)


class Control:
    """A labelled slider with a live value readout, laid out in panel space."""

    def __init__(self, parent, x, y, width, name, lo, hi, value, fmt,
                 on_change, unit=""):
        self.fmt, self.unit, self.on_change = fmt, unit, on_change
        self.title = _text(parent, name, x, y + 0.030, 0.030, DIM)
        self.value = _text(parent, "", x + width, y + 0.030, 0.032, ACCENT,
                           TextNode.ARight)
        self.slider = DirectSlider(
            parent=parent, pos=(x + width / 2, 0, y), scale=width / 2,
            range=(lo, hi), value=value, pageSize=(hi - lo) / 20.0,
            thumb_frameColor=ACCENT, thumb_relief=DGG.FLAT,
            thumb_frameSize=(-0.05, 0.05, -0.10, 0.10),
            frameColor=(0.20, 0.22, 0.26, 1), frameSize=(-1, 1, -0.035, 0.035),
            relief=DGG.FLAT, command=self._changed, suppressMouse=1)
        self._refresh()

    def _changed(self):
        self._refresh()
        if self.on_change:
            self.on_change(self.slider["value"])

    def _refresh(self):
        self.value.setText(f"{self.slider['value']:{self.fmt}}{self.unit}")

    def get(self) -> float:
        return float(self.slider["value"])

    def set(self, v: float) -> None:
        self.slider["value"] = float(v)
        self._refresh()

    def set_range(self, lo: float, hi: float) -> None:
        self.slider["range"] = (lo, hi)
        self.slider["value"] = min(max(self.slider["value"], lo), hi)
        self._refresh()


class Hud:
    def __init__(self, app):
        self.app = app
        self.pm = PanelManager(app)
        self._build_quake_panel()
        self._build_site_panel()
        self._build_intensity_panel()
        self._build_trace_panel()
        self._build_status_panel()
        self.toast = _text(app.aspect2d, "", 0.0, 0.58, 0.050, WARN,
                           TextNode.ACenter)
        self._toast_until = 0.0
        self.help = Overlay(app, "KEYS AND MOUSE", HELP_ROWS)
        self.pm.dock_all()

    # ------------------------------------------------------------------
    # panels
    # ------------------------------------------------------------------
    def _build_quake_panel(self):
        p = self.pm.add("quake", "EARTHQUAKE", 0.70, 1.94, "left")
        b = p.body
        x0, w = 0.03, 0.62
        self.menu_type = Dropdown(
            parent=b, pos=(x0, 0, -0.045), scale=0.038, width=w,
            items=[safe(TYPES[k].name) for k in TYPE_ORDER], initialitem=0,
            command=self._type_changed, max_rows=12)
        self.type_label = _text(b, "", x0, -0.140, 0.025, DIM)

        y, step = -0.225, 0.098
        self.c_mag = Control(b, x0, y, w, "Moment magnitude  Mw", 1.0, MW_MAX, 6.5,
                             ".2f", self._param)
        y -= step
        self.c_depth = Control(b, x0, y, w, "Focal depth", 0.5, 700.0, 10.0,
                               ".0f", self._param, " km")
        y -= step
        self.c_dist = Control(b, x0, y, w, "Epicentral distance", 0.5, 900.0,
                              20.0, ".0f", self._param, " km")
        y -= step
        self.c_stress = Control(b, x0, y, w, "Stress drop", 1.0, 1500.0, 70.0,
                                ".0f", self._param, " bar")
        y -= step
        self.c_az = Control(b, x0, y, w, "Azimuth to epicentre", 0.0, 359.0,
                            35.0, ".0f", self._param, "°")
        y -= step
        self.c_dir = Control(b, x0, y, w, "Rupture directivity", -1.0, 1.0,
                             0.0, "+.2f", self._param)
        y -= step

        _text(b, "Ground beneath the site", x0, y + 0.030, 0.030, DIM)
        self.site_keys = list(SITES.keys())
        self.menu_site = Dropdown(
            parent=b, pos=(x0, 0, y + 0.005), scale=0.034, width=w,
            items=[safe(SITES[k].name) for k in self.site_keys], initialitem=1,
            command=self._site_changed)
        y -= 0.150

        self.c_gain = Control(b, x0, y, w, "Shaking gain (×)", 0.0, 4.0, 1.0,
                              ".2f", self._gain_changed)
        y -= step
        self.c_speed = Control(b, x0, y, w, "Playback speed", 0.05, 2.0, 1.0,
                               ".2f", self._speed_changed, "×")
        y -= 0.095

        button(b, "RUN QUAKE  (R)", (x0 + 0.20, y), 0.040, self.app.run_quake,
               0.40, primary=True)
        button(b, "RESET  (F)", (x0 + 0.51, y), 0.036, self.app.reset_scene, 0.21)
        y -= 0.080
        # One click to the earthquake that brings this scene's structure
        # down (each scene names its own), then RUN.
        self.demo_button = button(b, "SET COLLAPSE CASE  (K)", (x0 + 0.31, y), 0.036,
                                  self.app.set_collapse_case, w,
                                  colour=(0.86, 0.40, 0.22, 1))
        y -= 0.070
        self.blurb = _text(b, "", x0, y, 0.025, DIM, wrap=25)

    def _build_site_panel(self):
        p = self.pm.add("site", "AT THIS SITE", 0.62, 1.17, "right", 0)
        self.readout = _text(p.body, "", 0.025, -0.045, 0.0275, TEXT)

    def _build_intensity_panel(self):
        p = self.pm.add("intensity", "INTENSITY", 0.62, 0.74, "right", 1)
        b = p.body
        self.intensity_box = DirectFrame(
            parent=b, frameColor=(0.3, 0.3, 0.3, 1),
            frameSize=(0.02, 0.60, -0.120, -0.015), pos=(0, 0, 0))
        self.intensity_text = _text(b, "", 0.04, -0.060, 0.0300, INK)
        self.intensity_text2 = _text(b, "", 0.04, -0.103, 0.0265, INK)
        self.effects = _text(b, "", 0.025, -0.160, 0.0245, DIM, wrap=23)

    def _build_trace_panel(self):
        p = self.pm.add("trace", "SEISMOGRAM   East / North / Up", 1.70, 0.53,
                        "bottom")
        self.trace_panel = p
        b = p.body
        self.trace_np: NodePath | None = None
        self.trace_label = _text(b, "", 0.02, -0.352, 0.024, DIM)
        ls = LineSegs()
        ls.setThickness(1.8)
        ls.setColor(1, 1, 1, 0.85)
        ls.moveTo(0, 0, -0.005)
        ls.drawTo(0, 0, -0.335)
        self.cursor_np = b.attachNewNode(ls.create())
        self.cursor_np.setX(0.02)
        # Transport: play/pause, skip, and a scrubber along the record. The
        # scrubber seeks (see QuakeSim.seek): forward it fast-forwards the
        # physics, backward it replays from the start, so what you land on
        # is what would really be there.
        y = -0.428
        self.play_button = button(b, "PLAY", (0.080, y), 0.034,
                                  self.app.toggle_pause, 0.12, primary=True)
        button(b, "-10 s", (0.200, y), 0.030, self.app.seek_relative, 0.10,
               extra_args=[-10.0])
        button(b, "+10 s", (0.310, y), 0.030, self.app.seek_relative, 0.10,
               extra_args=[10.0])
        xs0, xs1 = 0.39, p.width - 0.30
        self._scrub_set: deque = deque(maxlen=4)
        self._scrub_pending = None
        sx = (xs1 - xs0) / 2
        self.scrub = DirectSlider(
            parent=b, pos=((xs0 + xs1) / 2, 0, y + 0.012), scale=sx,
            range=(0.0, 1.0), value=0.0, pageSize=0.02,
            thumb_frameColor=(ACCENT, ACCENT, (0.55, 0.85, 1.0, 1), ACCENT),
            thumb_relief=DGG.FLAT,
            thumb_frameSize=(-0.010 / sx, 0.010 / sx, -0.024 / sx, 0.024 / sx),
            frameColor=(0.20, 0.22, 0.26, 1),
            frameSize=(-1, 1, -0.006 / sx, 0.006 / sx),
            relief=DGG.FLAT, command=self._scrubbed, suppressMouse=1)
        self.time_text = _text(b, "", p.width - 0.02, y, 0.027, TEXT,
                               TextNode.ARight)

    def _build_status_panel(self):
        p = self.pm.add("status", "STATUS     drag title bars to move panels  ·  "
                        "H hide  ·  L reset layout  ·  F1 help",
                        1.70, 0.40, "top")
        b = p.body
        # Scene and viewpoint choosers -- the same choices as the N key and
        # the number keys, but pickable.
        from ..world.scenes import SCENES, SCENE_ORDER
        y = -0.030
        _text(b, "Scene", 0.02, y - 0.036, 0.028, DIM)
        self.scene_keys = list(SCENE_ORDER)
        self.menu_scene = Dropdown(
            parent=b, pos=(0.13, 0, y), scale=0.036, width=0.44,
            items=[safe(SCENES[k].spec.short or SCENES[k].spec.name)
                   for k in self.scene_keys], initialitem=0,
            command=self._scene_changed)
        _text(b, "Viewpoint", 0.63, y - 0.036, 0.028, DIM)
        self.menu_vp = Dropdown(
            parent=b, pos=(0.80, 0, y), scale=0.036, width=0.62,
            items=["-"], initialitem=0, command=self._vp_changed)
        # Wrapped inside the panel: the renderer's name alone can be forty
        # characters, and nothing may run out of the box.
        self.status = _text(b, "", 0.02, -0.118, 0.0255, TEXT,
                            wrap=(p.width - 0.04) / 0.0255)

    # ------------------------------------------------------------------
    # callbacks
    # ------------------------------------------------------------------
    def _param(self, _=None):
        self.app.on_parameters_changed()

    def _type_changed(self, name):
        for k in TYPE_ORDER:
            if safe(TYPES[k].name) == name:
                self.app.on_type_changed(k)
                return

    def _scene_changed(self, name):
        from ..world.scenes import SCENES
        for k in self.scene_keys:
            if safe(SCENES[k].spec.short or SCENES[k].spec.name) == name:
                if k != self.app.scene_key:
                    self.app.request_scene(k)
                return

    def _vp_changed(self, name):
        for i, vp in enumerate(self.app.scene.spec.viewpoints):
            if safe(vp.name) == name:
                if i != self.app.vp_index:
                    self.app.apply_viewpoint(i)
                return

    def set_scene(self, key: str):
        """Reflect a scene load in the choosers (no callback)."""
        self.menu_scene.set(self.scene_keys.index(key), fCommand=0)
        names = [safe(vp.name) for vp in self.app.scene.spec.viewpoints] or ["-"]
        self.menu_vp["items"] = names
        self.menu_vp.set(0, fCommand=0)

    def set_viewpoint(self, index: int):
        if 0 <= index < len(self.menu_vp["items"]):
            self.menu_vp.set(index, fCommand=0)

    def _site_changed(self, name):
        for k in self.site_keys:
            if safe(SITES[k].name) == name:
                self.app.on_site_changed(k)
                return

    def _gain_changed(self, v):
        self.app.shaker.gain = float(v)

    def _speed_changed(self, v):
        self.app.shaker.speed = float(v)

    # ------------------------------------------------------------------
    # updates
    # ------------------------------------------------------------------
    def set_type(self, key: str):
        t = TYPES[key]
        self.menu_type.set(TYPE_ORDER.index(key), fCommand=0)
        lo, hi = t.mw_range
        d0, d1 = t.depth_km
        mech = {"strike_slip": "strike-slip", "clvd": "CLVD source",
                "isotropic": "isotropic source"}.get(t.mechanism,
                                                     t.mechanism.replace("_", " "))
        self.type_label.setText(
            f"{mech}  ·  typically M{max(1.0, lo):.1f}-{hi:.1f}"
            f"  ·  {d0:.0f}-{d1:.0f} km deep")
        self.blurb.setText(t.blurb)
        self.c_depth.set_range(*t.depth_km)
        self.c_stress.set_range(*t.stress_drop_bar)
        self.c_mag.set_range(max(1.0, t.mw_range[0]), MW_MAX)

    def set_site(self, key: str):
        self.menu_site.set(self.site_keys.index(key), fCommand=0)

    def show_results(self, src, site, gm, response, damage_lines):
        m = gm.meta
        mmi = m["mmi_text"]
        d = src.describe()
        arr = gm.arrivals
        cn = china_intensity(m["pga_g"] * K.G0, m["pgv_cms"] / 100.0)

        lines = [
            f"Peak acceleration   {m['pga_g']*100:8.2f} %g",
            f"                    {m['pga_g']*981:8.1f} cm/s²",
            f"Peak velocity       {m['pgv_cms']:8.2f} cm/s",
            f"Peak displacement   {m['pgd_cm']:8.2f} cm",
            f"Arias intensity     {m['arias_ms']:8.3f} m/s",
            f"Shaking (D5-95)     {m['d5_95_s']:8.1f} s",
            "",
            f"P wave arrives      {arr['P']:8.1f} s",
            f"S wave arrives      {arr['S']:8.1f} s",
            f"S minus P           {arr['S_minus_P']:8.1f} s",
            "",
            f"Site amplification  ×{m['site_amp']:6.2f}",
            f"Amplitude basis     "
            f"{'GMPE-anchored' if m['anchor_mode'] == 'gmpe' else 'physics model'}"
            f"{'  (capped at 3 g)' if m.get('saturated') else ''}",
            "",
            f"Seismic moment      {d['M0_Nm']:.2e} N·m",
            f"Energy              {d['energy_J']:.2e} J",
            f"  = TNT equivalent  {d['energy_tons_TNT']/1e3:.3g} kt",
            f"Corner frequency    {d['corner_freq_Hz']:8.3f} Hz",
            f"Rupture             {d['rupture_length_km']:.1f} × "
            f"{d['rupture_width_km']:.1f} km",
            f"Average slip        {d['avg_slip_m']:8.2f} m",
        ]
        if response is not None:
            roof = float(np.max(np.abs(response.floor_acc[-1]))) / K.G0 * 100
            lines += [
                "",
                f"Building T1         {response.periods[0]:8.2f} s",
                f"Roof acceleration   {roof:8.2f} %g",
                f"Peak drift ratio    {float(np.max(response.peak_drift))*100:8.3f} %",
                f"Residual drift      {float(np.max(response.residual_drift))*100:8.3f} %",
                f"Damage state        {response.damage_name}",
            ]
        self.readout.setText("\n".join(lines))

        col = shaking_colour(m["mmi"])
        self.intensity_box["frameColor"] = (col[0], col[1], col[2], 1)
        self.intensity_text.setText(f"MMI {mmi['roman']}   ·   {mmi['word']}")
        self.intensity_text2.setText(
            f"JMA shindo {m['jma']['class']}   ·   CSIS {cn:.1f}")
        self.effects.setText(mmi["text"] + "\n\n" + "\n".join(damage_lines[:7]))

    def draw_trace(self, gm):
        """Three-component accelerogram with the phase arrivals marked."""
        if self.trace_np:
            self.trace_np.removeNode()
            self.trace_np = None
        if gm is None:
            return
        body = self.trace_panel.body
        x0, x1 = 0.02, self.trace_panel.width - 0.02
        span = x1 - x0
        row_h, amp = 0.105, 0.046
        n = gm.n
        step = max(1, n // 1400)
        acc = gm.acc[:, ::step]
        t = np.arange(acc.shape[1]) * gm.dt * step
        peak = float(np.max(np.abs(gm.acc))) or 1.0
        dur = gm.duration or 1.0

        ls = LineSegs()
        ls.setThickness(1.2)
        colours = [(0.42, 0.78, 0.96, 1), (0.55, 0.92, 0.62, 1),
                   (0.98, 0.74, 0.40, 1)]
        for ci in range(3):
            base_y = -0.06 - ci * row_h
            ls.setColor(0.30, 0.32, 0.36, 1)
            ls.moveTo(x0, 0, base_y)
            ls.drawTo(x1, 0, base_y)
            ls.setColor(*colours[ci])
            for i in range(acc.shape[1]):
                x = x0 + span * (t[i] / dur)
                y = base_y + acc[ci, i] / peak * amp
                (ls.moveTo if i == 0 else ls.drawTo)(x, 0, y)

        for name, colour in (("P", (0.95, 0.95, 0.55, 1)),
                             ("S", (0.98, 0.55, 0.55, 1)),
                             ("Rayleigh", (0.72, 0.62, 0.98, 1))):
            ta = gm.arrivals.get(name, 0.0)
            if 0 < ta < dur:
                x = x0 + span * (ta / dur)
                ls.setColor(*colour)
                ls.moveTo(x, 0, -0.005)
                ls.drawTo(x, 0, -0.335)

        self.trace_np = body.attachNewNode(ls.create())
        self.trace_label.setText(
            f"full record {dur:.0f} s   ·   peak {peak/K.G0*100:.1f} %g   ·   "
            f"P {gm.arrivals['P']:.1f} s   S {gm.arrivals['S']:.1f} s   "
            f"Rayleigh {gm.arrivals['Rayleigh']:.1f} s")

    # -- transport ---------------------------------------------------------
    def _scrubbed(self):
        """Slider callback: fires a frame after a programmatic set too, so a
        value we set ourselves is not a seek. A drag is applied when the
        button is released (set_transport polls for that)."""
        v = float(self.scrub["value"])
        if any(abs(v - u) < 1e-7 for u in self._scrub_set):
            return
        self._scrub_pending = v
        dur = self.app.shaker.duration if self.app.shaker else 0.0
        if dur > 0:
            self.time_text.setText(f"{v * dur:6.1f} / {dur:.1f} s")

    def set_transport(self, shaker, seek_target):
        """Called every frame: the scrubber follows the clock, the button
        says what a click will do, and a finished drag becomes a seek."""
        dur = shaker.duration
        mouse_down = (self.app.mouseWatcherNode.isButtonDown(MouseButton.one())
                      if self.app.mouseWatcherNode else False)
        if self._scrub_pending is not None and not mouse_down:
            target, self._scrub_pending = self._scrub_pending, None
            if dur > 0:
                self.app.seek(target * dur)
        if self._scrub_pending is not None:
            return                          # the user is dragging: leave it
        if dur > 0:
            # While a seek runs the thumb sits where it was dropped; the
            # clock text shows the physics catching up to it.
            shown = shaker.time if seek_target is None else seek_target
            v = min(max(shown / dur, 0.0), 1.0)
            if abs(v - float(self.scrub["value"])) > 1e-7:
                self._scrub_set.append(v)
                self.scrub["value"] = v
            if seek_target is None:
                self.time_text.setText(f"{shaker.time:6.1f} / {dur:.1f} s")
            else:
                self.time_text.setText(f"{shaker.time:5.1f} → {seek_target:.1f} / {dur:.1f} s")
        else:
            self.time_text.setText("")
        if seek_target is not None:
            label = "STOP"
        elif shaker.playing and not shaker.paused:
            label = "PAUSE"
        else:
            label = "PLAY"
        if self.play_button["text"] != label:
            self.play_button["text"] = label

    def draw_cursor(self, gm, t_now: float):
        if gm is None or gm.duration <= 0:
            return
        x0, x1 = 0.02, self.trace_panel.width - 0.02
        self.cursor_np.setX(x0 + (x1 - x0) * min(t_now / gm.duration, 1.0))

    def set_status(self, text: str):
        self.status.setText(text)

    def show_toast(self, text: str, seconds: float = 2.5, now: float = 0.0):
        self.toast.setText(text)
        self._toast_until = now + seconds

    def tick_toast(self, now: float):
        self.pm.tick()
        if self._toast_until and now > self._toast_until:
            self.toast.setText("")
            self._toast_until = 0.0

    # -- layout ------------------------------------------------------------
    def toggle_controls(self):
        self.pm.toggle_all()

    def reset_layout(self):
        self.pm.reset_layout()

    def is_over_panel(self) -> bool:
        return (self.pm.dragging is not None or self.pm.is_over_panel()
                or self.help.visible or self.popup_open())

    def popup_open(self) -> bool:
        return any(m.open for m in (self.menu_type, self.menu_site,
                                    self.menu_scene, self.menu_vp))

    def toggle_help(self):
        self.help.toggle()
