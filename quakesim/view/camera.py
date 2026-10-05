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

"""Camera modes.

Stellarium's control scheme is the reference: you stand in one spot, drag to
swing your view anywhere on the sphere including straight up at the zenith and
straight down at your feet, and scroll to narrow the field of view rather than
to move. That is the `look` mode here. The others add walking, orbiting the
outside of a building, and a plan view from above looking back down at the
ground from any bearing.

Head-shake matters too: standing on a floor that is accelerating, your head
goes with it. In first-person modes the camera is offset by the floor's own
displacement, so the view shakes because the floor shakes, not because a
random jitter was added to it.
"""

from __future__ import annotations

import math

from panda3d.core import MouseButton, NodePath, Vec3, WindowProperties

MODES = ["look", "walk", "orbit", "top", "fly"]
MODE_NAMES = {
    "look": "Look around (fixed stance)",
    "walk": "Walk",
    "orbit": "Orbit exterior",
    "top": "Plan view from above",
    "fly": "Free fly",
}


class CameraRig:
    """Owns the camera and translates input into movement."""

    def __init__(self, base, shake_gain: float = 1.0):
        self.base = base
        self.cam = base.camera
        self.mode = "look"
        self.heading = 0.0
        self.pitch = 0.0
        self.fov = 65.0
        self.base_pos = Vec3(0, 0, 1.65)
        self.orbit_target = Vec3(0, 0, 5.0)
        self.orbit_distance = 40.0
        self.top_altitude = 60.0
        self.move_speed = 4.2
        self.eye_height = 1.65
        self.shake_gain = shake_gain
        self.head_offset = Vec3(0, 0, 0)
        # where the floor you stand on has gone (a released slab): added to
        # the walking/looking position so the camera rides it down
        self.ride_offset = Vec3(0, 0, 0)
        self.mouse_captured = False
        self._last_mouse = None
        self._dragging = False
        self.keys = {k: False for k in
                     ("fwd", "back", "left", "right", "up", "down", "fast",
                      "yaw_l", "yaw_r")}

    # -- placement ---------------------------------------------------------
    def apply_viewpoint(self, vp) -> None:
        self.mode = vp.mode if vp.mode in MODES else "look"
        self.base_pos = Vec3(*vp.pos)
        self.heading, self.pitch = vp.hpr[0], vp.hpr[1]
        self.fov = vp.fov
        self.eye_height = vp.pos[2]
        if self.mode == "orbit":
            self.orbit_distance = max(Vec3(vp.pos).length(), 10.0)
        if self.mode == "top":
            self.top_altitude = vp.pos[2]
        self._clamp()

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            return
        if mode == "top" and self.mode != "top":
            self.top_altitude = max(self.top_altitude, 40.0)
            self.pitch = -89.0
        if mode in ("look", "walk") and self.mode in ("top", "orbit"):
            self.pitch = 0.0
        self.mode = mode

    # -- input -------------------------------------------------------------
    def set_key(self, name: str, state: bool) -> None:
        if name in self.keys:
            self.keys[name] = state

    def toggle_capture(self) -> bool:
        self.mouse_captured = not self.mouse_captured
        if self.base.win is None or not hasattr(self.base.win, "requestProperties"):
            return self.mouse_captured
        props = WindowProperties()
        props.setCursorHidden(self.mouse_captured)
        props.setMouseMode(WindowProperties.M_relative if self.mouse_captured
                           else WindowProperties.M_absolute)
        self.base.win.requestProperties(props)
        self._last_mouse = None
        return self.mouse_captured

    def set_dragging(self, state: bool) -> None:
        self._dragging = state
        self._last_mouse = None

    def zoom(self, direction: int) -> None:
        if self.mode == "orbit":
            self.orbit_distance = max(2.0, min(600.0,
                                               self.orbit_distance * (0.88 if direction > 0 else 1.14)))
        elif self.mode == "top":
            self.top_altitude = max(6.0, min(1200.0,
                                             self.top_altitude * (0.85 if direction > 0 else 1.18)))
        else:
            # Stellarium-style: scrolling narrows the field of view.
            self.fov = max(9.0, min(110.0, self.fov * (0.88 if direction > 0 else 1.14)))

    def _clamp(self) -> None:
        # Full sphere: straight up and straight down are both reachable.
        self.pitch = max(-89.9, min(89.9, self.pitch))
        self.heading %= 360.0

    # -- per-frame ---------------------------------------------------------
    def update(self, dt: float, floor_disp: Vec3 | None = None) -> None:
        self._mouse_look()
        self._move(dt)
        if floor_disp is not None:
            self.head_offset = floor_disp * self.shake_gain
        self._place()

    def _mouse_look(self) -> None:
        win = self.base.win
        if win is None or not hasattr(win, "getPointer"):
            return                      # offscreen buffer: no pointer to read
        md = win.getPointer(0)
        x, y = md.getX(), md.getY()
        if self._dragging:
            # The release event is swallowed when the button comes up over a
            # panel, so the button state itself is the authority.
            try:
                if not self.base.mouseWatcherNode.isButtonDown(MouseButton.one()):
                    self._dragging = False
            except Exception:
                pass
        active = self.mouse_captured or self._dragging
        if not active:
            self._last_mouse = (x, y)
            return
        if self._last_mouse is None:
            self._last_mouse = (x, y)
            if self.mouse_captured:
                self.base.win.movePointer(0, self.base.win.getXSize() // 2,
                                          self.base.win.getYSize() // 2)
                self._last_mouse = (self.base.win.getXSize() // 2,
                                    self.base.win.getYSize() // 2)
            return
        dx = x - self._last_mouse[0]
        dy = y - self._last_mouse[1]
        # Sensitivity scales with field of view, so zoomed in you get fine
        # control instead of the view flying past what you were aiming at.
        s = 0.16 * (self.fov / 65.0)
        self.heading -= dx * s
        self.pitch -= dy * s
        self._clamp()
        if self.mouse_captured:
            cx, cy = self.base.win.getXSize() // 2, self.base.win.getYSize() // 2
            self.base.win.movePointer(0, cx, cy)
            self._last_mouse = (cx, cy)
        else:
            self._last_mouse = (x, y)

    def _move(self, dt: float) -> None:
        k = self.keys
        speed = self.move_speed * (3.4 if k["fast"] else 1.0)
        if self.mode == "top":
            speed = max(speed, self.top_altitude * 0.35)

        fwd = back = strafe = lift = 0.0
        if k["fwd"]:
            fwd += 1
        if k["back"]:
            fwd -= 1
        if k["right"]:
            strafe += 1
        if k["left"]:
            strafe -= 1
        if k["up"]:
            lift += 1
        if k["down"]:
            lift -= 1
        if k["yaw_l"]:
            self.heading += 70.0 * dt
        if k["yaw_r"]:
            self.heading -= 70.0 * dt
        self._clamp()

        if self.mode == "look":
            return                      # fixed stance, like Stellarium

        h = math.radians(self.heading)
        if self.mode == "walk":
            f = Vec3(-math.sin(h), math.cos(h), 0)
            r = Vec3(math.cos(h), math.sin(h), 0)
            self.base_pos += (f * fwd + r * strafe) * speed * dt
            self.base_pos.setZ(self.eye_height + lift * speed * dt * 0.0)
        elif self.mode == "fly":
            p = math.radians(self.pitch)
            f = Vec3(-math.sin(h) * math.cos(p), math.cos(h) * math.cos(p),
                     math.sin(p))
            r = Vec3(math.cos(h), math.sin(h), 0)
            self.base_pos += (f * fwd + r * strafe + Vec3(0, 0, 1) * lift) * speed * dt
        elif self.mode == "top":
            f = Vec3(-math.sin(h), math.cos(h), 0)
            r = Vec3(math.cos(h), math.sin(h), 0)
            self.base_pos += (f * fwd + r * strafe) * speed * dt
            self.top_altitude = max(4.0, self.top_altitude - lift * speed * dt)
        elif self.mode == "orbit":
            self.orbit_target += Vec3(0, 0, 1) * lift * speed * dt
            self.orbit_distance = max(2.0, self.orbit_distance - fwd * speed * dt)

    def _place(self) -> None:
        lens = self.base.camLens
        lens.setFov(self.fov)
        off = self.head_offset

        if self.mode == "orbit":
            h = math.radians(self.heading)
            p = math.radians(max(min(self.pitch, 88.0), -88.0))
            d = self.orbit_distance
            pos = self.orbit_target + Vec3(math.sin(h) * math.cos(p) * d,
                                           -math.cos(h) * math.cos(p) * d,
                                           math.sin(-p) * d + 0.0)
            self.cam.setPos(pos + off)
            self.cam.lookAt(self.orbit_target)
        elif self.mode == "top":
            pos = Vec3(self.base_pos.getX(), self.base_pos.getY(),
                       self.top_altitude)
            self.cam.setPos(pos + off)
            self.cam.setHpr(self.heading, self.pitch, 0)
        else:
            self.cam.setPos(self.base_pos + off + self.ride_offset)
            self.cam.setHpr(self.heading, self.pitch, 0)

    # -- reporting ---------------------------------------------------------
    def describe(self) -> str:
        p = self.cam.getPos()
        return (f"{MODE_NAMES[self.mode]}  |  "
                f"x {p[0]:.1f}  y {p[1]:.1f}  z {p[2]:.1f}  |  "
                f"az {self.heading % 360:.0f}°  alt {self.pitch:+.0f}°  |  "
                f"FOV {self.fov:.0f}°")

    def altitude_word(self) -> str:
        if self.pitch > 80:
            return "zenith"
        if self.pitch < -80:
            return "nadir"
        if self.pitch > 20:
            return "sky"
        if self.pitch < -20:
            return "ground"
        return "horizon"
