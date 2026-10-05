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

"""The simulator application.

Ties the chain together: source -> path -> site -> building -> room -> objects,
and puts a camera anywhere you like in it.
"""

from __future__ import annotations

import math
import time

import numpy as np
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import (AntialiasAttrib, ClockObject, ConfigVariableString,
                          NodePath, Vec3, WindowProperties, loadPrcFileData)

from dataclasses import dataclass

from ..seismo import constants as K
from ..seismo.intensity import object_response_summary
from ..seismo.site import SITES
from ..seismo.source import TYPES, TYPE_ORDER, Source
from ..seismo.synth import synthesize
from ..sim.physics import PhysicsWorld, Shaker
from ..structure import damage as dmg
from ..structure import mdof
from ..world.scenes import SCENES, SCENE_ORDER
from .camera import MODES, CameraRig
from .hud import Hud
from .loading import LoadingScreen
from .sky import Sky

# Free objects a scene builds at a given detail, measured, so the loading bar
# can say how far along the build is (it is clamped, so an estimate is
# enough). The count of a scene built this session replaces its entry.
EXPECTED_BODIES = {
    "warehouse":   [(0.35, 904), (0.6, 1418), (1.0, 4685), (1.5, 6500)],
    "kl_highrise": [(0.35, 87), (0.6, 87), (1.0, 87), (1.5, 87)],
    "ranau_house": [(0.35, 87), (0.6, 115), (1.0, 159), (1.5, 215)],
    "supermarket": [(0.35, 1873), (0.6, 6045), (1.0, 10840), (1.5, 15427)],
    "shophouse":   [(0.35, 147), (0.6, 187), (1.0, 287), (1.5, 390)],
}


def expected_bodies(key: str, detail: float) -> int:
    pts = EXPECTED_BODIES.get(key)
    if not pts:
        return 2000
    pts = sorted(pts)
    if detail <= pts[0][0]:
        return pts[0][1]
    for (d0, n0), (d1, n1) in zip(pts, pts[1:]):
        if d0 <= detail <= d1:
            return int(n0 + (n1 - n0) * (detail - d0) / max(d1 - d0, 1e-9))
    (d0, n0), (d1, n1) = pts[-2], pts[-1]
    return int(n1 + (n1 - n0) * (detail - d1) / max(d1 - d0, 1e-9))


class HeadResponse:
    """The observer's own body, as a single-degree-of-freedom oscillator.

    Standing on a shaking floor, your feet go with the floor and your head does
    not: you are an inverted pendulum with a period near two thirds of a second
    and heavy damping. The camera is offset by head-minus-floor, so the view
    lurches the way a real one does rather than being given an arbitrary
    jitter. Sitting or bracing raises the frequency and the damping.
    """

    def __init__(self, period: float = 0.62, damping: float = 0.28):
        self.w = 2.0 * math.pi / period
        self.z = damping
        self.u = np.zeros(3)      # head displacement relative to the floor
        self.v = np.zeros(3)

    def reset(self):
        self.u[:] = 0.0
        self.v[:] = 0.0

    def step(self, floor_acc: np.ndarray, dt: float) -> np.ndarray:
        dt = min(dt, 0.02)
        a = -floor_acc - 2.0 * self.z * self.w * self.v - self.w ** 2 * self.u
        self.v += a * dt
        self.u += self.v * dt
        # Your legs only let your head move so far before you take a step.
        np.clip(self.u, -0.28, 0.28, out=self.u)
        return self.u


@dataclass
class Snapshot:
    """The whole simulation at one instant of the record: every body, the
    structural record, the scene's own flags and the observer's head. Taken
    every few seconds while the record plays so that a seek backward, or
    into any second already simulated, is a restore plus a short catch-up
    rather than a replay from the start -- and so that a seek can be
    cancelled back to exactly where it began."""
    time: float
    world: object
    scene: dict
    head_u: np.ndarray
    head_v: np.ndarray


SNAPSHOTS_MAX = 150


# Rendering quality presets. "auto" picks from what the GPU reports.
QUALITY = {
    #          msaa  shadows  shadow-map  normal-maps
    "low":    (0,    False,   0,          False),
    "medium": (2,    True,    1024,       False),
    "high":   (4,    True,    2048,       True),
}


class QuakeSim(ShowBase):
    def __init__(self, scene_key: str = "warehouse", detail: float = 0.6,
                 seed: int = 4, use_pbr: bool = True, quality: str = "auto",
                 physics_hz: float = 120.0):
        loadPrcFileData("", "window-title Earthquake Simulator")
        loadPrcFileData("", "sync-video #t")
        loadPrcFileData("", "show-frame-rate-meter #f")
        loadPrcFileData("", "audio-library-name null")
        # Ten solver iterations, Bullet's default. Six was tried for speed and
        # measured: a four-plate stack jitters at 36 mm/s and walks 8 cm in
        # twenty seconds; at ten it is 2 mm/s and sleeps. Contact detection,
        # not the solver, is where the time goes anyway.
        loadPrcFileData("", "bullet-solver-iterations 10")
        auto = quality == "auto"
        if auto:
            # MSAA must be chosen before the window exists, so auto settles for
            # 2x; the shadow and normal-map choice waits until the GPU is known.
            msaa, shadows, shadow_size, normal_maps = 2, True, 2048, True
        else:
            msaa, shadows, shadow_size, normal_maps = QUALITY.get(quality, QUALITY["high"])
        loadPrcFileData("", f"framebuffer-multisample {1 if msaa else 0}")
        loadPrcFileData("", f"multisamples {msaa}")
        self._size_window_to_display()
        super().__init__()

        self.disableMouse()
        self.setBackgroundColor(0.35, 0.45, 0.58)
        if msaa:
            self.render.setAntialias(AntialiasAttrib.MAuto)
        self.camLens.setNearFar(0.06, 6000.0)
        self.physics_hz = physics_hz
        self.quality = quality

        # Which GPU did we actually get? On a laptop with two, Windows may hand
        # a Python process the integrated one; the status panel shows this so
        # it is never a mystery.
        self.renderer = "unknown renderer"
        try:
            gsg = self.win.getGsg()
            self.renderer = self._renderer_name(gsg.getDriverVendor(),
                                                gsg.getDriverRenderer())
            print(f"[quakesim] rendering on: {self.renderer}  "
                  f"(driver {gsg.getDriverVersion()})")
        except Exception:
            pass
        low_end = any(k in self.renderer.lower()
                      for k in ("intel", "llvmpipe", "microsoft", "software"))
        if auto:
            quality = "medium" if low_end else "high"
            if low_end:
                shadows, shadow_size, normal_maps = True, 1024, False
            print(f"[quakesim] quality: {quality} (auto)")
        if low_end and "intel" in self.renderer.lower():
            print("[quakesim] this is the integrated GPU. If the machine also "
                  "has a discrete one, set python.exe to 'High performance' in "
                  "Windows Settings > System > Display > Graphics.")

        self.pbr = None
        if use_pbr:
            try:
                import simplepbr
                self.pbr = simplepbr.init(msaa_samples=msaa, max_lights=4,
                                          use_normal_maps=normal_maps,
                                          enable_shadows=shadows,
                                          shadow_bias=0.004)
            except Exception as exc:      # driver too old, or no GLSL support
                print(f"[quakesim] physically-based shading unavailable "
                      f"({exc}); falling back to basic shading.")
        if self.pbr is None:
            self.render.setShaderAuto()
        self.shadow_size = shadow_size if shadows else 0

        from .fonts import install_font
        self.ui_font = install_font(self)
        # The splash goes up before the interface and the first scene are
        # built, so the window never sits blank.
        self.loading = LoadingScreen(self)
        self.loading.begin("Starting", "Building the interface", mode="splash")

        self.detail = detail
        self.seed = seed
        self.scene = None
        self.scene_key = None
        self.phys = None
        self.sky = None
        self.motion = None
        self.response = None
        self.source = Source()
        self.site = SITES["stiff_soil"]
        self.shaker = None
        self.head = HeadResponse()
        self.vp_index = 0
        self._dirty = False
        self._dirty_at = 0.0
        self._busy = False
        self._loading = False
        self._pending_scene = None
        self._applied_key = None
        self._seek_target = None
        self._seek_from = 0.0
        self._seek_resume = False
        self._seek_backup = None            # (Snapshot, playing, paused) for CANCEL
        self._snaps: list[Snapshot] = []    # keyframes along the record, by time
        self._snap_dt = 2.0
        self._next_snap_t = 0.0

        self.rig = CameraRig(self)
        self.hud = Hud(self)
        self._bind_keys()

        self.load_scene(scene_key)
        self.taskMgr.add(self._update, "quakesim-update")

    @staticmethod
    def _size_window_to_display() -> None:
        """Open the window at most of the display rather than Panda's
        800 x 600, so the panels are laid out once, for the real aspect
        ratio. An explicit win-size in a prc file or the environment wins."""
        try:
            from panda3d.core import GraphicsPipeSelection
            if ConfigVariableString("win-size").getValue() not in ("", "800 600"):
                return
            if ConfigVariableString("window-type").getValue() in ("offscreen", "none"):
                return
            pipe = GraphicsPipeSelection.getGlobalPtr().makeDefaultPipe()
            if pipe is None:
                return
            dw, dh = pipe.getDisplayWidth(), pipe.getDisplayHeight()
            if dw < 800 or dh < 600:
                return
            w, h = int(dw * 0.92), int(dh * 0.86)
            loadPrcFileData("", f"win-size {w} {h}")
            loadPrcFileData("", f"win-origin {(dw - w) // 2} {max(0, (dh - h) // 2 - 24)}")
        except Exception:
            pass

    @staticmethod
    def _renderer_name(vendor: str, renderer: str) -> str:
        """'NVIDIA Corporation' + 'NVIDIA GeForce GTX 1650/PCIe/SSE2' ->
        'NVIDIA GeForce GTX 1650': the status line has a width."""
        import re
        vendor, renderer = (vendor or "").strip(), (renderer or "").strip()
        renderer = re.sub(r"(/(PCIe|SSE2|SSE|3DNOW!?))+$", "", renderer)
        first = vendor.split(" ")[0].lower() if vendor else ""
        if first and renderer.lower().startswith(first):
            return renderer
        return f"{vendor} {renderer}".strip() or "unknown renderer"

    # ------------------------------------------------------------------
    # Scene lifecycle
    # ------------------------------------------------------------------
    def load_scene(self, key: str) -> None:
        self._loading = True
        self._seek_target = None
        self._seek_backup = None
        self._clear_snapshots()
        cls = SCENES[key]
        spec = cls.spec
        ls = self.loading
        ls.begin(f"Loading  {spec.name}", "Clearing the previous scene",
                 mode="splash")
        if self.scene is not None:
            self.scene.destroy()
            self.scene = None
        if self.phys is not None:
            self.phys.destroy()
            self.phys = None

        self.scene_key = key
        self.phys = PhysicsWorld(self.render, substep=1.0 / self.physics_hz,
                                 max_substeps=4)
        expected = expected_bodies(key, self.detail)
        self.phys.progress = lambda n: ls.set(
            0.03 + 0.70 * min(1.0, n / max(expected, 1)),
            f"Building the scene:  {n:,} objects")
        self.phys.stage = lambda label, f: ls.set(0.74 + 0.10 * f, label)
        t0 = time.time()
        self.scene = cls(self.phys, self.render, self.seed, self.detail)
        build_s = time.time() - t0
        EXPECTED_BODIES[key] = sorted(
            [(d, n) for d, n in EXPECTED_BODIES.get(key, [])
             if abs(d - self.detail) > 1e-6] + [(self.detail, len(self.phys.dynamic))])

        ls.set(0.85, "Sky and lighting")
        if self.sky is not None:
            self.sky.destroy()
        self.sky = Sky(self.render, spec.sky_time, shadow_size=self.shadow_size)

        self.shaker = Shaker(self.phys)
        self.site = spec.site

        qt = TYPES[spec.default_quake]
        self.source = Source(mw=spec.default_mw, qtype=qt,
                             depth_km=spec.default_depth_km or qt.depth_default,
                             stress_drop_bar=qt.stress_drop_default)
        self.distance_km = spec.default_distance_km
        self.azimuth = 35.0

        self.hud.set_type(spec.default_quake)
        site_key = next((k for k, v in SITES.items() if v is spec.site), "rock")
        self.hud.set_site(site_key)
        self.hud.c_mag.set(spec.default_mw)
        self.hud.c_depth.set(self.source.depth_km)
        self.hud.c_dist.set(self.distance_km)
        self.hud.c_stress.set(self.source.stress_drop_bar)
        self.hud.c_az.set(self.azimuth)

        self._loading = False
        self.vp_index = 0
        self.hud.set_scene(key)
        self.apply_viewpoint(0)
        self.rebuild_quake(progress=ls.span(0.87, 1.0))
        ls.end()
        self.hud.show_toast(f"{spec.name}  ·  {len(self.phys.dynamic)} objects  "
                            f"·  {self.phys.draw_call_estimate()} draw calls  "
                            f"·  built in {build_s:.1f} s", 3.5,
                            globalClock.getFrameTime())

    def next_scene(self, delta: int = 1) -> None:
        i = (SCENE_ORDER.index(self.scene_key) + delta) % len(SCENE_ORDER)
        self.request_scene(SCENE_ORDER[i])

    def request_scene(self, key: str) -> None:
        """Load a scene on the next frame, after a 'loading' notice has
        been drawn -- building the supermarket takes a few seconds and the
        window should say so rather than freeze."""
        if self._loading or key == self.scene_key:
            return
        self.hud.show_toast(f"Loading {SCENES[key].spec.name} ...", 6.0,
                            globalClock.getFrameTime())
        self._pending_scene = key
        self.taskMgr.doMethodLater(0.05, self._load_pending, "quakesim-load-scene")

    def _load_pending(self, task):
        key = self._pending_scene
        self._pending_scene = None
        if key is not None and key != self.scene_key:
            self.load_scene(key)
        return Task.done

    def apply_viewpoint(self, index: int) -> None:
        vps = self.scene.spec.viewpoints
        if not vps:
            return
        self.vp_index = index % len(vps)
        vp = vps[self.vp_index]
        self.rig.apply_viewpoint(vp)
        self.hud.set_viewpoint(self.vp_index)
        self.head.reset()
        note = f"  —  {vp.note}" if vp.note else ""
        self.hud.show_toast(f"{vp.name}{note}", 3.0, globalClock.getFrameTime())

    # ------------------------------------------------------------------
    # Earthquake
    # ------------------------------------------------------------------
    def _parameter_key(self) -> tuple:
        """Everything the record depends on, as read from the interface."""
        h = self.hud
        site_key = next((k for k, v in SITES.items() if v is self.site), "")
        return (self.source.qtype.key, round(h.c_mag.get(), 3), round(h.c_depth.get(), 2),
                round(h.c_dist.get(), 2), round(h.c_stress.get(), 1),
                round(h.c_dir.get(), 3), round(h.c_az.get(), 1), site_key)

    def on_parameters_changed(self) -> None:
        if self._loading:
            return
        # A slider's command fires on the frame AFTER its value is set, so
        # programmatic settings (the collapse case, a scene load) arrive here
        # late, when the record they describe has already been built. Only a
        # real change marks the record dirty; otherwise the debounced rebuild
        # would re-synthesise the same record and reset the clock.
        if self._parameter_key() == self._applied_key:
            return
        self._dirty = True
        self._dirty_at = globalClock.getFrameTime()

    def on_type_changed(self, key: str) -> None:
        qt = TYPES[key]
        self.source = self.source.with_defaults_for_type(qt)
        self.hud.set_type(key)
        self.hud.c_depth.set(qt.depth_default)
        self.hud.c_stress.set(qt.stress_drop_default)
        self.hud.c_mag.set(qt.clamp_mw(self.hud.c_mag.get()))
        self.on_parameters_changed()

    def on_site_changed(self, key: str) -> None:
        self.site = SITES[key]
        self.on_parameters_changed()

    def _gather_parameters(self) -> None:
        qt = self.source.qtype
        self.source = Source(
            mw=self.hud.c_mag.get(), qtype=qt,
            depth_km=qt.clamp_depth(self.hud.c_depth.get()),
            stress_drop_bar=self.hud.c_stress.get(),
            directivity=self.hud.c_dir.get())
        self.distance_km = self.hud.c_dist.get()
        self.azimuth = self.hud.c_az.get()

    def rebuild_quake(self, progress=None) -> None:
        """Synthesise the record and solve every structure for it. Shows a
        progress card of its own unless the caller passes a `progress`
        function (a scene load, which owns the loading screen)."""
        if self._busy:
            return
        self._busy = True
        own_card = progress is None
        try:
            self._gather_parameters()
            spec = self.scene.spec
            if own_card:
                self.loading.begin(
                    "Building the earthquake",
                    f"M{self.source.mw:.1f} {self.source.qtype.name} at "
                    f"{self.distance_km:.0f} km", mode="card", delay=0.25)
                progress = self.loading.span(0.0, 1.0)
            progress(0.02, "Synthesising the ground motion")
            self.motion = synthesize(self.source, self.site, self.distance_km,
                                     azimuth_deg=self.azimuth, seed=self.seed,
                                     max_duration=420.0)
            self.response = None
            if spec.building is not None:
                progress(0.14, "Solving the building response")
                self.response = mdof.solve_2d(
                    spec.building, self.motion.acc[0], self.motion.acc[1],
                    self.motion.dt,
                    progress=lambda f: progress(0.14 + 0.66 * f,
                                                "Solving the building response"))
            progress(0.80, "Loading the record")
            was_playing = self.shaker.playing
            was_paused = self.shaker.paused
            was_at = self.shaker.time
            self.shaker.load(self.motion, self.response, spec.story_of_floor)
            # Keyframes were simulated under the old record: no longer the
            # states this record leads to.
            self._clear_snapshots()
            self._snap_dt = max(2.0, self.motion.duration / 120.0)
            self._applied_key = self._parameter_key()
            # Scenes with their own secondary structures (pallet racking, say)
            # attach them here, driven by the floor motion they stand on.
            if hasattr(self.scene, "attach_structures"):
                self.scene.attach_structures(
                    self.shaker, self.motion, self.response,
                    progress=lambda f, label: progress(0.82 + 0.16 * f, label))
            self.shaker.gain = self.hud.c_gain.get()
            self.shaker.speed = self.hud.c_speed.get()
            if was_playing or was_at > 0.0:
                # Changing a parameter mid-quake re-synthesises the record but
                # must not rewind the one you are watching; after a quake has
                # run, the clock stays where it stopped and the scene stays as
                # the quake left it until RUN or RESET.
                self.shaker.time = min(was_at, self.motion.duration)
                self.shaker.playing = was_playing
                self.shaker.paused = was_paused
            progress(0.99, "Ready")

            lines = object_response_summary(self.motion.meta["pga_g"],
                                            self.motion.meta["pgv_cms"])
            if self.response is not None:
                if spec.observer_story < 0:
                    # standing on the ground floor: the ground's own motion
                    acc_g = self.motion.meta["pga_g"]
                else:
                    s_obs = min(spec.observer_story, self.response.stories - 1)
                    acc_g = float(np.max(np.abs(self.response.floor_acc[s_obs]))) / K.G0
                rep = dmg.assess(acc_g, float(np.max(self.response.peak_drift)))
                lines = rep.summary() + lines
            self.hud.show_results(self.source, self.site, self.motion,
                                  self.response, lines)
            self.hud.draw_trace(self.motion)
        finally:
            if own_card:
                self.loading.end()
            self._busy = False
            self._dirty = False

    def run_quake(self) -> None:
        self._seek_target = None
        self._seek_backup = None
        self.loading.end()
        if self._dirty:
            self.rebuild_quake()
        self.reset_scene(quiet=True)
        self.head.reset()
        self.shaker.start()
        self._keyframe()
        self.hud.show_toast("Rupture", 1.4, globalClock.getFrameTime())

    # -- keyframes -----------------------------------------------------------
    def _clear_snapshots(self) -> None:
        self._snaps = []
        self._next_snap_t = 0.0

    def _take_snapshot(self) -> Snapshot:
        return Snapshot(self.shaker.time, self.phys.snapshot(),
                        self.scene.snapshot_state(), self.head.u.copy(),
                        self.head.v.copy())

    def _keyframe(self, snap: Snapshot | None = None) -> None:
        """Record the current instant; replaces a keyframe already at it."""
        if snap is None:
            snap = self._take_snapshot()
        t = snap.time
        self._snaps = [k for k in self._snaps if abs(k.time - t) > 0.25 * self._snap_dt]
        self._snaps.append(snap)
        self._snaps.sort(key=lambda k: k.time)
        while len(self._snaps) > SNAPSHOTS_MAX:
            # thin the middle, keep the start and the newest
            self._snaps.pop(1 + (len(self._snaps) - 2) // 2)
        self._next_snap_t = (math.floor(t / self._snap_dt) + 1) * self._snap_dt

    def _keyframe_if_due(self) -> None:
        if self.shaker.time >= self._next_snap_t - 1e-6:
            self._keyframe()

    def _restore_snapshot(self, snap: Snapshot) -> None:
        self.phys.restore(snap.world)
        self.scene.restore_state(snap.scene)
        self.shaker.time = snap.time
        self.head.u[:] = snap.head_u
        self.head.v[:] = snap.head_v
        self._next_snap_t = (math.floor(snap.time / self._snap_dt) + 1) * self._snap_dt

    def _keyframe_before(self, t: float):
        best = None
        for k in self._snaps:
            if k.time <= t + 1e-6 and (best is None or k.time > best.time):
                best = k
        return best

    def set_collapse_case(self) -> None:
        """Dial in the earthquake this scene's author found brings it down."""
        demo = self.scene.spec.collapse_demo
        if demo is None:
            self.hud.show_toast("This scene has no collapse case", 2.0,
                                globalClock.getFrameTime())
            return
        mw, dist, qtype = demo
        spec = self.scene.spec
        self._loading = True
        try:
            # The whole parameter set, so the case is the same every time
            # whatever was dialled in before it.
            self.on_type_changed(qtype)
            qt = TYPES[qtype]
            self.hud.c_mag.set(qt.clamp_mw(mw))
            self.hud.c_dist.set(dist)
            self.hud.c_depth.set(qt.clamp_depth(spec.default_depth_km or qt.depth_default))
            self.hud.c_stress.set(qt.stress_drop_default)
            self.hud.c_dir.set(0.0)
            self.hud.c_az.set(35.0)
            self.hud.c_gain.set(1.0)
            self.shaker.gain = 1.0
            site_key = next((k for k, v in SITES.items() if v is spec.site), "rock")
            self.site = spec.site
            self.hud.set_site(site_key)
        finally:
            self._loading = False
        self.rebuild_quake()
        self.hud.show_toast(f"Collapse case: M{mw:.1f} {qt.name} at {dist:.0f} km  "
                            f"·  press R or RUN QUAKE", 4.0, globalClock.getFrameTime())

    def reset_scene(self, quiet: bool = False) -> None:
        self._seek_target = None
        self.phys.reset()
        if self.scene is not None:
            self.scene.on_reset()
        self.shaker.time = 0.0
        self.shaker.playing = False
        self.shaker.paused = False
        self._clear_snapshots()
        self.loading.end()
        self.head.reset()
        if not quiet:
            self.hud.show_toast("Scene reset", 1.4, globalClock.getFrameTime())

    def toggle_pause(self) -> None:
        """PLAY / PAUSE. A pause freezes everything -- the record, the
        shaking and every falling object -- until PLAY."""
        s = self.shaker
        if s.motion is None:
            return
        if self._seek_target is not None:          # STOP: hold where the seek got to
            self._end_seek(resume=False)
            return
        if s.paused:                               # PLAY: carry on from the frozen instant
            s.paused = False
            if 0.0 < s.time < s.duration:
                s.playing = True
            return
        if s.playing:                              # PAUSE
            s.playing = False
            s.paused = True
            return
        # Not playing, not paused: at rest before a run, or the record has
        # ended -- PLAY runs the earthquake (again).
        self.run_quake()

    # ------------------------------------------------------------------
    # Seeking. Rigid bodies cannot be rewound, so a seek is honest about
    # what it does: forward, the record is simulated up to the target at
    # the fastest rate that keeps the window responsive; backward, the scene
    # is reset and replayed from t = 0 the same way. What you see at the
    # target is what would have been there -- the physics never skipped.
    # ------------------------------------------------------------------
    def seek(self, t_target: float) -> None:
        s = self.shaker
        if s.motion is None:
            return
        dur = s.duration
        t_target = min(max(float(t_target), 0.0), dur)
        if self._seek_target is not None:
            resume = self._seek_resume
        else:
            resume = s.playing and not s.paused
        if abs(t_target - s.time) < 1e-3 and self._seek_target is None:
            return
        if self._seek_target is None:
            # Where we were, for CANCEL -- and a keyframe in its own right.
            backup = self._take_snapshot()
            self._seek_backup = (backup, s.playing, s.paused)
            if s.time > 0.0 or s.playing:
                self._keyframe(backup)
        # The still picture is taken now, before anything moves.
        self.loading.begin(f"Moving to  {t_target:.1f} s", "", mode="freeze",
                           delay=0.20, cancel=self.cancel_seek)
        back = self._keyframe_before(t_target)
        if t_target < s.time - 0.05 or (back is not None and back.time > s.time + 1e-6):
            # Rigid bodies cannot be rewound, but a keyframe can be put back:
            # jump to the last one before the target and catch up from there.
            if back is not None:
                self._restore_snapshot(back)
                self.loading.set(0.0, f"From the keyframe at {back.time:.0f} s",
                                 render=False)
            else:
                self.reset_scene(quiet=True)
                self.loading.begin(f"Moving to  {t_target:.1f} s",
                                   "Replaying from the start", mode="freeze",
                                   delay=0.20, cancel=self.cancel_seek)
                s.start()
                self._keyframe()
        elif s.time <= 0.0 and not s.playing:
            self.reset_scene(quiet=True)
            self.loading.begin(f"Moving to  {t_target:.1f} s", "", mode="freeze",
                               delay=0.20, cancel=self.cancel_seek)
            s.start()
            self._keyframe()
        self._seek_from = s.time
        self._seek_target = t_target
        self._seek_resume = resume
        s.paused = False
        s.playing = True

    def cancel_seek(self) -> None:
        """Abandon a seek and go straight back to where it started."""
        if self._seek_target is None or self._seek_backup is None:
            return
        snap, playing, paused = self._seek_backup
        self._seek_target = None
        self._seek_backup = None
        self.loading.end()
        self._restore_snapshot(snap)
        self.shaker.playing, self.shaker.paused = playing, paused
        self.hud.show_toast("Back to where you were", 1.4, globalClock.getFrameTime())

    def seek_relative(self, seconds: float) -> None:
        if self.shaker.motion is None:
            return
        target = self._seek_target if self._seek_target is not None else self.shaker.time
        self.seek(target + seconds)

    def _end_seek(self, resume: bool) -> None:
        s = self.shaker
        self._seek_target = None
        self._seek_backup = None
        self.loading.end()
        self._keyframe()
        if resume and 0.0 < s.time < s.duration:
            s.playing, s.paused = True, False
        elif s.time >= s.duration or s.time <= 0.0:
            s.playing, s.paused = False, False       # ended, or back at rest
        else:
            s.playing, s.paused = False, True        # hold the instant reached

    def _run_seek(self) -> None:
        """Advance toward the seek target within this frame's time budget.
        The window shows the last frame as a still picture with a progress
        card meanwhile (QuakeSim.seek), never the catching-up itself."""
        if self._seek_target is None:
            return
        budget = 0.12
        t0 = time.perf_counter()
        speed = self.shaker.speed
        self.shaker.speed = 1.0
        try:
            while (self.shaker.time < self._seek_target - 1e-6
                   and time.perf_counter() - t0 < budget):
                step = min(1.0 / 30.0, self._seek_target - self.shaker.time)
                self.shaker.update(step)
                self.scene.update(self.shaker.time, self.shaker)
                self._keyframe_if_due()
        finally:
            self.shaker.speed = speed
        span = max(self._seek_target - self._seek_from, 1e-6)
        done = (self.shaker.time - self._seek_from) / span
        self.loading.set(done, f"Simulating  {self.shaker.time:.1f} s  of  "
                               f"{self._seek_target:.1f} s", render=False)
        if self.shaker.time >= self._seek_target - 1e-6 or not self.shaker.playing:
            self._end_seek(resume=bool(self._seek_resume))

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    def _bind_keys(self) -> None:
        a = self.accept
        for key, name in (("w", "fwd"), ("s", "back"), ("a", "left"),
                          ("d", "right"), ("space", "up"), ("control", "down"),
                          ("q", "yaw_l"), ("e", "yaw_r"), ("shift", "fast")):
            a(key, self.rig.set_key, [name, True])
            a(key + "-up", self.rig.set_key, [name, False])

        a("escape", self._quit)
        a("tab", self._toggle_capture)
        a("mouse1", self._mouse1_down)
        a("mouse1-up", self.rig.set_dragging, [False])
        a("wheel_up", self._wheel, [1])
        a("wheel_down", self._wheel, [-1])

        a("r", self.run_quake)
        a("k", self.set_collapse_case)
        a("f", self.reset_scene)
        a("p", self.toggle_pause)
        a("v", self._cycle_mode)
        a("h", self.hud.toggle_controls)
        a("l", self.hud.reset_layout)
        a("f1", self._show_help)
        a("[", self._prev_vp)
        a("]", self._next_vp)
        a("n", self.next_scene, [1])
        a("shift-n", self.next_scene, [-1])
        a("t", self._cycle_time)
        for i in range(1, 10):
            a(str(i), self.apply_viewpoint, [i - 1])
        a("0", self._reset_view)
        a("g", self._toggle_debug)

    def _quit(self):
        if self._seek_target is not None:    # Esc cancels a seek first
            self.cancel_seek()
            return
        if self.hud.help.visible:            # ... or closes the help
            self.hud.help.hide()
            return
        self.userExit()

    def _mouse1_down(self):
        # A click on a slider, a button or a panel's title bar belongs to the
        # panel, not to the camera.
        if self.hud.is_over_panel():
            return
        self.rig.set_dragging(True)

    def _toggle_capture(self):
        on = self.rig.toggle_capture()
        self.hud.show_toast("Mouse look ON — Tab to release" if on
                            else "Mouse released", 1.6,
                            globalClock.getFrameTime())

    def _cycle_mode(self):
        i = (MODES.index(self.rig.mode) + 1) % len(MODES)
        self.rig.set_mode(MODES[i])
        from .camera import MODE_NAMES
        self.hud.show_toast(MODE_NAMES[MODES[i]], 1.8,
                            globalClock.getFrameTime())

    def _prev_vp(self):
        self.apply_viewpoint(self.vp_index - 1)

    def _next_vp(self):
        self.apply_viewpoint(self.vp_index + 1)

    def _reset_view(self):
        self.apply_viewpoint(self.vp_index)

    def _cycle_time(self):
        self.sky.set_hour(self.sky.hour + 3.0)
        self.hud.show_toast(f"{int(self.sky.hour):02d}:"
                            f"{int((self.sky.hour % 1) * 60):02d}", 1.4,
                            globalClock.getFrameTime())

    def _toggle_debug(self):
        if self.phys.debug_np is None:
            self.phys.enable_debug()
            self.hud.show_toast("Collision shapes shown", 1.6,
                                globalClock.getFrameTime())
        else:
            if self.phys.debug_np.isHidden():
                self.phys.debug_np.show()
            else:
                self.phys.debug_np.hide()

    def _show_help(self):
        self.hud.toggle_help()

    def _wheel(self, direction):
        # The wheel belongs to whatever list or panel is under the pointer.
        if self.hud.is_over_panel():
            return
        self.rig.zoom(direction)

    # ------------------------------------------------------------------
    def _update(self, task):
        # Never ask the physics to catch up more than a thirtieth of a second
        # in one frame; below that, time slows rather than the frame rate
        # collapsing further.
        dt = min(globalClock.getDt(), 1.0 / 30.0)
        now = globalClock.getFrameTime()
        frame = globalClock.getFrameCount()

        if (self._dirty and (now - self._dirty_at) > 0.35 and not self._busy
                and not self.shaker.playing):
            self.rebuild_quake()

        paused = self.shaker is not None and self.shaker.paused
        if self.shaker is not None:
            if self._seek_target is None:
                self.shaker.update(dt)
                if self.scene is not None and not paused:
                    self.scene.update(self.shaker.time, self.shaker)
                    if self.shaker.playing:
                        self._keyframe_if_due()
            else:
                self._run_seek()
            self.hud.set_transport(self.shaker, self._seek_target)

        # Camera shake comes from the observer's own body response, so it is
        # the floor motion filtered through a person, not noise. Frozen with
        # everything else while paused.
        acc = np.zeros(3)
        if self.shaker is not None and self.shaker.motion is not None:
            a = self.shaker._acc_cache.get(0, (0.0, 0.0, 0.0))
            acc = np.array(a, dtype=float)
        head = self.head.step(acc, 0.0 if paused else dt)
        shake = Vec3(float(head[0]), float(head[1]), float(head[2]) * 0.55)
        if self.rig.mode in ("orbit", "top", "fly"):
            shake = shake * 0.25
        # If the floor you are standing on has been released, ride it.
        ride = Vec3(0, 0, 0)
        ob = getattr(self.scene, "observer_body", None)
        if (ob is not None and ob.released and not ob.crushed
                and self.rig.mode in ("walk", "look")):
            h = ob.home[0]
            cz = self.rig.base_pos.getZ()
            if h[2] - 0.5 <= cz <= h[2] + 4.0:        # standing on that slab
                p = ob.path.getPos()
                ride = Vec3(p[0] - h[0], p[1] - h[1], p[2] - h[2])
        self.rig.ride_offset = ride
        self.rig.update(dt, shake)
        self.sky.follow(self.camera)

        self.hud.draw_cursor(self.motion, self.shaker.time if self.shaker else 0)
        self.hud.tick_toast(now)

        # Text regeneration and walking every body's transform are both a few
        # milliseconds; ten times a second is plenty.
        if self.motion is not None and frame % 6 == 0:
            ax, ay, az = self.shaker.current_accel_g()
            mag = math.hypot(ax, ay)
            moved, toppled, fallen = self.phys.counts(frame)
            if self._seek_target is not None:
                state = "seeking"
            elif self.shaker.paused:
                state = "paused"
            elif self.shaker.playing:
                state = "running"
            elif self.shaker.time <= 0:
                state = "ready"
            elif self.shaker.time >= self.motion.duration:
                state = "record ended"
            else:
                state = "stopped"
            summ = self.scene.structure.summary()
            released = summ["failed"]
            struct = (f"   ·   STRUCTURE: {released} of {summ['elements']} "
                      f"elements down" if released else
                      (f"   ·   {summ['elements']} structural elements intact"
                       if summ["elements"] else ""))
            report = self.scene.damage_report()
            last = ("\n" + "   ·   ".join(report))[:150] if report else ""
            self.hud.set_status(
                f"{self.scene.spec.name}   ·   {self.renderer}   ·   "
                f"{self.globalClock.getAverageFrameRate():.0f} fps   ·   "
                f"{self.phys.draw_call_estimate()} draw calls   ·   "
                f"physics {self.phys.last_ms:.1f} ms\n"
                f"{self.rig.describe()}  ·  looking at {self.rig.altitude_word()}\n"
                f"t = {self.shaker.time:6.2f} s / {self.motion.duration:.0f} s   "
                f"[{state}]   now {mag*100:5.1f} %g horizontal, "
                f"{abs(az)*100:4.1f} %g vertical\n"
                f"{len(self.phys.dynamic)} objects:  {moved} shifted   "
                f"{toppled} toppled   {fallen} fallen{struct}{last}")
        return Task.cont

    @property
    def globalClock(self):
        return globalClock
