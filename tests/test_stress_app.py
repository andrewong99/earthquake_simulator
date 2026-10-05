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

"""Application-level stress test.

`test_stress.py` beats on the physics and the structural model. This one
beats on the application the way a user does, through the same code paths
as the buttons and keys: switching scenes back and forth, changing every
parameter to both ends of its range, every earthquake type and every site,
changing parameters *while* a quake plays, pausing, resetting after a
collapse and running it again, and checks that nothing leaks between
scenes (bodies, draw calls, memory) and that a collapse is reproducible.

    python tests/test_stress_app.py            everything
    python tests/test_stress_app.py quick      one scene, the short list
"""

from __future__ import annotations

import gc
import math
import os
import sys
import time

import numpy as np
from panda3d.core import loadPrcFileData

loadPrcFileData("", "window-type offscreen")
loadPrcFileData("", "audio-library-name null")
loadPrcFileData("", "win-size 800 450")
loadPrcFileData("", "framebuffer-multisample 0")
loadPrcFileData("", "multisamples 0")

DT = 1.0 / 30.0


def _rss_mb() -> float:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def _finite_state(app) -> list[str]:
    errs = []
    for b in app.phys.dynamic:
        p = b.path.getPos()
        if not all(math.isfinite(v) for v in (p[0], p[1], p[2])):
            errs.append(f"non-finite position on {b.label}")
            break
        if max(abs(p[0]), abs(p[1]), abs(p[2])) > 1000.0:
            errs.append(f"{b.label} beyond 1000 m at {tuple(round(v, 1) for v in (p[0], p[1], p[2]))}")
            break
    m = app.motion
    if m is not None and not np.all(np.isfinite(m.acc)):
        errs.append("NaN in ground motion")
    if app.response is not None and not np.all(np.isfinite(app.response.floor_acc)):
        errs.append("NaN in structural response")
    return errs


def _step(app, seconds: float) -> None:
    for _ in range(int(seconds / DT)):
        app.shaker.update(DT)
        app.scene.update(app.shaker.time, app.shaker)


def _frames(app, n: int = 3) -> None:
    for _ in range(n):
        app.taskMgr.step()


def _switch(app, key: str) -> None:
    """Through the chooser, as the dropdown does."""
    from quakesim.view.fonts import safe
    from quakesim.world.scenes import SCENES
    spec = SCENES[key].spec
    app.hud._scene_changed(safe(spec.short or spec.name))
    for _ in range(12):
        app.taskMgr.step()
        time.sleep(0.005)
        if app.scene_key == key:
            break


# ---------------------------------------------------------------------------
def check_parameter_extremes(app) -> list[str]:
    """Every slider to both ends, every type, every site: the chain must
    produce a finite record and a finite response every time."""
    from quakesim.seismo.site import SITES
    from quakesim.seismo.source import MW_MAX, TYPES, TYPE_ORDER
    errs = []
    h = app.hud
    combos = []
    for key in TYPE_ORDER:
        qt = TYPES[key]
        lo, hi = qt.mw_range
        combos.append((key, hi, qt.depth_default, 10.0, qt.stress_drop_default, 0.0, 35.0))
        combos.append((key, lo, qt.depth_default, 50.0, qt.stress_drop_default, 0.0, 35.0))
        # the slider's ceiling, beyond the type's observed band
        combos.append((key, MW_MAX, qt.depth_default, 1.0, qt.stress_drop_default, 0.0, 35.0))
    combos += [
        ("strike_slip", 6.5, 0.5, 0.5, 70.0, 1.0, 0.0),
        ("strike_slip", 6.5, 0.5, 0.5, 1500.0, -1.0, 359.0),
        ("deep_focus", 8.0, 700.0, 900.0, 200.0, 0.0, 180.0),
        ("megathrust", 10.0, 25.0, 900.0, 30.0, 1.0, 90.0),
        ("megathrust", 10.0, 25.0, 0.5, 1500.0, -1.0, 270.0),
        ("strike_slip", 10.0, 0.5, 0.5, 1500.0, 1.0, 0.0),
        ("collapse", 10.0, 0.5, 0.5, 1.0, 0.0, 0.0),
        ("explosion", 5.0, 0.5, 0.5, 1500.0, 0.0, 0.0),
        ("impact", 6.0, 0.5, 2.0, 1500.0, 0.0, 0.0),
        ("strike_slip", 1.0, 0.5, 900.0, 1.0, 0.0, 0.0),
    ]
    t0 = time.time()
    n = 0
    for key, mw, depth, dist, stress, dirn, az in combos:
        qt = TYPES[key]
        app.on_type_changed(key)
        h.c_mag.set(qt.clamp_mw(mw))
        h.c_depth.set(qt.clamp_depth(depth))
        h.c_dist.set(dist)
        h.c_stress.set(min(max(stress, h.c_stress.slider["range"][0]),
                           h.c_stress.slider["range"][1]))
        h.c_dir.set(dirn)
        h.c_az.set(az)
        try:
            app.rebuild_quake()
        except Exception as exc:              # noqa: BLE001
            errs.append(f"{key} M{mw} d{depth} r{dist} s{stress}: EXCEPTION {exc!r}")
            continue
        n += 1
        errs += [f"{key} M{mw} d{depth} r{dist} s{stress}: {e}" for e in _finite_state(app)]
        m = app.motion.meta
        if not (0.0 <= m["pga_g"] < 20.0):
            errs.append(f"{key} M{mw} d{depth} r{dist}: PGA {m['pga_g']:.2f} g out of range")
        if app.motion.duration > 420.0 + 1e-6 or app.motion.duration < 1.0:
            errs.append(f"{key} M{mw}: record {app.motion.duration:.1f} s")
    for site in SITES:
        app.on_site_changed(site)
        app.rebuild_quake()
        errs += [f"site {site}: {e}" for e in _finite_state(app)]
        n += 1
    print(f"    {n} parameter combinations synthesised and solved in "
          f"{time.time() - t0:.1f} s, {len(errs)} problems")
    return errs


def check_live_changes(app) -> list[str]:
    """Change things while the earthquake is playing: the record must be
    re-synthesised without rewinding or stopping, and pausing must hold."""
    errs = []
    spec = app.scene.spec
    app.on_type_changed(spec.default_quake)
    app.hud.c_mag.set(spec.default_mw)
    app.hud.c_dist.set(spec.default_distance_km)
    app.rebuild_quake()
    app.run_quake()
    _step(app, 3.0)
    t_before = app.shaker.time
    app.hud.c_mag.set(spec.default_mw + 0.6)
    app.on_parameters_changed()
    app.rebuild_quake()                     # what the debounce does
    if not app.shaker.playing:
        errs.append("changing magnitude mid-quake stopped playback")
    if app.shaker.time < t_before - 1e-6:
        errs.append("changing magnitude mid-quake rewound the record")
    app.on_site_changed("soft_soil")
    app.rebuild_quake()
    app.toggle_pause()
    t_paused = app.shaker.time
    held = [(b, b.path.getPos()) for b in app.phys.dynamic[:400]]
    _step(app, 1.0)
    if abs(app.shaker.time - t_paused) > 1e-9:
        errs.append("time advanced while paused")
    if not app.shaker.paused:
        errs.append("pause did not freeze the world")
    crept = sum(1 for b, p0 in held if (b.path.getPos() - p0).length() > 1e-6)
    if crept:
        errs.append(f"{crept} objects moved while paused (a pause must freeze everything)")
    app.toggle_pause()
    _step(app, 1.0)
    if app.shaker.time <= t_paused:
        errs.append("time did not advance after unpausing")
    app.hud.c_speed.set(2.0)
    app.hud._speed_changed(2.0) if hasattr(app.hud, "_speed_changed") else None
    t0 = app.shaker.time
    _step(app, 1.0)
    if app.shaker.playing and app.shaker.time - t0 < 1.5:
        errs.append(f"playback speed 2x advanced only {app.shaker.time - t0:.2f} s in 1 s")
    app.hud.c_speed.set(1.0)
    app.hud._speed_changed(1.0) if hasattr(app.hud, "_speed_changed") else None
    app.hud.c_gain.set(4.0)
    app.hud._gain_changed(4.0) if hasattr(app.hud, "_gain_changed") else None
    _step(app, 3.0)
    errs += _finite_state(app)
    app.hud.c_gain.set(1.0)
    app.hud._gain_changed(1.0) if hasattr(app.hud, "_gain_changed") else None
    app.on_site_changed(next(k for k, v in __import__("quakesim.seismo.site", fromlist=["SITES"]).SITES.items() if v is spec.site))
    app.reset_scene(quiet=True)
    print(f"    live parameter changes, pause/resume, 2x speed, 4x gain: {len(errs)} problems")
    return errs


def check_collapse_repeat(app, repeats: int = 3) -> list[str]:
    """Run the collapse case, reset, run it again: the same elements must
    fail at the same time each run, and every reset must restore the scene."""
    errs = []
    if app.scene.spec.collapse_demo is None:
        return errs
    results = []
    n_dyn0 = len(app.phys.dynamic)
    for k in range(repeats):
        app.set_collapse_case()
        app.run_quake()
        m = app.motion
        a2 = np.sum(m.acc[:2] ** 2, axis=0)
        cum = np.cumsum(a2)
        t95 = float(np.searchsorted(cum, 0.95 * cum[-1])) * m.dt
        horizon = min(m.duration, t95 + 6.0, 150.0)
        # pause and resume half way, as a viewer would
        _step(app, horizon * 0.5)
        app.toggle_pause()
        _frames(app, 2)
        app.toggle_pause()
        _step(app, horizon * 0.5)
        app.shaker.playing = False
        _step(app, 3.0)
        summ = app.scene.structure.summary()
        results.append((summ["failed"], round(summ["first_failure"] or -1.0, 2),
                        app.phys.displaced_count(), app.phys.fallen_count()))
        errs += [f"collapse run {k + 1}: {e}" for e in _finite_state(app)]
        app.reset_scene(quiet=True)
        if app.phys.released_count() or app.scene.structure.summary()["failed"]:
            errs.append(f"run {k + 1}: reset left elements released")
        if len(app.phys.dynamic) != n_dyn0:
            errs.append(f"run {k + 1}: {len(app.phys.dynamic)} dynamic bodies after reset, "
                        f"was {n_dyn0}")
        _step(app, 1.5)
        moved = app.phys.displaced_count(0.05)
        if moved > max(2, 0.01 * n_dyn0):
            errs.append(f"run {k + 1}: {moved} objects not back in place after reset")
    failed = {r[0] for r in results}
    first = {r[1] for r in results}
    if len(failed) > 1 or len(first) > 1:
        errs.append(f"collapse not reproducible across runs: {results}")
    print(f"    collapse case run {repeats}x with reset: failed/first/shifted/fallen = "
          f"{results}")
    return errs


def check_views(app) -> list[str]:
    from quakesim.view.camera import MODES
    from quakesim.view.fonts import safe
    errs = []
    for i, vp in enumerate(app.scene.spec.viewpoints):
        app.hud._vp_changed(safe(vp.name))
        _frames(app, 1)
        if app.vp_index != i:
            errs.append(f"viewpoint chooser landed on {app.vp_index}, wanted {i}")
    for mode in MODES:
        app.rig.set_mode(mode)
        for key in ("fwd", "left", "up"):
            app.rig.set_key(key, True)
        app.rig.update(0.05, None)
        for key in ("fwd", "left", "up"):
            app.rig.set_key(key, False)
        app.rig.zoom(1)
        app.rig.zoom(-1)
        _frames(app, 1)
    for _ in range(4):
        app._cycle_time()
    app.hud.toggle_controls()
    app.hud.toggle_controls()
    app.hud.reset_layout()
    app._toggle_debug()
    _frames(app, 1)
    app._toggle_debug()
    app.apply_viewpoint(0)
    errs += check_widgets(app)
    return errs


def check_widgets(app) -> list[str]:
    """The drop-down lists, the help card and the transport controls, driven
    the way their mouse handlers drive them."""
    errs = []
    hud = app.hud
    for name in ("menu_type", "menu_site", "menu_scene", "menu_vp"):
        m = getattr(hud, name)
        before = m.get()
        m.show()
        _frames(app, 1)
        if not m.open or m.popup.isHidden():
            errs.append(f"{name}: list did not open")
        m.bar["value"] = 1.0                       # scroll to the bottom
        m._wheel(-1)
        _frames(app, 1)
        m.close()
        _frames(app, 1)
        if m.open or not m.popup.isHidden() or not m.cancel.isHidden():
            errs.append(f"{name}: list did not close")
        if m.get() != before:
            errs.append(f"{name}: opening the list changed the choice")
    # Pick the second viewpoint through its row, as a click would.
    vp = hud.menu_vp
    if len(vp.items) > 1:
        vp.show()
        vp._choose(1)
        _frames(app, 1)
        if app.vp_index != 1 or vp.get() != vp.items[1]:
            errs.append(f"viewpoint row 1 chose {vp.get()!r}, index {app.vp_index}")
        vp._choose(0)
        _frames(app, 1)
    # Pick the ground type through its row and make sure it took.
    site = hud.menu_site
    site.show()
    site._choose(0)
    _frames(app, 1)
    from quakesim.seismo.site import SITES
    if app.site is not SITES[hud.site_keys[0]]:
        errs.append(f"ground row 0 did not select {hud.site_keys[0]}")
    site.show()
    site._choose(1)
    _frames(app, 1)
    # Help card.
    hud.toggle_help()
    _frames(app, 1)
    if not hud.help.visible or not hud.is_over_panel():
        errs.append("help card did not show (or does not block the camera)")
    if hud.help.visible:
        app._quit()                                # Esc closes it, does not quit
        _frames(app, 1)
        if hud.help.visible:
            errs.append("Esc did not close the help card")
    # Transport: run, seek forward, back, relative, then the button.
    app.run_quake()
    _frames(app, 5)
    app.seek(8.0)
    for _ in range(600):
        if app._seek_target is None:
            break
        _frames(app, 1)
    if abs(app.shaker.time - 8.0) > 0.05:
        errs.append(f"seek to 8 s landed at {app.shaker.time:.2f} s")
    if not app.shaker.playing:
        errs.append("seek did not resume playing")
    app.seek_relative(-6.0)
    for _ in range(600):
        if app._seek_target is None:
            break
        _frames(app, 1)
    if abs(app.shaker.time - 2.0) > 0.05:
        errs.append(f"seek -6 s landed at {app.shaker.time:.2f} s")
    app.toggle_pause()
    _frames(app, 2)
    if app.shaker.playing or hud.play_button["text"] != "PLAY":
        errs.append(f"pause: playing={app.shaker.playing}, button {hud.play_button['text']!r}")
    app.toggle_pause()
    _frames(app, 2)
    if not app.shaker.playing or hud.play_button["text"] != "PAUSE":
        errs.append(f"play: playing={app.shaker.playing}, button {hud.play_button['text']!r}")
    hud.scrub["value"] = 0.5                       # a drag, released
    _frames(app, 3)
    if app._seek_target is None and abs(app.shaker.time / app.shaker.duration - 0.5) > 0.02:
        errs.append("scrubber drag did not seek")
    # A seek simulates every second between here and the target, at about
    # 30 ms of physics per frame; a long record on a slow machine takes a
    # few thousand frames, so wait on the clock rather than a frame count.
    t_wall = time.time()
    while app._seek_target is not None and time.time() - t_wall < 300:
        _frames(app, 1)
    if abs(app.shaker.time / app.shaker.duration - 0.5) > 0.02:
        errs.append(f"scrubber seek landed at {app.shaker.time / app.shaker.duration:.2f}"
                    f" of {app.shaker.duration:.0f} s after {time.time() - t_wall:.0f} s")
    app.reset_scene(quiet=True)
    _frames(app, 2)
    return errs


# ---------------------------------------------------------------------------
def main(argv) -> int:
    from quakesim.view.app import QuakeSim
    from quakesim.world.scenes import SCENE_ORDER
    quick = "quick" in argv
    keys = [a for a in argv if a in SCENE_ORDER] or (["ranau_house"] if quick else list(SCENE_ORDER))
    detail = 0.35
    errs = []

    t0 = time.time()
    app = QuakeSim(keys[0], detail, seed=4, use_pbr=False)
    print(f"=== application stress: start in {keys[0]} ({time.time() - t0:.1f} s)")

    print("--- parameter extremes")
    errs += check_parameter_extremes(app)

    for key in keys:
        print(f"--- {key}")
        _switch(app, key)
        if app.scene_key != key:
            errs.append(f"chooser did not switch to {key}")
            continue
        errs += [f"{key}: {e}" for e in check_views(app)]
        errs += [f"{key}: {e}" for e in check_live_changes(app)]
        errs += [f"{key}: {e}" for e in check_collapse_repeat(app, 2 if quick else 3)]

    # --- scene switching leaks -------------------------------------------
    print("--- scene switching")
    gc.collect()
    rss0 = _rss_mb()
    counts = {}
    draws = {}
    cycle = keys * (2 if quick else 3)
    for key in cycle:
        _switch(app, key)
        _frames(app, 2)
        n, d = len(app.phys.dynamic), app.phys.draw_call_estimate()
        if key in counts and (n != counts[key] or d != draws[key]):
            errs.append(f"{key}: {n} bodies / {d} draw calls on revisit, "
                        f"was {counts[key]} / {draws[key]}")
        counts[key], draws[key] = n, d
        if d > 600:
            errs.append(f"{key}: {d} draw calls -- old scene left behind?")
    gc.collect()
    rss1 = _rss_mb()
    print(f"    {len(cycle)} scene loads; bodies {counts}; draw calls {draws}; "
          f"RSS {rss0:.0f} -> {rss1:.0f} MB")
    if rss1 - rss0 > 600:
        errs.append(f"memory grew {rss1 - rss0:.0f} MB over {len(cycle)} scene loads")

    print()
    if errs:
        print("FAIL:")
        for e in errs:
            print("  - " + e)
        return 1
    print("PASS: the application survived everything the interface can do to it.")
    return 0


if __name__ == "__main__":
    code = main(sys.argv[1:])
    sys.stdout.flush()
    os._exit(code)
