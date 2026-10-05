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

"""Stress test.

Every scene through a ladder of earthquakes, from one that must do nothing
to ones well beyond anything the parameter ranges were designed around, with
the physics stepped at a fixed rate so the result does not depend on the
machine. For each run it checks:

    * no exception, no NaN anywhere, no body flung beyond +-1000 m
    * a small earthquake releases no structural element at all
    * the collapse suggestion each scene makes actually collapses something
    * after a collapse, reset puts every element and object back and the
      scene is stable again
    * the record never runs past the duration cap, and long megathrust
      records still play

    python tests/test_stress.py                  all scenes, full ladder
    python tests/test_stress.py ranau_house      one scene
    python tests/test_stress.py ranau_house quick   the short ladder
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np
from panda3d.core import loadPrcFileData

loadPrcFileData("", "window-type offscreen")
loadPrcFileData("", "audio-library-name null")
loadPrcFileData("", "win-size 640 400")
loadPrcFileData("", "framebuffer-multisample 0")
loadPrcFileData("", "multisamples 0")

# (label, Mw, distance km, type, depth km or None, stress drop bar or None,
#  expectation: "none" = nothing structural may fail, "collapse" = something
#  must, "any" = no expectation)
LADDER = [
    ("M4.0 @ 20 km  (must do nothing)", 4.0, 20.0, "strike_slip", None, None, "none"),
    ("M5.0 @ 8 km   (must not collapse)", 5.0, 8.0, "strike_slip", None, None, "none"),
    ("scene default", None, None, None, None, None, "any"),
    ("scene collapse suggestion", "demo", "demo", "demo", None, None, "collapse"),
    ("M7.5 @ 2 km reverse, 1500 bar", 7.5, 2.0, "reverse", None, 1500.0, "any"),
    ("M6.5 @ 0.5 km, 3 km deep", 6.5, 0.5, "strike_slip", 3.0, None, "any"),
    ("M8.0 deep-focus, 700 km deep", 8.0, 60.0, "deep_focus", 700.0, None, "any"),
    ("M9.5 megathrust @ 5 km", 9.5, 5.0, "megathrust", None, None, "collapse"),
    ("M9.0 megathrust @ 500 km (long record)", 9.0, 500.0, "megathrust", None, None, "any"),
    ("M7.0 injection-induced @ 3 km", 7.0, 3.0, "induced", 4.0, None, "any"),
]
QUICK = [LADDER[0], LADDER[3], LADDER[7]]


def _all_positions(phys) -> np.ndarray:
    out = np.empty((len(phys.dynamic), 3))
    for k, b in enumerate(phys.dynamic):
        p = b.path.getPos()
        out[k] = (p[0], p[1], p[2])
    return out


def _check_sane(phys, label: str) -> list[str]:
    errs = []
    pos = _all_positions(phys)
    if pos.size:
        if not np.all(np.isfinite(pos)):
            bad = [phys.dynamic[k].label for k in np.where(~np.isfinite(pos).all(1))[0][:5]]
            errs.append(f"{label}: non-finite position on {bad}")
        far = np.abs(pos).max(1) > 1000.0
        if far.any():
            bad = [phys.dynamic[k].label for k in np.where(far)[0][:5]]
            errs.append(f"{label}: bodies beyond 1000 m: {bad}")
    return errs


def _set_case(app, case):
    label, mw, dist, qtype, depth, stress, expect = case
    spec = app.scene.spec
    if mw == "demo":
        if spec.collapse_demo is None:
            return False
        mw, dist, qtype = spec.collapse_demo
    if mw is None:
        # the scene's own defaults
        app.on_type_changed(spec.default_quake)
        app.hud.c_mag.set(spec.default_mw)
        app.hud.c_dist.set(spec.default_distance_km)
        if spec.default_depth_km is not None:
            app.hud.c_depth.set(spec.default_depth_km)
    else:
        app.on_type_changed(qtype)
        qt = app.source.qtype
        app.hud.c_mag.set(qt.clamp_mw(mw))
        app.hud.c_dist.set(dist)
        if depth is not None:
            app.hud.c_depth.set(qt.clamp_depth(depth))
        if stress is not None:
            app.hud.c_stress.set(stress)
    app.rebuild_quake()
    return True


def run_case(app, case, dt=1.0 / 30.0, tail=4.0):
    label, *_, expect = case
    if not _set_case(app, case):
        print(f"    {label:44s} (no suggestion for this scene)")
        return []
    errs = []
    m = app.motion.meta
    if not np.all(np.isfinite(app.motion.acc)):
        errs.append(f"{label}: NaN in the ground motion")
    if app.response is not None and not np.all(np.isfinite(app.response.floor_acc)):
        errs.append(f"{label}: NaN in the structural response")
    if app.motion.duration > 420.0 + 1e-6:
        errs.append(f"{label}: record {app.motion.duration:.0f} s exceeds the cap")

    app.run_quake()
    # Simulate until 95% of the Arias intensity has arrived (plus a margin),
    # so a long subduction record is judged on its strong phase, not on the
    # first thirty seconds after S.
    a2 = np.sum(app.motion.acc[:2] ** 2, axis=0)
    cum = np.cumsum(a2)
    t95 = float(np.searchsorted(cum, 0.95 * cum[-1])) * app.motion.dt
    horizon = min(app.motion.duration, t95 + 6.0, 150.0)
    steps = int(horizon / dt) + int(tail / dt)
    t0 = time.time()
    n_dyn0 = len(app.phys.dynamic)
    for k in range(steps):
        app.shaker.update(dt)
        app.scene.update(app.shaker.time, app.shaker)
        if k % 150 == 0:
            bad = _check_sane(app.phys, label)
            if bad:
                errs += bad
                break
        if k == int(horizon / dt):
            app.shaker.playing = False      # stop the record, let things settle
    el = time.time() - t0
    errs += _check_sane(app.phys, label)

    released = app.phys.released_count()
    summ = app.scene.structure.summary()
    failed = summ["failed"]
    moved, toppled, fallen = app.phys.counts()
    peak_drift = (float(np.max(app.response.peak_drift)) * 100
                  if app.response is not None else 0.0)
    print(f"    {label:44s} PGA {m['pga_g']*100:5.1f}%g  drift {peak_drift:5.2f}%  "
          f"{failed:3d}/{summ['elements']} elements down  "
          f"{moved:4d} shifted {fallen:4d} fallen  "
          f"[{horizon:.0f} s in {el:.1f} s]")
    for line in app.scene.damage_report():
        print(f"        {line}")
    if expect == "none" and failed:
        errs.append(f"{label}: released {failed} structural elements "
                    f"({summ['by_tag']}) -- too fragile")
    if expect == "collapse" and not failed:
        errs.append(f"{label}: nothing structural failed")
    crushed = app.phys.crushed_count()
    if len(app.phys.dynamic) != n_dyn0 + released - crushed:
        errs.append(f"{label}: dynamic count {len(app.phys.dynamic)} != "
                    f"{n_dyn0} + {released} released - {crushed} crushed")

    # -- reset must restore everything -------------------------------------
    app.reset_scene(quiet=True)
    if app.phys.released_count():
        errs.append(f"{label}: {app.phys.released_count()} elements still released after reset")
    if app.scene.structure.summary()["failed"]:
        errs.append(f"{label}: structural model not reset")
    if app.scene.cracks.count():
        errs.append(f"{label}: cracks not cleared by reset")
    for _ in range(int(2.0 / dt)):
        app.shaker.update(dt)
        app.scene.update(app.shaker.time, app.shaker)
    moved = app.phys.displaced_count(0.05)
    if moved > max(2, 0.01 * len(app.phys.dynamic)):
        errs.append(f"{label}: {moved} objects not back in place after reset")
    return errs


def run_scene(key: str, ladder, detail: float = 0.35) -> list[str]:
    from quakesim.view.app import QuakeSim
    t0 = time.time()
    app = QuakeSim(key, detail, seed=4, use_pbr=False)
    print(f"\n=== {key}: {len(app.phys.dynamic)} bodies, "
          f"{app.phys.n_breakable} breakable elements, built in {time.time()-t0:.1f} s")
    errs = []
    for case in ladder:
        try:
            errs += run_case(app, case)
        except Exception as exc:          # noqa: BLE001 - the point is to catch it
            import traceback
            traceback.print_exc()
            errs.append(f"{case[0]}: EXCEPTION {exc!r}")
    # switch scenes and back: rebuilding must not leak or explode
    from quakesim.world.scenes import SCENE_ORDER
    other = SCENE_ORDER[(SCENE_ORDER.index(key) + 1) % len(SCENE_ORDER)]
    try:
        app.load_scene(other)
        app.load_scene(key)
        app.taskMgr.step()
        print(f"    scene switch {key} -> {other} -> {key}: OK")
    except Exception as exc:              # noqa: BLE001
        import traceback
        traceback.print_exc()
        errs.append(f"scene switch: EXCEPTION {exc!r}")
    return errs


def main(argv):
    from quakesim.world.scenes import SCENE_ORDER
    keys = [a for a in argv if a in SCENE_ORDER] or list(SCENE_ORDER)
    ladder = QUICK if "quick" in argv else LADDER
    detail = 0.35
    for a in argv:
        if a.startswith("detail="):
            detail = float(a.split("=")[1])
    all_errs = []
    for key in keys:
        all_errs += run_scene(key, ladder, detail)
    print()
    if all_errs:
        print("FAIL:")
        for e in all_errs:
            print("  - " + e)
        return 1
    print("PASS: every scene survived the ladder.")
    return 0


if __name__ == "__main__":
    code = main(sys.argv[1:])
    sys.stdout.flush()
    # Skip interpreter teardown: Panda's offscreen buffer aborts on exit
    # after a long run, and the verdict has already been printed.
    import os
    os._exit(code)
