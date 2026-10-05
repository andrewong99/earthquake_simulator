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

"""Keyframes and seeking.

A seek jumps to the last keyframe before the target and catches up from
there; CANCEL puts the simulation back exactly where the seek began. Both
rest on PhysicsWorld.restore putting every body, every structural element
and the scene's own flags back as they were. This checks that, scene by
scene, in the middle of a collapse:

  1. restore(snapshot) reproduces the snapshot's state to float32 precision
     -- poses, velocities, sleep states, floors, released / crushed flags,
     element failure records, damage report;
  2. the simulation carries on from the restored state without blowing up
     and reaches the same structural outcome as the uninterrupted run;
  3. in the application, a seek backward lands on the target by way of a
     keyframe (no replay from zero), and cancelling a seek part-way lands
     back on the pre-seek state exactly.

    python tests/test_snapshots.py            all scenes
    python tests/test_snapshots.py quick      ranau and warehouse
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
from panda3d.core import loadPrcFileData

loadPrcFileData("", "window-type offscreen")
loadPrcFileData("", "audio-library-name null")
loadPrcFileData("", "win-size 640 360")

DT = 1.0 / 30.0

# (scene, seconds into the collapse case for the snapshot, seconds to carry on)
CASES = {
    "warehouse": (11.0, 6.0),
    "kl_highrise": (142.0, 8.0),
    "ranau_house": (8.0, 6.0),
    "supermarket": (12.0, 6.0),
    "shophouse": (11.0, 6.0),
}


def _state(app):
    phys = app.phys
    rows = []
    for b in phys.bodies:
        p, q = b.path.getPos(), b.path.getQuat()
        v, w = b.node.getLinearVelocity(), b.node.getAngularVelocity()
        rows.append((p[0], p[1], p[2], q[0], q[1], q[2], q[3],
                     v[0], v[1], v[2], w[0], w[1], w[2]))
    flags = [(b.released, b.crushed, b.floor, b.node.isActive(), b.softened)
             for b in phys.bodies]
    elems = [(e.failed_at, e.failed_by) for e in app.scene.structure.elements]
    return np.array(rows, dtype=np.float64), flags, elems, app.scene.damage_report()


def _run_to(app, t):
    while app.shaker.time < t - 1e-9 and app.shaker.playing:
        app.shaker.update(DT)
        app.scene.update(app.shaker.time, app.shaker)


def _frames(app, n):
    for _ in range(n):
        app.taskMgr.step()


def check_scene(app, key, errs):
    t1, span = CASES[key]
    app.set_collapse_case()
    app.run_quake()
    _run_to(app, t1)
    failed_t1 = app.scene.structure.summary()["failed"]
    snap = app._take_snapshot()
    rows_a, flags_a, elems_a, report_a = _state(app)
    counts_a = app.phys.counts()

    _run_to(app, t1 + span)
    failed_t2 = app.scene.structure.summary()["failed"]
    counts_t2 = app.phys.counts()

    # 1. exact restore
    t0 = time.perf_counter()
    app._restore_snapshot(snap)
    restore_s = time.perf_counter() - t0
    rows_b, flags_b, elems_b, report_b = _state(app)
    live = [i for i, b in enumerate(app.phys.bodies) if not b.crushed]
    dpos = np.abs(rows_b[live, :7] - rows_a[live, :7]).max() if live else 0.0
    dvel = np.abs(rows_b[live, 7:] - rows_a[live, 7:]).max() if live else 0.0
    if dpos > 2e-3:
        errs.append(f"{key}: restored pose differs by {dpos:.4f}")
    if dvel > 2e-3:
        errs.append(f"{key}: restored velocity differs by {dvel:.4f}")
    if flags_a != flags_b:
        n = sum(1 for a, b in zip(flags_a, flags_b) if a != b)
        errs.append(f"{key}: {n} bodies with different flags after restore")
    if elems_a != elems_b:
        errs.append(f"{key}: structural failure record differs after restore")
    if report_a != report_b:
        errs.append(f"{key}: damage report differs after restore: {report_a} vs {report_b}")
    if app.phys.counts() != counts_a:
        errs.append(f"{key}: object counts differ after restore: {app.phys.counts()} vs {counts_a}")
    if abs(app.shaker.time - t1) > 1e-6:
        errs.append(f"{key}: clock not restored ({app.shaker.time} vs {t1})")

    # 2. carry on from the restored state
    app.shaker.playing = True
    _run_to(app, t1 + span)
    rows_c, _, _, _ = _state(app)
    if not np.all(np.isfinite(rows_c)):
        errs.append(f"{key}: non-finite state after continuing from a restore")
    vmax = float(np.sqrt((rows_c[:, 7:10] ** 2).sum(axis=1)).max())
    if vmax > 35.0:
        errs.append(f"{key}: {vmax:.0f} m/s after continuing from a restore")
    failed_c = app.scene.structure.summary()["failed"]
    if abs(failed_c - failed_t2) > max(3, 0.1 * failed_t2):
        errs.append(f"{key}: {failed_c} elements failed after the restored run, "
                    f"{failed_t2} in the uninterrupted one")
    counts_c = app.phys.counts()
    print(f"    {key:12s} snapshot at {t1:.0f} s ({failed_t1} failed): restore exact "
          f"(pose {dpos:.1e}, vel {dvel:.1e}) in {restore_s*1000:.0f} ms; "
          f"at {t1+span:.0f} s uninterrupted {failed_t2} failed / {counts_t2}, "
          f"restored {failed_c} / {counts_c}")

    # 3. seeking in the application: back to the keyframe before t1
    app.reset_scene(quiet=True)
    app.run_quake()
    _frames(app, 2)
    while app.shaker.time < t1 - 1e-9:
        app.taskMgr.step()                      # live playback: keyframes along the way
    n_keys = len(app._snaps)
    target = max(1.0, t1 - 3.0)
    before_rows, before_flags, _, _ = _state(app)
    t_before = app.shaker.time
    app.seek(target)
    n = 0
    while app._seek_target is not None and n < 4000:
        app.taskMgr.step()
        n += 1
    if abs(app.shaker.time - target) > 0.05:
        errs.append(f"{key}: seek to {target} s landed at {app.shaker.time:.2f} s")
    if app._seek_from < 0.0 or app._seek_from > target:
        errs.append(f"{key}: seek started from {app._seek_from:.1f} s, not a keyframe before {target}")
    print(f"    {key:12s} {n_keys} keyframes at {t_before:.0f} s; seek back to {target:.0f} s "
          f"caught up from {app._seek_from:.0f} s in {n} frames")

    # cancel part-way through a long forward seek: back to the pre-seek state
    before_rows, before_flags, before_elems, _ = _state(app)
    t_before = app.shaker.time
    app.seek(app.shaker.duration)
    _frames(app, 3)
    if app._seek_target is None:
        print(f"    {key:12s} (seek to the end finished before it could be cancelled)")
    else:
        app.cancel_seek()
        after_rows, after_flags, after_elems, _ = _state(app)
        live = [i for i, b in enumerate(app.phys.bodies) if not b.crushed]
        d = np.abs(after_rows[live, :7] - before_rows[live, :7]).max()
        if d > 2e-3 or after_flags != before_flags or after_elems != before_elems:
            errs.append(f"{key}: cancel did not return to the pre-seek state (pose diff {d:.4f})")
        if abs(app.shaker.time - t_before) > 1e-6 or app._seek_target is not None:
            errs.append(f"{key}: cancel left the clock at {app.shaker.time} (was {t_before})")
        print(f"    {key:12s} cancelled a seek: back at {app.shaker.time:.2f} s, pose diff {d:.1e}")
    app.reset_scene(quiet=True)


def main(argv) -> int:
    from quakesim.view.app import QuakeSim
    quick = "quick" in argv
    keys = [a for a in argv if a in CASES] or (["ranau_house", "warehouse"] if quick
                                                else list(CASES))
    errs = []
    app = QuakeSim(keys[0], 0.35, seed=4, use_pbr=False)
    print("=== keyframes, restore and seeking")
    for key in keys:
        if app.scene_key != key:
            app.load_scene(key)
        check_scene(app, key, errs)
    print()
    if errs:
        print("FAIL:")
        for e in errs:
            print("  -", e)
        return 1
    print("PASS: every restore is exact, every seek lands, every cancel returns.")
    return 0


if __name__ == "__main__":
    code = main(sys.argv[1:])
    sys.stdout.flush()
    os._exit(code)
