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

"""End-to-end smoke test.

Builds the whole application offscreen, runs a real earthquake through it, and
checks that the right things happen: the record plays, objects that should
move do move, every camera mode and viewpoint works, and a frame renders.
"""

from __future__ import annotations

import os
import sys
import time

from panda3d.core import loadPrcFileData

loadPrcFileData("", "window-type offscreen")
loadPrcFileData("", "audio-library-name null")
loadPrcFileData("", "win-size 960 600")
loadPrcFileData("", "framebuffer-multisample 0")
loadPrcFileData("", "multisamples 0")


def run(scene="ranau_house", detail=0.35, shot=None, render_frames=45) -> int:
    from quakesim.view.app import QuakeSim

    t0 = time.time()
    app = QuakeSim(scene, detail, seed=4, use_pbr=False)
    build = time.time() - t0
    m = app.motion.meta
    print(f"  built {scene}: {len(app.phys.dynamic)} dynamic bodies in {build:.1f} s")
    print(f"  quake  {m['pga_g']*100:.1f} %g   PGV {m['pgv_cms']:.1f} cm/s   "
          f"MMI {m['mmi_text']['roman']}   JMA {m['jma']['class']}   "
          f"record {app.motion.duration:.0f} s")
    print(f"  arrivals: P {app.motion.arrivals['P']:.1f} s, "
          f"S {app.motion.arrivals['S']:.1f} s")
    if app.response is not None:
        print(f"  building T1 {app.response.periods[0]:.2f} s, "
              f"peak drift {float(app.response.peak_drift.max())*100:.3f} %, "
              f"damage {app.response.damage_name}")

    # --- physics phase, at a fixed timestep so the result does not depend
    #     on how fast this machine happens to render -----------------------
    app.run_quake()
    dt = 1.0 / 60.0
    horizon = min(app.motion.duration, app.motion.arrivals["S"] + 30.0)
    steps = int(horizon / dt)
    t0 = time.time()
    for _ in range(steps):
        app.shaker.update(dt)
        app.scene.update(app.shaker.time, app.shaker)
    sim = time.time() - t0
    stirred = app.phys.displaced_count(0.005)
    moved = app.phys.displaced_count()
    toppled = app.phys.toppled_count()
    fallen = app.phys.fallen_count()
    print(f"  simulated {horizon:.1f} s of shaking in {sim:.1f} s CPU "
          f"({horizon/sim:.2f}x real time)")
    print(f"  of {len(app.phys.dynamic)} objects: {stirred} stirred (>5 mm), "
          f"{moved} shifted (>2 cm), {toppled} rotated past 25 deg, "
          f"{fallen} fell")

    # At moderate intensity most things correctly stay put -- a stiff room at
    # MMI VI should show crockery sliding a centimetre, not a wrecked house.
    # What would be wrong is nothing responding at all.
    if stirred == 0 and m["pga_g"] > 0.05:
        print("  !! nothing responded at all despite significant shaking")
        return 1

    # --- render phase ----------------------------------------------------
    t0 = time.time()
    for _ in range(render_frames):
        app.taskMgr.step()
    el = time.time() - t0
    print(f"  {render_frames} frames rendered in {el:.1f} s "
          f"({render_frames/el:.1f} fps, software renderer)")

    for vp in range(len(app.scene.spec.viewpoints)):
        app.apply_viewpoint(vp)
        app.taskMgr.step()
    from quakesim.view.camera import MODES
    for mode in MODES:
        app.rig.set_mode(mode)
        app.taskMgr.step()
    app.rig.zoom(1)
    app.rig.zoom(-1)
    app._cycle_time()
    app.hud.toggle_controls()
    app.hud.toggle_controls()
    app.reset_scene()
    app.taskMgr.step()
    print(f"  {len(app.scene.spec.viewpoints)} viewpoints, {len(MODES)} camera "
          f"modes, zoom, time-of-day, panels and reset all OK")

    # the collapse case, through the same path the K key and button use
    app.set_collapse_case()
    app.run_quake()
    for _ in range(int(min(app.motion.duration, 40.0) / dt)):
        app.shaker.update(dt)
        app.scene.update(app.shaker.time, app.shaker)
        if app.scene.structure.summary()["failed"] and app.shaker.time > 3.0:
            break
    for _ in range(10):
        app.taskMgr.step()
    summ = app.scene.structure.summary()
    print(f"  collapse case: {summ['failed']} of {summ['elements']} structural "
          f"elements released by t = {app.shaker.time:.1f} s; "
          f"{'; '.join(app.scene.damage_report()) or 'nothing reported'}")
    app.reset_scene()
    app.taskMgr.step()

    # switch scene through the in-app chooser, as the dropdown does, and
    # make sure the old scene is really gone (its bodies used to stay drawn)
    from quakesim.world.scenes import SCENE_ORDER, SCENES
    from quakesim.view.fonts import safe
    other = SCENE_ORDER[(SCENE_ORDER.index(scene) + 1) % len(SCENE_ORDER)]
    app.hud._scene_changed(safe(SCENES[other].spec.short or SCENES[other].spec.name))
    for _ in range(12):
        app.taskMgr.step()
        time.sleep(0.01)
    if app.scene_key != other:
        print(f"  !! scene chooser did not switch to {other}")
        return 1
    dc = app.phys.draw_call_estimate()
    print(f"  chooser switched to {other}: {len(app.phys.dynamic)} bodies, "
          f"{dc} draw calls, camera {app.rig.describe()}")
    if dc > 600:
        print("  !! draw calls too high after a scene switch -- old scene left behind?")
        return 1
    app.hud._vp_changed(safe(app.scene.spec.viewpoints[1].name))
    app.taskMgr.step()
    if app.vp_index != 1:
        print("  !! viewpoint chooser did not apply")
        return 1

    if shot:
        app.taskMgr.step()
        app.win.saveScreenshot(shot)
        print(f"  screenshot -> {shot}")
    return 0


if __name__ == "__main__":
    scene = sys.argv[1] if len(sys.argv) > 1 else "ranau_house"
    detail = float(sys.argv[2]) if len(sys.argv) > 2 else 0.35
    shot = sys.argv[3] if len(sys.argv) > 3 else None
    print(f"\n--- smoke test: {scene} (detail {detail}) ---")
    code = run(scene, detail, shot)
    sys.stdout.flush()
    os._exit(code)      # skip interpreter teardown: Panda's Bullet objects abort in it
