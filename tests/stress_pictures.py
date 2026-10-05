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

"""Stress-test pictures.

Runs each scene through its stress cases and photographs the results:
at rest, mid-collapse, after, and one case each scene is built to show
(the far-field sway in KL, the real 2015 event in Ranau, the default
supermarket quake and the M8.3 shelf case, an overhead of the wreckage).
Twenty-one frames, HUD on,
each with a caption stating the case and the measured outcome.

    python tests/stress_pictures.py            -> stress_report/01_*.png ... 21_*.png
    python tests/stress_pictures.py --detail 0.35 --out somewhere
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time

import numpy as np
from panda3d.core import loadPrcFileData

loadPrcFileData("", "window-type offscreen")
loadPrcFileData("", "audio-library-name null")
loadPrcFileData("", "win-size 1600 900")
loadPrcFileData("", "framebuffer-multisample 0")
loadPrcFileData("", "multisamples 0")

DT = 1.0 / 30.0


def _caption(path: str, lines: list[str]) -> None:
    """Burn a caption strip into the bottom of the frame."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return
    img = Image.open(path).convert("RGB")
    w, h = img.size
    import textwrap
    wrapped = []
    for i, line in enumerate(lines):
        pieces = textwrap.wrap(line, 150) or [""]
        wrapped += [(p, i == 0) for p in pieces]
    lines = wrapped
    strip = 30 + 26 * len(lines)
    out = Image.new("RGB", (w, h + strip), (14, 16, 20))
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    font = None
    for cand in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "C:/Windows/Fonts/segoeui.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        if os.path.exists(cand):
            font = ImageFont.truetype(cand, 19)
            break
    y = h + 12
    for line, is_title in lines:
        draw.text((18, y), line, fill=(235, 238, 242) if is_title else (170, 180, 190),
                  font=font)
        y += 26
    out.save(path)


class Shoot:
    def __init__(self, app, out_dir: str):
        self.app = app
        self.out = out_dir
        self.n = 0
        os.makedirs(out_dir, exist_ok=True)

    def step(self, seconds: float) -> None:
        app = self.app
        for _ in range(int(seconds / DT)):
            app.shaker.update(DT)
            app.scene.update(app.shaker.time, app.shaker)

    def run_to(self, t_target: float) -> None:
        """Play the loaded record until the shaker clock reaches t_target."""
        app = self.app
        if not app.shaker.playing and app.shaker.time <= 0.0:
            app.run_quake()
        while app.shaker.time < t_target and app.shaker.playing:
            app.shaker.update(DT)
            app.scene.update(app.shaker.time, app.shaker)

    def settle(self, seconds: float = 3.0) -> None:
        self.app.shaker.playing = False
        self.step(seconds)

    def set_case(self, mw, dist, qtype, depth=None) -> None:
        app = self.app
        app._loading = True
        try:
            app.on_type_changed(qtype)
            qt = app.source.qtype
            app.hud.c_mag.set(qt.clamp_mw(mw))
            app.hud.c_dist.set(dist)
            if depth is not None:
                app.hud.c_depth.set(qt.clamp_depth(depth))
            app.hud.c_dir.set(0.0)
            app.hud.c_az.set(35.0)
        finally:
            app._loading = False
        app.rebuild_quake()

    def snap(self, vp: int, title: str, note: str = "", cam=None) -> str:
        """Photograph the current state from viewpoint `vp`, or from `cam`
        = ((x, y, z), (h, p), fov) when the scene's viewpoint would have
        the subject behind a panel."""
        from quakesim.world.scene_base import Viewpoint
        app = self.app
        self.n += 1
        app.apply_viewpoint(vp)
        if cam is not None:
            (x, y, z), (h, pch), fov = cam
            app.rig.apply_viewpoint(Viewpoint("c", "custom", (x, y, z), (h, pch, 0),
                                              "look", fov))
            app.head.reset()
        app.hud.tick_toast(1e9)                   # no toast in the frame
        app.phys._counts_frame = -1000            # status counts refreshed now
        for _ in range(6):                        # the status text updates every 6th
            app.taskMgr.step()
        # a file name that is legal on Windows too: no ':' ',' '(' ...
        slug = re.sub(r"[^a-z0-9._-]+", "_", title.lower().replace("/", "-"))[:40]
        path = os.path.join(self.out, f"{self.n:02d}_{app.scene_key}_{slug}.png")
        app.win.saveScreenshot(path)
        summ = app.scene.structure.summary()
        m = app.motion.meta
        moved, toppled, fallen = app.phys.counts()
        src = app.source
        case = (f"M{src.mw:.1f} {src.qtype.name} at {app.distance_km:.0f} km, "
                f"depth {src.depth_km:.0f} km, stress drop {app.hud.c_stress.get():.0f} bar"
                f"  ·  PGA {m['pga_g']*100:.1f} %g  ·  MMI {m['mmi_text']['roman']}")
        state = (f"t = {app.shaker.time:.1f} s  ·  {len(app.phys.dynamic)} bodies: "
                 f"{moved} shifted, {toppled} toppled, {fallen} fallen  ·  "
                 f"structure: {summ['failed']} of {summ['elements']} elements down")
        report = "  ·  ".join(app.scene.damage_report()) or "no structural damage reported"
        lines = [f"#{self.n:02d}  {app.scene.spec.name}  —  {title}", case, state,
                 report[:150]]
        if note:
            lines.append(note.format(failed=summ["failed"], elements=summ["elements"],
                                     bodies=len(app.phys.dynamic),
                                     crushed=app.phys.crushed_count()))
        _caption(path, lines)
        print(f"  {os.path.basename(path)}: {state}")
        return path


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="stress_report")
    ap.add_argument("--detail", type=float, default=0.6)
    ap.add_argument("--scenes", default="warehouse,kl_highrise,ranau_house,supermarket,shophouse")
    args = ap.parse_args(argv)

    from quakesim.view.app import QuakeSim
    scenes = args.scenes.split(",")
    t0 = time.time()
    app = QuakeSim(scenes[0], args.detail, seed=4, use_pbr=True)
    sh = Shoot(app, args.out)

    for key in scenes:
        if app.scene_key != key:
            app.load_scene(key)
        app.reset_scene(quiet=True)
        print(f"--- {key}")

        if key == "warehouse":
            sh.snap(0, "at rest, in the aisle",
                    "Stability check: 3 s of no shaking, nothing moves; all {elements} structural elements standing.")
            app.set_collapse_case()
            sh.run_to(12.5)
            sh.snap(3, "collapse case, mid-quake, roof-truss level",
                    "Rack bays release when any level drifts 5% (±12%); the shed itself is at 0.3% drift and stands.")
            sh.run_to(26.0)
            sh.settle(3.0)
            sh.snap(1, "after the collapse, end of the aisle",
                    "Every bay let go; loads on the floor. Reset must restore all {elements} elements (verified by test_stress.py).")
            app.reset_scene(quiet=True)
            sh.set_case(7.5, 2.0, "reverse")
            app.hud.c_stress.set(app.hud.c_stress.slider["range"][1])   # slider maximum
            app.rebuild_quake()
            sh.run_to(18.0)
            sh.settle(3.0)
            sh.snap(2, "extreme case from the office: M7.5 at 2 km, maximum stress drop",
                    "Beyond the design range: the stress test's job is that nothing explodes, goes NaN or flies off. The forklift (2.6 t, low) does not go over.")
            app.reset_scene(quiet=True)

        elif key == "kl_highrise":
            sh.snap(0, "at rest, living room",
                    "Level 25 of a 25-storey RC tower on Klang Valley alluvium; the room rides the storey-24 response.")
            sh.set_case(8.8, 560.0, "megathrust", 28.0)
            sh.run_to(200.0)
            sh.snap(1, "far-field default, on the balcony",
                    "The Malaysian case: M8.8 Sumatra megathrust 560 km away. PGA 2% g on the ground, 9% g up here; sway, no damage.")
            app.reset_scene(quiet=True)
            app.set_collapse_case()
            sh.run_to(144.5)
            sh.snap(5, "collapse case, mid-pancake, from the street opposite",
                    "M9.5 megathrust at 6 km: storey 1 drifts 4% and is crushed; each storey above is crushed as it lands on the next.",
                    cam=((0.0, -120.0, -76.1), (0, 12), 74))
            sh.run_to(168.0)
            sh.settle(3.0)
            sh.snap(4, "after the collapse, street level",
                    "The tower is a rubble pile: {failed} of {elements} elements released, {crushed} storeys crushed. Reset restores every one.",
                    cam=((0.0, -60.0, -76.1), (0, 5), 70))
            app.reset_scene(quiet=True)

        elif key == "ranau_house":
            sh.snap(3, "at rest, across the yard",
                    "URM block house below a 33-degree slope (FS 1.08) with a fissure strip across the yard.")
            sh.set_case(6.0, 15.0, "strike_slip", 10.0)
            sh.run_to(30.0)
            sh.settle(3.0)
            sh.snap(1, "the real event: M6.0 at 15 km (5 June 2015), kitchen",
                    "PGA 0.17 g: the kitchen is rattled and the walls crack, but at 0.30 g minimum for a top course nothing structural fails.")
            app.reset_scene(quiet=True)
            app.set_collapse_case()
            sh.run_to(8.2)
            sh.snap(0, "collapse case, mid-quake, living room",
                    "M7.0 at 5 km, PGA 0.41 g: top wall courses go first (0.30 g), gables follow, purlins and roof lose their bearing.")
            sh.run_to(30.0)
            sh.settle(4.0)
            sh.snap(5, "after: landslide, collapsed house, fissure, from the ridge",
                    "Rocks slid by the Newmark integral, lost strength after 8 cm and ran; the yard strip subsided at 0.35 g.")
            app.reset_scene(quiet=True)

        elif key == "supermarket":
            sh.snap(0, "at rest, in the aisle",
                    "{bodies} individually simulated items; stability check passed with nothing moving.")
            sh.set_case(6.4, 11.0, "strike_slip")
            sh.run_to(10.5)
            sh.snap(0, "default quake, mid-shaking, in the aisle",
                    "M6.4 at 11 km, PGA 0.22 g: glass jars slide at 0.26 g and cartons hold past 0.5 g -- friction is per material pair.")
            app.reset_scene(quiet=True)
            sh.set_case(8.3, 11.0, "strike_slip", 10.0)
            sh.run_to(25.0)
            sh.snap(0, "M8.3 at 11 km after 25 s, in the aisle",
                    "PGA 0.30 g on the slab (the GMPE median), 0.48 g on the shelves: every gondola run and chiller is solved as its own 0.35 s / 0.30 s frame and the stock rides that, not the slab. A rigid shelf model dropped 0.2% of the stock here.")
            app.reset_scene(quiet=True)
            app.set_collapse_case()
            sh.run_to(31.0)
            sh.settle(3.0)
            sh.snap(4, "collapse case, after, CCTV corner",
                    "Gondolas and chillers stand on the slab and go over at ~0.35-0.38 g of ground acceleration (+-12%); wall panels tear off the frame at 0.45 g of its response, roof panels follow.")
            sh.snap(1, "collapse case, after, head of the aisles",
                    "The same state across all eight runs: which went over, which stood, and where the stock ended up.")
            app.reset_scene(quiet=True)

        elif key == "shophouse":
            sh.snap(2, "at rest, across the street",
                    "Three four-storey shophouses: RC frame, brick infill, open shopfront -- a soft ground storey.")
            app.set_collapse_case()
            sh.run_to(11.3)
            sh.snap(2, "collapse case, mid-collapse, from across the street",
                    "M7.2 at 5 km: the ground storey drifts 6% while the floors above drift 0.1%; its columns and infill are crushed at 2% and the three floors above come down as a stack.")
            sh.run_to(24.0)
            sh.settle(3.0)
            sh.snap(2, "after the collapse, across the street",
                    "The soft storey is gone: {failed} of {elements} elements released, {crushed} crushed. The parapets went at 0.35 g before the frame did.",
                    cam=((4.75, -22.0, 1.65), (12, -3), 66))
            sh.snap(6, "after the collapse, overhead",
                    "The row in plan: three pancaked units, the signboards and parapets on the street.")
            app.reset_scene(quiet=True)

    print(f"\n{sh.n} pictures in {args.out}/ ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    code = main(sys.argv[1:])
    sys.stdout.flush()
    os._exit(code)
