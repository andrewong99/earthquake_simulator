#!/usr/bin/env python3
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

"""Earthquake Simulator -- entry point.

    python earthquake.py                          open the default scene
    python earthquake.py --scene kl_highrise      open a particular scene
    python earthquake.py --detail high            more objects (needs more CPU)
    python earthquake.py --list                   list scenes and quake types
    python earthquake.py --report ...             headless analysis, no 3D window

Run `python earthquake.py --help` for everything.
"""

from __future__ import annotations

import argparse
import sys

DETAIL = {"low": 0.35, "medium": 0.6, "high": 1.0, "ultra": 1.5}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="earthquake.py",
        description="A physically-grounded earthquake simulator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
controls
  mouse drag / Tab   look around, all the way up to the zenith and down to
                     the nadir, Stellarium style
  W A S D            move (in walking, flying and plan views)
  Space / Ctrl       up / down          Shift  move faster
  mouse wheel        zoom (narrows the field of view)
  1 - 9   [  ]       jump between the scene's viewpoints
  V                  cycle camera mode        N  next scene
  R                  run the earthquake       P  pause      F  reset objects
  K                  set this scene's collapse case, then R
  T                  advance the time of day  H  hide the panels
  drag a title bar   move a panel             L  reset the panel layout
  G                  show collision shapes    F1 help       Esc quit
""")
    p.add_argument("--scene", default="warehouse",
                   help="scene to open first (any can be chosen from the Scene menu "
                        "inside): warehouse | kl_highrise | ranau_house | "
                        "supermarket | shophouse")
    p.add_argument("--detail", default="medium", choices=list(DETAIL),
                   help="how many individually simulated objects to build")
    p.add_argument("--seed", type=int, default=4)
    p.add_argument("--quality", default="auto",
                   choices=["auto", "low", "medium", "high"],
                   help="rendering quality: shadows, anti-aliasing, normal maps")
    p.add_argument("--physics-hz", type=float, default=120.0,
                   help="physics substep rate (default 120; 240 is more exact "
                        "and twice the cost)")
    p.add_argument("--no-pbr", action="store_true",
                   help="use basic shading instead of physically-based")
    p.add_argument("--list", action="store_true",
                   help="list scenes, earthquake types and site classes")
    p.add_argument("--report", action="store_true",
                   help="headless: print the ground-motion analysis and exit")
    p.add_argument("--mw", type=float, help="magnitude, for --report")
    p.add_argument("--type", dest="qtype", help="quake type key, for --report")
    p.add_argument("--distance", type=float, help="epicentral distance in km")
    p.add_argument("--depth", type=float, help="focal depth in km")
    p.add_argument("--site", help="site key, for --report")
    p.add_argument("--export", metavar="FILE",
                   help="write the synthesized accelerogram to a CSV file")
    args = p.parse_args(argv)

    if args.list:
        from quakesim.cli import print_catalogue
        print_catalogue()
        return 0
    if args.report or args.export:
        from quakesim.cli import report
        return report(args)

    from quakesim.world.scenes import SCENE_ORDER
    if args.scene not in SCENE_ORDER:
        print(f"Unknown scene {args.scene!r}. Choose from: "
              f"{', '.join(SCENE_ORDER)}", file=sys.stderr)
        return 2

    from quakesim.view.app import QuakeSim
    app = QuakeSim(args.scene, DETAIL[args.detail], args.seed,
                   use_pbr=not args.no_pbr, quality=args.quality,
                   physics_hz=args.physics_hz)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
