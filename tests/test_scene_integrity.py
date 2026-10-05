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

"""Scene integrity audit.

Two failure modes ruin a physics scene before the earthquake even starts:

  interpenetration  two rigid bodies built overlapping each other. Bullet
                    resolves the overlap by firing them apart, so the room
                    explodes on load.
  unsupported       an object placed in mid-air because the surface it was
                    meant to rest on is at a different height than assumed.
                    It free-falls at t = 0.

Both look like "the earthquake did it" and neither is physics. This audit
settles each scene with no shaking at all and asserts that nothing moves.
"""

from __future__ import annotations

import math
import sys

from panda3d.core import loadPrcFileData

loadPrcFileData("", "window-type none")
loadPrcFileData("", "audio-library-name null")

from direct.showbase.ShowBase import ShowBase  # noqa: E402

_BASE = None


def _base():
    global _BASE
    if _BASE is None:
        _BASE = ShowBase()
    return _BASE


def _support(world, body) -> str:
    """What is directly under an object at its build position, and how far.

    An object placed even a few millimetres past the edge of its shelf has
    nothing under it and free-falls at t = 0. That reads as "the earthquake
    did it" but happens with no earthquake at all, so it is worth naming the
    surface explicitly rather than inferring it from how far things flew.
    """
    from panda3d.core import Point3
    (hx, hy, hz), _ = body.home
    start = Point3(hx, hy, hz)
    end = Point3(hx, hy, hz - 3.0)
    hit = world.world.rayTestClosest(start, end)
    if not hit.hasHit():
        return "NOTHING BELOW"
    node = hit.getNode()
    gap = (hz - body.size[2] * 0.5) - hit.getHitPos().getZ()
    return f"{node.getName()}, {gap*1000:+.0f} mm gap under base"


def audit(scene_key: str, detail: float = 0.5, settle_s: float = 3.0,
          tol_move: float = 0.03, verbose: bool = True) -> dict:
    from quakesim.sim.physics import PhysicsWorld
    from quakesim.world.scenes import SCENES

    base = _base()
    world = PhysicsWorld(base.render)
    scene = SCENES[scene_key](world, base.render, detail=detail)

    # --- overlap check ------------------------------------------------------
    # One real substep, then read Bullet's own persistent manifolds: their
    # point distances are what the solver acted on. Two things this replaces
    # were wrong. contactTest() results in Panda 1.10 hold a reference to a
    # manifold point that has gone out of scope, so every contact of a body
    # reports the distance of whichever pair was computed last (a 15 cm
    # overlap reads as -0.0). And the probe used to be a 1 microsecond fixed
    # step: Bullet's position correction is erp * penetration / dt, so 0.1 mm
    # of contact noise -- normal for a bottle on an 18 m shelf -- became an
    # 18 m/s kick, and the audit reported items "launched" that the
    # simulator itself never moves.
    #
    # Static bodies sleep (PhysicsWorld.finalise), and Bullet skips a pair
    # when neither body is active, so for this one step every static body is
    # woken: the interesting overlaps are between two *static* elements --
    # a wall built through a column -- which cost nothing while both stand
    # and become a shove the moment the collapse model releases one.
    info = {b.node: b for b in world.bodies}
    for b in world.bodies:
        if b.node.isStatic():
            b.node.setActive(True, True)
    world.world.doPhysics(world.substep, 1, world.substep)
    for b in world.bodies:
        world._sleep_static(b)

    def releasable(b):
        return b.mass > 0.0 or b.breakable or b.driven

    overlaps = []           # at least one body can move: a real defect
    static_overlaps = []    # two fixed bodies: costs nothing, reported only
    for m in world.world.getManifolds():
        pts = m.getManifoldPoints()
        if not pts:
            continue
        d = min(p.getDistance() for p in pts)
        if d >= -0.004:
            continue
        a, c = info.get(m.getNode0()), info.get(m.getNode1())
        if a is None or c is None:
            continue
        row = (d, a.label or a.node.getName(), c.label or c.node.getName())
        (overlaps if releasable(a) or releasable(c) else static_overlaps).append(row)
    overlaps.sort()
    static_overlaps.sort()

    # --- settle with no shaking ------------------------------------------
    for _ in range(int(settle_s * 60)):
        world.step(1.0 / 60.0)

    moved = []
    for b in world.dynamic:
        if not b.home:
            continue
        p = b.path.getPos()
        h = b.home[0]
        d = math.dist((p[0], p[1], p[2]), h)
        if d > tol_move:
            moved.append((d, b.label or b.node.getName(), b.kind,
                          _support(world, b)))
    moved.sort(reverse=True)

    report = {
        "scene": scene_key,
        "dynamic": len(world.dynamic),
        "overlaps": overlaps,
        "static_overlaps": static_overlaps,
        "struct_overlaps": overlaps,        # older name
        "moved": moved,
        "worst_move_cm": (moved[0][0] * 100 if moved else 0.0),
        "moved_fraction": len(moved) / max(len(world.dynamic), 1),
    }
    if verbose:
        print(f"\n=== {scene_key} (detail {detail}) : {len(world.dynamic)} dynamic bodies")
        print(f"    overlapping at build (movable or releasable) : {len(overlaps)}")
        for d, a, bnm in overlaps[:10]:
            print(f"        {d*1000:7.1f} mm  {a}  <->  {bnm}")
        print(f"    fixed-with-fixed overlaps (no effect)        : {len(static_overlaps)}")
        print(f"    moved after {settle_s:.0f} s of no shaking : {len(moved)}"
              f"  ({report['moved_fraction']*100:.1f}%)")
        for d, lbl, kind, sup in moved[:12]:
            print(f"        {d*100:8.1f} cm  {lbl} [{kind}]   <- {sup}")
    scene.destroy()
    return report


def main(keys=None, detail=0.5):
    from quakesim.world.scenes import SCENE_ORDER
    keys = keys or SCENE_ORDER
    bad = []
    for k in keys:
        r = audit(k, detail)
        # Loose contents settling a couple of centimetres is real physics --
        # fruit finds its level in a bin. Anything that travels further than
        # its own size, or a systematic fraction of the scene moving, is a
        # build error.
        big = [m for m in r["moved"] if m[0] > 0.30]
        if r["moved_fraction"] > 0.02 or len(big) > 0 or r["overlaps"]:
            bad.append(k)
            if big:
                print(f"    -> {len(big)} object(s) travelled more than 30 cm "
                      f"with no shaking")
    print()
    if bad:
        print(f"FAIL: scenes not stable at rest: {', '.join(bad)}")
        return 1
    print("PASS: every scene is stable at rest.")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    sys.exit(main(args or None))
