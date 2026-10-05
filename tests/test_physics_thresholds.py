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

"""Does the physics engine reproduce the analytic thresholds?

Two results decide almost everything that happens to loose objects in an
earthquake, and both have exact closed-form answers:

    sliding      a body starts to slide when  a/g > mu
    overturning  a rigid block starts to tip when  a/g > b/h

If the simulation does not reproduce those, nothing built on top of it can be
trusted. Each threshold is found by bisection under a *held* acceleration,
because ramping the load lets the body's own inertia lag behind the ramp and
reports a threshold that is too high by however fast you ramped.
"""

from __future__ import annotations

import math
import sys

from panda3d.core import loadPrcFileData

loadPrcFileData("", "window-type none")
loadPrcFileData("", "audio-library-name null")

from direct.showbase.ShowBase import ShowBase  # noqa: E402

_BASE = None
TOL = 0.05          # 5% of the analytic value


def _base():
    global _BASE
    if _BASE is None:
        _BASE = ShowBase()
    return _BASE


def _moves(a_g, mu, bw, bh, rotating, hold=2.5, tol=0.005):
    from quakesim.sim.physics import PhysicsWorld
    world = PhysicsWorld(_base().render)
    world.add_ground_plane(0.0)
    world.bodies[-1].node.setFriction(math.sqrt(mu))
    blk = world.add_box((bw, bw, bh), (0, 0, bh / 2 + 0.0005), "wood",
                        mass=5.0, label="probe")
    blk.node.setFriction(math.sqrt(mu))
    blk.node.setDeactivationEnabled(False)
    for _ in range(60):
        world.step(1.0 / 240.0)
    x0 = blk.path.getPos()[0]
    for _ in range(int(hold * 240)):
        world.apply_floor_acceleration({0: (a_g * 9.80665, 0.0, 0.0)})
        world.step(1.0 / 240.0)
        if rotating and abs(blk.path.getR()) > 20.0:
            return True
        if (not rotating) and abs(blk.path.getPos()[0] - x0) > tol:
            return True
    return False


def _bisect(mu, bw, bh, rotating, lo=0.01, hi=2.0, steps=16):
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if _moves(mid, mu, bw, bh, rotating):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def main() -> int:
    fails = 0
    print("\nSLIDING   squat block, tipping impossible   theory: a/g = mu")
    for mu in (0.20, 0.35, 0.50, 0.70):
        got = _bisect(mu, 0.40, 0.10, rotating=False)
        err = (got - mu) / mu
        ok = abs(err) <= TOL
        fails += not ok
        print(f"   mu = {mu:.2f}   measured {got:.3f} g   "
              f"error {err*100:+5.1f}%   {'ok' if ok else 'FAIL'}")

    print("\nOVERTURNING   high friction, so it must tip   theory: a/g = b/h")
    for bw, bh in ((0.10, 1.00), (0.20, 1.00), (0.15, 0.45),
                   (0.30, 0.60), (0.40, 0.50)):
        theory = bw / bh
        got = _bisect(2.0, bw, bh, rotating=True)
        err = (got - theory) / theory
        ok = abs(err) <= TOL
        fails += not ok
        print(f"   {bw*100:3.0f} x {bh*100:3.0f} cm   b/h = {theory:.3f}   "
              f"measured {got:.3f} g   error {err*100:+5.1f}%   "
              f"{'ok' if ok else 'FAIL'}")

    print("\nINERTIAL FORCING   a free body must accelerate at exactly -a")
    from quakesim.sim.physics import PhysicsWorld
    world = PhysicsWorld(_base().render)
    b = world.add_box((0.1, 0.1, 0.1), (0, 0, 10), "plastic", mass=1.0,
                      label="probe")
    b.node.setDeactivationEnabled(False)
    world.apply_floor_acceleration({0: (3.0, 0.0, 0.0)})
    for _ in range(240):
        world.step(1.0 / 240.0)
    v = b.node.getLinearVelocity()
    ok_x = abs(v[0] + 3.0) < 0.05
    ok_z = abs(v[2] + 9.80665) < 0.05
    fails += not (ok_x and ok_z)
    print(f"   after 1.0 s: vx = {v[0]:+.4f} m/s (want -3.0000), "
          f"vz = {v[2]:+.4f} (want -9.8066)   "
          f"{'ok' if ok_x and ok_z else 'FAIL'}")

    print()
    if fails:
        print(f"FAIL: {fails} threshold(s) outside {TOL*100:.0f}% of theory")
        return 1
    print(f"PASS: every threshold within {TOL*100:.0f}% of theory")
    return 0


if __name__ == "__main__":
    sys.exit(main())
