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

"""Structural collapse.

The shear-building solver in `mdof.py` tells us how far each storey drifts
and how hard each floor accelerates, as functions of time. This module turns
that into things you can see: it holds every structural element that is
allowed to fail, checks each one's failure rule against the solver's time
series as the earthquake plays, and *releases* the element -- hands it to the
physics engine as a free body -- the moment its rule is met. After that the
fall is pure rigid-body dynamics: a released slab drops onto whatever is
below, a released wall panel topples outward, columns above a failed storey
lose their footing and come down with the storey they carry.

Failure rules
    drift    the storey's inter-storey drift ratio reaches the limit
             (columns, walls, glazing, connectors -- anything racked by
             the frame)
    acc      the floor's horizontal acceleration reaches the limit, in g
             (parapets, gables, out-of-plane masonry, ceiling panels --
             anything thrown by its own inertia)
    support  enough of the elements it stands on have failed
             (a slab on its columns; the columns of the next storey on that
             slab -- this is what lets a failure propagate upward)

Each element's limit is scattered by a few percent so a wall comes apart panel
by panel rather than all at once, which is both how real materials behave and
what makes a collapse read as a collapse rather than a switch being thrown.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from ..seismo import constants as K


@dataclass
class Element:
    body: object                        # BodyInfo
    rule: str                           # drift | acc | support
    story: int = 0
    limit: float = 0.0
    group: int = 0                      # floor group whose response applies
    depends: list = field(default_factory=list)
    need: int = 1                       # support rule: how many must fail
    floor_after: int = 0                # floor group once released (0 = ground)
    tag: str = ""
    # Extra drift limit, on top of the rule: a wall panel that is thrown by
    # acceleration also comes apart when the storey it belongs to racks past
    # the material's collapse drift.
    drift_limit: float = 0.0
    # Read the acceleration from the ground motion rather than the storey's
    # response. Right for the walls of a single-storey masonry house: they
    # *are* the structure, and once it yields the solver caps the roof
    # acceleration at the wall strength -- which says nothing about the
    # out-of-plane demand on the wall, which is the ground's.
    ground_acc: bool = False
    # A storey that fails by its own rule is crushed -- its body is taken
    # out of the world and whatever stood on it drops through -- rather than
    # released as a free block. Support failures never crush.
    crush: bool = False
    failed_at: float | None = None
    failed_by: str = ""

    @property
    def failed(self) -> bool:
        return self.failed_at is not None


class StructuralModel:
    def __init__(self, phys, seed: int = 4, scatter: float = 0.12):
        self.phys = phys
        self.rng = random.Random(seed)
        self.scatter = scatter
        self.elements: list[Element] = []
        self.events: list[tuple[float, str]] = []
        self._last_t = -1.0

    # -- building the model ------------------------------------------------
    def add(self, body, rule: str, limit: float = 0.0, story: int = 0,
            group: int = 0, depends=None, need: int = 1, floor_after: int = 0,
            tag: str = "", scatter: bool = True, drift_limit: float = 0.0,
            ground_acc: bool = False, crush: bool = False) -> Element:
        if scatter and rule in ("drift", "acc"):
            limit = limit * (1.0 + self.rng.uniform(-self.scatter, self.scatter))
        if scatter and drift_limit > 0.0:
            drift_limit *= 1.0 + self.rng.uniform(-self.scatter, self.scatter)
        e = Element(body, rule, story, limit, group, list(depends or []),
                    need, floor_after, tag, drift_limit, ground_acc, crush)
        self.elements.append(e)
        return e

    def reset(self) -> None:
        for e in self.elements:
            e.failed_at = None
            e.failed_by = ""
        self.events = []
        self._last_t = -1.0

    # -- evaluation ------------------------------------------------------
    @staticmethod
    def _slice(dt: float, n: int, t0: float, t1: float):
        """Sample indices covering (t0, t1] -- the peak inside a frame, not
        the value at the instant the frame happened to land on."""
        i1 = int(t1 / dt)
        i0 = int(t0 / dt) + 1 if t0 >= 0.0 else 0
        i0 = max(0, min(i0, i1))
        if i1 < 0 or i0 >= n:
            return None
        return i0, min(i1, n - 1) + 1

    def _ground_acc(self, shaker, t0: float, t1: float) -> float:
        m = shaker.motion
        if m is None:
            return 0.0
        sl = self._slice(m.dt, m.n, t0, t1)
        if sl is None:
            return 0.0
        h = np.hypot(m.acc[0, sl[0]:sl[1]], m.acc[1, sl[0]:sl[1]])
        return float(h.max()) * shaker.gain / K.G0

    def _demand(self, shaker, group: int, story: int, t0: float, t1: float,
                ground_acc: bool = False):
        """(peak drift ratio, peak horizontal acceleration in g) over the
        interval (t0, t1]."""
        entry = shaker.responses.get(group)
        if entry is None:
            # No structure for this group: the ground motion itself.
            return 0.0, self._ground_acc(shaker, t0, t1)
        resp, _ = entry
        sl = self._slice(resp.dt, resp.drift.shape[1], t0, t1)
        if sl is None:
            return 0.0, 0.0
        # story < 0: the worst storey of the structure (a rack frame goes
        # wherever along its height the drift is worst)
        s = slice(None) if story < 0 else min(story, resp.stories - 1)
        d = float(np.max(np.abs(resp.drift[s, sl[0]:sl[1]])))
        if resp.drift_y is not None:
            d = max(d, float(np.max(np.abs(resp.drift_y[s, sl[0]:sl[1]]))))
        if ground_acc:
            return d * shaker.gain, self._ground_acc(shaker, t0, t1)
        ax = resp.floor_acc[s, sl[0]:sl[1]]
        if resp.floor_acc_y is not None:
            a = float(np.max(np.hypot(ax, resp.floor_acc_y[s, sl[0]:sl[1]])))
        else:
            a = float(np.max(np.abs(ax)))
        return d * shaker.gain, a * shaker.gain / K.G0

    def update(self, t: float, shaker) -> list[Element]:
        """Release every element whose rule is met at time t. Returns them."""
        if shaker.motion is None or t <= self._last_t:
            return []
        t0, self._last_t = self._last_t, t
        released = []
        cache: dict[tuple, tuple] = {}
        for e in self.elements:
            if e.failed:
                continue
            # Anything with supports goes when enough of them have gone,
            # whatever its own rule -- a gable course also falls when the
            # course under it does.
            if e.depends and sum(1 for d in e.depends if d.failed) >= e.need:
                e.failed_at = t
                e.failed_by = "support"
                released.append(e)
                continue
            if e.rule == "support":
                continue
            key = (e.group, e.story, e.ground_acc)
            if key not in cache:
                cache[key] = self._demand(shaker, e.group, e.story, t0, t,
                                          e.ground_acc)
            drift, acc_g = cache[key]
            if ((e.rule == "drift" and drift >= e.limit)
                    or (e.rule == "acc" and acc_g >= e.limit)
                    or (e.drift_limit > 0.0 and drift >= e.drift_limit)):
                e.failed_at = t
                e.failed_by = e.rule
                released.append(e)

        # A support failure can cascade within the same instant (slab -> the
        # columns standing on it -> the slab above), so iterate to a fixed point.
        while True:
            more = []
            for e in self.elements:
                if e.failed or not e.depends:
                    continue
                if sum(1 for d in e.depends if d.failed) >= e.need:
                    e.failed_at = t
                    e.failed_by = "support"
                    more.append(e)
            if not more:
                break
            released.extend(more)

        for e in released:
            crush = e.crush and e.failed_by != "support"
            self.phys.release(e.body, floor=e.floor_after, crush=crush)
            if not crush:
                if e.failed_by == "acc":
                    self._tip_over(e, shaker, t)
                elif not e.ground_acc:
                    self._release_velocity(e, shaker, t)
            if e.tag:
                self.events.append((t, e.tag))
        return released

    def _tip_over(self, e, shaker, t: float) -> None:
        """An element whose acceleration rule has fired is going over: the
        rule is the overturning fragility (a loaded gondola's median is far
        below the rigid-block b/h of its empty carcass, because of rocking
        amplification and the stock shifting on it), so the block is set
        rotating about its base edge with just enough energy to carry its
        centre of mass over the edge, in the direction the inertial force
        is pushing it, and gravity does the rest. Without this a released
        gondola sat rocking on the slab -- the rule said it had gone over
        and the picture showed it standing."""
        body = e.body
        m = shaker.motion
        lo, hi = body.local_lo, body.local_hi
        if m is None or lo is None:
            return
        ax, ay, _ = m.sample(t)
        fx, fy = -ax, -ay                     # the inertial force on the block
        if abs(fx) < 1e-6 and abs(fy) < 1e-6:
            return
        from panda3d.core import Vec3
        q = body.path.getQuat()
        best = None
        for axis_local, i in ((Vec3(1, 0, 0), 0), (Vec3(0, 1, 0), 1)):
            axis = q.xform(axis_local)
            f = fx * axis[0] + fy * axis[1]
            b = hi[i] if f > 0 else -lo[i]
            if b <= 1e-3:
                continue
            score = abs(f) / b
            if best is None or score > best[0]:
                best = (score, axis * (1.0 if f > 0 else -1.0), b)
        if best is None:
            return
        _, d, b = best
        d = Vec3(d[0], d[1], 0.0)
        if d.length() < 1e-6:
            return
        d.normalize()
        h_c = -lo[2]                           # centre of mass above the base
        if h_c <= 1e-3:
            return
        r = math.hypot(b, h_c)
        drop = r - h_c                         # rise of the CoM to the tipping point
        omega = 2.0 * math.sqrt(1.5 * K.G0 * drop) / r
        axis = Vec3(0, 0, 1).cross(d)          # z x d: turns the top toward d
        node = body.node
        node.setAngularVelocity(axis * omega)
        node.setLinearVelocity(d * (omega * h_c) + Vec3(0, 0, omega * b))

    def _release_velocity(self, e, shaker, t: float) -> None:
        """A frame lets go while it is moving. The element is a static body
        drawn at its home pose until this instant, so it is handed the
        lateral velocity its storey has at this instant, relative to the
        ground. Without it a soft-storey frame released at 6% drift stood as
        a set of loose, perfectly upright blocks -- statically stable,
        which a frame leaning 20 cm at the moment it fails is not."""
        entry = shaker.responses.get(e.group)
        if entry is None or e.story < 0:
            return
        resp, _ = entry
        n = resp.floor_disp.shape[1]
        i = int(t / resp.dt)
        # the response is solved on a coarser grid than the record and held
        # between its samples, so difference across 30 ms, not one sample
        k = max(1, int(round(0.015 / resp.dt)))
        i0, i1 = max(0, i - k), min(n - 1, i + k)
        if i1 <= i0:
            return
        s = min(e.story, resp.stories - 1)
        span = (i1 - i0) * resp.dt
        vx = float(resp.floor_disp[s, i1] - resp.floor_disp[s, i0]) / span
        vy = 0.0
        if getattr(resp, "floor_disp_y", None) is not None:
            vy = float(resp.floor_disp_y[s, i1] - resp.floor_disp_y[s, i0]) / span
        g = shaker.gain
        from panda3d.core import Vec3
        e.body.node.setLinearVelocity(Vec3(vx * g, vy * g, 0.0))

    # -- snapshots -----------------------------------------------------------
    def snapshot(self) -> tuple:
        return ([(e.failed_at, e.failed_by) for e in self.elements],
                list(self.events), self._last_t)

    def restore(self, snap: tuple) -> None:
        """The elements' failure record; the bodies themselves are the
        physics world's business (PhysicsWorld.restore)."""
        flags, events, last_t = snap
        for e, (at, by) in zip(self.elements, flags):
            e.failed_at = at
            e.failed_by = by
        self.events = list(events)
        self._last_t = last_t

    # -- reporting ---------------------------------------------------------
    def summary(self) -> dict:
        total = len(self.elements)
        failed = sum(1 for e in self.elements if e.failed)
        by_tag: dict[str, int] = {}
        for e in self.elements:
            if e.failed and e.tag:
                by_tag[e.tag] = by_tag.get(e.tag, 0) + 1
        return {"elements": total, "failed": failed, "by_tag": by_tag,
                "first_failure": min((e.failed_at for e in self.elements if e.failed),
                                     default=None)}
