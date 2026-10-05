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

"""Bullet physics world and the non-inertial "shaker".

How the shaking is applied
--------------------------
The obvious approach -- move the floor and walls along the ground displacement
and let contact drag everything else along -- works but behaves badly: the
camera lurches, kinematic bodies tunnel through contacts at high frequency,
and the whole world translates metres away from the origin during a big event.

So the simulation runs in the *frame of the floor*. The room stays where it
is, and every free body instead feels the inertial (d'Alembert) force

    F = -m * a_floor(t)

which is exactly equivalent by the equivalence principle -- it is the same
physics you feel as being pushed sideways in a braking bus. Because that force
is proportional to mass, it can be applied as a per-body gravity vector
instead of a force, which costs nothing:

    g_effective = (-ax, -ay, -g - az)

Objects then slide, rock, topple and fall entirely on their own, from real
friction and real geometry. Nothing about their motion is scripted.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from panda3d.bullet import (BulletBoxShape, BulletCapsuleShape,
                            BulletCylinderShape, BulletDebugNode,
                            BulletGenericConstraint, BulletPlaneShape,
                            BulletRigidBodyNode, BulletSphereShape,
                            BulletTriangleMesh, BulletTriangleMeshShape,
                            BulletWorld, ZUp)
from panda3d.core import (BitMask32, NodePath, Point3, Quat,
                          RigidBodyCombiner, TransformState, Vec3)

from . import materials as mat
from ..seismo import constants as K

def _margin_for(min_dim: float) -> float:
    """Bullet's collision margin, scaled to the object.

    The default margin is 40 mm. That is fine for vehicles and catastrophic
    for groceries: a 74 mm drinks can is almost entirely margin, so two cans
    standing 1 mm apart read as deeply interpenetrating and get fired across
    the room. Scaling the margin to the object keeps small items stable
    without giving up the numerical cushioning that large ones need.
    """
    return float(min(max(0.04 * min_dim, 0.0004), 0.02))


# Sleep thresholds. A body below both for SLEEP_TIME is frozen exactly.
#
# These are deliberately not tiny. The sequential-impulse solver leaves a
# stack of bodies slightly compliant -- a pallet with twelve cartons on it
# moves at v = -a * 47 ms under a base acceleration a, whatever the iteration
# count -- so under shaking well below its sliding threshold the stack
# jitters at tens of mm/s and, if kept awake, walks a few centimetres a
# minute. At 0.10 m/s that jitter is below the threshold and the stack
# sleeps, which is the physically right answer: it should not be moving.
# Anything whose own sliding or tipping threshold the shaking has reached is
# woken by apply_floor_acceleration and stays awake while it moves. The cost
# of the thresholds is that a body sliding out slower than 0.10 m/s for half
# a second is stopped a millimetre or two early.
SLEEP_LIN = 0.10          # m/s
SLEEP_ANG = 0.50          # rad/s
SLEEP_TIME = 0.5          # s

MASK_STATIC = BitMask32.bit(0)
MASK_DYNAMIC = BitMask32.bit(1)
MASK_DEBRIS = BitMask32.bit(2)


@dataclass
class BodyInfo:
    """Bookkeeping for one simulated object."""
    node: BulletRigidBodyNode
    path: NodePath
    material: str
    mass: float
    size: tuple[float, float, float]
    floor: int = 0
    kind: str = "prop"
    anchored: bool = False
    fragile: bool = False
    broken: bool = False
    home: tuple = ()
    label: str = ""
    wake_threshold: float = 0.0     # |a| in m/s^2 below which it cannot move
    breakable: bool = False         # a structural element that can be released
    com_offset: tuple = (0.0, 0.0, 0.0)   # compound: parts' centroid in the given frame
    local_lo: tuple | None = None         # bounding box about the centre of mass
    local_hi: tuple | None = None
    released: bool = False
    break_mass: float = 0.0         # the mass it gets once released
    driven: bool = False            # position set from a structural solution
    home_floor: int = 0             # floor group it was built in
    driven_home: tuple = ()         # (pos, hpr, drift_gain) for a driven body
    crushed: bool = False           # taken out of the world entirely
    peak_mu: float = 0.0            # slope material: peak friction (0 = n/a)
    softened: bool = False          # slope material: dropped to residual

    @property
    def slenderness(self) -> float:
        w = min(self.size[0], self.size[1])
        return w / max(self.size[2], 1e-4)

    def tipping_threshold_g(self) -> float:
        return self.slenderness


@dataclass
class WorldSnapshot:
    """The complete state of every body at one instant -- enough to put the
    world back exactly there (PhysicsWorld.restore). Rows follow
    PhysicsWorld.bodies, whose membership never changes after the build."""
    state: np.ndarray         # (n, 13) float32: pos xyz, quat rijk, lin vel, ang vel
    active: np.ndarray        # (n,) bool: awake
    floor: np.ndarray         # (n,) int32
    released: np.ndarray      # (n,) bool
    crushed: np.ndarray       # (n,) bool
    softened: np.ndarray      # (n,) bool: slope rock dropped to residual friction
    friction: np.ndarray      # (n,) float32
    driven_last: dict
    last_mag: dict
    n_released: int

    def nbytes(self) -> int:
        return int(self.state.nbytes + self.active.nbytes + self.floor.nbytes
                   + self.released.nbytes + self.crushed.nbytes
                   + self.softened.nbytes + self.friction.nbytes)


class PhysicsWorld:
    """Wraps a Bullet world with the bookkeeping the simulator needs."""

    def __init__(self, render: NodePath, substep: float = 1.0 / 120.0,
                 max_substeps: int = 4):
        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0, -K.G0))
        self.render = render
        # 120 Hz is enough for objects a few centimetres across once continuous
        # collision detection is on, and it costs half of what 240 Hz does.
        # Capping the substeps per frame stops the death spiral where a slow
        # frame demands more physics, which makes the next frame slower.
        self.substep = substep
        self.max_substeps = max_substeps

        self.bodies: list[BodyInfo] = []
        self.dynamic: list[BodyInfo] = []
        self.by_floor: dict[int, list[BodyInfo]] = {}
        self.kinematic: list[tuple[BodyInfo, tuple]] = []
        self.breakable: list[BodyInfo] = []
        self._driven_last: dict[int, tuple] = {}
        self.n_released = 0
        self._limit_frame = 0
        # Two trees. Everything that can move -- dynamic and kinematic bodies --
        # lives under a RigidBodyCombiner, which draws all of it in a handful
        # of batches and re-transforms the vertices itself as the bodies move.
        # Everything that cannot move lives under a plain node for physics,
        # with its geometry in a separate visual tree that is flattened once.
        # Together these take a scene from thousands of draw calls to tens.
        self._rbc = RigidBodyCombiner("moving-bodies")
        self._root = render.attachNewNode(self._rbc)
        self._static_root = render.attachNewNode("static-bodies")
        self.static_visual_root = render.attachNewNode("static-visuals")
        self.debug_np: NodePath | None = None
        self._counts_cache = (0, 0, 0)
        self._counts_frame = -1
        self.last_ms = 0.0
        self.n_breakable = 0
        self._thresh = np.zeros(0, dtype=np.float32)
        self._floor_idx = np.zeros(0, dtype=np.int32)
        self._thresh_dirty = True
        self._last_mag: dict[int, float] = {}
        # Every body may sleep. It is not only a speed matter: a stack that is
        # never allowed to sleep integrates the solver's residual jitter for
        # as long as the scene is open and slowly walks off its shelf (a
        # four-plate stack, ten solver iterations: 2 mm/s of jitter, a
        # centimetre a minute). A sleeping body is exactly still. Nothing
        # sleeps through an arrival because apply_floor_acceleration wakes
        # each body the moment the shaking crosses its own threshold, and
        # drive_kinematic wakes whatever stands on a support that moved.
        self.always_awake = False
        # Build progress, for the loading screen: `progress(n_objects)` every
        # few free objects while the scene is built, `stage(label, fraction)`
        # through finalise. Both optional.
        self.progress = None
        self.stage = None

    # -- construction ------------------------------------------------------
    def _register(self, info: BodyInfo) -> BodyInfo:
        self.bodies.append(info)
        info.home_floor = info.floor
        if info.mass > 0.0:
            if self.progress is not None and len(self.dynamic) % 20 == 0:
                self.progress(len(self.dynamic) + 1)
            # The smallest horizontal acceleration that can do anything to this
            # object: it either slides (at mu*g) or tips (at b/h * g), whichever
            # comes first. Below that it is genuinely inert and may sleep, which
            # is what makes a twelve-thousand-object scene run at all.
            m = mat.get(info.material)
            thresh = min(m.friction, max(info.slenderness, 0.02)) * K.G0
            info.wake_threshold = thresh * 0.40
            self.dynamic.append(info)
            self.by_floor.setdefault(info.floor, []).append(info)
            self._thresh_dirty = True
        return info

    def _rebuild_threshold_index(self) -> None:
        self._thresh = np.array([b.wake_threshold for b in self.dynamic],
                                dtype=np.float32)
        self._floor_idx = np.array([b.floor for b in self.dynamic],
                                   dtype=np.int32)
        self._thresh_dirty = False

    def add_box(self, size, pos, material="plastic", mass=None, hpr=(0, 0, 0),
                floor=0, kind="prop", static=False, anchored=False,
                label="", parent=None, mass_override=None) -> BodyInfo:
        sx, sy, sz = size
        m = mat.get(material)
        if mass is None:
            mass = 0.0 if static else m.density * sx * sy * sz
        if mass_override is not None:
            mass = mass_override

        shape = BulletBoxShape(Vec3(sx * 0.5, sy * 0.5, sz * 0.5))
        shape.setMargin(_margin_for(min(sx, sy, sz)))
        node = BulletRigidBodyNode(label or f"{material}_box")
        node.addShape(shape)
        node.setMass(0.0 if (static or anchored) else mass)
        # Bullet multiplies the two bodies' friction, so store sqrt(mu)
        node.setFriction(math.sqrt(max(m.friction, 1e-3)))
        node.setRestitution(m.restitution)
        node.setAnisotropicFriction(Vec3(1, 1, 1))
        node.setDeactivationEnabled(False)

        if not static and not anchored:
            r = 0.5 * min(sx, sy, sz)
            node.setCcdMotionThreshold(r)
            node.setCcdSweptSphereRadius(r * 0.8)
            node.setDeactivationEnabled(True)
            node.setLinearSleepThreshold(SLEEP_LIN)
            node.setAngularSleepThreshold(SLEEP_ANG)
            node.setDeactivationTime(SLEEP_TIME)

        root = self._static_root if (static or anchored) else self._root
        np_ = (parent or root).attachNewNode(node)
        np_.setPos(*pos)
        np_.setHpr(*hpr)
        self.world.attachRigidBody(node)
        info = BodyInfo(node=node, path=np_, material=material, mass=mass,
                        size=(sx, sy, sz), floor=floor, kind=kind,
                        anchored=static or anchored, fragile=m.fragile,
                        home=(tuple(pos), tuple(hpr)), label=label)
        return self._register(info)

    def add_compound(self, parts, pos, material="wood", mass=1.0, hpr=(0, 0, 0),
                     floor=0, kind="furniture", static=False, anchored=False,
                     label="") -> BodyInfo:
        """One rigid body made of several boxes.

        A table is not a slab floating above four posts -- it is one object.
        Building it from separate bodies makes them overlap at the joints and
        blow apart on the first frame, and gets the tipping behaviour wrong
        besides, because what decides whether a table goes over is its leg
        spread against the height of its centre of mass. `parts` is a list of
        ((sx, sy, sz), (ox, oy, oz)) in body-local coordinates.
        """
        m = mat.get(material)
        node = BulletRigidBodyNode(label or f"{material}_compound")
        # Panda's Bullet body has its centre of mass AT THE NODE ORIGIN, so a
        # compound described from its base (a table at floor level, a
        # gondola on the slab) must have its node placed at the parts'
        # centroid and the parts offset back, or its mass sits on the floor
        # and it can never tip: measured, a 1 m wide, 2 m tall box built
        # with its origin at the base slid 3.5 m under 0.7 g and stayed
        # upright, the same box built about its centre went over.
        vol = 0.0
        com = [0.0, 0.0, 0.0]
        for size, off in parts:
            v = size[0] * size[1] * size[2]
            vol += v
            for i in range(3):
                com[i] += off[i] * v
        com = [c / vol for c in com] if vol > 0 else [0.0, 0.0, 0.0]
        lo = [1e9] * 3
        hi = [-1e9] * 3
        for size, off in parts:
            shape = BulletBoxShape(Vec3(size[0] * 0.5, size[1] * 0.5, size[2] * 0.5))
            shape.setMargin(_margin_for(min(size)))
            node.addShape(shape, TransformState.makePos(
                Point3(off[0] - com[0], off[1] - com[1], off[2] - com[2])))
            for i in range(3):
                lo[i] = min(lo[i], off[i] - size[i] * 0.5)
                hi[i] = max(hi[i], off[i] + size[i] * 0.5)
        q = Quat()
        q.setHpr(Vec3(*hpr))
        shift = q.xform(Vec3(*com))
        pos = (pos[0] + shift[0], pos[1] + shift[1], pos[2] + shift[2])
        node.setMass(0.0 if (static or anchored) else mass)
        node.setFriction(math.sqrt(max(m.friction, 1e-3)))
        node.setRestitution(m.restitution)
        if not static and not anchored:
            node.setDeactivationEnabled(True)
            node.setLinearSleepThreshold(SLEEP_LIN)
            node.setAngularSleepThreshold(SLEEP_ANG)
            node.setDeactivationTime(SLEEP_TIME)
        root = self._static_root if (static or anchored) else self._root
        np_ = root.attachNewNode(node)
        np_.setPos(*pos)
        np_.setHpr(*hpr)
        self.world.attachRigidBody(node)
        size = tuple(hi[i] - lo[i] for i in range(3))
        info = BodyInfo(node=node, path=np_, material=material, mass=mass,
                        size=size, floor=floor, kind=kind,
                        anchored=static or anchored, fragile=m.fragile,
                        home=(tuple(pos), tuple(hpr)), label=label)
        info.com_offset = tuple(com)      # subtract from part offsets for visuals
        info.local_lo = tuple(lo[i] - com[i] for i in range(3))   # bbox about the CoM
        info.local_hi = tuple(hi[i] - com[i] for i in range(3))
        return self._register(info)

    def add_cylinder(self, radius, height, pos, material="glass", mass=None,
                     hpr=(0, 0, 0), floor=0, kind="prop", static=False,
                     anchored=False, label="", parent=None) -> BodyInfo:
        m = mat.get(material)
        if mass is None:
            mass = 0.0 if static else m.density * math.pi * radius ** 2 * height
        shape = BulletCylinderShape(radius, height, ZUp)
        shape.setMargin(_margin_for(min(radius * 2.0, height)))
        node = BulletRigidBodyNode(label or f"{material}_cyl")
        node.addShape(shape)
        node.setMass(0.0 if (static or anchored) else mass)
        node.setFriction(math.sqrt(max(m.friction, 1e-3)))
        node.setRestitution(m.restitution)
        if not static and not anchored:
            node.setDeactivationEnabled(True)
            node.setLinearSleepThreshold(SLEEP_LIN)
            node.setAngularSleepThreshold(SLEEP_ANG)
            node.setDeactivationTime(SLEEP_TIME)
            node.setCcdMotionThreshold(radius)
            node.setCcdSweptSphereRadius(radius * 0.8)
        root = self._static_root if (static or anchored) else self._root
        np_ = (parent or root).attachNewNode(node)
        np_.setPos(*pos)
        np_.setHpr(*hpr)
        self.world.attachRigidBody(node)
        info = BodyInfo(node=node, path=np_, material=material, mass=mass,
                        size=(radius * 2, radius * 2, height), floor=floor,
                        kind=kind, anchored=static or anchored,
                        fragile=m.fragile,
                        home=(tuple(pos), tuple(hpr)), label=label)
        return self._register(info)

    def add_sphere(self, radius, pos, material="produce", mass=None, floor=0,
                   kind="prop", label="", parent=None) -> BodyInfo:
        m = mat.get(material)
        if mass is None:
            mass = m.density * 4.0 / 3.0 * math.pi * radius ** 3
        node = BulletRigidBodyNode(label or f"{material}_sph")
        _sph = BulletSphereShape(radius)
        _sph.setMargin(_margin_for(radius * 2.0))
        node.addShape(_sph)
        node.setMass(mass)
        node.setFriction(math.sqrt(max(m.friction, 1e-3)))
        node.setRestitution(m.restitution)
        node.setDeactivationEnabled(True)
        node.setLinearSleepThreshold(SLEEP_LIN)
        node.setAngularSleepThreshold(SLEEP_ANG)
        node.setDeactivationTime(SLEEP_TIME)
        node.setCcdMotionThreshold(radius)
        node.setCcdSweptSphereRadius(radius * 0.8)
        np_ = (parent or self._root).attachNewNode(node)
        np_.setPos(*pos)
        self.world.attachRigidBody(node)
        info = BodyInfo(node=node, path=np_, material=material, mass=mass,
                        size=(radius * 2,) * 3, floor=floor, kind=kind,
                        anchored=False, fragile=m.fragile,
                        home=(tuple(pos), (0.0, 0.0, 0.0)), label=label)
        return self._register(info)

    def add_ground_plane(self, z=0.0, material="screed") -> BodyInfo:
        m = mat.get(material)
        node = BulletRigidBodyNode("ground")
        node.addShape(BulletPlaneShape(Vec3(0, 0, 1), z))
        node.setMass(0.0)
        node.setFriction(math.sqrt(m.friction))
        node.setRestitution(m.restitution)
        np_ = self._static_root.attachNewNode(node)
        self.world.attachRigidBody(node)
        info = BodyInfo(node=node, path=np_, material=material, mass=0.0,
                        size=(0.0, 0.0, 0.0), floor=0, kind="ground",
                        anchored=True)
        return self._register(info)

    def add_driven_box(self, size, pos, material="steel", hpr=(0, 0, 0),
                       floor=0, kind="frame", label="",
                       drift_gain: float = 1.0, parent=None,
                       mass: float | None = None) -> BodyInfo:
        """A body whose position comes from a structural solution.

        Rack uprights, shelf beams, a driven floor slab. It is a *static*
        body moved by setting its transform -- deliberately not a Bullet
        kinematic body, because anything resting on a kinematic body is never
        allowed to sleep, which with a thousand palletised cartons is the
        difference between 16 fps and 60. Its contents are woken explicitly
        whenever it actually moves (see drive_kinematic).
        """
        info = self.add_box(size, pos, material, hpr=hpr, floor=floor,
                            kind=kind, static=True, label=label, parent=parent)
        info.driven = True
        if mass is not None:
            info.break_mass = float(mass)         # what it weighs once released
        info.path.reparentTo(self._root)          # it moves: batch it with movers
        info.driven_home = (tuple(pos), tuple(hpr), drift_gain)
        self.kinematic.append((info, info.driven_home))
        return info



    add_kinematic_box = add_driven_box

    def add_breakable(self, size, pos, material="concrete", mass=None,
                      hpr=(0, 0, 0), floor=0, kind="structure", label="",
                      floor_after: int = 0) -> BodyInfo:
        """A structural element that stands until the structure says otherwise.

        Built static, so it costs nothing and holds everything up. When the
        structural solver decides its storey has lost its lateral system, or
        the floor acceleration exceeds what it can take out-of-plane, it is
        *released*: given its real mass and handed to the physics engine,
        which decides what happens next. Nothing about the fall is scripted.
        """
        sx, sy, sz = size
        m = mat.get(material)
        if mass is None:
            mass = m.density * sx * sy * sz
        info = self.add_box(size, pos, material, hpr=hpr, floor=floor,
                            kind=kind, static=True, label=label)
        info.breakable = True
        info.break_mass = float(mass)
        info.path.reparentTo(self._root)          # it may move: batch with movers
        self.breakable.append(info)
        return info

    def release(self, info: BodyInfo, floor: int | None = None,
                crush: bool = False) -> None:
        """Turn a breakable (or driven) element into a free rigid body.

        With `crush` the element is instead removed from the world -- a
        storey whose columns have gone does not fall as a block, it loses
        its height, and what stood on it comes down through it.
        """
        if info.released or not (info.breakable or info.driven):
            return
        info.released = True
        info.anchored = False
        self.n_released += 1
        node = info.node
        if info.driven:
            self.kinematic = [(i, h) for (i, h) in self.kinematic if i is not info]
            m = mat.get(info.material)
            if info.break_mass <= 0.0:
                info.break_mass = m.density * info.size[0] * info.size[1] * info.size[2]
        if crush:
            info.crushed = True
            self.world.removeRigidBody(node)
            self._park(info)
            return
        # Bullet sorts bodies into static and non-static when they are ADDED
        # to the world; a body added static is never integrated no matter
        # what its mass is set to afterwards. So it goes out and comes back.
        self.world.removeRigidBody(node)
        node.setMass(info.break_mass)
        node.setKinematic(False)
        node.setDeactivationEnabled(True)
        node.setLinearSleepThreshold(SLEEP_LIN)
        node.setAngularSleepThreshold(SLEEP_ANG)
        node.setDeactivationTime(SLEEP_TIME)
        r = 0.5 * min(info.size)
        node.setCcdMotionThreshold(r)
        node.setCcdSweptSphereRadius(r * 0.8)
        self.world.attachRigidBody(node)
        node.setActive(True)
        info.mass = info.break_mass
        if floor is not None:
            info.floor = floor
        self.dynamic.append(info)
        self.by_floor.setdefault(info.floor, []).append(info)
        m = mat.get(info.material)
        info.wake_threshold = min(m.friction, max(info.slenderness, 0.02)) * K.G0 * 0.4
        self._thresh_dirty = True

    def crush(self, info: BodyInfo) -> None:
        """Take an already-released element out of the world: a storey that
        has been landed on by everything above it."""
        if info.crushed:
            return
        if not info.released:
            self.release(info, crush=True)
            return
        info.crushed = True
        self.world.removeRigidBody(info.node)
        self._park(info)
        if info in self.dynamic:
            self.dynamic.remove(info)
        for lst in self.by_floor.values():
            if info in lst:
                lst.remove(info)
        self._thresh_dirty = True

    PARK_Z = -3000.0

    def _park(self, info: BodyInfo) -> None:
        """Put a crushed element out of sight. Its geometry lives in the
        RigidBodyCombiner, which bakes the vertices at collect time and only
        follows the node's TRANSFORM afterwards -- hide() does nothing to
        it -- so out of sight means a long way underground."""
        info.path.setPos(0.0, 0.0, self.PARK_Z)
        info.node.setLinearVelocity(Vec3(0, 0, 0))
        info.node.setAngularVelocity(Vec3(0, 0, 0))

    @staticmethod
    def _sleep_static(info: BodyInfo) -> None:
        """Put a static body into the ISLAND_SLEEPING state.

        Bullet only runs the narrowphase for a pair when one of the two is
        active, and a static body added to the world is meant to be asleep.
        Every body here is built with deactivation disabled, which for a
        static body means permanently *active*: every wall touching a floor,
        every ceiling tee crossing another, every shelf against a sleeping
        item, was being collided afresh each substep for nothing. Putting
        the statics to sleep removed every static-static manifold and took
        the supermarket's at-rest step from 64 to 44 ms. Contacts with
        active bodies are unaffected -- the active side is what triggers
        them -- and release() wakes an element when it is freed.
        """
        node = info.node
        if node.isStatic() and not node.isKinematic():
            node.setActive(False, True)

    def _refreeze(self, info: BodyInfo) -> None:
        """Undo release(): back to a static element at its home pose."""
        if not info.released:
            return
        node = info.node
        if info.crushed:
            info.crushed = False
        else:
            self.world.removeRigidBody(node)
        node.setMass(0.0)
        node.setLinearVelocity(Vec3(0, 0, 0))
        node.setAngularVelocity(Vec3(0, 0, 0))
        self.world.attachRigidBody(node)          # back in as a static body
        self._sleep_static(info)
        info.released = False
        self.n_released = max(0, self.n_released - 1)
        info.anchored = True
        info.mass = 0.0
        if info in self.dynamic:
            self.dynamic.remove(info)
        for lst in self.by_floor.values():
            if info in lst:
                lst.remove(info)
        info.floor = info.home_floor
        if info.driven:
            self.kinematic.append((info, info.driven_home))
        self._thresh_dirty = True

    # -- shaking -----------------------------------------------------------
    def apply_floor_acceleration(self, accel_by_floor: dict[int, tuple]) -> None:
        """Set each body's effective gravity from its floor's acceleration.

        Also wakes any sleeping body whose own sliding/tipping threshold the
        shaking has just crossed, so nothing sleeps through an arrival.
        """
        if self._thresh_dirty:
            self._rebuild_threshold_index()
        for floor, bodies in self.by_floor.items():
            ax, ay, az = accel_by_floor.get(floor, accel_by_floor.get(0, (0, 0, 0)))
            g = Vec3(-ax, -ay, -K.G0 - az)

            # An upward-negative vertical excursion unweights everything in the
            # room, and friction is proportional to the normal force. During
            # those moments a jar slides at a horizontal acceleration well
            # below mu*g. Real records have vertical peaks around 60% of the
            # horizontal, so this is not a small correction -- and ignoring it
            # let objects sleep straight through the instants when they would
            # actually have moved.
            unweight = max(1.0 + az / K.G0, 0.25)
            demand = math.hypot(ax, ay) / unweight

            prev = self._last_mag.get(floor, 0.0)
            self._last_mag[floor] = demand
            rising = demand >= prev
            for b in bodies:
                if b.anchored:
                    continue
                b.node.setGravity(g)
                if self.always_awake:
                    continue
                if rising and demand > b.wake_threshold and not b.node.isActive():
                    b.node.setActive(True)

    # A driven group's load is shifted once its support has moved this far
    # from where the load was last shifted to. Below it the load sits at
    # most this far off its beam, which is invisible, and stays asleep.
    CARRY_STEP = 0.004

    def drive_kinematic(self, disp_by_floor: dict[int, tuple]) -> None:
        """Move each driven group -- support and load together -- to its
        structurally computed displacement.

        The beams go to their computed positions every frame. Everything
        standing on them is shifted by the same displacement whenever it
        has accumulated CARRY_STEP, so the load's own dynamics run in the
        level's frame under the level's absolute acceleration
        (apply_floor_acceleration): x_world = x_relative + d(t), the frame
        transformation and nothing more. The earlier scheme moved only the
        beams: a static body moved by transform has no surface velocity, so
        Bullet's friction pinned the load to the *world* while the beam
        swayed under it, and a pallet that should have ridden a 2.5 cm sway
        sat still and appeared to slide 3 cm the other way (measured, M5.6
        at 25 km). Bullet kinematic bodies were measured too: their load
        never sleeps and, through Panda, did not ride them either.

        Two facts about Panda's Bullet binding shape the shift. Panda hands
        a body's transform to Bullet only when that node's own transform is
        set, so a common parent node cannot do this (measured: an awake
        stack under a parent moved 2.5 cm slipped 3 cm). And the transform
        Panda reads back lags the body's true state by one substep -- Bullet
        interpolates motion states with latency -- so setting a moving
        body's transform pulls it back by v*h and a sliding box covered
        half its distance. Each awake body is therefore set to p + d + v*h
        and rotated on by w*h, which reproduces the untouched trajectory to
        a micrometre. A sleeping body is woken by setPos and is put straight
        back to sleep: it is carried, Bullet learns its new position, and
        it wakes only when the level's acceleration crosses its own sliding
        or tipping threshold, like everything else.
        """
        h = self.substep
        # only groups that actually have a driven support (rack levels);
        # a building storey's contents live in the storey's own frame and
        # its displacement is never applied to them
        driven_floors = {info.floor for info, _ in self.kinematic}
        for floor, d in disp_by_floor.items():
            if floor not in driven_floors:
                continue
            last = self._driven_last.get(floor)
            if last is None:
                self._driven_last[floor] = last = (0.0, 0.0, 0.0)
            dx, dy, dz = d[0] - last[0], d[1] - last[1], d[2] - last[2]
            if (abs(dx) < self.CARRY_STEP and abs(dy) < self.CARRY_STEP
                    and abs(dz) < self.CARRY_STEP):
                continue
            self._driven_last[floor] = d
            shift = Vec3(dx, dy, dz)
            for b in self.by_floor.get(floor, ()):
                if b.anchored or b.crushed:
                    continue
                node, path = b.node, b.path
                if node.isActive():
                    v = node.getLinearVelocity()
                    path.setPos(path.getPos() + shift + v * h)
                    w = node.getAngularVelocity()
                    wl = w.length()
                    if wl > 0.02:
                        inc = Quat()
                        inc.setFromAxisAngleRad(wl * h, w / wl)
                        path.setQuat(path.getQuat() * inc)
                else:
                    path.setPos(path.getPos() + shift)
                    node.setActive(False, True)
        for info, (pos, hpr, gain) in self.kinematic:
            d = disp_by_floor.get(info.floor)
            if d is None:
                continue
            info.path.setPos(pos[0] + d[0] * gain, pos[1] + d[1] * gain,
                             pos[2] + d[2] * gain)

    def step(self, dt: float) -> None:
        t0 = time.perf_counter()
        self.world.doPhysics(dt, self.max_substeps, self.substep)
        # Smoothed so the readout is legible rather than flickering.
        ms = (time.perf_counter() - t0) * 1000.0
        self.last_ms = ms if self.last_ms <= 0 else 0.9 * self.last_ms + 0.1 * ms

    # -- queries -----------------------------------------------------------
    def finalise(self) -> None:
        """Called once the scene is fully built."""
        for b in self.dynamic:
            b.node.setDeactivationEnabled(not self.always_awake)
        for b in self.bodies:
            self._sleep_static(b)
        self._rebuild_threshold_index()
        self.n_breakable = len(self.breakable)
        # Batch the geometry. After this, adding more bodies is not supported.
        if self.stage is not None:
            self.stage("Batching the moving geometry", 0.0)
        self._rbc.collect()
        if self.stage is not None:
            self.stage("Flattening the static geometry", 0.55)
        self.static_visual_root.flattenStrong()
        if self.stage is not None:
            self.stage("Scene ready", 1.0)
        self.progress = None
        self.stage = None

    def draw_call_estimate(self) -> int:
        """Number of Geoms that reach the renderer -- a proxy for draw calls.

        The combiner's original children are still in the graph but are not
        drawn; what is drawn is its internal scene, so count that instead.
        """
        cached = getattr(self, "_draw_calls", None)
        if cached is not None:
            return cached

        def geoms(np_):
            n = np_.node().getNumGeoms() if np_.node().isGeomNode() else 0
            for g in np_.findAllMatches("**/+GeomNode"):
                n += g.node().getNumGeoms()
            return n
        total = geoms(self._rbc.getInternalScene())
        for child in self.render.getChildren():
            if child == self._root:
                continue
            total += geoms(child)
        self._draw_calls = total
        return total

    def wake_all(self) -> None:
        for b in self.dynamic:
            b.node.setActive(True)

    def reset(self) -> None:
        for b in self.bodies:
            if b.released:
                self._refreeze(b)
            if b.floor != b.home_floor and b.mass > 0.0:
                self.set_floor(b, b.home_floor)
        self._driven_last = {}
        for b in self.bodies:
            if not b.home:
                continue
            pos, hpr = b.home
            b.path.setPos(*pos)
            b.path.setHpr(*hpr)
            b.node.setLinearVelocity(Vec3(0, 0, 0))
            b.node.setAngularVelocity(Vec3(0, 0, 0))
            b.node.setGravity(Vec3(0, 0, -K.G0))
            b.broken = False
            b.path.show()
        self.world.setGravity(Vec3(0, 0, -K.G0))

    # -- snapshots ---------------------------------------------------------
    def snapshot(self) -> WorldSnapshot:
        """Capture every body. A sleeping body has zero velocity by Bullet's
        definition, so only its pose is read back -- most of a scene is
        asleep most of the time, and the read-back is what a snapshot costs."""
        bodies = self.bodies
        n = len(bodies)
        state = np.zeros((n, 13), dtype=np.float32)
        active = np.empty(n, dtype=bool)
        floor = np.empty(n, dtype=np.int32)
        released = np.empty(n, dtype=bool)
        crushed = np.empty(n, dtype=bool)
        softened = np.empty(n, dtype=bool)
        friction = np.empty(n, dtype=np.float32)
        for i, b in enumerate(bodies):
            node = b.node
            awake = node.isActive()
            active[i] = awake
            floor[i] = b.floor
            released[i] = b.released
            crushed[i] = b.crushed
            softened[i] = b.softened
            friction[i] = node.getFriction() if b.peak_mu > 0.0 else 0.0
            p = b.path.getPos()
            q = b.path.getQuat()
            state[i, 0] = p[0]; state[i, 1] = p[1]; state[i, 2] = p[2]
            state[i, 3] = q[0]; state[i, 4] = q[1]; state[i, 5] = q[2]; state[i, 6] = q[3]
            if awake and b.mass > 0.0:
                v = node.getLinearVelocity()
                w = node.getAngularVelocity()
                state[i, 7] = v[0]; state[i, 8] = v[1]; state[i, 9] = v[2]
                state[i, 10] = w[0]; state[i, 11] = w[1]; state[i, 12] = w[2]
        return WorldSnapshot(state, active, floor, released, crushed, softened,
                             friction, dict(self._driven_last), dict(self._last_mag),
                             self.n_released)

    def restore(self, snap: WorldSnapshot) -> None:
        """Put every body back exactly as the snapshot has it."""
        bodies = self.bodies
        if snap.state.shape[0] != len(bodies):
            raise ValueError("snapshot is from another scene")
        # Elements whose released / crushed state differs go out and come
        # back the way the collapse model does it, so Bullet holds them the
        # same way; the rest are left in place and only re-posed.
        for i, b in enumerate(bodies):
            want_rel, want_crush = bool(snap.released[i]), bool(snap.crushed[i])
            if b.released == want_rel and b.crushed == want_crush:
                continue
            if b.released:
                self._refreeze(b)
            if want_rel:
                self.release(b, floor=int(snap.floor[i]), crush=want_crush)
        st = snap.state
        for i, b in enumerate(bodies):
            if b.crushed:
                continue
            node = b.node
            row = st[i]
            b.path.setPosQuat(Point3(float(row[0]), float(row[1]), float(row[2])),
                              Quat(float(row[3]), float(row[4]), float(row[5]), float(row[6])))
            if b.mass > 0.0:
                node.setLinearVelocity(Vec3(float(row[7]), float(row[8]), float(row[9])))
                node.setAngularVelocity(Vec3(float(row[10]), float(row[11]), float(row[12])))
                if snap.active[i]:
                    node.setActive(True, True)
                else:
                    node.setActive(False, True)
                if int(snap.floor[i]) != b.floor:
                    self.set_floor(b, int(snap.floor[i]))
            elif not snap.active[i]:
                # setting a transform wakes a static body too
                self._sleep_static(b)
            if b.peak_mu > 0.0:
                b.softened = bool(snap.softened[i])
                node.setFriction(float(snap.friction[i]))
        self._driven_last = dict(snap.driven_last)
        self._last_mag = dict(snap.last_mag)
        self.n_released = int(snap.n_released)
        self._thresh_dirty = True
        self._counts_frame = -1000

    def counts(self, frame: int | None = None) -> tuple[int, int, int]:
        """(shifted, toppled, fallen), recomputed at most every 12 frames.

        Walking every body's transform from Python costs a couple of
        milliseconds per pass, which is fine at 5 Hz and wasteful at 60.
        """
        if frame is None or frame - self._counts_frame >= 12:
            self._counts_cache = (self.displaced_count(), self.toppled_count(),
                                  self.fallen_count())
            self._counts_frame = frame if frame is not None else -1
        return self._counts_cache

    def displaced_count(self, tol: float = 0.02) -> int:
        """Objects that have moved more than `tol` from where they started."""
        n = 0
        for b in self.dynamic:
            if not b.home:
                continue
            p = b.path.getPos()
            hx, hy, hz = b.home[0]
            if (p[0] - hx) ** 2 + (p[1] - hy) ** 2 + (p[2] - hz) ** 2 > tol * tol:
                n += 1
        return n

    def toppled_count(self, degrees: float = 25.0) -> int:
        """Objects that have rotated past `degrees` -- rocked over, not slid.

        Worth counting separately: a body can rock hard without its centre
        moving far, and overturning needs enough energy rather than merely
        enough peak acceleration (Housner 1963), so the two counts diverge.
        """
        n = 0
        for b in self.dynamic:
            if not b.home:
                continue
            h = b.home[1]
            p = b.path.getHpr()
            if max(abs(p[1] - h[1]), abs(p[2] - h[2])) > degrees:
                n += 1
        return n

    def set_floor(self, info: BodyInfo, floor: int) -> None:
        """Move a body to another floor group (e.g. rack contents to the
        ground once the rack they stood on has gone)."""
        if info.floor == floor:
            return
        lst = self.by_floor.get(info.floor)
        if lst is not None and info in lst:
            lst.remove(info)
        info.floor = floor
        self.by_floor.setdefault(floor, []).append(info)
        self._thresh_dirty = True

    # Nothing in a room outruns this. A slab hitting a rubble pile at 38 m/s
    # (level 25, free fall) launches a rigid glass caught between it and the
    # floor at 100 m/s and more, 190 m from the tower; the real glass is
    # powder. Bodies over the limit are slowed to it. 30 m/s is the order
    # of the collapse front itself and of what debris leaves a pancake
    # collapse at; free fall from a shelf, a table or a roof never reaches
    # it, so nothing outside a collapse is touched.
    SPEED_LIMIT = 30.0

    def limit_speeds(self) -> None:
        """Called each frame while structure is released; checks every
        third frame, awake bodies only (a sleeping body is still)."""
        if not self.n_released:
            return
        self._limit_frame += 1
        if self._limit_frame % 3:
            return
        vmax = self.SPEED_LIMIT
        v2max = vmax * vmax
        for b in self.dynamic:
            node = b.node
            if not node.isActive():
                continue
            v = node.getLinearVelocity()
            v2 = v.lengthSquared()
            if v2 > v2max:
                node.setLinearVelocity(v * (vmax / math.sqrt(v2)))
                w = node.getAngularVelocity()
                if w.lengthSquared() > 400.0:
                    node.setAngularVelocity(w * (20.0 / w.length()))

    def released_count(self) -> int:
        """Structural elements the solver has let go of."""
        return sum(1 for b in self.bodies if b.released)

    def crushed_count(self) -> int:
        return sum(1 for b in self.bodies if b.crushed)

    def fallen_count(self, drop: float = 0.30) -> int:
        """Objects that have dropped -- off a shelf, out of a rack."""
        n = 0
        for b in self.dynamic:
            if not b.home:
                continue
            if b.home[0][2] - b.path.getPos()[2] > drop:
                n += 1
        return n

    def check_breakage(self, min_speed: float | None = None) -> list[BodyInfo]:
        """Fragile objects that just hit something hard enough to shatter."""
        broken = []
        for b in self.dynamic:
            if not b.fragile or b.broken:
                continue
            m = mat.get(b.material)
            limit = min_speed if min_speed is not None else m.break_speed
            v = b.node.getLinearVelocity()
            if v.length() < limit:
                continue
            result = self.world.contactTest(b.node)
            if result.getNumContacts() > 0:
                b.broken = True
                broken.append(b)
        return broken

    def destroy(self) -> None:
        """Take every body out of the Bullet world and every node this world
        put into the scene graph. Without this a scene switch leaves the
        old scene's moving bodies -- rack frames, pallets, cartons -- drawn
        in the new one, and their draw calls with them."""
        for c in list(self.world.getConstraints()):
            self.world.removeConstraint(c)
        for b in self.bodies:
            if not b.crushed:
                try:
                    self.world.removeRigidBody(b.node)
                except Exception:
                    pass
        for np_ in (self._root, self._static_root, self.static_visual_root,
                    self.debug_np):
            if np_ is not None:
                np_.removeNode()
        self.bodies = []
        self.dynamic = []
        self.by_floor = {}
        self.kinematic = []
        self.breakable = []
        self.n_released = 0

    def enable_debug(self) -> None:
        node = BulletDebugNode("debug")
        node.showWireframe(True)
        node.showConstraints(True)
        self.debug_np = self.render.attachNewNode(node)
        self.world.setDebugNode(node)
        self.debug_np.show()


# ---------------------------------------------------------------------------
class Shaker:
    """Drives a PhysicsWorld from a ground motion and structural responses.

    Bodies are grouped by a `floor` id. Each group can be driven by a
    different structure: the room's own floor of the building, or -- in the
    warehouse -- the level of a pallet rack, which is a far more flexible
    structure than the shed around it and moves quite differently. Anything
    standing on that level then feels the rack's motion, not the floor's,
    which is the whole point.
    """

    def __init__(self, world: PhysicsWorld):
        self.world = world
        self.motion = None
        self.responses: dict[int, tuple] = {}   # floor id -> (response, story)
        self.time = 0.0
        self.playing = False        # the record clock is advancing
        self.paused = False         # frozen by the user: nothing moves at all
        self.speed = 1.0
        self.gain = 1.0
        self.vertical_enabled = True
        self._acc_cache: dict[int, tuple] = {}

    # -- setup -------------------------------------------------------------
    def load(self, motion, response=None, story_of_floor=None) -> None:
        self.motion = motion
        self.responses = {}
        if response is not None and story_of_floor:
            for floor, story in story_of_floor.items():
                self.responses[floor] = (response, story)
        self.time = 0.0
        self.playing = False
        self.paused = False

    def attach(self, floors: dict[int, int], response) -> None:
        """Drive these floor groups from an additional structure."""
        for floor, story in floors.items():
            self.responses[floor] = (response, story)

    @property
    def duration(self) -> float:
        return self.motion.duration if self.motion else 0.0

    @property
    def response(self):
        """The main building's response, for the readouts."""
        for floor in sorted(self.responses):
            return self.responses[floor][0]
        return None

    # -- per-frame ---------------------------------------------------------
    def accel_at(self, t: float) -> dict[int, tuple]:
        """Acceleration for each floor group, m/s^2, in world axes."""
        if self.motion is None:
            return {0: (0.0, 0.0, 0.0)}
        e, n, z = self.motion.sample(t)
        if not self.vertical_enabled:
            z = 0.0
        g = self.gain
        ground = (e * g, n * g, z * g)

        out = {}
        for floor in self.world.by_floor:
            entry = self.responses.get(floor)
            if entry is None:
                out[floor] = ground
                continue
            response, story = entry
            i = int(t / response.dt)
            if not (0 <= i < response.floor_acc.shape[1]):
                out[floor] = ground
                continue
            s = min(story, response.stories - 1)
            # Both horizontal components are solved independently, so each
            # carries its own floor acceleration rather than being scaled by
            # the other's ratio.
            ae = float(response.floor_acc[s, i]) * g
            an = (float(response.floor_acc_y[s, i]) * g
                  if getattr(response, "floor_acc_y", None) is not None
                  else n * g)
            out[floor] = (ae, an, z * g)
        if not out:
            out[0] = ground
        return out

    def disp_at(self, t: float) -> dict[int, tuple]:
        """Displacement of each driven group relative to the ground, metres."""
        out = {}
        for floor, (response, story) in self.responses.items():
            i = int(t / response.dt)
            if not (0 <= i < response.floor_disp.shape[1]):
                continue
            s = min(story, response.stories - 1)
            dx = float(response.floor_disp[s, i]) * self.gain
            dy = (float(response.floor_disp_y[s, i]) * self.gain
                  if getattr(response, "floor_disp_y", None) is not None else 0.0)
            out[floor] = (dx, dy, 0.0)
        return out

    def update(self, dt: float) -> None:
        # A pause freezes the world: the clock, the shaking and the physics
        # all stop together, and the picture holds until PLAY. (With the
        # record stopped but the physics still stepping, everything in the
        # air kept falling under the acceleration of the paused instant --
        # that was not a pause, it was a different earthquake.)
        if self.paused:
            return
        if self.playing and self.motion is not None:
            self.time += dt * self.speed
            if self.time > self.duration:
                self.time = self.duration
                self.playing = False
        self._acc_cache = self.accel_at(self.time)
        self.world.apply_floor_acceleration(self._acc_cache)
        d = self.disp_at(self.time)
        if d:
            self.world.drive_kinematic(d)
        self.world.limit_speeds()
        self.world.step(dt)

    def current_accel_g(self) -> tuple:
        a = self._acc_cache.get(0) or next(iter(self._acc_cache.values()),
                                           (0.0, 0.0, 0.0))
        return tuple(x / K.G0 for x in a)

    def start(self) -> None:
        self.time = 0.0
        self.playing = True
        self.paused = False
        self.world.wake_all()
