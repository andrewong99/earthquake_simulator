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

"""Scene interface.

A scene bundles the ground it stands on, the structure that filters the
shaking, the geometry you look at, the camera positions worth looking from,
and -- now -- the structural elements that are allowed to fail and the
surfaces that crack.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from panda3d.core import NodePath

from ..seismo import constants as K
from ..seismo.site import Site
from ..structure.collapse import StructuralModel
from ..structure.mdof import DAMAGE_DRIFT, Building
from .cracks import CrackLayer, Surface
from .props import Builder


@dataclass
class Viewpoint:
    key: str
    name: str
    pos: tuple[float, float, float]
    hpr: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mode: str = "look"        # look | walk | orbit | top | fly
    fov: float = 60.0
    note: str = ""


@dataclass
class SceneSpec:
    key: str
    name: str
    blurb: str
    site: Site
    short: str = ""                 # for the scene chooser; name if empty
    building: Building | None = None
    default_distance_km: float = 20.0
    default_quake: str = "strike_slip"
    default_mw: float = 6.5
    default_depth_km: float | None = None
    sky_time: float = 14.0
    outdoor: bool = False
    viewpoints: list[Viewpoint] = field(default_factory=list)
    story_of_floor: dict[int, int] = field(default_factory=dict)
    observer_story: int = 0        # -1: on the ground floor (ground motion)
    note: str = ""
    # (magnitude, distance km, quake type) the scene author found gives a
    # collapse -- shown in the interface as a suggestion
    collapse_demo: tuple | None = None


class Scene:
    """Base class for all scenes."""

    spec: SceneSpec

    def __init__(self, phys, render: NodePath, seed: int = 4,
                 detail: float = 1.0):
        self.phys = phys
        self.detail = detail
        self.seed = seed
        self.root = render.attachNewNode(f"scene_{self.spec.key}")
        self.builder = Builder(phys, self.root, seed, detail)
        self.structure = StructuralModel(phys, seed)
        self.cracks = CrackLayer(render, seed)
        self.extras = {}
        self._drift_max: dict[tuple, float] = {}
        self._damage_level: dict[tuple, int] = {}
        self._track_t = -1.0
        self._systems: dict[int, str] = {}      # floor group -> structural system
        self.observer_body = None               # a body the camera should ride
        self.build(self.builder)
        phys.finalise()

    def build(self, b: Builder) -> None:      # pragma: no cover - overridden
        raise NotImplementedError

    # -- hooks ---------------------------------------------------------------
    def attach_structures(self, shaker, motion, building_response,
                          progress=None) -> None:
        """Scenes with secondary structures (racking) attach them here.
        `progress(fraction, label)`, if given, feeds the loading screen."""

    def register_system(self, group: int, system: str) -> None:
        """Which structural system's damage thresholds a floor group uses."""
        self._systems[group] = system

    def crack_panels(self, panels, normal: str, story: int, group: int,
                     weight: float = 0.15, thickness: float = 0.15) -> None:
        """Register a crack surface on the face of each panel body, so the
        cracks fall with the panel."""
        for body in panels:
            x, y, z = body.home[0]
            sx, sy, sz = body.size
            if normal in ("+x", "-x"):
                c = (x + (thickness / 2 if normal == "+x" else -thickness / 2), y, z)
                size = (sy, sz)
            elif normal in ("+y", "-y"):
                c = (x, y + (thickness / 2 if normal == "+y" else -thickness / 2), z)
                size = (sx, sz)
            else:
                c = (x, y, z + (sz / 2 if normal == "+z" else -sz / 2))
                size = (sx, sy)
            self.cracks.register(Surface(c, size, normal, story, group, weight, body))

    def update(self, t: float, shaker) -> None:
        self.builder.update_pendants()
        if shaker.motion is None:
            return
        if self.structure.update(t, shaker):
            self.cracks.on_release()
        self._track_damage(t, shaker)

    def _track_damage(self, t: float, shaker) -> None:
        """Running peak drift per storey -> damage level -> cracks."""
        t0, self._track_t = self._track_t, t
        if t <= t0:
            return
        for group, (resp, story) in list(shaker.responses.items()):
            sl = StructuralModel._slice(resp.dt, resp.drift.shape[1], t0, t)
            if sl is None:
                continue
            sys_key = self._systems.get(group) or (
                self.spec.building.system if self.spec.building else "concrete_moment_frame")
            thr = DAMAGE_DRIFT.get(sys_key, (0.003, 0.006, 0.016, 0.04))
            for s in range(resp.stories):
                d = float(np.max(np.abs(resp.drift[s, sl[0]:sl[1]])))
                if resp.drift_y is not None:
                    d = max(d, float(np.max(np.abs(resp.drift_y[s, sl[0]:sl[1]]))))
                d *= shaker.gain
                key = (group, s)
                if d > self._drift_max.get(key, 0.0):
                    self._drift_max[key] = d
                    level = sum(1 for th in thr if d >= th)
                    if level > self._damage_level.get(key, 0):
                        self._damage_level[key] = level
                        self.cracks.set_damage(group, s, level)

    def on_reset(self) -> None:
        self.structure.reset()
        self.cracks.reset()
        self._drift_max = {}
        self._damage_level = {}
        self._track_t = -1.0

    # -- snapshots -------------------------------------------------------------
    # Scene attributes that change as the earthquake plays, beyond what the
    # bodies and the structural elements carry; a scene lists its own.
    SNAPSHOT_ATTRS: tuple = ()

    def snapshot_state(self) -> dict:
        import copy
        d = {"structure": self.structure.snapshot(),
             "cracks": (list(self.cracks.placed), dict(self.cracks.level_by_story)),
             "_drift_max": dict(self._drift_max),
             "_damage_level": dict(self._damage_level),
             "_track_t": self._track_t}
        for name in self.SNAPSHOT_ATTRS:
            d[name] = copy.deepcopy(getattr(self, name))
        return d

    def restore_state(self, d: dict) -> None:
        import copy
        self.structure.restore(d["structure"])
        placed, levels = d["cracks"]
        self.cracks.placed = list(placed)
        self.cracks.level_by_story = dict(levels)
        self.cracks._rebuild()
        self._drift_max = dict(d["_drift_max"])
        self._damage_level = dict(d["_damage_level"])
        self._track_t = d["_track_t"]
        for name in self.SNAPSHOT_ATTRS:
            setattr(self, name, copy.deepcopy(d[name]))

    def destroy(self) -> None:
        self.cracks.root.removeNode()
        self.root.removeNode()

    @property
    def viewpoints(self):
        return self.spec.viewpoints

    def object_count(self) -> int:
        return len(self.phys.dynamic)

    def damage_report(self) -> list[str]:
        out = []
        summ = self.structure.summary()
        if summ["failed"]:
            parts = ", ".join(f"{n} {tag}" for tag, n in sorted(summ["by_tag"].items()))
            out.append(f"Structural failure: {parts} "
                       f"(first at {summ['first_failure']:.1f} s)")
        if self.cracks.count():
            out.append(f"{self.cracks.count()} cracks")
        return out


SCENES: dict[str, type] = {}


def register(cls):
    SCENES[cls.spec.key] = cls
    return cls
