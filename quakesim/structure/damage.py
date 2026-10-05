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

"""Component-level damage.

A shear-building model tells you how much the frame drifts. It does not tell
you that the gable wall fell into the garden, that the suspended ceiling came
down, or that the glazing popped out -- and those are what people actually
see. Non-structural damage also dominates real earthquake losses.

Two demand parameters drive everything here:

    drift   racks the frame, so it breaks anything spanning between floors:
            glazing, partitions, cladding, stair flights
    floor acceleration
            throws anything supported by one floor: ceilings, parapets,
            out-of-plane walls, equipment, contents

Unreinforced masonry is the clearest case. Its walls are strong in their own
plane and nearly helpless across it, so they fail by acceleration long before
the building drifts enough to matter. That is why a URM parapet at 0.1 g is
already dangerous while the frame below is untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..seismo import constants as K


@dataclass
class Component:
    """A damageable non-structural element."""
    key: str
    name: str
    demand: str                 # "acc" | "drift"
    thresholds: tuple[float, float, float]   # onset, widespread, collapse
    note: str = ""


COMPONENTS = [
    Component("urm_oop", "Unreinforced wall, out-of-plane", "acc",
              (0.10, 0.22, 0.45),
              "Slender masonry spanning vertically between floors. Fails by "
              "cracking at mid-height then hinging outward."),
    Component("parapet", "Parapet / gable / chimney", "acc",
              (0.06, 0.14, 0.28),
              "Cantilevered above the roof with nothing bracing the top, so "
              "it sees the largest acceleration in the building."),
    Component("ceiling", "Suspended ceiling grid", "acc",
              (0.25, 0.45, 0.80),
              "Lay-in tiles pop out of the grid first; then the grid itself "
              "loses its perimeter clips."),
    Component("light_fixture", "Pendant / recessed lighting", "acc",
              (0.30, 0.55, 0.95), ""),
    Component("sprinkler", "Sprinkler pipework", "acc",
              (0.35, 0.60, 1.00),
              "Branch lines whip and shear off heads where they pass through "
              "the ceiling."),
    Component("glazing", "Window glazing / curtain wall", "drift",
              (0.008, 0.017, 0.030),
              "Glass is stiff and brittle; it fails when the frame racks the "
              "opening out of square."),
    Component("partition", "Interior partition wall", "drift",
              (0.004, 0.008, 0.020), ""),
    Component("cladding", "Facade cladding panels", "drift",
              (0.006, 0.013, 0.025), ""),
    Component("stair", "Stair flight connection", "drift",
              (0.010, 0.020, 0.035),
              "Stairs act as unintended braces and get torn at the landings."),
    Component("equipment", "Unanchored equipment / cabinets", "acc",
              (0.20, 0.35, 0.60), ""),
    Component("rack_conn", "Rack beam-to-column connector", "drift",
              (0.012, 0.025, 0.045), ""),
]
COMPONENT_BY_KEY = {c.key: c for c in COMPONENTS}

LEVELS = ["Intact", "Onset", "Widespread", "Failed"]


def _level(value: float, thr: tuple[float, float, float]) -> int:
    return sum(1 for t in thr if value >= t)


@dataclass
class DamageReport:
    per_component: dict[str, int] = field(default_factory=dict)
    peak_floor_acc_g: float = 0.0
    peak_drift: float = 0.0
    roof_acc_g: float = 0.0
    notes: list[str] = field(default_factory=list)

    def level(self, key: str) -> int:
        return self.per_component.get(key, 0)

    def failed(self, key: str) -> bool:
        return self.per_component.get(key, 0) >= 3

    def summary(self) -> list[str]:
        out = []
        for c in COMPONENTS:
            lv = self.per_component.get(c.key, 0)
            if lv > 0:
                out.append(f"{c.name}: {LEVELS[lv]}")
        return out


def assess(floor_acc_g: float, drift_ratio: float,
           present: list[str] | None = None,
           anchored: bool = False) -> DamageReport:
    """Damage state of every component at one floor."""
    rep = DamageReport(peak_floor_acc_g=floor_acc_g, peak_drift=drift_ratio)
    keys = present or [c.key for c in COMPONENTS]
    boost = 1.8 if anchored else 1.0        # anchoring raises the thresholds
    for key in keys:
        c = COMPONENT_BY_KEY.get(key)
        if c is None:
            continue
        thr = tuple(t * boost for t in c.thresholds)
        d = floor_acc_g if c.demand == "acc" else drift_ratio
        rep.per_component[key] = _level(d, thr)
    return rep


# ---------------------------------------------------------------------------
# Rigid-body response of individual objects
# ---------------------------------------------------------------------------
def sliding_threshold(friction: float) -> float:
    """Horizontal acceleration (in g) at which a body starts to slide."""
    return friction


def rocking_threshold(width: float, height: float) -> float:
    """Acceleration (in g) at which a rigid block starts to rock about its edge.

    A block of half-width b and half-height h tips when the overturning moment
    m*a*h exceeds the restoring moment m*g*b, i.e. a/g > b/h. Slender objects
    (a bottle, a bookcase) go over at a fraction of a g; squat ones never do.
    """
    if height <= 0.0:
        return 99.0
    return (width * 0.5) / (height * 0.5)


def topple_probability(width: float, height: float, pga_g: float,
                       pgv_cms: float) -> float:
    """Probability a free-standing block overturns.

    Rocking is notoriously sensitive: a block that starts rocking does not
    necessarily fall, because overturning needs enough *energy*, not just
    enough peak acceleration. Following the classic Housner (1963) result the
    controlling quantity combines the static tipping threshold with the
    velocity of the shaking.
    """
    a_t = rocking_threshold(width, height)
    if pga_g < a_t:
        return 0.0
    r = math.hypot(width * 0.5, height * 0.5)
    p = math.sqrt(K.G0 / max(r, 1e-3) * 0.75)       # rocking frequency parameter
    energy_ratio = (pgv_cms / 100.0) * p / (K.G0 * a_t)
    return float(min(max(1.0 - math.exp(-1.6 * max(energy_ratio - 0.35, 0.0)), 0.0), 1.0))


def behaviour(width: float, height: float, friction: float,
              pga_g: float, pgv_cms: float) -> str:
    """One-word verdict used for tooltips and the object inspector."""
    a_slide = sliding_threshold(friction)
    a_rock = rocking_threshold(width, height)
    if pga_g < min(a_slide, a_rock):
        return "stays put"
    if a_rock < a_slide:
        p = topple_probability(width, height, pga_g, pgv_cms)
        return "topples" if p > 0.5 else "rocks"
    return "slides"


def floor_acc_profile(response, story_height: float) -> np.ndarray:
    """Peak absolute acceleration at each floor, in g."""
    return np.max(np.abs(response.floor_acc), axis=1) / K.G0


def assess_building(response, systems: dict[int, list[str]] | None = None,
                    anchored: bool = False) -> dict[int, DamageReport]:
    """Damage at every floor of a building."""
    acc = np.max(np.abs(response.floor_acc), axis=1) / K.G0
    drift = response.peak_drift
    out = {}
    for i in range(response.stories):
        present = (systems or {}).get(i)
        out[i] = assess(float(acc[i]), float(drift[i]), present, anchored)
    return out
