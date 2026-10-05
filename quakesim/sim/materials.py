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

"""Material properties.

Whether a bottle stays on the shelf is decided by two numbers: the friction
coefficient between it and the shelf, and the ratio of its width to its
height. Both are real, measurable quantities, so they are tabulated here
rather than tuned by eye.

Friction values are static coefficients for the named pair of surfaces, taken
from the usual engineering ranges. Restitution ("bounciness") is the fraction
of approach velocity returned in a collision.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    key: str
    name: str
    density: float          # kg/m^3
    friction: float         # static coefficient against a typical shelf
    restitution: float
    colour: tuple[float, float, float]
    roughness: float = 0.6
    metallic: float = 0.0
    fragile: bool = False   # breaks when impact speed exceeds break_speed
    break_speed: float = 3.0    # m/s


_M = [
    Material("concrete", "Concrete", 2400, 0.65, 0.12, (0.62, 0.61, 0.58), 0.92),
    Material("screed", "Floor screed", 2200, 0.60, 0.10, (0.55, 0.54, 0.52), 0.88),
    Material("steel", "Steel", 7850, 0.42, 0.38, (0.56, 0.57, 0.60), 0.35, 1.0),
    Material("galv_steel", "Galvanised steel", 7850, 0.38, 0.35, (0.70, 0.72, 0.74), 0.42, 1.0),
    Material("alu", "Aluminium", 2700, 0.35, 0.40, (0.78, 0.79, 0.81), 0.30, 1.0),
    Material("glass", "Glass", 2500, 0.28, 0.22, (0.72, 0.82, 0.84), 0.06, 0.0, True, 1.6),
    Material("bottle_glass", "Glass bottle (filled)", 1150, 0.26, 0.18,
             (0.42, 0.55, 0.40), 0.10, 0.0, True, 2.0),
    Material("ceramic", "Glazed ceramic / porcelain", 2300, 0.34, 0.20,
             (0.90, 0.89, 0.86), 0.25, 0.0, True, 1.8),
    Material("wood", "Timber", 650, 0.48, 0.30, (0.52, 0.36, 0.21), 0.70),
    Material("plywood", "Plywood / MDF", 700, 0.46, 0.26, (0.66, 0.51, 0.33), 0.75),
    Material("cardboard", "Cardboard carton", 190, 0.52, 0.08, (0.66, 0.51, 0.34), 0.90),
    Material("carton_full", "Filled carton", 320, 0.54, 0.06, (0.63, 0.48, 0.32), 0.90),
    Material("plastic", "Plastic", 950, 0.32, 0.44, (0.80, 0.80, 0.82), 0.45),
    Material("pet_bottle", "PET bottle (filled)", 1010, 0.24, 0.30,
             (0.80, 0.86, 0.88), 0.18),
    Material("can", "Steel/aluminium can", 780, 0.30, 0.36, (0.72, 0.70, 0.66), 0.30, 0.8),
    Material("paper", "Paper / books", 780, 0.44, 0.08, (0.74, 0.70, 0.60), 0.85),
    Material("fabric", "Fabric / upholstery", 260, 0.68, 0.04, (0.36, 0.38, 0.45), 0.95),
    Material("rubber", "Rubber", 1200, 0.88, 0.66, (0.16, 0.16, 0.17), 0.85),
    Material("brick", "Clay brick", 1900, 0.66, 0.10, (0.58, 0.32, 0.25), 0.92),
    Material("plaster", "Plaster / render", 1500, 0.62, 0.10, (0.86, 0.85, 0.82), 0.90),
    Material("tile", "Polished floor tile", 2300, 0.36, 0.18, (0.80, 0.78, 0.74), 0.22),
    Material("laminate", "Laminate worktop", 850, 0.30, 0.22, (0.72, 0.68, 0.62), 0.35),
    Material("mineral_fibre", "Mineral fibre ceiling tile", 300, 0.50, 0.05,
             (0.90, 0.90, 0.88), 0.95),
    Material("liquid", "Liquid in container", 1000, 0.20, 0.05, (0.35, 0.55, 0.70), 0.10),
    Material("produce", "Fruit / vegetables", 700, 0.55, 0.18, (0.55, 0.60, 0.28), 0.80),
    Material("electronics", "Electronics", 900, 0.36, 0.25, (0.22, 0.23, 0.26), 0.40),
]

MATERIALS = {m.key: m for m in _M}


def get(key: str) -> Material:
    return MATERIALS.get(key, MATERIALS["plastic"])


def pair_friction(a: str, b: str) -> float:
    """Effective friction for a contact between two materials.

    Bullet combines per-body friction by multiplying, so each body is given
    sqrt(mu) and the product recovers the intended pair value. This helper
    returns the pair value we are aiming for.
    """
    ma, mb = get(a), get(b)
    return (ma.friction * mb.friction) ** 0.5
