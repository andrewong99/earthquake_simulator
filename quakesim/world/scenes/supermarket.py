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

"""Supermarket sales floor.

This is the scene everyone has watched on CCTV: long parallel gondola runs,
a few thousand small objects each free to slide, roll, rock and fall, and a
suspended ceiling overhead that starts dropping tiles well before anything
structural is in trouble.

It is also the most honest test of the physics, because nothing here is
choreographed. Whether a jar goes over is decided by its own width-to-height
ratio and the friction of the shelf it stands on, and the answer differs
shelf by shelf and item by item -- which is exactly why real footage looks
the way it does.
"""

from __future__ import annotations

import math

from ...seismo.site import SITES
from ...structure.mdof import Building
from ..cracks import Surface
from ..props import Builder
from ..scene_base import Scene, SceneSpec, Viewpoint, register

W, D, H = 34.0, 24.0, 4.2
CEIL = 3.15
COLS_X, COLS_Y = (-12.0, 1.7, 12.0), (-6.5, 6.5)     # interior columns, in the aisles
RESP = 99                                             # floor group carrying the shell's response
GOND = 200                                            # gondola shelves: the run's own response
CHILL = 201                                           # chiller shelves: the cabinet's response


@register
class Supermarket(Scene):
    spec = SceneSpec(
        key="supermarket",
        name="Supermarket sales floor",
        short="Supermarket",
        blurb="34 x 24 m sales floor with eight gondola runs, chillers, a "
              "checkout line and a suspended ceiling. Around two thousand "
              "individually simulated items, each of which slides or topples "
              "on its own merits.",
        site=SITES["stiff_soil"],
        building=Building("Single-storey retail shell", stories=1,
                          story_height=4.2, floor_mass=310_000,
                          system="concrete_shear_wall", damping=0.05,
                          period_override=0.32),
        default_distance_km=11.0,
        default_quake="strike_slip",
        default_mw=6.4,
        sky_time=10.0,
        # Everything stands on the slab and feels the ground motion; group
        # RESP carries the shell's response for the structural rules.
        story_of_floor={99: 0},
        observer_story=-1,          # you stand on the ground floor
        note="Shelf friction is per material pair, so glass jars start "
             "sliding around 0.26 g while cartons hold on past 0.5 g. "
             "Gondolas are bolted to nothing and go over at about 0.35 g "
             "of ground acceleration (±12%, the FEMA P-58 range for "
             "unanchored shelving); the wall panels tear off their roof "
             "anchorage at 0.45 g of the shell's own response, the way "
             "tilt-up shells did at Northridge, and the roof follows.",
        collapse_demo=(7.6, 1.0, "reverse"),
        viewpoints=[
            Viewpoint("aisle", "In the aisle", (1.7, -5.5, 1.62), (0, -2, 0),
                      "walk", 72, "Between two full gondolas."),
            Viewpoint("aisle_end", "Head of the aisles", (0.0, -10.8, 1.62),
                      (0, 2, 0), "walk", 76, "Looking across all eight runs."),
            Viewpoint("drinks", "Drinks aisle", (-6.8, 2.0, 1.55), (-90, 0, 0),
                      "walk", 70, "Glass bottles: the lowest friction in the shop."),
            Viewpoint("checkout", "Checkout line", (12.0, -9.0, 1.62),
                      (-140, -4, 0), "walk", 70, ""),
            Viewpoint("cctv", "CCTV corner", (-15.5, -10.5, 3.05),
                      (-42, -22, 0), "look", 82,
                      "The angle all the real footage is shot from."),
            Viewpoint("top", "Overhead", (0, 0, 42.0), (0, -90, 0), "top", 62, ""),
        ],
    )

    # A loaded steel gondola is not rigid: it racks across the aisle at
    # 2-4 Hz, and shake-table work on store shelving puts shelf-level
    # accelerations at two to three times the floor's. Each run is solved as
    # a one-storey frame (T = 0.35 s, 5% damping) and its stock feels the
    # frame's motion, exactly as pallets feel the rack's in the warehouse.
    # Rigid shelves gave a supermarket at M8.3 in which the bottles stood --
    # right for 0.30 g at the slab, wrong for what the shelf actually does.
    GONDOLA = Building("Loaded gondola shelving (across the aisle)", stories=1,
                       story_height=1.85, floor_mass=320.0, system="steel_rack",
                       damping=0.05, period_override=0.35)
    CHILLER = Building("Upright chiller cabinet", stories=1, story_height=2.05,
                       floor_mass=420.0, system="steel_rack", damping=0.05,
                       period_override=0.30)

    def attach_structures(self, shaker, motion, building_response,
                          progress=None) -> None:
        """Solve the shelving responses and bind the shelf groups to them.
        The units stand on the slab, so their base motion is the ground's."""
        from ...structure import mdof as _mdof
        base_x, base_y = motion.acc[0], motion.acc[1]
        pg = (lambda f: progress(0.5 * f, "Solving the shelving response")) if progress else None
        pc = (lambda f: progress(0.5 + 0.5 * f, "Solving the chiller response")) if progress else None
        self.gondola_response = _mdof.solve_2d(self.GONDOLA, base_x, base_y, motion.dt,
                                               progress=pg)
        self.chiller_response = _mdof.solve_2d(self.CHILLER, base_x, base_y, motion.dt,
                                               progress=pc)
        shaker.attach({GOND: 0}, self.gondola_response)
        shaker.attach({CHILL: 0}, self.chiller_response)

    def build(self, b: Builder) -> None:
        rng = b.rng
        st = self.structure
        self.gondola_response = None
        self.chiller_response = None
        self.units = []             # (structural element, its stock) per unit
        self.register_system(RESP, "concrete_shear_wall")
        self._shell(b)
        # the car park: a real surface, so a wall panel that goes over
        # outward has something to land on
        b.box((240, 240, 0.3), (0, 0, -0.30), "concrete", "asphalt", static=True,
              kind="structure", label="car park", uv=1.0 / 4.0)

        # suspended ceiling: individually simulated lay-in tiles
        # (the grid hangs from the roof, so the tiles feel the roof's motion)
        b.ceiling_grid(W - 2.0, D - 2.0, CEIL, (0, 0), 0.6, floor=RESP,
                       avoid=[(x, y, 0.22) for x in COLS_X for y in COLS_Y])
        for i in range(-4, 5):
            for j in range(-3, 4):
                b.decor((1.18, 0.28, 0.09), (i * 3.6, j * 3.2, CEIL + 0.02),
                        "alu", colour=(0.98, 0.98, 0.94), roughness=0.25)
        # sprinkler branch lines
        for j in range(-3, 4):
            b.decor((W - 3.0, 0.06, 0.06), (0, j * 3.2 + 1.2, CEIL + 0.30),
                    "steel", colour=(0.72, 0.16, 0.12))

        # --- gondola runs -------------------------------------------------
        # Eight double-sided runs, 1.0 m deep, 1.85 m high, 4 shelf levels.
        runs = (-13.6, -10.2, -6.8, -3.4, 0.0, 3.4, 6.8, 10.2)
        if b.detail < 0.5:
            runs = runs[::2]
        for run, gx in enumerate(runs):
            self._gondola_run(b, gx, rng, run)

        # --- perimeter chillers -------------------------------------------
        for i in range(max(2, int(6 * b.detail))):
            x = -14.0 + i * 3.4
            # An open carcass, not a solid block -- the stock lives inside
            # it. One body, unanchored: 2.05 m tall on a 1.05 m base (b/h
            # 0.5 static), released at 0.38 g of ground acceleration.
            cw, cd, chh, pnl = 3.2, 1.05, 2.05, 0.018
            parts = [((cw, pnl, chh - pnl), (0, cd / 2 - pnl / 2, pnl + (chh - pnl) / 2)),
                     ((pnl, cd, chh - pnl), (-(cw / 2 - pnl / 2), 0, pnl + (chh - pnl) / 2)),
                     ((pnl, cd, chh - pnl), (cw / 2 - pnl / 2, 0, pnl + (chh - pnl) / 2)),
                     ((cw, cd, pnl), (0, 0, chh - pnl / 2)),
                     ((cw, cd, pnl), (0, 0, pnl / 2))]
            nlv = 4 if b.detail >= 0.6 else 2
            for lv in range(nlv):
                parts.append(((3.0, 0.9, 0.02), (0, 0, 0.28 + lv * 0.44)))
            chiller = b.breakable_compound(parts, (x, D / 2 - 1.0, 0.0), "steel",
                                           "steel", mass=260.0, label="upright chiller",
                                           uv=1.0 / 0.4)
            el = st.add(chiller, "acc", 0.38, story=0, group=RESP, tag="chiller toppled",
                        ground_acc=True)    # it stands on the slab, not the roof
            n0 = len(self.phys.dynamic)
            for lv in range(nlv):
                z = 0.28 + lv * 0.44
                for k in range(max(3, int(9 * b.detail))):
                    px = x - 1.30 + k * b.spacing(0.34)
                    if px > x + 1.30:
                        continue
                    if rng.random() < 0.5:
                        b.bottle((px, D / 2 - 1.05, z + 0.011), r=0.033,
                                 h=0.245, material="pet_bottle", floor=CHILL,
                                 label="chilled bottle",
                                 colour=(0.85, 0.92, 0.95))
                    else:
                        b.box((0.10, 0.10, 0.20), (px, D / 2 - 1.05, z + 0.111),
                              "carton_full", None, mass=1.05, kind="contents",
                              floor=CHILL, label="milk carton",
                              colour=(0.92, 0.92, 0.9))
            self.units.append((el, self.phys.dynamic[n0:]))
            b.decor((3.2, 0.04, 2.0), (x, D / 2 - 1.53, 1.05), None,
                    colour=(0.8, 0.9, 0.95), transparent=0.22)

        # chest freezers: squat and heavy, they barely move -- a deliberate
        # contrast with everything else in the room
        for i in range(4):
            b.box((2.0, 0.95, 0.90), (-13.0 + i * 3.0, -D / 2 + 0.9, 0.45),
                  "steel", "steel", mass=190.0, kind="furniture",
                  label="chest freezer", colour=(0.88, 0.89, 0.9))

        # --- checkout -----------------------------------------------------
        for i in range(4):
            y = -10.6 + i * 2.6
            b.box((2.4, 0.75, 0.92), (12.4, y, 0.46), "laminate", "laminate",
                  static=True, kind="structure", label="checkout counter",
                  uv=1.0 / 0.5)
            b.box((0.34, 0.28, 0.36), (13.3, y + 0.1, 1.10), "electronics",
                  None, mass=6.5, kind="contents", label="till",
                  colour=(0.18, 0.19, 0.22))
            b.box((0.55, 0.45, 0.75), (14.6, y - 1.0, 0.375), "steel", "steel",
                  mass=16.0, kind="contents", label="shopping trolley",
                  colour=(0.72, 0.73, 0.75), metallic=0.7)

        # --- produce ------------------------------------------------------
        for i in range(3):
            x = -12.6 + i * 3.4
            b.box((2.8, 1.6, 0.70), (x, -D / 2 + 2.6, 0.35), "plywood",
                  "wood_light", static=True, kind="structure",
                  label="produce bin", uv=1.0 / 0.4)
            for sx2 in (-1, 1):
                b.box((0.05, 1.6, 0.16), (x + sx2 * 1.375, -D / 2 + 2.6, 0.78),
                      "plywood", "wood_light", static=True, kind="structure")
            for sy2 in (-1, 1):
                b.box((2.8, 0.05, 0.16), (x, -D / 2 + 2.6 + sy2 * 0.775, 0.78),
                      "plywood", "wood_light", static=True, kind="structure")
            # Laid out on a jittered grid: scattering fruit at random guarantees
            # some of it starts inside its neighbours, and Bullet resolves that
            # by launching it across the shop.
            rmax = 0.054
            pitch = 2.2 * rmax
            ncols = int(2.2 / pitch)
            nrows = int(1.1 / pitch)
            placed = 0
            budget = int(34 * b.detail)
            for gi in range(ncols):
                for gj in range(nrows):
                    if placed >= budget:
                        break
                    placed += 1
                    rad = 0.036 + rng.random() * 0.018
                    b.sphere(rad,
                             (x - 1.1 + (gi + 0.5) * pitch + rng.uniform(-0.008, 0.008),
                              -D / 2 + 2.6 - 0.55 + (gj + 0.5) * pitch
                              + rng.uniform(-0.008, 0.008),
                              0.702 + rad),
                         "produce", label="produce",
                         colour=(0.35 + rng.random() * 0.5,
                                 0.35 + rng.random() * 0.4,
                                 0.10 + rng.random() * 0.2))

        b.pendant_lamp((0, -11.0, CEIL - 0.02), cord=0.50, bob_r=0.09,
                       mass=1.1, label="hanging sign")

    # ------------------------------------------------------------------
    def _shell(self, b: Builder) -> None:
        """Floor, walls, roof and columns -- the walls and roof in panels the
        collapse model can release. Shear-wall shell: HAZUS "Complete" at
        2.5% drift, roof panels going with the columns and walls under them."""
        st = self.structure
        b.box((W, D, 0.25), (0, 0, -0.125), "concrete", "tile_grey", static=True,
              kind="structure", label="floor", uv=1.0 / 0.6)
        # interior columns, six, standing in the aisles
        cols = {}
        for i, x in enumerate(COLS_X):
            for j, y in enumerate(COLS_Y):
                c = b.breakable((0.40, 0.40, H - 0.05), (x, y, (H - 0.05) / 2),
                                "concrete", "concrete", label="column", uv=1.0 / 0.5)
                cols[(i, j)] = st.add(c, "drift", 0.025, story=0, group=RESP, tag="column")
        # wall panels
        walls = {}
        t = 0.25
        for sy, name in ((-1, "s"), (1, "n")):
            for k in range(8):
                x = -W / 2 + (k + 0.5) * W / 8
                p = b.breakable((W / 8, t, H), (x, sy * D / 2, H / 2), "concrete",
                                "white", label="wall panel", uv=1.0 / 0.8)
                walls[(name, k)] = st.add(p, "acc", 0.45, story=0, group=RESP,
                                         drift_limit=0.025, tag="wall panel")
        for sx, name in ((-1, "w"), (1, "e")):
            for k in range(6):
                # between the north and south walls, not through them
                span = (D - 2 * t) / 6
                y = -D / 2 + t + (k + 0.5) * span
                p = b.breakable((t, span - 0.001, H), (sx * W / 2, y, H / 2), "concrete",
                                "white", label="wall panel", uv=1.0 / 0.8)
                walls[(name, k)] = st.add(p, "acc", 0.45, story=0, group=RESP,
                                         drift_limit=0.025, tag="wall panel")
        # roof panels: 4 x 3, each on the columns and wall panels nearest it
        xe = [-W / 2, -8.5, 0.0, 8.5, W / 2]
        ye = [-D / 2, -4.0, 4.0, D / 2]
        for i in range(4):
            for j in range(3):
                xc, yc = (xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2
                deps = []
                for (ci, cj), el in cols.items():
                    cx, cy = COLS_X[ci], COLS_Y[cj]
                    if xe[i] - 0.5 <= cx <= xe[i + 1] + 0.5 and ye[j] - 0.5 <= cy <= ye[j + 1] + 0.5:
                        deps.append(el)
                if j == 0:
                    deps += [walls[("s", k)] for k in range(8) if xe[i] <= -W / 2 + (k + 0.5) * W / 8 <= xe[i + 1]]
                if j == 2:
                    deps += [walls[("n", k)] for k in range(8) if xe[i] <= -W / 2 + (k + 0.5) * W / 8 <= xe[i + 1]]
                if i == 0:
                    deps += [walls[("w", k)] for k in range(6) if ye[j] - 0.5 <= -D / 2 + (k + 0.5) * D / 6 <= ye[j + 1] + 0.5]
                if i == 3:
                    deps += [walls[("e", k)] for k in range(6) if ye[j] - 0.5 <= -D / 2 + (k + 0.5) * D / 6 <= ye[j + 1] + 0.5]
                p = b.breakable((xe[i + 1] - xe[i] - 0.02, ye[j + 1] - ye[j] - 0.02, 0.12),
                                (xc, yc, H + 0.06), "concrete", "white",
                                mass=(xe[i + 1] - xe[i]) * (ye[j + 1] - ye[j]) * 45.0,
                                label="roof panel", uv=1.0 / 0.6)
                st.add(p, "drift", 0.03, story=0, group=RESP, depends=deps,
                       need=max(1, len(deps) // 3), tag="roof panel")
        self.cracks.register(Surface((0, 0, 0.0), (W, D), "+z", 0, RESP, 2.0))
        self.crack_panels([walls[("n", k)].body for k in range(8)], "-y", 0, RESP, 0.15, t)
        self.crack_panels([walls[("s", k)].body for k in range(8)], "+y", 0, RESP, 0.15, t)

    def update(self, t: float, shaker) -> None:
        super().update(t, shaker)
        # Once a unit has gone over its stock is on the floor or on its way
        # there, and feels the ground rather than a shelf that no longer holds it.
        for el, stock in self.units:
            if el.failed and stock and stock[0].floor != 0:
                for body in stock:
                    self.phys.set_floor(body, 0)

    def _gondola_run(self, b: Builder, gx, rng, run):
        """One double-sided gondola run with four shelf levels each side.

        The run is one body -- base, spine and eight shelves -- standing on
        the floor and bolted to nothing, which is how most of them are. It
        is held as structure until the ground acceleration reaches what
        tips a 1.02 m wide, 1.85 m tall unit with stock on every level:
        the static tipping ratio b/h is 0.5, and the overturning fragility
        of unanchored shelving under real ground motion (FEMA P-58, HAZUS
        "Moderate" for unanchored contents) has its median at 0.35-0.40 g,
        so the rule uses 0.35 g with the model's +-12% scatter. It is then
        let go to do what it does.
        """
        # Longest run leaves a 0.4 m aisle to the chillers on the back wall
        # (at full detail an 18 m run was built 10 cm into them, which is a
        # shove the moment both are released).
        length = min(17.0, 18.0 * min(1.0, 0.55 + 0.45 * b.detail))
        y_mid = 1.6
        h = 1.85
        parts = [((1.02, length, 0.28), (0, 0, 0.14)),
                 ((0.10, length, h), (0, 0, h / 2 + 0.28))]
        for side in (-1, 1):
            for lv in range(4):
                z = 0.40 + lv * 0.42
                depth = 0.44 - lv * 0.02
                parts.append(((depth, length, 0.022), (side * (0.055 + depth / 2), 0, z)))
                # front lip: 15 mm. Real shelves have one, and it is the
                # difference between things sliding off and merely sliding.
                parts.append(((0.012, length, 0.015),
                              (side * (0.055 + depth - 0.006), 0, z + 0.019)))
        body = b.breakable_compound(parts, (gx, y_mid, 0.0), "steel", "shelf_white",
                                    mass=22.0 * length, label="gondola run",
                                    uv=1.0 / 0.5)
        # The demand is the ground's: the run stands on the floor slab, and
        # the shed's own response (its roof sees about twice the ground
        # acceleration) is what the wall and roof panels feel, not this.
        el = self.structure.add(body, "acc", 0.35, story=0, group=RESP,
                                tag="gondola toppled", ground_acc=True)
        from .. import geom as g
        cx, cy, cz = body.com_offset
        for i in range(int(length / 1.25) + 1):
            v = g.make_box(1.02, 0.05, h, 1.0, colour=(0.86, 0.87, 0.88))
            g.apply_material(v, "shelf_white", (0.86, 0.87, 0.88))
            v.reparentTo(body.path)
            v.setPos(-cx, -length / 2 + i * 1.25 - cy, h / 2 + 0.28 - cz)

        theme = run % 5
        n0 = len(self.phys.dynamic)
        for side in (-1, 1):
            for lv in range(4):
                z = 0.40 + lv * 0.42
                depth = 0.44 - lv * 0.02
                self._stock_shelf(b, gx + side * (0.055 + depth / 2), z + 0.011,
                                  length, depth, theme, lv, rng, y_mid)
        self.units.append((el, self.phys.dynamic[n0:]))

    def _stock_shelf(self, b: Builder, x, z, length, depth, theme, lv, rng,
                     y_mid=1.6):
        # Denser at higher detail, but never denser than the items are deep:
        # a 0.19 m cereal box on a 0.15 m pitch is built into its neighbour,
        # and Bullet resolves that by firing both off the shelf.
        min_step = {0: 0.10, 1: 0.09, 2: 0.215, 3: 0.09, 4: 0.13}[theme]
        step = max(b.spacing(0.15), min_step)
        y0 = y_mid - length / 2 + 0.25
        n = int((length - 0.5) / step)
        i = 0
        while i < n:
            y = y0 + i * step
            r = rng.random()
            if theme == 0:      # drinks: tall glass, low friction
                if r < 0.55:
                    b.bottle((x, y, z), r=0.037, h=0.29 + rng.random() * 0.06,
                             floor=GOND, label="glass bottle",
                             colour=(0.25 + rng.random() * 0.3,
                                     0.35 + rng.random() * 0.3, 0.25))
                    i += 1
                elif r < 0.8:
                    b.bottle((x, y, z), r=0.043, h=0.32, material="pet_bottle",
                             floor=GOND, label="PET bottle", colour=(0.8, 0.88, 0.92))
                    i += 1
                else:
                    for k in range(2):
                        b.can((x + (k - 0.5) * 0.075, y, z), floor=GOND)
                    i += 1
            elif theme == 1:    # tinned goods: squat, heavy, well behaved
                for k in range(2):
                    b.can((x + (k - 0.5) * 0.075, y, z), r=0.037, h=0.10, floor=GOND)
                i += 1
            elif theme == 2:    # cereal and dry goods: tall but light cartons
                hh = 0.30 + rng.random() * 0.06
                b.box((0.075, 0.19, hh), (x, y + 0.02, z + hh / 2),
                      "carton_full", "cardboard", mass=0.55, kind="contents",
                      floor=GOND, label="cereal box", uv=1.0 / 0.2)
                i += 1
            elif theme == 3:    # jars and sauces: the great topplers
                hh = 0.15 + rng.random() * 0.09
                b.cyl(0.038, hh, (x, y, z + hh / 2), "bottle_glass", None,
                      mass=0.62, kind="contents", floor=GOND, label="glass jar",
                      colour=(0.55 + rng.random() * 0.3, 0.42, 0.20))
                i += 1
            else:               # household: mixed
                if r < 0.5:
                    hh = 0.24
                    b.box((0.11, 0.11, hh), (x, y + 0.01, z + hh / 2),
                          "plastic", "plastic", mass=0.75, kind="contents",
                          floor=GOND, label="detergent box", colour=(0.2, 0.45, 0.7))
                else:
                    b.bottle((x, y, z), r=0.045, h=0.26, material="plastic",
                             mass=0.85, floor=GOND, label="bottle",
                             colour=(0.9, 0.75, 0.2))
                i += 1
