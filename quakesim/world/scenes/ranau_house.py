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

"""A village house near Ranau, Sabah, below a rock slope.

The near-field case, built around the real Mw 6.0 of 5 June 2015 -- shallow,
15 km from Ranau, directly under Mount Kinabalu. Sharp, violent, full of high
frequency, over in seconds. It damaged masonry across the district and set
off rockfalls and debris flows on the mountain.

Unreinforced block masonry is strong in its own plane and nearly helpless
across it, so its walls fail by *acceleration*, not by the building bending:
the top course of an unbraced wall goes first, the roof it carried follows.
Every wall here is built from panels the structural model can release, with
the top row weakest. Behind the house is a 30-degree rock slope whose loose
rock sits at a static factor of safety of about 1.1; it lets go when the
shaking exceeds its Newmark yield acceleration, and there is nothing between
it and the back wall (33 degrees, friction angle 35: factor of safety 1.08,
k_y = 0.035 g). Across the front yard is a strip of ground over a
shallow void that subsides when the ground acceleration exceeds 0.35 g --
surface fissuring is not rigid-body physics, so that trigger is stated
plainly rather than dressed up.

Try M6.0 at 15 km (the real event: cracks, gables, a rattled kitchen), then
M6.8 at 6 km (walls start to go, the slope lets go), then M7.3 at 5 km.
"""

from __future__ import annotations

import math

from ...seismo.site import SITES
from ...structure.mdof import Building
from ..cracks import Surface
from ..props import Builder
from ..scene_base import Scene, SceneSpec, Viewpoint, register

W, D, H = 9.0, 7.5, 2.85
T = 0.15                      # block wall thickness
SLOPE_DEG, SLOPE_HEADING = 33.0, 90.0
ROCK_MU = 0.70                # rock-on-slope friction: phi = 35 deg, FS = 1.08
SLOPE_CENTRE = (0.0, 17.5, 5.5)      # toe of the rock at y ~ 8 m, z ~ 0
# the slot in the yard the fissure strip sits in (x centre, y centre, w, l)
FISSURE_X, FISSURE_Y, FISSURE_W, FISSURE_L = -2.0, -12.5, 0.92, 12.0
RESP = 77                     # floor group that carries the house's response


@register
class RanauHouse(Scene):
    spec = SceneSpec(
        key="ranau_house",
        name="Village house, Ranau (Sabah)",
        short="Ranau village house",
        blurb="Single-storey unreinforced block house on residual soil below "
              "a rock slope, with a corrugated roof and unbraced gables. The "
              "walls fail course by course when the floor acceleration exceeds "
              "what masonry can take out-of-plane; the roof follows the walls; "
              "the slope behind lets go at its own yield acceleration.",
        site=SITES["sabah_residual"],
        building=Building("URM block + timber house", stories=1,
                          story_height=2.85, floor_mass=52_000,
                          system="masonry", damping=0.07, ductile=False,
                          post_yield_ratio=-0.10, period_override=0.18),
        default_distance_km=15.0,
        default_quake="strike_slip",
        default_mw=6.0,
        default_depth_km=10.0,
        sky_time=7.2,
        # Floor group 0 -- everything standing on the ground slab -- feels
        # the ground motion itself; the house's structural response is
        # carried on group RESP for the wall and roof rules.
        story_of_floor={77: 0},
        observer_story=-1,          # you stand on the ground floor
        note="Near-field, shallow, high-frequency. Walls: acceleration rule, "
             "top course 0.30 g, middle 0.42 g, bottom 0.60 g (±12%). Slope: "
             "33°, k_y ≈ 0.035 g. Fissure: 0.35 g.",
        collapse_demo=(7.0, 5.0, "strike_slip"),
        viewpoints=[
            Viewpoint("living", "Living room", (1.0, -2.9, 1.58), (40, -4, 0),
                      "walk", 76, "Settee, low table, the display cabinet beyond."),
            Viewpoint("kitchen", "Kitchen", (3.2, 0.3, 1.58),
                      (8, -10, 0), "walk", 72, "The back wall is the one the slope hits."),
            Viewpoint("veranda", "Front veranda", (0.0, -5.6, 1.58),
                      (180, -4, 0), "walk", 72,
                      "Outside but under the roof - where people run to. "
                      "The fissure strip crosses the yard in front."),
            Viewpoint("yard", "Across the yard", (7.0, -16.0, 1.65),
                      (24, 4, 0), "walk", 66,
                      "The whole house, gable end on, slope behind."),
            Viewpoint("slope", "Looking up at the slope", (0.0, 6.5, 1.65),
                      (0, 18, 0), "look", 74,
                      "Between the back wall and the rock. Do not stand here."),
            Viewpoint("ridge", "Up on the ridge", (22.0, 30.0, 16.0),
                      (150, -22, 0), "look", 70, "The slope and the house below it."),
            Viewpoint("top", "Overhead", (0, 4, 48.0), (0, -90, 0), "top", 60, ""),
        ],
    )

    def build(self, b: Builder) -> None:
        rng = b.rng
        st = self.structure
        self.register_system(RESP, "masonry")

        # --- ground -------------------------------------------------------
        # Built in pieces around a slot in the front yard: the fissure strip
        # sits in that slot flush with the surface, over a void, so when it is
        # released it has somewhere to go and the hole it leaves is a real
        # hole with soil walls, not a decal.
        hx0, hx1 = FISSURE_X - FISSURE_W / 2, FISSURE_X + FISSURE_W / 2
        hy0, hy1 = FISSURE_Y - FISSURE_L / 2, FISSURE_Y + FISSURE_L / 2

        def with_slot(x0, x1, y0, y1, z, thick, texture, uv, static):
            """Cover the rectangle x0..x1, y0..y1 except the fissure slot."""
            pieces = [(x0, hx0, y0, y1), (hx1, x1, y0, y1),
                      (hx0, hx1, hy1, y1), (hx0, hx1, y0, hy0)]
            for px0, px1, py0, py1 in pieces:
                if px1 - px0 < 1e-3 or py1 - py0 < 1e-3:
                    continue
                size = (px1 - px0, py1 - py0, thick)
                pos = ((px0 + px1) / 2, (py0 + py1) / 2, z)
                if static:
                    b.box(size, pos, "concrete", texture, static=True,
                          kind="structure", label="ground", uv=uv)
                else:
                    b.decor(size, pos, texture, uv=uv)

        with_slot(-160, 160, -160, 160, -0.025, 0.02, "grass", 1.0 / 4.0, False)
        with_slot(-15, 15, -20, 9, -0.06, 0.10, "soil", 1.0 / 2.0, False)
        with_slot(-30, 30, -36, 24, -0.23, 0.40, "soil", 1.0 / 2.0, True)
        # low tree-covered knolls to the south, stepped so they read as
        # mounds rather than blocks
        for i in range(7):
            a = rng.uniform(math.pi * 1.05, math.pi * 1.95)     # south side only
            r = rng.uniform(40, 130)
            hh = rng.uniform(4, 11)
            cx, cy = math.cos(a) * r, math.sin(a) * r
            for tier in range(3):
                w = hh * (4.5 - 1.3 * tier)
                h_t = hh * (tier + 1) / 3
                shade = 0.50 + 0.06 * tier
                b.decor((w, w * rng.uniform(0.8, 1.2), h_t), (cx, cy, -0.5 + h_t / 2),
                        "grass", colour=(shade * 0.95, shade * 1.25, shade * 0.8),
                        uv=1.0 / 3.0)

        # --- the slope behind the house --------------------------------------
        self._slope(b)

        # --- house structure --------------------------------------------------
        b.box((W + 0.6, D + 0.6, 0.35), (0, 0, -0.175), "concrete", "concrete",
              static=True, kind="structure", label="slab", uv=1.0 / 0.8)
        self.cracks.register(Surface((0, 0, 0.0), (W, D), "+z", 0, RESP, 1.4))

        # Walls as breakable panel courses. Two ways out: thrown out of plane
        # by the ground acceleration (thresholds rise toward the base, where
        # the weight above clamps the course), or racked apart in plane once
        # the storey drift passes what masonry holds together (HAZUS
        # "Complete" for URM is 2%; the top course lets go first).
        top, mid, bot = 0.30, 0.42, 0.60
        drift_lim = (0.020, 0.016, 0.012)
        # The side walls run the full depth; the front and rear walls butt
        # against their inner faces. (Run full width they share the corner
        # with the side wall -- 70 mm of interpenetration per course, which
        # Bullet resolves the moment both are released.)
        xi = W / 2 - T / 2
        walls = {
            "rear": b.wall_panels((-xi, D / 2), (xi, D / 2), 0.0, H, T,
                                  rows=3, texture="plaster_cream", label="rear wall"),
            "left": b.wall_panels((-W / 2, -D / 2), (-W / 2, D / 2), 0.0, H, T,
                                  rows=3, texture="plaster_cream", label="side wall"),
            "right": b.wall_panels((W / 2, D / 2), (W / 2, -D / 2), 0.0, H, T,
                                   rows=3, texture="plaster_cream", label="side wall"),
            "front_l": b.wall_panels((-xi, -D / 2), (-0.9, -D / 2), 0.0, H, T,
                                     rows=3, cols=2, texture="plaster_cream",
                                     label="front wall"),
            "front_r": b.wall_panels((0.9, -D / 2), (xi, -D / 2), 0.0, H, T,
                                     rows=3, cols=2, texture="plaster_cream",
                                     label="front wall"),
        }
        top_row = []
        top_by_wall = {}
        for name, rows in walls.items():
            under = None
            for r, row in enumerate(rows):
                limit = (bot, mid, top)[r]
                els = []
                for c, panel in enumerate(row):
                    # a course also goes when the course under it goes
                    dep = [under[c]] if under and c < len(under) else None
                    els.append(st.add(panel, "acc", limit, story=0, group=RESP,
                                      tag="wall panel", depends=dep, need=1,
                                      drift_limit=drift_lim[r], ground_acc=True))
                under = els
                if r == 2:
                    top_row += els
                    top_by_wall[name] = els
        # crack surfaces on the inner faces of the two long walls
        # crack surfaces on the inner face of each wall panel, so the cracks
        # go with the panel when it does
        flat = lambda rows: [p for row in rows for p in row]
        self.crack_panels(flat(walls["rear"]), "-y", 0, RESP, 0.16, T)
        self.crack_panels(flat(walls["left"]), "+x", 0, RESP, 0.14, T)
        self.crack_panels(flat(walls["right"]), "-x", 0, RESP, 0.14, T)
        self.crack_panels(flat(walls["front_l"]) + flat(walls["front_r"]), "+y", 0, RESP, 0.12, T)

        lintel = b.breakable((1.8, T, 0.55), (0, -D / 2, H - 0.275), "brick",
                             "plaster_cream", label="lintel")
        st.add(lintel, "acc", 0.50, tag="lintel", ground_acc=True, drift_limit=0.016, group=RESP)
        part = b.wall_panels((1.6, -D * 0.06), (1.6, D * 0.47), 0.0, H, T,
                             rows=3, texture="plaster_blue", label="partition")
        under = None
        for r, row in enumerate(part):
            els = []
            for c, panel in enumerate(row):
                dep = [under[c]] if under and c < len(under) else None
                els.append(st.add(panel, "acc", (0.70, 0.55, 0.40)[r],
                                  tag="partition panel", depends=dep, need=1,
                                  drift_limit=(0.024, 0.020, 0.016)[r],
                                  ground_acc=True, group=RESP))
            under = els
        self.crack_panels([p for row in part for p in row], "-x", 0, RESP, 0.14, T)

        # --- roof geometry: ridge along x at y = 0, eaves at y = +-D/2 ----------
        # 22-degree pitch: the ridge sits 1.52 m above the eave walls; the
        # gables are the triangles at x = +-W/2. Purlins rest on the gables,
        # sheets rest on the purlins, and all of it comes down when the walls
        # that carry the gables give way.
        pitch = math.radians(22.0)
        rise = (D / 2) * math.tan(pitch)
        top_of = lambda y: H + (D / 2 - abs(y)) * math.tan(pitch)   # roof line

        # gable courses on the wall head at each x-end: mortared block
        # courses braced by nothing, which is the classic out-of-plane
        # failure. Each course is one breakable body. Capacity falls toward
        # the apex (less weight above, longer lever), and a course also goes
        # when the course under it goes.
        for sx in (-1, 1):
            head = top_by_wall["left" if sx < 0 else "right"]
            below = None
            for row in range(7):
                z = H + 0.10 + row * 0.202
                halfw = (D / 2) * (1.0 - (0.20 + row * 0.202) / rise)
                if halfw < 0.25:
                    break
                course = b.breakable((T, 2 * halfw * 0.98, 0.20), (sx * W / 2, 0.0, z),
                                     "brick", "brick",
                                     mass=1900.0 * T * 2 * halfw * 0.20 * 0.75,
                                     kind="gable", label="gable course", uv=1.0 / 0.4)
                limit = 0.42 - 0.025 * row          # 0.42 g bottom -> 0.27 g top
                if below is None:
                    # sits on the wall head: goes when half of that wall's
                    # top course has gone
                    el = st.add(course, "acc", limit, tag="gable course",
                                depends=head, need=max(1, len(head) // 2),
                                ground_acc=True, drift_limit=0.012, group=RESP)
                else:
                    el = st.add(course, "acc", limit, tag="gable course",
                                depends=[below], need=1, ground_acc=True,
                                drift_limit=0.012, group=RESP)
                below = el

        # purlins along x, inside the gable span so they do not pierce it;
        # released when 40% of the top wall course has gone
        need = max(1, int(0.4 * len(top_row)))
        purlins = []
        for y in (-3.4, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.4):
            z = top_of(y) + 0.045
            p = b.breakable((W - 0.2, 0.09, 0.09), (0, y, z), "wood", "wood",
                            mass=12.0, label="purlin", colour=(0.5, 0.36, 0.24))
            st.add(p, "support", depends=top_row, need=need, tag="purlin")
            purlins.append(st.elements[-1])
        self.purlins = purlins

        # corrugated sheets, eave to ridge, four panels per slope. A sheet
        # lies on the purlins' upper corners: the purlins are level and the
        # sheet is pitched, so its underside clears the purlin top by
        # 0.045 tan(22 deg) at the down-slope corner. Each sheet stops 3 cm
        # short of the ridge line so the two slopes do not meet inside each
        # other, and the ridge cap sits above the sheet ends.
        slope_len = (D / 2) / math.cos(pitch) - 0.03
        lift = 0.045 * math.tan(pitch) + 0.002
        for sy in (-1, 1):
            for i in range(4):
                x = -W / 2 - 0.6 + (i + 0.5) * (W + 1.2) / 4
                yc = sy * (D / 4 + 0.015 * math.cos(pitch))
                zc = top_of(yc) + 0.09 + 0.015 + lift
                sheet = b.breakable(((W + 1.2) / 4 * 0.98, slope_len, 0.03),
                                    (x, yc, zc), "galv_steel", "corrugated_red",
                                    mass=8.0 * (W + 1.2) / 4 * slope_len,
                                    hpr=(0, -sy * 22.0, 0), label="roof sheet",
                                    uv=1.0 / 0.9)
                st.add(sheet, "support", depends=purlins, need=3, tag="roof sheet")
        # highest point of a sheet: the top inner corner of its ridge end
        # (mid-plane at 0.03 cos(pitch) from the ridge, plus half the sheet
        # thickness along its normal)
        sheet_top = (top_of(0.03 * math.cos(pitch)) + 0.09 + 0.015 + lift
                     + 0.015 * math.cos(pitch))
        ridge = b.breakable((W + 1.3, 0.30, 0.08), (0, 0, sheet_top + 0.04 + 0.003),
                            "galv_steel", "corrugated_red", mass=30.0, label="ridge cap")
        st.add(ridge, "support", depends=purlins, need=3, tag="roof sheet")

        # veranda: posts, and a lean-to sheet tucked under the front eave
        b.box((W, 2.2, 0.14), (0, -D / 2 - 1.1, -0.07), "concrete", "screed",
              static=True, kind="structure", label="veranda slab", uv=1.0 / 0.6)
        posts = []
        for sx in (-1, 1):
            p = b.breakable((0.14, 0.14, 2.45), (sx * (W / 2 - 0.4), -D / 2 - 2.0, 1.225),
                            "wood", "wood", mass=18.0, label="veranda post")
            st.add(p, "acc", 0.55, tag="veranda post", ground_acc=True, group=RESP)
            posts.append(st.elements[-1])
        vroof = b.breakable((W, 2.6, 0.04), (0, -D / 2 - 1.45, H - 0.25), "galv_steel",
                            "corrugated_red", mass=8.0 * W * 2.6, hpr=(0, 10, 0),
                            label="veranda roof", uv=1.0 / 0.9)
        st.add(vroof, "support", depends=posts, need=1, tag="roof sheet")

        # --- the fissure strip across the front yard ---------------------------
        # A strip of ground over a shallow void, in the slot the ground was
        # built around. Released at 0.35 g it drops 0.30 m onto the bedrock
        # below -- a triggered subsidence, stated as such. It lies wholly in
        # the yard, south of the veranda slab, so nothing in the house rests
        # on it.
        sw, sl, sth = FISSURE_W - 0.04, FISSURE_L - 0.10, 0.30
        b.box((FISSURE_W, FISSURE_L, 0.30), (FISSURE_X, FISSURE_Y, -0.01 - sth - 0.30 - 0.15),
              "concrete", "soil", static=True, kind="structure",
              label="bedrock under fissure", uv=1.0 / 2.0)
        strip = b.breakable((sw, sl, sth), (FISSURE_X, FISSURE_Y, -0.01 - sth / 2),
                            "concrete", "soil", mass=2200.0 * sw * sl * sth * 0.6,
                            label="fissure strip", uv=1.0 / 2.0)
        st.add(strip, "acc", 0.35, story=0, group=RESP, tag="ground fissure",
               scatter=False, ground_acc=True)
        self.fissure = st.elements[-1]

        # --- contents -----------------------------------------------------
        b.box((1.90, 0.80, 0.42), (-2.2, -2.0, 0.21), "fabric", "carpet_red",
              mass=52.0, kind="furniture", label="settee", uv=1.0 / 0.5)
        b.table(1.0, 0.55, 0.45, (-2.2, -0.5, 0), "wood", "wood", mass=14.0,
                label="low table")
        b.cyl(0.05, 0.20, (-2.45, -0.42, 0.552), "glass", None, mass=0.5,
              kind="contents", label="water jug", colour=(0.7, 0.85, 0.88))
        for i in range(3):
            b.cyl(0.032, 0.09, (-2.30 + i * 0.14, -0.55, 0.497), "glass", None,
                  mass=0.16, kind="contents", label="glass", colour=(0.78, 0.88, 0.9))

        cab = b.cabinet_shell(1.05, 0.40, 1.85, (-3.80, 1.9, 0), levels=4,
                              texture="wood_dark", label="display cabinet",
                              static=False)
        for zz in cab:
            b.crockery_stack((-4.05, 1.92, zz), count=4, r=0.10)
            for i in range(2):
                b.cyl(0.045, 0.16, (-3.58 + i * 0.16, 1.92, zz + 0.082),
                      "ceramic", None, mass=0.30, kind="contents",
                      label="ornament", colour=(0.85, 0.78, 0.6))
        for i in range(3):
            b.box((0.20, 0.03, 0.25), (-4.15 + i * 0.32, 1.88, 1.852 + 0.125),
                  "plywood", "wood_dark", mass=0.55, kind="contents",
                  label="framed photograph")

        b.box((0.95, 0.42, 0.55), (-1.0, 2.6, 0.275), "plywood", "wood",
              mass=18.0, kind="furniture", label="TV stand")
        b.box((0.82, 0.07, 0.50), (-1.0, 2.6, 0.802), "electronics", None,
              mass=6.0, kind="contents", label="television", colour=(0.11, 0.11, 0.13))
        b.cyl(0.19, 1.28, (-4.10, -2.8, 0.64), "plastic", "plastic", mass=6.5,
              kind="furniture", label="standing fan", colour=(0.85, 0.85, 0.82))

        # kitchen
        # (starts clear of the partition's face at x = 1.675)
        b.box((2.5, 0.58, 0.85), (2.95, D / 2 - 0.5, 0.425), "plywood", "wood",
              static=True, kind="structure", label="kitchen counter", uv=1.0 / 0.5)
        b.box((2.5, 0.60, 0.04), (2.95, D / 2 - 0.5, 0.87), "tile", "tile",
              static=True, kind="structure", label="counter top", uv=1.0 / 0.3)
        for i in range(5):
            b.cyl(0.055, 0.13 + i * 0.02, (2.0 + i * 0.28, D / 2 - 0.5,
                                            0.892 + (0.13 + i * 0.02) / 2),
                  "ceramic", None, mass=0.45, kind="contents", label="pot",
                  colour=(0.75, 0.74, 0.70))
        for i in range(4):
            b.bottle((3.85, D / 2 - 0.40 - i * 0.10, 0.892), r=0.033, h=0.22,
                     label="kitchen bottle")
        b.cyl(0.055, 0.32, (1.88, D / 2 - 0.68, 0.892 + 0.16), "steel", "steel",
              mass=1.6, kind="contents", label="vacuum flask (termos)",
              colour=(0.72, 0.24, 0.20), metallic=0.5)
        for i in range(4):
            b.cyl(0.030, 0.26, (3.32 + i * 0.14, D / 2 - 0.70, 0.892 + 0.13),
                  "bottle_glass", None, mass=0.42, kind="contents",
                  label="sauce bottle", colour=(0.45, 0.26, 0.14))
        b.box((0.62, 0.62, 1.55), (3.90, D / 2 - 2.6, 0.775), "steel", "steel",
              mass=62.0, kind="furniture", label="refrigerator",
              colour=(0.86, 0.86, 0.84), roughness=0.4)

        b.table(1.35, 0.80, 0.74, (3.0, -1.6, 0), "wood", "wood", mass=26.0,
                label="dining table")
        for dx, dy, hh in ((-0.5, -0.7, 0), (0.5, -0.7, 0), (0.0, 0.7, 180)):
            b.chair((3.0 + dx, -1.6 + dy, 0), (hh, 0, 0))
        b.crockery_stack((3.0, -1.6, 0.742), count=3, r=0.105)
        b.cyl(0.07, 0.44, (2.55, -1.60, 0.742 + 0.22), "glass", None, mass=0.95,
              kind="contents", label="flower vase", colour=(0.55, 0.72, 0.66))

        # The fan and the light hang on rods from the purlins -- there is no
        # ceiling in this house -- and come down with the roof.
        for (x, y, bob_z, r, m, label) in ((-1.6, 0.2, H - 0.52, 0.10, 6.5, "ceiling fan"),
                                            (3.0, -1.6, H - 0.65, 0.11, 1.2, "pendant light")):
            purlin = min(self.purlins, key=lambda e: abs(e.body.home[0][1] - y))
            px, py, pz = purlin.body.home[0]
            b.pendant_lamp((x, py, pz - 0.045), cord=pz - 0.045 - bob_z, bob_r=r,
                           mass=m, label=label, anchor_body=purlin.body)

        # water tank on a stand outside
        b.box((1.3, 1.3, 3.0), (W / 2 + 2.4, 2.0, 1.5), "steel", "steel",
              static=True, kind="structure", label="tank stand",
              colour=(0.5, 0.5, 0.52), metallic=0.8)
        b.cyl(0.60, 1.10, (W / 2 + 2.4, 2.0, 3.552), "plastic", "plastic",
              mass=1200.0, kind="contents", label="water tank (1200 L)",
              colour=(0.15, 0.35, 0.55))

    # ------------------------------------------------------------------
    def _slope(self, b: Builder) -> None:
        """A 30-degree rock slope rising away from the back of the house."""
        u, v, n, hpr = b.slope_frame(SLOPE_DEG, SLOPE_HEADING)
        along, across = 30.0, 34.0
        # the slope body, thick enough that nothing can fall through it
        centre = b.slope_point(SLOPE_CENTRE, SLOPE_DEG, SLOPE_HEADING, 0.0, 0.0, -0.6)
        slope = b.box((across, along, 1.2), centre, "concrete", "soil", hpr=hpr,
                      static=True, kind="structure", label="rock slope", uv=1.0 / 3.0)
        # Bullet multiplies the two bodies' friction; the rocks carry
        # sqrt(ROCK_MU), so the slope carries the same and the pair is
        # ROCK_MU = 0.70: friction angle 35.0 degrees on a 33 degree slope,
        # static factor of safety 1.08, Newmark yield acceleration
        # k_y = tan(35 - 33) = 0.035 g. A cut slope in residual soil that
        # has been creeping since the last monsoon -- which is what stands
        # behind a great many houses in Sabah. Jibson's regression gives it
        # about half a metre of displacement at 0.4 g, and the physics here
        # agrees; a firmer slope (k_y = 0.15 g) moves a few centimetres,
        # which is correct and invisible.
        slope.node.setFriction(math.sqrt(ROCK_MU))
        # the ridge line and the hill mass behind, visual only
        # the hill mass behind the cut: tiers stepping back and up from the
        # crest, grass on top, so the skyline is a ridge and not a box
        crest = b.slope_point(SLOPE_CENTRE, SLOPE_DEG, SLOPE_HEADING, along / 2, 0.0, 0.0)
        for tier in range(4):
            depth = 40.0 - 8.0 * tier
            top = crest[2] + 3.0 + 5.0 * tier
            yc = crest[1] + 6.0 + depth / 2 + 10.0 * tier
            b.decor((across + 60 - 14 * tier, depth, top + 2.0),
                    (crest[0], yc, (top - 2.0) / 2), "grass",
                    colour=(0.42 + 0.03 * tier, 0.55 + 0.03 * tier, 0.36), uv=1.0 / 6.0)
        count = max(30, int(110 * b.detail))
        # The field stops short of the toe: the surface meets the ground at
        # a = -5.5 / sin(33) = -10.1 m along the slope, and a rock built
        # below that is inside the ground box.
        field_centre = b.slope_point(SLOPE_CENTRE, SLOPE_DEG, SLOPE_HEADING, 1.5, 0.0, 0.0)
        self.rocks = b.boulder_field(field_centre, 19.0, across - 6.0,
                                     SLOPE_DEG, SLOPE_HEADING, count, floor=0,
                                     friction=ROCK_MU, seed=self.seed + 7)

    # Residual strength. A slope that has started to move does not stop
    # where the shaking stops: the sliding surface loses its peak strength
    # and settles to a residual value (the mechanism behind every
    # earthquake-triggered landslide that ran rather than crept). Once a
    # rock has slid RUN_AFTER along the slope its friction drops to
    # RESIDUAL of its peak, which on a 33-degree slope is below tan(33) --
    # and it runs.
    RUN_AFTER = 0.08          # m of Newmark displacement before strength loss
    RESIDUAL = 0.86           # residual / peak friction: phi 35 -> 31 degrees

    def update(self, t: float, shaker) -> None:
        super().update(t, shaker)
        for r in getattr(self, "rocks", ()):
            if r.softened or r.peak_mu <= 0.0:
                continue
            p = r.path.getPos()
            if math.dist((p[0], p[1], p[2]), r.home[0]) > self.RUN_AFTER:
                r.softened = True
                r.node.setFriction(math.sqrt(r.peak_mu * self.RESIDUAL))
                r.node.setActive(True)

    def on_reset(self) -> None:
        super().on_reset()
        for r in getattr(self, "rocks", ()):
            if r.softened:
                r.softened = False
                r.node.setFriction(math.sqrt(r.peak_mu))

    def damage_report(self) -> list[str]:
        out = super().damage_report()
        moved = 0
        for r in getattr(self, "rocks", ()):
            p = r.path.getPos()
            if math.dist((p[0], p[1], p[2]), r.home[0]) > 1.0:
                moved += 1
        if moved:
            out.append(f"Landslide: {moved} of {len(self.rocks)} rocks ran")
        else:
            crept = sum(1 for r in getattr(self, "rocks", ())
                        if math.dist(r.path.getPos(), r.home[0]) > 0.02)
            if crept:
                out.append(f"Slope creeping: {crept} of {len(self.rocks)} rocks shifted")
        if getattr(self, "fissure", None) is not None and self.fissure.failed:
            out.append("Ground fissure opened across the yard")
        return out
