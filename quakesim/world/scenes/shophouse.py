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

"""A row of shophouses on a Malaysian town street.

The commonest building type in every town between Penang and Johor Bahru:
two, three or four storeys of reinforced-concrete frame with brick infill,
a sundry shop or coffee shop on the ground floor and the family living
above, a covered five-foot way along the front. Thousands were put up in
the 1960s-80s with no seismic provision at all, and they carry the one
weakness that kills more people in earthquakes than any other: a soft
storey. The ground floor is open -- a shopfront, a roller shutter, columns
and nothing else -- while every floor above it is stiffened by brick walls.
Under lateral load the whole building's drift concentrates in that one
storey, its non-ductile columns fail in shear, and the floors above come
down onto the shop as a stack.

Here the structure is solved with a soft ground storey (a third of the
stiffness of the infilled floors above), the ground-storey columns are
released when that storey's drift reaches 2% -- HAZUS "Complete" for a
pre-code concrete frame -- and the slabs above lose their support and fall.
Infill walls crack out at 1%. The parapet along the roof edge goes at
0.35 g and lands on the street, which is where the people who ran out are
standing.
"""

from __future__ import annotations

import math

from ...seismo.site import SITES
from ...structure.mdof import Building
from ..cracks import Surface
from ..props import Builder
from ..scene_base import Scene, SceneSpec, Viewpoint, register

UW, UD = 6.1, 15.0            # one unit: 20 ft wide, 15 m deep
NU = 3                        # units in the row
H0, H1 = 3.6, 3.2             # ground storey, upper storeys
STOREYS = 4
SLAB = 0.18
COL = 0.35
XB = [-(NU / 2) * UW + i * UW for i in range(NU + 1)]      # party-wall lines
YB = [-UD / 2, -UD / 2 + 5.0, UD / 2 - 5.0, UD / 2]           # column lines
LEVELS = [0.0, H0, H0 + H1, H0 + 2 * H1, H0 + 3 * H1]       # floor levels
WALKWAY = 1.5                                               # five-foot way
SHOP = 1                                                    # the unit we stand in
LIVING_GROUP = 1                                            # first-floor bodies


@register
class Shophouse(Scene):
    spec = SceneSpec(
        key="shophouse",
        name="Shophouse row, small-town Malaysia",
        short="Shophouse row",
        blurb="Three four-storey shophouses: RC frame, brick infill, an open "
              "shopfront under three floors of family home. A soft ground "
              "storey -- the building type that kills more people in "
              "earthquakes than any other. Stand in the sundry shop, on the "
              "five-foot way, or in the living room upstairs.",
        site=SITES["stiff_soil"],
        building=Building("4-storey non-ductile RC shophouse (soft storey)",
                          stories=STOREYS, story_height=3.3,
                          floor_mass=NU * UW * UD * 950.0,
                          system="nonductile_rc", damping=0.05,
                          base_shear_coeff=0.08, ductile=False,
                          post_yield_ratio=-0.05,
                          stiffness_shape=(0.35, 1.0, 1.0, 1.0)),
        default_distance_km=12.0,
        default_quake="reverse",
        default_mw=6.3,
        default_depth_km=12.0,
        sky_time=16.0,
        story_of_floor={LIVING_GROUP: 0},        # first floor rides storey 1
        observer_story=-1,          # you stand on the ground floor
        note="Soft-storey frame: ground storey at 35% of the upper storeys' "
             "stiffness, so it takes most of the drift. Its columns are "
             "crushed at 2% drift (±12%) and its infill at 1%; the floors "
             "above come down as a stack. Parapets go at 0.35 g.",
        collapse_demo=(7.2, 5.0, "reverse"),
        viewpoints=[
            Viewpoint("shop", "In the sundry shop", (0.3, -4.3, 1.60), (4, -3, 0),
                      "walk", 72, "Ground floor, under three storeys of home."),
            Viewpoint("walkway", "On the five-foot way", (0.0, -6.9, 1.60),
                      (0, 4, 0), "walk", 74,
                      "The covered walkway along the shopfronts."),
            Viewpoint("street", "Across the street", (4.0, -22.0, 1.65),
                      (10, 12, 0), "walk", 66,
                      "The whole row. Watch the ground storey."),
            Viewpoint("living", "Living room, first floor", (-1.2, -3.5, H0 + 1.60),
                      (-20, 0, 0), "walk", 68, "Family home above the shop."),
            Viewpoint("corner", "Street corner", (-17.0, -21.0, 1.65),
                      (-40, 12, 0), "walk", 64, "The row from the corner."),
            Viewpoint("roof", "Roof of the block opposite", (10.0, -36.0, 14.0),
                      (12, -8, 0), "look", 62, ""),
            Viewpoint("top", "Overhead", (0, 0, 60.0), (0, -90, 0), "top", 58, ""),
        ],
    )

    # ------------------------------------------------------------------
    def build(self, b: Builder) -> None:
        rng = b.rng
        st = self.structure
        self._freed = False
        self.register_system(LIVING_GROUP, "nonductile_rc")
        self._frame(b)
        self._shop(b)
        self._home(b)
        self._street(b)

    # ------------------------------------------------------------------
    def _frame(self, b: Builder) -> None:
        st = self.structure
        # ground slab, the five-foot way and the street, one surface
        b.box((90.0, 80.0, 0.30), (0, -10.0, -0.15), "concrete", "screed",
              static=True, kind="structure", label="ground", uv=1.0 / 0.6)

        # columns by storey and grid position
        self.columns = {}                    # (storey, i, j) -> Element
        self.slabs = {}                      # (unit, level) -> Element
        col_h = [LEVELS[s + 1] - LEVELS[s] - SLAB for s in range(STOREYS)]
        for s in range(STOREYS):
            for i, x in enumerate(XB):
                for j, y in enumerate(YB):
                    z0 = LEVELS[s]
                    c = b.breakable((COL, COL, col_h[s]), (x, y, z0 + col_h[s] / 2),
                                    "concrete", "plaster_cream", label="column",
                                    uv=1.0 / 0.5)
                    deps = None
                    if s > 0:
                        # stands on the slab(s) it is bounded by
                        deps = [self.slabs[(u, s)] for u in (i - 1, i)
                                if (u, s) in self.slabs]
                    # A ground-storey column that reaches 2% drift has lost
                    # its capacity -- crushed concrete, buckled bars -- and is
                    # taken out of the world, so the storey above comes down
                    # through it: the soft-storey pancake. Left in as an
                    # intact block it went on holding the slab up, upright,
                    # like a table leg. Columns higher up fail by losing the
                    # slab they stand on, and fall with it as blocks.
                    self.columns[(s, i, j)] = st.add(
                        c, "drift", 0.020, story=s, group=LIVING_GROUP,
                        depends=deps, need=1, tag="column", crush=(s == 0))
            # slabs of the level above this storey, one per unit
            for u in range(NU):
                xc = (XB[u] + XB[u + 1]) / 2
                z = LEVELS[s + 1]
                cols = [self.columns[(s, i, j)] for i in (u, u + 1) for j in range(4)]
                top = s == STOREYS - 1
                slab = b.breakable((UW - 0.02, UD, SLAB), (xc, 0, z - SLAB / 2),
                                   "concrete", "screed" if not top else "concrete",
                                   label="roof slab" if top else "floor slab",
                                   uv=1.0 / 0.6)
                self.slabs[(u, s + 1)] = st.add(slab, "support", depends=cols,
                                                need=3, tag="floor slab",
                                                floor_after=0, story=s,
                                                group=LIVING_GROUP)
        self.observer_slab = self.slabs[(SHOP, 1)]
        # the camera in the living room rides the first-floor slab down
        self.observer_body = self.slabs[(SHOP, 1)].body

        # infill walls: party walls every storey, front and rear walls on the
        # upper storeys (the ground floor is an open shopfront). Each wall in
        # three panels along its length so it comes apart, not all at once.
        # A ground-storey panel that has cracked through at 1% drift is
        # crushed out of the frame like the columns: as an intact free block
        # under 39 t of slab it is a bearing wall, and held the storey up
        # after every column had gone.
        self.walls = []
        for s in range(STOREYS):
            z0 = LEVELS[s]
            h = col_h[s]
            for i, x in enumerate(XB):
                for k in range(3):
                    yc = -UD / 2 + (k + 0.5) * UD / 3
                    p = b.breakable((0.15, UD / 3 - COL - 0.02, h), (x, yc, z0 + h / 2),
                                    "brick", "plaster_cream", label="party wall",
                                    uv=1.0 / 0.8)
                    deps = [self.slabs[(u, s)] for u in (i - 1, i)
                            if s > 0 and (u, s) in self.slabs] or None
                    self.walls.append(st.add(p, "drift", 0.010, story=s,
                                             group=LIVING_GROUP, depends=deps,
                                             need=1, tag="infill wall",
                                             crush=(s == 0)))
            for u in range(NU):
                xc = (XB[u] + XB[u + 1]) / 2
                faces = [(UD / 2, "rear wall")]
                if s > 0:
                    faces.append((-UD / 2, "front wall"))
                for y, label in faces:
                    p = b.breakable((UW - COL - 0.02, 0.15, h), (xc, y, z0 + h / 2),
                                    "brick", "plaster_cream" if y < 0 else "plaster",
                                    label=label, uv=1.0 / 0.8)
                    deps = [self.slabs[(u, s)]] if s > 0 else None
                    self.walls.append(st.add(p, "drift", 0.010, story=s,
                                             group=LIVING_GROUP, depends=deps,
                                             need=1, tag="infill wall",
                                             crush=(s == 0)))
                if s > 0:
                    # window strip on the front, visual only, riding the wall
                    pass
        # roof parapet along the street edge: unreinforced, goes at 0.35 g
        for u in range(NU):
            xc = (XB[u] + XB[u + 1]) / 2
            par = b.breakable((UW - 0.05, 0.15, 0.9), (xc, -UD / 2, LEVELS[-1] + 0.45),
                              "brick", "plaster_cream", label="parapet", uv=1.0 / 0.8)
            st.add(par, "acc", 0.35, story=STOREYS - 1, group=LIVING_GROUP,
                   depends=[self.slabs[(u, STOREYS)]], need=1, tag="parapet")
        # signboards over the five-foot way, one per shop
        for u in range(NU):
            xc = (XB[u] + XB[u + 1]) / 2
            sign = b.breakable((UW - 0.6, 0.06, 0.9), (xc, -UD / 2 - 0.10, H0 - 0.55),
                               "galv_steel", None, mass=45.0, label="signboard",
                               colour=((0.85, 0.15, 0.12), (0.95, 0.85, 0.20),
                                       (0.20, 0.45, 0.80))[u])
            st.add(sign, "acc", 0.60, story=0, group=LIVING_GROUP,
                   depends=[self.slabs[(u, 1)]], need=1, tag="signboard")

        # crack surfaces: the shop floor (ground slab), the shop's party
        # walls and rear wall, the living-room floor and its walls -- each
        # on the panel that carries it
        self.cracks.register(Surface((0, 0, 0.0), (UW, UD), "+z", 0, LIVING_GROUP, 1.5))
        shop_panels = [e.body for e in self.walls
                       if abs(e.body.home[0][0] - XB[SHOP]) < 0.01 and e.story == 0]
        self.crack_panels(shop_panels, "+x", 0, LIVING_GROUP, 0.3, 0.15)
        shop_panels = [e.body for e in self.walls
                       if abs(e.body.home[0][0] - XB[SHOP + 1]) < 0.01 and e.story == 0]
        self.crack_panels(shop_panels, "-x", 0, LIVING_GROUP, 0.3, 0.15)
        self.cracks.register(Surface((0, 0, H0), (UW, UD), "+z", 1, LIVING_GROUP, 1.0,
                                     body=self.slabs[(SHOP, 1)].body))
        living_walls = [e.body for e in self.walls
                        if e.story == 1 and abs(e.body.home[0][0]) < UW / 2 - 0.5
                        and e.body.home[0][1] > 0]
        self.crack_panels(living_walls, "-y", 1, LIVING_GROUP, 0.5, 0.15)

    # ------------------------------------------------------------------
    def _shop(self, b: Builder) -> None:
        """The sundry shop on the ground floor of the middle unit."""
        rng = b.rng
        st = self.structure
        # two runs of shelving along the length of the shop, open toward the
        # aisle between them, fixed to the floor and stiff enough to stay put
        for side in (-1, 1):
            xs = side * 1.55
            for k in range(2):
                y = -1.0 + k * 3.2
                levels = self._shelf_run(b, xs, y, 3.0, side)
                for lv, z in enumerate(levels):
                    n_items = max(4, int(11 * b.detail))
                    for m in range(n_items):
                        py = y - 1.35 + m * (2.7 / n_items)
                        r = rng.random()
                        if r < 0.35:
                            b.can((xs, py, z), r=0.036, h=0.11)
                        elif r < 0.60:
                            b.bottle((xs, py, z), r=0.034, h=0.22 + rng.random() * 0.06,
                                     material="pet_bottle", label="drink bottle",
                                     colour=(0.75 + rng.random() * 0.2,
                                             0.55 + rng.random() * 0.4, 0.3))
                        elif r < 0.85:
                            hh = 0.14 + rng.random() * 0.12
                            b.box((0.12, 0.09, hh), (xs, py, z + hh / 2), "cardboard",
                                  "cardboard", mass=0.4, kind="contents",
                                  label="packet", colour=(0.6 + rng.random() * 0.35,
                                                          0.5 + rng.random() * 0.3,
                                                          0.3 + rng.random() * 0.3))
                        else:
                            b.cyl(0.045, 0.13, (xs, py, z + 0.065), "glass", None,
                                  mass=0.55, kind="contents", label="jar",
                                  colour=(0.82, 0.72, 0.45))
        # counter by the front with the till and a glass display cabinet
        b.box((1.8, 0.6, 1.0), (1.4, -4.6, 0.5), "laminate", "laminate", static=True,
              kind="structure", label="counter", uv=1.0 / 0.5)
        b.box((0.36, 0.30, 0.32), (1.0, -4.6, 1.16), "electronics", None, mass=5.5,
              kind="contents", label="till", colour=(0.20, 0.21, 0.24))
        b.box((0.30, 0.22, 0.42), (2.0, -4.55, 1.21), "glass", None, mass=3.0,
              kind="contents", label="sweet jar display", colour=(0.85, 0.78, 0.55))
        cab = b.cabinet_shell(1.0, 0.5, 1.2, (-1.0, -4.6, 0), levels=2,
                              texture="wood_light", label="display cabinet",
                              static=False)
        for z in cab:
            for i in range(3):
                b.box((0.18, 0.12, 0.10), (-1.3 + i * 0.3, -4.6, z + 0.05), "cardboard",
                      None, mass=0.3, kind="contents", label="cigarette carton",
                      colour=(0.9, 0.9, 0.85))
        # rice sacks stacked by the rear wall, a gas cylinder rack, a chiller
        for i in range(2):
            for k in range(4):
                b.box((0.55, 0.38, 0.16), (-1.6 + i * 0.62, 5.6, 0.08 + k * 0.162),
                      "fabric", "cardboard", mass=10.0, kind="contents",
                      label="rice sack (10 kg)", colour=(0.92, 0.90, 0.82))
        for i in range(3):
            b.cyl(0.16, 0.74, (1.4 + i * 0.36, 5.9, 0.37), "steel", "steel", mass=28.0,
                  kind="contents", label="gas cylinder", colour=(0.75, 0.25, 0.20))
        chill = b.cabinet_shell(1.2, 0.68, 1.9, (2.3, 5.0, 0), levels=4,
                                texture="steel", label="drinks chiller",
                                breakable=True)
        # crushed with the ground storey, like the shelving: a rigid chiller
        # left standing is what the falling slab would otherwise rest on
        self.structure.add(b.last_body, "drift", 0.020, story=0, group=LIVING_GROUP,
                           tag="drinks chiller", crush=True)
        for z in chill:
            for i in range(4):
                b.bottle((1.9 + i * 0.27, 5.0, z), r=0.036, h=0.25, material="pet_bottle",
                         label="chilled drink", colour=(0.55, 0.75, 0.9))
        # a ceiling fan and two pendant lamps
        ceiling = self.slabs[(SHOP, 1)].body
        b.pendant_lamp((0, -1.5, H0 - SLAB - 0.05), cord=0.6, mass=1.2,
                       label="shop lamp", anchor_body=ceiling)
        b.pendant_lamp((0, 3.0, H0 - SLAB - 0.05), cord=0.6, mass=1.2,
                       label="shop lamp", anchor_body=ceiling)
        # a motorcycle and a stool on the five-foot way
        b.box((1.9, 0.6, 0.95), (1.6, -6.9, 0.48), "steel", "steel", mass=110.0,
              kind="vehicle", label="motorcycle", colour=(0.20, 0.22, 0.55))
        b.compound([((0.34, 0.34, 0.03), (0, 0, 0.44)),
                    ((0.03, 0.03, 0.43), (0.14, 0.14, 0.215)),
                    ((0.03, 0.03, 0.43), (-0.14, 0.14, 0.215)),
                    ((0.03, 0.03, 0.43), (0.14, -0.14, 0.215)),
                    ((0.03, 0.03, 0.43), (-0.14, -0.14, 0.215))],
                   (-2.2, -6.8, 0), "plastic", None, mass=1.4, label="plastic stool",
                   colour=(0.85, 0.20, 0.15))

    def _shelf_run(self, b: Builder, x, y_mid, length, side) -> list:
        """Fixed shop shelving: a back panel on the wall side, two ends and
        five shelves open toward the aisle, as one body. Returns the shelf
        surface heights.

        It stands as structure and is crushed with the ground storey (at 2%
        drift, with the columns): light steel shelving under a 39 t slab
        folds flat, whereas a rigid unit left in the world caught the
        falling storey 2 m up and held it there.
        """
        parts = [((0.03, length, 2.0), (side * 0.20, 0.0, 1.0))]
        for sy in (-1, 1):
            parts.append(((0.42, 0.03, 2.0), (0.0, sy * (length / 2 - 0.015), 1.0)))
        out = []
        for lv in range(5):
            z = 0.08 + lv * 0.42
            parts.append(((0.40, length - 0.06, 0.022), (0.0, 0.0, z)))
            out.append(z + 0.011 + 0.002)
        unit = b.breakable_compound(parts, (x, y_mid, 0.0), "steel", "shelf_white",
                                    mass=70.0, label="shop shelving", uv=1.0 / 0.5)
        self.structure.add(unit, "drift", 0.020, story=0, group=LIVING_GROUP,
                           tag="shop shelving", crush=True)
        return out

    # ------------------------------------------------------------------
    def _home(self, b: Builder) -> None:
        """The family's living room on the first floor of the shop unit."""
        rng = b.rng
        z0 = H0
        g = LIVING_GROUP
        b.box((2.1, 0.85, 0.42), (-1.6, -4.2, z0 + 0.21), "fabric", "carpet",
              mass=60.0, floor=g, kind="furniture", label="settee", uv=1.0 / 0.5)
        b.table(0.9, 0.5, 0.42, (-1.6, -3.0, z0), "wood", "wood_dark", mass=12.0,
                floor=g, label="coffee table")
        b.cyl(0.035, 0.11, (-1.8, -3.0, z0 + 0.477), "glass", None, mass=0.22,
              floor=g, kind="contents", label="glass", colour=(0.75, 0.88, 0.9))
        b.box((0.2, 0.14, 0.02), (-1.4, -2.95, z0 + 0.43), "paper", None, mass=0.4,
              floor=g, kind="contents", label="newspaper", colour=(0.85, 0.83, 0.78))
        b.box((1.4, 0.4, 0.45), (-1.6, -1.2, z0 + 0.225), "plywood", "wood_dark",
              mass=30.0, floor=g, kind="furniture", label="TV unit", uv=1.0 / 0.4)
        b.box((1.05, 0.07, 0.62), (-1.6, -1.2, z0 + 0.76), "electronics", None,
              mass=9.0, floor=g, kind="contents", label="television",
              colour=(0.1, 0.1, 0.12), roughness=0.25)
        # altar shelf on the party wall: a tall unanchored cabinet with ornaments
        alt = b.cabinet_shell(0.9, 0.42, 1.7, (2.5, 1.0, z0), levels=3,
                              texture="wood_dark", label="altar cabinet", static=False)
        for z in alt:
            b.cyl(0.05, 0.16, (2.5, 1.0, z + 0.08), "ceramic", None, mass=0.6, floor=g,
                  kind="contents", label="urn", colour=(0.75, 0.25, 0.20))
            b.box((0.12, 0.08, 0.14), (2.25, 1.0, z + 0.07), "ceramic", None, mass=0.5,
                  floor=g, kind="contents", label="figurine", colour=(0.9, 0.75, 0.3))
        b.table(1.3, 0.8, 0.74, (0.8, 3.5, z0), "wood", "wood", mass=28.0, floor=g,
                label="dining table")
        for dx, hh in ((-0.5, 0), (0.5, 180)):
            b.chair((0.8 + dx, 3.5 + (0.7 if hh else -0.7), z0), (hh, 0, 0), floor=g)
        b.crockery_stack((0.8, 3.5, z0 + 0.742), count=4, r=0.10, floor=g)
        for i in range(3):
            b.cyl(0.034, 0.11, (0.4 + i * 0.2, 3.2, z0 + 0.797), "glass", None,
                  mass=0.24, floor=g, kind="contents", label="glass",
                  colour=(0.78, 0.88, 0.9))
        b.cyl(0.17, 1.05, (-2.5, 4.5, z0 + 0.525), "plastic", None, mass=22.0, floor=g,
              kind="furniture", label="water dispenser", colour=(0.85, 0.88, 0.9))
        b.cabinet(0.8, 0.45, 1.85, (2.4, 5.4, z0), "wood_light", mass=45.0, floor=g,
                  label="wardrobe")
        b.pendant_lamp((-1.6, -3.0, z0 + H1 - SLAB - 0.05), cord=0.7, mass=1.4,
                       floor=g, label="living room lamp",
                       anchor_body=self.slabs[(SHOP, 2)].body)
        # the staircase: a solid block along the party wall, stopping clear
        # of the wall, the column and the slab above (it was built through
        # all three, which is harmless while they stand and fires them
        # sideways the moment the collapse model releases them). It goes
        # with the ground storey: left as a fixed block it held the whole
        # released stack up on its own, since a rigid slab resting on any
        # fixed thing rests on it.
        st = self.structure
        stair_h = H0 - SLAB - 0.02
        stair = b.breakable((0.7, 3.6, stair_h), (2.5, -1.0, stair_h / 2),
                            "concrete", "screed", mass=2600.0, label="staircase",
                            uv=1.0 / 0.5)
        st.add(stair, "drift", 0.020, story=0, group=LIVING_GROUP, tag="staircase",
               crush=True)              # goes with the ground-storey columns

    # ------------------------------------------------------------------
    def _street(self, b: Builder) -> None:
        rng = b.rng
        b.decor((320, 320, 0.2), (0, 0, -0.42), "asphalt", uv=1.0 / 4.0)
        b.decor((NU * UW + 40, 1.2, 0.12), (0, -UD / 2 - WALKWAY - 0.6, 0.06), "concrete",
                colour=(0.6, 0.6, 0.58), uv=1.0 / 0.6)          # kerb / drain cover
        # the block opposite, and the rest of this row
        for sx in (-1, 1):
            b.decor((UW * 3, UD, LEVELS[-1] - 1.0), (sx * (NU * UW / 2 + UW * 1.5 + 0.3), 0,
                                                     (LEVELS[-1] - 1.0) / 2), "plaster_cream",
                    colour=(0.85, 0.80, 0.68), uv=1.0 / 2.0)
        for i in range(6):
            x = -22 + i * 9.0
            hh = rng.choice((7.0, 10.4, 13.6))
            b.decor((8.4, 14.0, hh), (x, -UD / 2 - 24.0, hh / 2), "plaster_cream",
                    colour=(0.78 + rng.random() * 0.15, 0.74 + rng.random() * 0.15,
                            0.62 + rng.random() * 0.15), uv=1.0 / 2.0)
        # a few cars on the street
        for i in range(4):
            b.box((4.3, 1.75, 1.45), (-14 + i * 7.5, -UD / 2 - WALKWAY - 4.0, 0.727),
                  "steel", "steel", mass=1300.0, kind="vehicle", label="car",
                  colour=((0.8, 0.8, 0.82), (0.15, 0.15, 0.18), (0.6, 0.1, 0.1),
                          (0.85, 0.85, 0.85))[i], roughness=0.35, metallic=0.6)

    # ------------------------------------------------------------------
    def update(self, t: float, shaker) -> None:
        super().update(t, shaker)
        # Once the shop's ceiling slab has gone the living room is on its
        # way down: nothing up there feels a storey response any more.
        if self.observer_slab.failed and not self._freed:
            self._freed = True
            for body in list(self.phys.dynamic):
                if body.home_floor == LIVING_GROUP:
                    self.phys.set_floor(body, 0)

    SNAPSHOT_ATTRS = ("_freed",)

    def on_reset(self) -> None:
        super().on_reset()
        self._freed = False

    def damage_report(self) -> list[str]:
        out = super().damage_report()
        gone = sum(1 for (s, i, j), e in self.columns.items() if s == 0 and e.failed)
        if gone:
            out.append(f"Soft storey: {gone} of {len(XB) * len(YB)} ground-floor "
                       f"columns failed")
        slabs = sum(1 for e in self.slabs.values() if e.failed)
        if slabs:
            out.append(f"{slabs} floor slabs down")
        return out
