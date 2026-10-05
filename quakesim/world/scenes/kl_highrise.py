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

"""Level 25 of a Kuala Lumpur condominium.

Peninsular Malaysia has almost no local seismicity, so KL's earthquake
experience is entirely second-hand: great ruptures on the Sumatran subduction
zone and the Sumatran Fault, 350 to 700 km away across the Strait of Malacca.

By the time that motion reaches the Klang Valley every trace of high frequency
has been absorbed, and what is left is long-period energy in the 2-6 second
band. That is unfortunate, because it is also the band that soft Klang Valley
alluvium amplifies, and the band that a 25-storey concrete tower is tuned to.
The result is the thing KL residents actually report: no bang, no rattle, just
water moving in a glass and a slow, nauseating sway that goes on for minutes.
Street level barely notices.

Set the quake to a megathrust at 500-700 km and watch the pendant lamp.
"""

from __future__ import annotations

import math

from ...seismo.site import SITES
from ...structure.mdof import Building
from ..cracks import Surface
from ..props import Builder
from ..scene_base import Scene, SceneSpec, Viewpoint, register

W, D, H = 13.5, 11.0, 2.95        # apartment internal size
LEVEL = 25
FLOOR_Z = 0.0
STOREY = 3.10
GROUND_Z = -LEVEL * STOREY - 0.3          # street level in the room's frame
TOWER_W, TOWER_D = W + 3.0, D + 3.0
RUBBLE_PER_STOREY = 7
FREEFALL_GROUP = 999                      # no structural response: ground motion


@register
class KLHighrise(Scene):
    spec = SceneSpec(
        key="kl_highrise",
        name="KL high-rise apartment, level 25",
        short="KL high-rise, level 25",
        blurb="A 25th-floor unit in a 25-storey reinforced-concrete tower on "
              "Klang Valley alluvium. Built for the far-field case: a great "
              "Sumatran earthquake several hundred kilometres away, whose "
              "long-period motion the soil and the building both amplify.",
        site=SITES["kl_alluvium"],
        building=Building("25-storey RC residential tower", stories=25,
                          story_height=3.10, floor_mass=520_000,
                          system="concrete_moment_frame", damping=0.03,
                          base_shear_coeff=0.045),
        default_distance_km=560.0,
        default_quake="megathrust",
        default_mw=8.8,
        default_depth_km=28.0,
        sky_time=17.4,
        story_of_floor={0: 24, 1: 24},
        observer_story=24,
        note="Tower T1 = 2.34 s; site period ~ 4.9 s. Compare the same "
             "earthquake seen from the ground-level viewpoint. A storey is "
             "crushed when its drift reaches 4% (±12%, HAZUS Complete for a "
             "concrete frame); everything above it comes down.",
        collapse_demo=(9.5, 6.0, "megathrust"),
        viewpoints=[
            Viewpoint("living", "Living room", (2.4, 3.6, 1.62), (143, -4, 0),
                      "walk", 70, "Sofa, coffee table, and the balcony beyond."),
            Viewpoint("balcony", "On the balcony", (0.0, -6.6, 1.62),
                      (180, -6, 0), "look", 74,
                      "78 m up, looking back at the city. Best place to feel "
                      "the sway against a fixed horizon."),
            Viewpoint("kitchen", "Kitchen", (4.6, 2.6, 1.60), (-120, -8, 0),
                      "walk", 66, "Where the crockery lives."),
            Viewpoint("bedroom", "Bedroom", (-4.4, 3.0, 1.60), (-40, 0, 0),
                      "walk", 66, ""),
            Viewpoint("outside", "Outside the tower, street level",
                      (0, -120.0, GROUND_Z + 1.7), (0, 16, 0), "walk", 66,
                      "The whole tower, from the ground. Notice how little "
                      "happens down here."),
            Viewpoint("across", "From the tower opposite",
                      (70.0, -90.0, -20.0), (38, -14, 0), "look", 62,
                      "Level with the upper storeys, 110 m away."),
            Viewpoint("top", "Above the tower", (0, 0, 150.0), (0, -90, 0),
                      "top", 60, ""),
        ],
    )

    def build(self, b: Builder) -> None:
        rng = b.rng
        st = self.structure
        self._rubble_out = {}
        self._crush_z = {}
        self._contents_freed = False
        self.register_system(0, "concrete_moment_frame")
        # --- the tower below, and the city --------------------------------
        self._tower_and_city(b)

        # --- the unit -----------------------------------------------------
        # Every part of the shell is a structural element: the floor rides
        # on the storey below and the walls and ceiling ride on the floor.
        shell = b.room_shell(W, D, H, (0, 0, FLOOR_Z), "tile", "plaster",
                             "plaster", 0.22, floor=0, open_sides=("s",),
                             breakable=True)
        below = self.storeys[-1]
        floor_el = st.add(shell["floor"], "support", depends=[below], need=1,
                          tag="apartment floor", floor_after=FREEFALL_GROUP)
        self.observer_body = shell["floor"]
        wall_els = []
        for k, body in shell.items():
            if k == "floor":
                continue
            wall_els.append(st.add(body, "drift", 0.04, story=LEVEL - 1, group=0,
                                   depends=[floor_el], need=1, tag="apartment wall",
                                   floor_after=FREEFALL_GROUP))
        st.add(self.roof, "support", depends=wall_els, need=2, tag="roof",
               floor_after=FREEFALL_GROUP)

        def fixed(body, tag="fixture"):
            """Anything built into the unit comes down with its floor."""
            return st.add(body, "support", depends=[floor_el], need=1, tag=tag,
                          floor_after=FREEFALL_GROUP)
        self._fixed = fixed
        self.cracks.register(Surface((0, 0, 0.0), (W, D), "+z", LEVEL - 1, 0, 1.2,
                                     body=shell["floor"]))
        self.cracks.register(Surface((0, D / 2 - 0.11, H / 2), (W, H), "-y",
                                     LEVEL - 1, 0, 1.0, body=shell["wall_n"]))
        self.cracks.register(Surface((-W / 2 + 0.11, 0, H / 2), (D, H), "+x",
                                     LEVEL - 1, 0, 0.8, body=shell["wall_w"]))
        self.cracks.register(Surface((W / 2 - 0.11, 0, H / 2), (D, H), "-x",
                                     LEVEL - 1, 0, 0.8, body=shell["wall_e"]))

        # balcony and its glazing
        fixed = self._fixed
        fixed(b.breakable((W * 0.62, 2.6, 0.18), (0, -D / 2 - 1.3, -0.09), "concrete",
                          "tile_grey", label="balcony slab"), "balcony")
        fixed(b.breakable((W * 0.62 - 0.22, 0.08, 1.10), (0, -D / 2 - 2.55, 0.55), "glass",
                          None, mass=60.0, label="balcony balustrade",
                          colour=(0.62, 0.76, 0.82)))
        for sx in (-1, 1):
            fixed(b.breakable((0.10, 2.50, 1.10), (sx * W * 0.31, -D / 2 - 0.08 - 1.25, 0.55),
                              "glass", None, mass=40.0, label="balcony glazing",
                              colour=(0.62, 0.76, 0.82)))
        # sliding door frame
        for sx in (-1, 1):
            fixed(b.breakable((W - W * 0.62, 0.14, H - 0.02),
                              (sx * (W / 2 - (W - W * 0.62) / 4 + 0.4), -D / 2, H / 2 - 0.01),
                              "plaster", "plaster", label="door frame"))

        # --- living room --------------------------------------------------
        b.box((2.30, 0.92, 0.42), (-2.6, -2.4, 0.21), "fabric", "carpet",
              mass=78.0, kind="furniture", label="sofa", uv=1.0 / 0.5)
        b.box((2.30, 0.28, 0.46), (-2.6, -2.0, 0.65), "fabric", "carpet",
              mass=22.0, kind="furniture", label="sofa back", uv=1.0 / 0.5)
        b.table(1.10, 0.58, 0.40, (-2.6, -0.6, 0), "plywood", "wood_dark",
                mass=18.0, label="coffee table")

        # A glass of water: the thing everyone remembers noticing first.
        b.cyl(0.036, 0.13, (-2.9, -0.6, 0.467), "glass", None, mass=0.42,
              kind="contents", label="glass of water", colour=(0.72, 0.86, 0.90))
        b.cyl(0.033, 0.10, (-2.35, -0.5, 0.452), "ceramic", None, mass=0.30,
              kind="contents", label="coffee mug", colour=(0.85, 0.86, 0.88))
        b.box((0.26, 0.19, 0.03), (-2.6, -0.78, 0.417), "paper", None, mass=0.9,
              kind="contents", label="magazine", colour=(0.60, 0.30, 0.28))

        # TV on a low unit: heavy, slender, unanchored -- a classic casualty
        b.box((1.70, 0.42, 0.46), (-2.6, 1.9, 0.23), "plywood", "wood_dark",
              mass=42.0, kind="furniture", label="TV unit", uv=1.0 / 0.4)
        b.box((1.26, 0.075, 0.74), (-2.6, 1.9, 0.832), "electronics", None,
              mass=11.5, kind="contents", label='55" television',
              colour=(0.10, 0.10, 0.12), roughness=0.25)

        # Tall bookcase. Depth 0.30 m against height 1.95 m means it tips
        # forwards at about 0.15 g -- which is why bookcases always fall
        # forwards, and why anchoring one to the wall is worth doing.
        shelves = b.cabinet_shell(0.92, 0.32, 1.95, (-6.1, 0.4, 0), levels=5,
                                  texture="wood_dark", label="bookcase",
                                  static=False)
        for zz in shelves:
            b.book_row(-6.48, 0.42, zz, count=8, height=0.26)

        for i in range(2):
            b.box((0.22, 0.03, 0.28), (-2.95 + i * 0.34, 1.82, 0.60), "plywood",
                  "wood_dark", mass=0.6, kind="contents",
                  label="framed photograph")
        b.cyl(0.085, 0.46, (1.62, -1.40, 0.752 + 0.23), "ceramic", None,
              mass=1.7, kind="contents", label="tall vase",
              colour=(0.52, 0.60, 0.58))
        b.cyl(0.16, 1.55, (-4.55, -2.95, 0.775), "steel", "steel", mass=7.5,
              kind="furniture", label="floor lamp", colour=(0.28, 0.29, 0.31))
        b.pendant_lamp((-2.6, -0.6, H - 0.05), cord=0.95, mass=1.6,
                       label="pendant lamp over table", anchor_body=shell["ceiling"])
        b.pendant_lamp((3.9, 2.2, H - 0.05), cord=0.55, mass=0.9,
                       label="kitchen pendant", anchor_body=shell["ceiling"])

        # --- dining -------------------------------------------------------
        b.table(1.55, 0.90, 0.75, (2.2, -1.4, 0), "plywood", "wood_light",
                mass=38.0, label="dining table")
        for i, (dx, dy, hh) in enumerate(((-0.55, -0.75, 0), (0.55, -0.75, 0),
                                          (-0.55, 0.75, 180), (0.55, 0.75, 180))):
            b.chair((2.2 + dx, -1.4 + dy, 0), (hh, 0, 0), label=f"dining chair {i+1}")
        b.crockery_stack((2.2, -1.4, 0.752), count=4, r=0.11)
        for i in range(3):
            b.cyl(0.035, 0.115, (1.75 + i * 0.22, -1.72, 0.810), "glass", None,
                  mass=0.28, kind="contents", label="drinking glass",
                  colour=(0.75, 0.87, 0.90))

        # --- kitchen ------------------------------------------------------
        fixed(b.breakable((3.6, 0.62, 0.90), (4.6, D / 2 - 0.55, 0.45), "plywood",
                          "wood_light", mass=120.0, label="kitchen base units",
                          uv=1.0 / 0.5))
        fixed(b.breakable((3.6, 0.64, 0.04), (4.6, D / 2 - 0.55, 0.92), "laminate",
                          "laminate", mass=30.0, label="worktop"))
        # wall cabinets full of crockery -- doors are not modelled, so the
        # contents behave as they would with the doors swinging open
        wall_shelves = b.cabinet_shell(3.6, 0.34, 0.72, (4.6, D / 2 - 0.40, 1.50),
                                       levels=2, texture="wood_light",
                                       label="wall cabinet", breakable=True)
        fixed(b.last_body, "wall cabinet")
        for zz in wall_shelves:
            for i in range(4):
                b.crockery_stack((3.4 + i * 0.55, D / 2 - 0.40, zz), count=4,
                                 r=0.105)
        for i in range(6):
            b.cyl(0.035, 0.12, (3.2 + i * 0.30, D / 2 - 0.62, 1.002), "glass",
                  None, mass=0.26, kind="contents", label="glass",
                  colour=(0.78, 0.88, 0.90))
        for i in range(4):
            b.bottle((5.75, D / 2 - 0.42 - i * 0.14, 0.942), r=0.036,
                     h=0.24 + i * 0.03, label="bottle")
        # fridge: tall, heavy, slender. b/h = 0.70/1.80 = 0.39
        b.box((0.70, 0.70, 1.80), (6.20, D / 2 - 2.4, 0.90), "steel", "steel",
              mass=88.0, kind="furniture", label="refrigerator",
              colour=(0.80, 0.81, 0.83), roughness=0.35, metallic=0.5)

        # --- bedroom ------------------------------------------------------
        b.box((1.55, 2.05, 0.45), (-4.6, 3.2, 0.225), "fabric", "carpet",
              mass=95.0, kind="furniture", label="bed", uv=1.0 / 0.6)
        b.cabinet(1.20, 0.58, 2.10, (-6.00, 1.60, 0), "wood", mass=95.0,
                  label="wardrobe (unanchored, 95 kg)")
        b.box((0.45, 0.40, 0.55), (-3.25, 4.60, 0.275), "plywood", "wood_dark",
              mass=12.0, kind="furniture", label="bedside table")
        b.cyl(0.055, 0.30, (-3.25, 4.60, 0.702), "ceramic", None, mass=0.9,
              kind="contents", label="table lamp", colour=(0.88, 0.84, 0.72))

        # --- ceiling and services: visuals riding the ceiling slab --------
        from .. import geom as g
        for i in range(-1, 2):
            v = g.make_box(0.30, 0.30, 0.06, 1.0, colour=(0.95, 0.95, 0.92))
            g.apply_material(v, "alu", (0.95, 0.95, 0.92), roughness=0.3)
            v.reparentTo(shell["ceiling"].path)
            v.setPos(i * 3.4, 3.4, -0.08)

    # ------------------------------------------------------------------
    def _tower_and_city(self, b: Builder) -> None:
        """The building we are standing in, and the skyline around it.

        The tower shaft is built storey by storey. Each storey is a
        structural element with the building's own drift rule: when the
        solver says a storey has drifted past what its columns can take it is
        crushed -- removed from the world and replaced by rubble -- and every
        storey above it, and the apartment, comes down through the gap.
        While the tower stands, each storey is placed at its own solved
        displacement relative to the apartment, so from the balcony the
        world sways beneath you and from the street the tower leans.
        """
        rng = b.rng
        st = self.structure
        self.storeys = []
        self.rubble = {}
        for s in range(LEVEL - 1):
            zc = GROUND_Z + (s + 0.5) * STOREY
            shade = 0.70 + 0.02 * (s % 2)
            blk = b.breakable((TOWER_W, TOWER_D, STOREY), (0, 0, zc), "concrete",
                              "concrete", mass=520_000.0, label=f"storey {s + 1}",
                              colour=(shade, shade - 0.02, shade - 0.05), uv=1.0 / 3.0)
            # slab edge band and window strips, riding on the storey body
            from .. import geom as g
            band = g.make_box(TOWER_W + 0.2, TOWER_D + 0.2, 0.22, 1.0 / 2.0,
                              colour=(0.58, 0.56, 0.53))
            g.apply_material(band, "concrete", None)
            band.reparentTo(blk.path)
            band.setPos(0, 0, STOREY / 2 - 0.11)
            for sy in (-1, 1):
                win = g.make_box(TOWER_W - 1.5, 0.06, 1.5, 1.0,
                                 colour=(0.30, 0.42, 0.48))
                g.apply_material(win, None, (0.30, 0.42, 0.48), roughness=0.15)
                win.reparentTo(blk.path)
                win.setPos(0, sy * (TOWER_D / 2 + 0.05), -0.2)
            dep = [self.storeys[-1]] if self.storeys else None
            el = st.add(blk, "drift", 0.04, story=s, group=0, depends=dep,
                        need=1, tag="storey", crush=True)
            self.storeys.append(el)
            # rubble for this storey, parked far below the city until needed
            pile = []
            for k in range(RUBBLE_PER_STOREY):
                # sized to fit one cell of the 3 x 3 grid the pile is laid
                # out on when the storey is crushed (see update)
                sz = (rng.uniform(3.0, 5.0), rng.uniform(2.5, 4.2), rng.uniform(0.5, 1.0))
                r = b.breakable(sz, ((k - RUBBLE_PER_STOREY / 2) * 8.0, 0.0,
                                     -600.0 - s * 3.0), "concrete",
                                "concrete", mass=2400.0 * sz[0] * sz[1] * sz[2] * 0.6,
                                label="rubble", colour=(0.62, 0.60, 0.56))
                pile.append(r)
            self.rubble[s] = pile
        # roof slab over the apartment: the top of the tower, and the last
        # thing to come down
        self.roof = b.breakable((W + 3.0, D + 3.0, 0.5), (0, 0, H + 0.4), "concrete",
                                "concrete", label="roof", colour=(0.66, 0.64, 0.61),
                                uv=1.0 / 2.0)

        # ground and neighbouring towers
        b.box((900, 900, 0.4), (0, 0, GROUND_Z - 0.5), "concrete", "asphalt",
              static=True, kind="structure", label="street", uv=1.0 / 8.0)
        rng2 = rng
        for i in range(46):
            a = rng2.uniform(0, 2 * math.pi)
            r = rng2.uniform(70, 380)
            hh = rng2.uniform(24, 180)
            w = rng2.uniform(14, 34)
            shade = 0.45 + rng2.random() * 0.30
            b.decor((w, w * rng2.uniform(0.7, 1.3), hh),
                    (math.cos(a) * r, math.sin(a) * r, GROUND_Z - 0.3 + hh / 2),
                    "concrete", colour=(shade, shade * 1.02, shade * 1.06),
                    uv=1.0 / 4.0)

    def update(self, t: float, shaker) -> None:
        super().update(t, shaker)
        resp = shaker.response
        gain = shaker.gain
        # Sway: each standing storey at its own displacement relative to
        # the apartment's, because the room is the frame we simulate in.
        if resp is not None and shaker.motion is not None:
            i = int(t / resp.dt)
            if 0 <= i < resp.floor_disp.shape[1]:
                top_x = float(resp.floor_disp[-1, i])
                top_y = (float(resp.floor_disp_y[-1, i])
                         if resp.floor_disp_y is not None else 0.0)
                for s, el in enumerate(self.storeys):
                    if el.failed:
                        continue
                    dx = (float(resp.floor_disp[s, i]) - top_x) * gain
                    dy = ((float(resp.floor_disp_y[s, i]) if resp.floor_disp_y is not None
                           else 0.0) - top_y) * gain
                    home = el.body.home[0]
                    el.body.path.setPos(home[0] + dx, home[1] + dy, home[2])
        # Progressive collapse. The lowest standing storey of the falling
        # stack lands on whatever is below with the whole tower above it
        # behind the impact -- far beyond what any storey carries -- and is
        # crushed in turn; the stack drops again onto the next. Landing is
        # read from the block: it has fallen a clear part of a storey (what
        # it lands on is a rubble layer up to a metre thick, so it never
        # drops a whole one) and its downward velocity has been arrested.
        for s, el in enumerate(self.storeys):
            body = el.body
            if body.crushed:
                continue
            if body.released:
                dropped = body.home[0][2] - body.path.getZ()
                vz = body.node.getLinearVelocity()[2]
                if dropped > STOREY * 0.4 and vz > -1.0:
                    self._crush_z[s] = float(body.path.getZ())
                    self.phys.crush(body)
            break                       # only the lowest standing storey
        # A crushed storey becomes rubble where it was crushed: one layer of
        # slabs on a 3 x 3 grid inside the volume the storey vacated, each
        # in its own cell so none is built into another or into the storey
        # now landing on it. (A column of slabs reaching into the storey
        # above was resolved by Bullet as a shove that sometimes held that
        # storey up short of the drop the crush rule asked for, and the
        # pancake stalled at one storey in one run out of three.)
        for s, el in enumerate(self.storeys):
            if not el.body.crushed or self._rubble_out.get(s):
                continue
            self._rubble_out[s] = True
            zc = self._crush_z.get(s, el.body.home[0][2])
            rng = self.builder.rng
            cells = [(i, j) for i in range(3) for j in range(3)]
            rng.shuffle(cells)
            cw, cd = TOWER_W / 3.0, TOWER_D / 3.0
            for r, (i, j) in zip(self.rubble[s], cells):
                r.path.setPos(-TOWER_W / 2 + cw * (i + 0.5) + rng.uniform(-0.08, 0.08),
                              -TOWER_D / 2 + cd * (j + 0.5) + rng.uniform(-0.08, 0.08),
                              zc - STOREY / 2 + r.size[2] / 2 + 0.05)
                r.path.setHpr(rng.uniform(-3, 3), 0.0, 0.0)
                self.phys.release(r, floor=0)
        # Once the floor has gone the room's contents are in free fall with
        # it: no storey response applies to them any more.
        if self.observer_body is not None and self.observer_body.released \
                and not self._contents_freed:
            self._contents_freed = True
            for body in list(self.phys.dynamic):
                if body.home_floor in (0, 1) and body is not self.observer_body:
                    self.phys.set_floor(body, FREEFALL_GROUP)

    SNAPSHOT_ATTRS = ("_rubble_out", "_crush_z", "_contents_freed")

    def on_reset(self) -> None:
        super().on_reset()
        self._rubble_out = {}
        self._crush_z = {}
        self._contents_freed = False

    def damage_report(self) -> list[str]:
        out = super().damage_report()
        crushed = [s + 1 for s, el in enumerate(self.storeys) if el.body.crushed]
        if crushed:
            out.append(f"Storey {', '.join(str(c) for c in crushed)} crushed -- "
                       f"tower down from there")
        elif self.observer_body is not None and self.observer_body.released:
            out.append("Apartment floor has gone")
        return out
