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

"""Distribution warehouse with pallet racking.

The classic earthquake failure of a modern warehouse is not the building --
portal frames are light and flexible and ride it out. It is the racking.
Loaded pallet racks carry an enormous mass three, four, five metres up on
slender uprights, and they are far more flexible than the shed around them.
They sway further, they sway slower, and a rack that goes over takes its
neighbours with it.

The racks here are driven by their own structural solution, not the building's,
so they visibly lag and whip relative to the floor.
"""

from __future__ import annotations

import math

from ...seismo.site import SITES
from ...structure.mdof import Building
from ..props import Builder
from ..scene_base import Scene, SceneSpec, Viewpoint, register

W, D, H = 46.0, 30.0, 11.0        # clear internal dimensions

# Each rack beam level is its own group of bodies, driven by the rack's own
# structural response rather than the building's floor. Ids start above any
# building floor id so the two never collide.
RACK_FLOOR_BASE = 100
RESP = 99                          # floor group carrying the shed's response


@register
class Warehouse(Scene):
    spec = SceneSpec(
        key="warehouse",
        name="Distribution warehouse",
        short="Warehouse",
        blurb="Steel portal-frame shed, 46 x 30 m, with six runs of loaded "
              "pallet racking 7.2 m high. The racking is the story here: it "
              "is far more flexible than the building and carries most of the "
              "mass well above the floor.",
        site=SITES["stiff_soil"],
        building=Building("Portal-frame shed", stories=1, story_height=11.0,
                          floor_mass=260_000, system="steel_moment_frame",
                          damping=0.04, period_override=0.55),
        default_distance_km=14.0,
        default_quake="reverse",
        default_mw=6.6,
        sky_time=11.0,
        # Floor group 0 stands on the slab and feels the ground; the mezzanine
        # (group 1) rides the shed's response; group RESP carries the shed's
        # response for the frame's own failure rules.
        story_of_floor={1: 0, 99: 0},
        observer_story=-1,          # you stand on the ground floor
        note="The racking is solved as its own 5-level structure (T = 1.05 s "
             "down-aisle, about twice the shed's 0.55 s), driven by the floor "
             "motion. Everything stored on a beam level feels that level's "
             "motion, not the floor's. A rack bay lets go when any level "
             "drifts 5% (±12%); the portal frame at 6%.",
        collapse_demo=(7.2, 6.0, "reverse"),
        viewpoints=[
            Viewpoint("aisle", "Standing in the aisle", (-12.0, -4.75, 1.65),
                      (-90, -2, 0), "walk", 70,
                      "Between two loaded racks - where you would not want to be."),
            Viewpoint("aisle_end", "End of the aisle", (-20.5, 3.45, 1.65),
                      (-90, 0, 0), "walk", 70, "Looking down the full rack run."),
            Viewpoint("mezz", "Mezzanine office window", (20.0, -12.0, 6.0),
                      (59, -10, 0), "look", 62, "Over the racks from the office."),
            Viewpoint("high", "Roof truss level", (0, 0, 9.6), (0, -35, 0),
                      "look", 75, "Watching the whole floor from the trusses."),
            Viewpoint("outside", "Outside, loading yard", (0, -46.0, 1.7),
                      (0, 4, 0), "walk", 65, "The building from the yard."),
            Viewpoint("top", "Directly overhead", (0, 0, 60.0), (0, -90, 0),
                      "top", 55, "Plan view of the whole floor."),
        ],
    )

    # The racking, as a structure in its own right. Loaded pallet racking is
    # slender, lightly damped and carries most of its mass high up, so it is
    # far more flexible than the portal frame around it: about 1 s down-aisle
    # against the shed's 0.55 s. That mismatch is why racks fail in
    # earthquakes that leave the building itself untouched.
    RACK = Building("Loaded pallet racking (down-aisle)", stories=5,
                    story_height=1.45, floor_mass=2_400,
                    system="steel_rack", damping=0.03,
                    period_override=1.05)

    def attach_structures(self, shaker, motion, building_response,
                          progress=None) -> None:
        """Solve the rack response and bind the beam levels to it."""
        from ...structure import mdof as _mdof
        if building_response is not None:
            base_x = building_response.floor_acc[0]
            base_y = (building_response.floor_acc_y[0]
                      if building_response.floor_acc_y is not None
                      else motion.acc[1])
        else:
            base_x, base_y = motion.acc[0], motion.acc[1]
        pr = (lambda f: progress(f, "Solving the racking response")) if progress else None
        self.rack_response = _mdof.solve_2d(self.RACK, base_x, base_y, motion.dt,
                                            progress=pr)
        shaker.attach({RACK_FLOOR_BASE + lv: lv for lv in range(self.RACK.stories)},
                      self.rack_response)

    def build(self, b: Builder) -> None:
        rng = b.rng
        self.rack_response = None
        self.bays = []
        # --- shell --------------------------------------------------------
        b.box((W + 1.0, D + 1.0, 0.35), (0, 0, -0.175), "concrete", "concrete",
              static=True, kind="structure", label="slab", uv=1.0 / 1.2)
        # the yard: a real surface for anything that comes off the building
        b.box((260, 260, 0.3), (0, 0, -0.36), "concrete", "asphalt", static=True,
              kind="structure", label="yard", uv=1.0 / 3.0)

        st = self.structure
        self.register_system(RESP, "steel_moment_frame")
        # Portal frames: seven, from one end wall to the other. Columns fail
        # by storey drift (steel moment frame, HAZUS "Complete" 6%); the
        # rafter goes when either of its columns goes; roof and wall sheeting
        # goes with the frame it hangs on, or tears off on its own at 5%.
        fx = [-(W / 2 - 0.4) + i * (W - 0.8) / 6 for i in range(7)]
        cy = D / 2 - 0.4
        ch = H - 0.06                       # clear of the roof sheets above
        frames = []
        for x in fx:
            cols = []
            for sy in (-1, 1):
                c = b.breakable((0.34, 0.34, ch), (x, sy * cy, ch / 2), "steel",
                                "steel", mass=1900.0, label="portal column",
                                roughness=0.4, metallic=0.9)
                cols.append(st.add(c, "drift", 0.06, story=0, group=RESP,
                                   tag="portal column"))
            r = b.breakable((0.30, 2 * cy - 0.34, 0.9), (x, 0, H - 0.5), "steel",
                            "steel", mass=2600.0, label="rafter",
                            roughness=0.4, metallic=0.9)
            frames.append((cols, st.add(r, "support", depends=cols, need=1,
                                        tag="rafter")))
        for j in range(-6, 7):
            b.decor((W, 0.16, 0.16), (0, j * 2.3, H - 1.1), "steel",
                    roughness=0.5, metallic=0.9)

        edges = [-(W / 2 - 0.15)] + fx[1:-1] + [W / 2 - 0.15]   # inside the end walls
        for i in range(6):
            x0, x1 = edges[i], edges[i + 1]
            xc, xw = (x0 + x1) / 2, x1 - x0
            # roof: a sheet per bay per slope, resting on the rafters at each end
            rafters = [frames[i][1], frames[i + 1][1]]
            for sy in (-1, 1):
                p = b.breakable((xw, (D + 1.0) / 2, 0.25),
                                (xc, sy * (D + 1.0) / 4, H + 0.135), "galv_steel",
                                "corrugated", mass=xw * (D + 1) / 2 * 14.0,
                                label="roof sheet", uv=1.0 / 2.0)
                st.add(p, "drift", 0.05, story=0, group=RESP, depends=rafters,
                       need=1, tag="roof sheet")
            # side walls: a sheet per bay, hung on the columns at each end
            for k, sy in enumerate((-1, 1)):
                cols = [frames[i][0][k], frames[i + 1][0][k]]
                p = b.breakable((xw, 0.30, H), (xc, sy * (D / 2), H / 2),
                                "galv_steel", "corrugated", mass=xw * H * 12.0,
                                label="wall sheet", uv=1.0 / 1.0)
                st.add(p, "drift", 0.05, story=0, group=RESP, depends=cols,
                       need=1, tag="wall sheet")
        # end walls: six sheets each, hung on that end's frame
        for sx, fr in ((-1, frames[0]), (1, frames[-1])):
            for k in range(-3, 3):
                y = (k + 0.5) * (D + 1.0) / 6
                p = b.breakable((0.30, (D + 1.0) / 6, H), (sx * (W / 2), y, H / 2),
                                "galv_steel", "corrugated",
                                mass=(D + 1.0) / 6 * H * 12.0, label="wall sheet",
                                uv=1.0 / 1.0)
                st.add(p, "drift", 0.05, story=0, group=RESP, depends=fr[0],
                       need=1, tag="wall sheet")

        # high-bay lights and sprinkler main
        for i in range(-2, 3):
            for j in (-1, 1):
                b.decor((0.55, 0.55, 0.18), (i * 9.0, j * 8.0, H - 1.5),
                        "alu", colour=(0.95, 0.95, 0.9), roughness=0.3)
        b.decor((W, 0.10, 0.10), (0, 0, H - 1.3), "steel", colour=(0.75, 0.15, 0.12))

        # --- racking ------------------------------------------------------
        # Six runs, three aisles. Bay: 2.7 m wide, 1.1 m deep, 5 beam levels.
        self.racks = []
        bay_w, bay_d, lift = 2.75, 1.10, 1.45
        levels = 5
        for run, ry in enumerate((-9.5, -8.2, -1.3, 0.0, 6.9, 8.2)):
            back_to_back = run % 2 == 1
            nbay = max(3, int(6 * b.detail ** 0.6))
            # the two southern runs stop short of the mezzanine office
            # (x > 13.5) whatever the detail level
            hi = min(nbay, 3) if ry < -7.0 else nbay
            shared = None          # adjacent bays share an upright frame
            for bay in range(-nbay, hi + 1):
                bx = bay * bay_w
                shared = self._rack_bay(b, bx, ry, bay_w, bay_d, lift, levels,
                                        rng, fill=0.86, run=run, left=shared)

        # cross-aisle floor stock and pallet stacks, on slots 1.7 m apart so
        # two can never be built through each other (random x used to do
        # exactly that at some detail levels, and the overlap resolved by
        # firing cartons across the shed at load)
        slots = [(x, y) for y in (-13.5, 13.5, 3.4)
                 for x in [-19.5 + i * 1.7 for i in range(24)]
                 if not (y < -12.0 and x > 11.0)]        # forklift and office
        rng.shuffle(slots)
        for x, y in slots[:max(4, int(14 * b.detail))]:
            b.pallet((x, y, 0.0), label="floor pallet")
            zz = b.pallet_top((x, y, 0.0))
            for m in range(rng.randint(1, 3)):
                sz = (0.55 + rng.random() * 0.2, 0.42 + rng.random() * 0.15,
                      0.34 + rng.random() * 0.2)
                b.carton(sz, (x + rng.uniform(-0.15, 0.15),
                              y + rng.uniform(-0.12, 0.12), zz),
                         label="stock carton")
                zz += sz[2] + 0.002

        # drums, and an IBC tote
        for k in range(max(3, int(9 * b.detail))):
            b.drum((-21.0 + (k % 3) * 0.62, 12.0 + (k // 3) * 0.62, 0.0),
                   full=rng.random() > 0.3)
        b.box((1.20, 1.00, 1.16), (18.5, 12.5, 0.58), "plastic", "plastic",
              mass=1080.0, kind="contents", label="IBC tote (1000 L)",
              colour=(0.85, 0.88, 0.86))

        # forklift (heavy, low, will not go over - a useful contrast)
        b.box((2.30, 1.15, 1.20), (14.0, -13.0, 0.60), "steel", "steel",
              mass=2600.0, kind="vehicle", label="forklift",
              colour=(0.85, 0.55, 0.06), roughness=0.5)
        b.decor((0.14, 0.14, 3.4), (15.0, -13.4, 1.7), "steel", metallic=0.9)
        b.decor((0.14, 0.14, 3.4), (15.0, -12.6, 1.7), "steel", metallic=0.9)

        # mezzanine office
        # (the slab and its glazing stop 3 cm clear of the portal columns
        # at y = -14.6, which are released in a collapse)
        b.box((9.0, 6.9, 0.22), (18.0, -10.95, 4.4), "concrete", "screed",
              static=True, kind="structure", label="mezzanine", uv=1.0 / 0.8)
        for sx, sy in ((13.8, -14.2), (13.8, -7.8), (22.2, -14.2), (22.2, -7.8)):
            b.decor((0.20, 0.20, 4.4), (sx, sy, 2.2), "steel", metallic=0.9)
        b.box((9.0, 0.12, 1.1), (18.0, -14.3, 5.05), "glass", None,
              static=True, kind="structure", label="office glazing",
              colour=(0.55, 0.72, 0.78))
        b.table(1.6, 0.8, 0.74, (17.0, -11.0, 4.51), "plywood", "wood_light",
                mass=28, floor=1, label="office desk")
        b.chair((17.0, -12.0, 4.51), floor=1)
        for i in range(5):
            b.box((0.30, 0.24, 0.06), (17.0, -10.9 + i * 0.005, 5.28 + i * 0.061),
                  "paper", None, mass=2.0, floor=1, kind="contents",
                  label="box file", colour=(0.55, 0.45, 0.35))

        b.pendant_lamp((18.0, -9.0, 6.6), cord=0.75, mass=1.1, floor=1,
                       label="office pendant")

    # ------------------------------------------------------------------
    def update(self, t: float, shaker) -> None:
        super().update(t, shaker)
        # Once a bay has gone its load is on the floor, or on its way there,
        # and should feel the floor's motion rather than a rack level's.
        for bay in self.bays:
            if bay["down"]:
                continue
            if any(e.failed for e in bay["uprights"]):
                bay["down"] = True
                for body in bay["contents"]:
                    self.phys.set_floor(body, 0)

    def on_reset(self) -> None:
        super().on_reset()
        for bay in self.bays:
            bay["down"] = False

    def snapshot_state(self) -> dict:
        d = super().snapshot_state()
        d["bays_down"] = [bool(bay["down"]) for bay in self.bays]
        return d

    def restore_state(self, d: dict) -> None:
        super().restore_state(d)
        for bay, down in zip(self.bays, d["bays_down"]):
            bay["down"] = down

    def damage_report(self) -> list[str]:
        out = super().damage_report()
        down = sum(1 for bay in self.bays if bay["down"])
        if down:
            out.append(f"Racking: {down} of {len(self.bays)} bays collapsed")
        return out

    def _rack_bay(self, b: Builder, x, y, bw, bd, lift, levels, rng,
                  fill=0.85, run=0, left=None):
        """One bay of pallet racking: two uprights, beam pairs, loaded pallets.

        The uprights and beams are kinematic -- driven by the rack's own
        structural response -- while everything on them is free. Adjacent
        bays share an upright frame: `left` is the previous bay's right-hand
        upright element, and the new right-hand one is returned. (Building
        both uprights per bay put two frames in the same place, and the
        90 mm of interpenetration was resolved, on release, as a shove.)
        """
        colour = (0.78, 0.34, 0.08) if run % 2 == 0 else (0.13, 0.29, 0.55)
        tex = "rack_orange" if run % 2 == 0 else "rack_blue"
        top = lift * levels
        st = self.structure
        from .. import geom as g

        # The frame is driven by the rack's own structural response while it
        # stands. When the bottom level's drift reaches what the beam-to-
        # upright connectors and baseplates can take (5%, between HAZUS
        # "Extensive" 3.5% and "Complete" 7% for storage racks) the uprights
        # are released with their real mass, the beams follow, and the bay
        # comes down under its load.
        uprights = [left] if left is not None else []
        for sx in ((1,) if left is not None else (-1, 1)):
            px = x + sx * bw / 2
            up = self.phys.add_kinematic_box(
                (0.09, bd, top + 0.25), (px, y, (top + 0.25) / 2),
                "steel", floor=RACK_FLOOR_BASE + levels // 2, kind="rack",
                label="rack upright", drift_gain=1.0, mass=95.0)
            v = g.make_box(0.09, bd, top + 0.25, 1.0 / 0.4, colour=colour)
            g.apply_material(v, tex, None, roughness=0.45, metallic=0.4)
            v.reparentTo(up.path)
            uprights.append(st.add(up, "drift", 0.05, story=-1,
                                   group=RACK_FLOOR_BASE, tag="rack bay"))

        contents = []
        n0 = len(self.phys.dynamic)
        for lv in range(levels):
            z = 0.28 + lv * lift
            for dy in (-0.30, 0.30):
                bm = self.phys.add_kinematic_box(
                    (bw - 0.09, 0.10, 0.10), (x, y + dy, z), "steel",
                    floor=RACK_FLOOR_BASE + lv, kind="rack",
                    label="rack beam", drift_gain=1.0, mass=16.0)
                v = g.make_box(bw - 0.09, 0.10, 0.10, 1.0 / 0.3, colour=colour)
                g.apply_material(v, tex, None, roughness=0.45, metallic=0.4)
                v.reparentTo(bm.path)
                st.add(bm, "support", depends=uprights, need=1, tag="rack beam")

            if rng.random() > fill:
                continue
            for slot in (-1, 1):
                px = x + slot * 0.66
                base = z + 0.05                     # top of the beam pair
                lvl = RACK_FLOOR_BASE + lv
                b.pallet((px, y, base), floor=lvl, label="rack pallet")
                load = b.pallet_top((px, y, base))  # real load surface
                kind = rng.random()
                if kind < 0.55:
                    # three layers at most: four reach the beam of the level
                    # above (0.146 + 4 x 0.302 = 1.35 m under a 1.40 m clear
                    # height, less the pallet's seating) and are built into it
                    nz = max(1, min(3, int(rng.randint(2, 4) * min(b.detail, 1.0))))
                    nx, ny = (2, 2) if b.detail >= 0.9 else (1, 2)
                    for ix in range(nx):
                        for iy in range(ny):
                            for iz in range(nz):
                                b.carton((0.40, 0.34, 0.30),
                                         (px - 0.20 + ix * 0.41,
                                          y - 0.17 + iy * 0.35,
                                          load + iz * 0.302),
                                         floor=lvl, label="palletised carton")
                elif kind < 0.80:
                    b.box((1.05, 0.72, 0.95), (px, y, load + 0.475),
                          "carton_full", "cardboard", mass=320.0, floor=lvl,
                          kind="contents", label="shrink-wrapped load",
                          uv=1.0 / 0.4)
                else:
                    for ix in range(2 if b.detail >= 0.6 else 1):
                        b.drum((px - 0.30 + ix * 0.60, y, load), floor=lvl,
                               full=rng.random() > 0.4)
        contents = self.phys.dynamic[n0:]
        self.bays.append({"uprights": uprights, "contents": contents, "down": False})
        return uprights[-1]
