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

"""Object library.

Every item carries its real dimensions and its real mass. Those two numbers,
plus the friction of the surface it stands on, are the whole story of what
happens to it in an earthquake -- nothing here is animated or scripted.

Masses are for the object *as it would be found*: a filled carton, a bottle
with liquid in it, a wardrobe with clothes. Hollow objects get an explicit
mass rather than density times bounding volume, because a ceramic bowl is not
a solid block of ceramic.
"""

from __future__ import annotations

import math
import random

from panda3d.bullet import BulletSphericalConstraint
from panda3d.core import NodePath, Point3, Vec3

from . import geom
from ..sim.physics import PhysicsWorld


class Builder:
    """Creates physics bodies together with their visuals."""

    def __init__(self, phys: PhysicsWorld, root: NodePath, seed: int = 4,
                 detail: float = 1.0):
        self.phys = phys
        self.root = root
        self.rng = random.Random(seed)
        self.detail = max(0.15, float(detail))
        self.groups: dict[str, list] = {}

    def spacing(self, step: float) -> float:
        """Widen the spacing between stocked items at lower detail levels."""
        return step / self.detail

    def keep(self, base_probability: float = 1.0) -> bool:
        return self.rng.random() < base_probability * self.detail

    # -- primitives --------------------------------------------------------
    def box(self, size, pos, material="plastic", texture=None, mass=None,
            hpr=(0, 0, 0), static=False, anchored=False, floor=0, kind="prop",
            label="", colour=None, uv=1.0, roughness=None, metallic=None):
        b = self.phys.add_box(size, pos, material, mass=mass, hpr=hpr,
                              floor=floor, kind=kind, static=static,
                              anchored=anchored, label=label)
        geom.visual_for_body(b, self.root, texture or material, colour, uv,
                             "box", roughness, metallic,
                             static_root=self.phys.static_visual_root)
        self.groups.setdefault(kind, []).append(b)
        return b

    def cyl(self, radius, height, pos, material="glass", texture=None, mass=None,
            hpr=(0, 0, 0), static=False, floor=0, kind="prop", label="",
            colour=None, uv=1.0, roughness=None, metallic=None):
        b = self.phys.add_cylinder(radius, height, pos, material, mass=mass,
                                   hpr=hpr, floor=floor, kind=kind,
                                   static=static, label=label)
        geom.visual_for_body(b, self.root, texture or material, colour, uv,
                             "cylinder", roughness, metallic,
                             static_root=self.phys.static_visual_root)
        self.groups.setdefault(kind, []).append(b)
        return b

    def sphere(self, radius, pos, material="produce", mass=None, floor=0,
               kind="contents", label="", colour=None):
        b = self.phys.add_sphere(radius, pos, material, mass=mass, floor=floor,
                                 kind=kind, label=label)
        geom.visual_for_body(b, self.root, None, colour or (0.6, 0.65, 0.3),
                             1.0, "sphere",
                             static_root=self.phys.static_visual_root)
        self.groups.setdefault(kind, []).append(b)
        return b

    def decor(self, size, pos, texture, hpr=(0, 0, 0), colour=None, uv=1.0,
              shape="box", roughness=None, metallic=None, transparent=None,
              moving: bool = False):
        """Pure visual, no physics -- walls seen from outside, sky props, etc.

        Static by default, which lets it be flattened with the rest of the
        fixed geometry. Pass `moving=True` for anything the scene animates.
        """
        if shape == "cylinder":
            v = geom.make_cylinder(size[0] * 0.5, size[2], 18, uv, colour=colour)
        elif shape == "plane":
            v = geom.make_plane(size[0], size[1], uv, colour=colour)
        else:
            v = geom.make_box(*size, uv, colour=colour)
        geom.apply_material(v, texture, colour, uv, True, roughness, metallic,
                            transparent)
        v.reparentTo(self.root if moving else self.phys.static_visual_root)
        v.setPos(*pos)
        v.setHpr(*hpr)
        return v

    # -- structural shell --------------------------------------------------
    def room_shell(self, w, d, h, pos=(0, 0, 0), floor_tex="tile",
                   wall_tex="plaster", ceiling_tex="ceiling_tile",
                   thickness=0.20, floor=0, open_sides=(), ceiling=True,
                   breakable=False) -> dict:
        """Floor slab, four walls and a ceiling, all rigid.

        With `breakable` every part is a structural element the collapse
        model may release; the parts are returned by name either way.
        """
        x0, y0, z0 = pos
        out = {}

        def part(size, p, material, texture, label, uv, mass=None):
            if breakable:
                return self.breakable(size, p, material, texture, mass=mass,
                                      floor=floor, label=label, uv=uv)
            return self.box(size, p, material, texture, static=True, floor=floor,
                            kind="structure", label=label, uv=uv)

        out["floor"] = part((w, d, thickness), (x0, y0, z0 - thickness / 2),
                            "concrete", floor_tex, "floor", 1.0 / 0.6)
        if ceiling:
            out["ceiling"] = part((w, d, 0.12), (x0, y0, z0 + h + 0.06), "concrete",
                                  ceiling_tex, "ceiling", 1.0 / 0.6)
        # east/west walls sit between the north/south walls rather than
        # through them: once these can be released, an overlap at the corner
        # is an impulse waiting to happen
        walls = {
            "n": ((w, thickness, h), (x0, y0 + d / 2, z0 + h / 2)),
            "s": ((w, thickness, h), (x0, y0 - d / 2, z0 + h / 2)),
            "e": ((thickness, d - 2 * thickness - 0.002, h), (x0 + w / 2, y0, z0 + h / 2)),
            "w": ((thickness, d - 2 * thickness - 0.002, h), (x0 - w / 2, y0, z0 + h / 2)),
        }
        for k, (size, p) in walls.items():
            if k in open_sides:
                continue
            out[f"wall_{k}"] = part(size, p, "plaster", wall_tex, f"wall_{k}",
                                    1.0 / 0.8, mass=1500.0 * size[0] * size[1] * size[2])
        return out

    # -- fixtures ----------------------------------------------------------
    def pendant_lamp(self, pos, cord=0.9, bob_r=0.13, mass=1.4, floor=0,
                     label="pendant lamp", anchor_body=None):
        """A hanging lamp on a ball joint.

        The single most honest instrument in the room: its swing period is
        2*pi*sqrt(L/g), so it picks out ground motion near its own period and
        ignores everything else. A distant megathrust sets it swinging in
        slow arcs; a sharp local quake barely disturbs it.

        With `anchor_body` -- a ceiling slab that may be released -- the lamp
        hangs from that body and comes down with it.
        """
        x, y, z = pos
        if anchor_body is None:
            anchor = self.phys.add_box((0.05, 0.05, 0.05), (x, y, z), "steel",
                                       static=True, floor=floor, kind="fixture",
                                       label="lamp anchor")
            pivot_a = Point3(0, 0, 0)
        else:
            anchor = anchor_body
            hx, hy, hz = anchor_body.home[0]
            pivot_a = Point3(x - hx, y - hy, z - hz)
        bob = self.phys.add_sphere(bob_r, (x, y, z - cord), "plastic",
                                   mass=mass, floor=floor, kind="fixture",
                                   label=label)
        v = geom.make_sphere(bob_r, colour=(0.95, 0.92, 0.80))
        geom.apply_material(v, None, None, roughness=0.3)
        v.reparentTo(bob.path)
        c = BulletSphericalConstraint(anchor.node, bob.node,
                                      pivot_a, Point3(0, 0, cord))
        self.phys.world.attachConstraint(c)
        bob.node.setLinearDamping(0.02)
        bob.node.setAngularDamping(0.05)
        cordvis = geom.make_cylinder(0.005, cord, 6, colour=(0.15, 0.15, 0.15))
        cordvis.reparentTo(self.root)
        self.groups.setdefault("pendant", []).append(
            (bob, cordvis, (x, y, z), cord, anchor, pivot_a))
        return bob

    def update_pendants(self):
        for bob, cordvis, anchor_pos, cord, anchor, pivot in self.groups.get("pendant", []):
            p = bob.path.getPos()
            if anchor is not None and anchor.mass > 0.0 and not anchor.anchored:
                a = anchor.path.getMat().xformPoint(pivot)
                anchor_pos = (a[0], a[1], a[2])
            mid = ((anchor_pos[0] + p[0]) / 2, (anchor_pos[1] + p[1]) / 2,
                   (anchor_pos[2] + p[2]) / 2)
            cordvis.setPos(*mid)
            cordvis.lookAt(p[0], p[1], p[2])
            cordvis.setP(cordvis.getP() + 90)

    def ceiling_grid(self, w, d, z, pos=(0, 0), tile=0.6, floor=0,
                     drop_when: float = 0.45, avoid=()):
        """Suspended ceiling made of individually simulated tiles.

        Lay-in tiles are not fixed to anything -- they simply rest in the grid,
        which is why they are among the first things to come down. `avoid`
        lists (x, y, half_size) squares -- columns -- that get no tile.
        """
        x0, y0 = pos
        tile = tile / max(self.detail, 0.2) ** 0.5
        nx, ny = max(int(w / tile), 1), max(int(d / tile), 1)
        # Snap the grid to a whole number of tiles so every tile has a tee on
        # all four edges; a partial bay at the end leaves one unsupported.
        w, d = nx * tile, ny * tile
        tee_w, tee_h, th = 0.026, 0.035, 0.016

        # Tees first: their top face is the plane the tiles lie on. A tee
        # stops short of a column and starts again beyond it, as a real grid
        # is trimmed -- a tee run straight through a column is 200 mm of
        # interpenetration that fires the column sideways the moment the
        # collapse model releases it.
        def runs(lo, hi, cuts):
            segs, a = [], lo
            for c0, c1 in sorted(cuts):
                if c0 > a:
                    segs.append((a, min(c0, hi)))
                a = max(a, c1)
            if a < hi:
                segs.append((a, hi))
            return [(p, q) for p, q in segs if q - p > 0.05]

        for i in range(nx + 1):
            x = x0 - w / 2 + tile * i
            cuts = [(ay - hs - 0.02, ay + hs + 0.02) for ax, ay, hs in avoid
                    if abs(x - ax) < hs + tee_w / 2 + 0.02]
            for ya, yb in runs(y0 - d / 2, y0 + d / 2, cuts):
                self.box((tee_w, yb - ya, tee_h), (x, (ya + yb) / 2, z - tee_h / 2),
                         "alu", "alu", static=True, floor=floor, kind="structure",
                         label="ceiling tee")
        for j in range(ny + 1):
            y = y0 - d / 2 + tile * j
            cuts = [(ax - hs - 0.02, ax + hs + 0.02) for ax, ay, hs in avoid
                    if abs(y - ay) < hs + tee_w / 2 + 0.02]
            for xa, xb in runs(x0 - w / 2, x0 + w / 2, cuts):
                self.box((xb - xa, tee_w, tee_h), ((xa + xb) / 2, y, z - tee_h / 2),
                         "alu", "alu", static=True, floor=floor, kind="structure",
                         label="ceiling tee")

        # Tiles are cut slightly oversize so each edge lands on its tee, which
        # is exactly how a lay-in tile is held: by gravity and nothing else.
        tiles = []
        for i in range(nx):
            for j in range(ny):
                px = x0 - w / 2 + tile * (i + 0.5)
                py = y0 - d / 2 + tile * (j + 0.5)
                if any(abs(px - ax) < hs + tile / 2 and abs(py - ay) < hs + tile / 2
                       for ax, ay, hs in avoid):
                    continue
                t = self.box((tile - tee_w + 0.022, tile - tee_w + 0.022, th),
                             (px, py, z + th / 2 + 0.001),
                             "mineral_fibre", "ceiling_tile", mass=1.6,
                             floor=floor, kind="ceiling_tile",
                             label="ceiling tile", uv=1.0 / tile)
                tiles.append(t)
        return tiles

    # -- furniture ---------------------------------------------------------
    def compound(self, parts, pos, material="wood", texture="wood", mass=10.0,
                 hpr=(0, 0, 0), floor=0, kind="furniture", label="",
                 uv=1.0, colour=None, roughness=None, metallic=None):
        """One rigid body assembled from several boxes, with matching visuals."""
        b = self.phys.add_compound(parts, pos, material, mass=mass, hpr=hpr,
                                   floor=floor, kind=kind, label=label)
        cx, cy, cz = b.com_offset
        for size, off in parts:
            v = geom.make_box(*size, uv, colour=colour)
            geom.apply_material(v, texture or material, colour, uv, True,
                                roughness, metallic)
            v.reparentTo(b.path)
            v.setPos(off[0] - cx, off[1] - cy, off[2] - cz)
        self.groups.setdefault(kind, []).append(b)
        return b

    def breakable_compound(self, parts, pos, material="steel", texture=None,
                           mass=100.0, hpr=(0, 0, 0), floor=0, kind="structure",
                           label="", uv=1.0, colour=None, roughness=None,
                           metallic=None, floor_after=0):
        """A compound that stands as structure until released -- a gondola
        run that is bolted to nothing, a chiller cabinet, a rack of shelves."""
        b = self.phys.add_compound(parts, pos, material, mass=0.0, hpr=hpr,
                                   floor=floor, kind=kind, static=True,
                                   label=label)
        b.breakable = True
        b.break_mass = float(mass)
        b.path.reparentTo(self.phys._root)
        self.phys.breakable.append(b)
        cx, cy, cz = b.com_offset
        for size, off in parts:
            v = geom.make_box(*size, uv, colour=colour)
            geom.apply_material(v, texture or material, colour, uv, True,
                                roughness, metallic)
            v.reparentTo(b.path)
            v.setPos(off[0] - cx, off[1] - cy, off[2] - cz)
        self.groups.setdefault(kind, []).append(b)
        return b

    def table(self, w, d, h, pos, material="wood", texture="wood", mass=32.0,
              floor=0, label="table"):
        """A table as one rigid body.

        Top plus four legs. Its footprint is the leg spread and its centre of
        mass sits just under the top, which is what makes a tall narrow table
        tip and a low wide one merely slide.
        """
        leg, t = 0.06, 0.04
        parts = [((w, d, t), (0, 0, h - t / 2))]
        for sx in (-1, 1):
            for sy in (-1, 1):
                parts.append(((leg, leg, h - t),
                              (sx * (w / 2 - leg), sy * (d / 2 - leg),
                               (h - t) / 2)))
        return self.compound(parts, pos, material, texture, mass=mass,
                             floor=floor, label=label, uv=1.0 / 0.5)

    def chair(self, pos, hpr=(0, 0, 0), material="wood", texture="wood",
              floor=0, label="chair"):
        """A chair as one rigid body: seat, back and four legs, about 6 kg."""
        parts = [((0.44, 0.44, 0.05), (0, 0, 0.425)),
                 ((0.44, 0.04, 0.44), (0, -0.20, 0.67))]
        for sx in (-1, 1):
            for sy in (-1, 1):
                parts.append(((0.04, 0.04, 0.40),
                              (sx * 0.19, sy * 0.19, 0.20)))
        return self.compound(parts, pos, material, texture, mass=6.2, hpr=hpr,
                             floor=floor, label=label, uv=1.0 / 0.4)

    def table_top_z(self, pos, h) -> float:
        return pos[2] + h + 0.002

    def cabinet(self, w, d, h, pos, texture="wood", mass=None, floor=0,
                anchored=False, label="cabinet"):
        """A solid tall cabinet with nothing stored inside it.

        Its slenderness w/h is what decides whether it goes over: a 0.92 m
        wide bookcase 1.95 m tall tips at 0.47 g, a 0.30 m deep one at 0.15 g
        across its depth -- which is why bookcases always fall forwards.
        """
        x, y, z = pos
        if mass is None:
            mass = 22.0 + 60.0 * (w * d * h)
        return self.box((w, d, h), (x, y, z + h / 2), "plywood", texture,
                        mass=mass, floor=floor, kind="furniture",
                        anchored=anchored, label=label, uv=1.0 / 0.5)

    def cabinet_shell(self, w, d, h, pos, levels=3, texture="wood", floor=0,
                      static=True, label="cabinet", panel=0.018,
                      open_front=True, mass=None, anchored=False,
                      breakable=False):
        """An open carcass: back, sides, top, base and shelves.

        Pass `static=False` for free-standing furniture and it is built as ONE
        rigid body, so it can rock, slide and go over with everything still on
        its shelves -- which is how a display cabinet actually fails, and why
        crockery inside one is far more at risk than crockery on a fixed
        counter. Wall-hung and built-in units stay static, which is the
        default.

        Returns the shelf surface heights so contents sit on real surfaces.
        """
        x, y, z = pos
        # The back and sides stand ON the base panel rather than reaching the
        # floor beside it. Measured: a free-standing carcass whose base, back
        # and both sides all touch the floor rocks at 0.07 rad/s under 10% g
        # (five overlapping contact patches fighting in the solver) and
        # anything on its shelves rattles; with one patch it is perfectly
        # still.
        hh = h - panel
        parts = [
            ((w, panel, hh), (0, d / 2 - panel / 2, panel + hh / 2)),        # back
            ((panel, d, hh), (-(w / 2 - panel / 2), 0, panel + hh / 2)),     # side
            ((panel, d, hh), (w / 2 - panel / 2, 0, panel + hh / 2)),        # side
            ((w, d, panel), (0, 0, h - panel / 2)),                          # top
            ((w, d, panel), (0, 0, panel / 2)),                              # base
        ]
        if not open_front:
            parts.append(((w, panel, hh), (0, -(d / 2 - panel / 2), panel + hh / 2)))

        surfaces = []
        inner = h - panel
        for i in range(levels):
            zz = panel + i * (inner - panel) / max(levels, 1)
            parts.append(((w - 2 * panel, d - panel, panel), (0, 0, zz - panel / 2)))
            surfaces.append(z + zz + 0.002)

        self.last_body = None
        if breakable:
            # one static compound the collapse model may release (a wall
            # cabinet that comes down with the wall it hangs on)
            if mass is None:
                mass = 14.0 + 210.0 * (w * d * h)
            self.last_body = self.breakable_compound(
                parts, (x, y, z), "plywood", texture, mass=mass, floor=floor,
                label=label, uv=1.0 / 0.4)
        elif static or anchored:
            kw = dict(floor=floor, kind="structure", static=True, uv=1.0 / 0.4)
            for size, off in parts:
                self.box(size, (x + off[0], y + off[1], z + off[2]),
                         "plywood", texture, label=f"{label} panel", **kw)
        else:
            if mass is None:
                # Carcass timber only; the contents add their own mass.
                mass = 14.0 + 210.0 * (w * d * h)
            self.compound(parts, (x, y, z), "plywood", texture, mass=mass,
                          floor=floor, kind="furniture", label=label,
                          uv=1.0 / 0.4)
        return surfaces

    # -- structure that can fail --------------------------------------------
    def breakable(self, size, pos, material="concrete", texture=None, mass=None,
                  hpr=(0, 0, 0), floor=0, kind="structure", label="",
                  colour=None, uv=1.0, roughness=None, metallic=None,
                  floor_after=0):
        """A structural element the collapse model may release."""
        b = self.phys.add_breakable(size, pos, material, mass=mass, hpr=hpr,
                                    floor=floor, kind=kind, label=label,
                                    floor_after=floor_after)
        geom.visual_for_body(b, self.root, texture or material, colour, uv,
                             "box", roughness, metallic, static_root=None)
        self.groups.setdefault(kind, []).append(b)
        return b

    def wall_panels(self, start, end, z0, height, thickness=0.15, rows=3,
                    cols=None, material="brick", texture="plaster",
                    mass_per_m2=None, floor=0, label="wall", colour=None,
                    uv=1.0 / 0.7, panel_len=1.5):
        """A wall built from a grid of breakable panels between two points.

        Returns the panels row by row, bottom first. Panel mass comes from the
        material density unless `mass_per_m2` is given (useful for framed
        walls, which weigh far less than their bounding box of brick would).
        """
        x0, y0 = start
        x1, y1 = end
        length = math.hypot(x1 - x0, y1 - y0)
        if cols is None:
            cols = max(1, int(round(length / panel_len)))
        heading = math.degrees(math.atan2(y1 - y0, x1 - x0))
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        pl, ph = length / cols, height / rows
        out = []
        for r in range(rows):
            row = []
            for c in range(cols):
                cx = x0 + ux * pl * (c + 0.5)
                cy = y0 + uy * pl * (c + 0.5)
                cz = z0 + ph * (r + 0.5)
                mass = None
                if mass_per_m2 is not None:
                    mass = mass_per_m2 * pl * ph
                row.append(self.breakable((pl * 0.995, thickness, ph * 0.995),
                                          (cx, cy, cz), material, texture,
                                          mass=mass, hpr=(heading, 0, 0),
                                          floor=floor, label=f"{label} panel",
                                          colour=colour, uv=uv))
            out.append(row)
        return out

    def column(self, pos, height, size=0.40, material="concrete",
               texture="concrete", floor=0, label="column", colour=None):
        x, y, z = pos
        return self.breakable((size, size, height), (x, y, z + height / 2),
                              material, texture, floor=floor, label=label,
                              colour=colour, uv=1.0 / 0.5)

    def slab(self, w, d, pos, thickness=0.18, material="concrete",
             texture="concrete", floor=0, label="slab", colour=None, uv=1.0 / 1.0,
             mass=None):
        x, y, z = pos
        return self.breakable((w, d, thickness), (x, y, z + thickness / 2),
                              material, texture, mass=mass, floor=floor,
                              label=label, colour=colour, uv=uv)

    @staticmethod
    def slope_frame(slope_deg, heading_deg):
        """Unit vectors (up-slope in plan, across, surface normal) and the
        Panda HPR that lays a box flat on that slope."""
        hd, sl = math.radians(heading_deg), math.radians(slope_deg)
        u = (math.cos(hd), math.sin(hd))
        v = (-math.sin(hd), math.cos(hd))
        n = (-math.sin(sl) * u[0], -math.sin(sl) * u[1], math.cos(sl))
        hpr = (heading_deg - 90.0, slope_deg, 0.0)
        return u, v, n, hpr

    def slope_point(self, centre, slope_deg, heading_deg, a, c, lift=0.0):
        """World point at distance `a` up the slope and `c` across, plus
        `lift` along the surface normal."""
        u, v, n, _ = self.slope_frame(slope_deg, heading_deg)
        sl = math.radians(slope_deg)
        return (centre[0] + u[0] * a * math.cos(sl) + v[0] * c + n[0] * lift,
                centre[1] + u[1] * a * math.cos(sl) + v[1] * c + n[1] * lift,
                centre[2] + a * math.sin(sl) + n[2] * lift)

    def boulder_field(self, centre, along, across, slope_deg, heading_deg,
                      count, floor=0, friction=0.80, seed=None):
        """Loose rock resting on a slope.

        At rest until the shaking exceeds its yield acceleration (Newmark
        1965): for a block on a slope of angle b with friction angle f,
        k_y = tan(f - b). Friction is scattered per rock so the slope lets
        go progressively rather than as one sheet.
        """
        rng = random.Random(seed if seed is not None else self.rng.random())
        u, v, n, hpr = self.slope_frame(slope_deg, heading_deg)
        out = []
        n_along = max(2, int(math.sqrt(count * along / max(across, 1e-3))))
        n_across = max(2, int(math.ceil(count / n_along)))
        pitch_a, pitch_c = along / n_along, across / n_across
        for i in range(n_along):
            for j in range(n_across):
                if len(out) >= count:
                    break
                # Flatter than they are wide: a rock whose height exceeds its
                # base tips at tan(slope) = b/h, and on a 30-degree slope that
                # is b/h = 0.58 -- it would go over at rest. Keeping b/h >= 1.3
                # means the rocks slide when the shaking reaches k_y, which is
                # the Newmark picture this field exists to show.
                w = rng.uniform(0.40, 1.0)
                d = rng.uniform(0.40, 0.9)
                sz = (w, d, min(w, d) * rng.uniform(0.45, 0.75))
                a = (i + 0.5) * pitch_a - along / 2 + rng.uniform(-0.08, 0.08) * pitch_a
                c = (j + 0.5) * pitch_c - across / 2 + rng.uniform(-0.08, 0.08) * pitch_c
                pos = self.slope_point(centre, slope_deg, heading_deg, a, c,
                                       sz[2] / 2 + 0.006)
                shade = rng.uniform(0.38, 0.62)
                b = self.box(sz, pos, "brick", "concrete",
                             mass=2400.0 * sz[0] * sz[1] * sz[2] * 0.85,
                             hpr=hpr, floor=floor, kind="rock", label="rock",
                             colour=(shade, shade * 0.93, shade * 0.82), uv=2.0)
                mu = friction * rng.uniform(0.92, 1.08)
                b.node.setFriction(math.sqrt(mu))
                b.peak_mu = mu
                # Wake at half the Newmark yield acceleration, not at the
                # level-ground sliding threshold, which on a slope is far
                # too high: a rock at k_y = 0.035 g must not sleep through
                # a 0.1 g pulse.
                k_y = math.tan(max(math.atan(mu) - math.radians(slope_deg), 0.005))
                b.wake_threshold = 0.5 * k_y * 9.81
                out.append(b)
        return out

    # -- contents ----------------------------------------------------------
    def bottle(self, pos, r=0.036, h=0.30, material="bottle_glass", mass=None,
               floor=0, label="bottle", colour=None):
        if mass is None:
            mass = 1.05 * math.pi * r ** 2 * h * 1000
        return self.cyl(r, h, (pos[0], pos[1], pos[2] + h / 2), material,
                        material, mass=mass, floor=floor, kind="contents",
                        label=label, colour=colour)

    def can(self, pos, r=0.033, h=0.115, floor=0, label="can"):
        return self.cyl(r, h, (pos[0], pos[1], pos[2] + h / 2), "can", "alu",
                        mass=0.40, floor=floor, kind="contents", label=label,
                        metallic=0.9, roughness=0.35)

    def carton(self, size, pos, mass=None, floor=0, label="carton", hpr=(0, 0, 0)):
        if mass is None:
            mass = 320.0 * size[0] * size[1] * size[2]
        return self.box(size, (pos[0], pos[1], pos[2] + size[2] / 2),
                        "carton_full", "cardboard", mass=mass, hpr=hpr,
                        floor=floor, kind="contents", label=label,
                        uv=1.0 / 0.35)

    def book_row(self, x0, y, z, count=10, height=0.24, floor=0):
        out = []
        x = x0
        for i in range(count):
            t = 0.028 + self.rng.random() * 0.022
            h = height * (0.85 + self.rng.random() * 0.3)
            x += t / 2 + (0.0015 if i else 0.0)
            out.append(self.box((t, 0.16, h), (x, y, z + h / 2 + 0.002),
                                "paper", None, mass=0.45,
                                colour=(0.25 + self.rng.random() * 0.5,
                                        0.20 + self.rng.random() * 0.5,
                                        0.20 + self.rng.random() * 0.5),
                                floor=floor, kind="contents", label="book"))
            x += t / 2
        return out

    def disc(self, radius, height, pos, material="ceramic", texture=None,
             mass=None, floor=0, kind="contents", label="", colour=None):
        """A plate, a lid, a coaster: drawn as a cylinder, collided as a box.

        Stacked cylinders are one of Bullet's weak spots: two flat cylinder
        faces meet through the general convex algorithm, which yields one
        contact point a step and so almost no resistance to spinning. A
        four-plate stack under 10% g shaking spins at 2 rad/s and walks off
        the shelf; the same plates as boxes sleep. The box is 85% of the
        diameter on a side, so its corners reach 20% past the rim and its
        edges sit 15% inside it -- invisible for a plate, and a plate is what
        this is for.
        """
        side = 2.0 * radius * 0.85
        b = self.phys.add_box((side, side, height), pos, material, mass=mass,
                              floor=floor, kind=kind, label=label)
        geom.visual_for_body(b, self.root, texture or material, colour, 1.0,
                             "cylinder", static_root=self.phys.static_visual_root,
                             visual_size=(2 * radius, 2 * radius, height))
        self.groups.setdefault(kind, []).append(b)
        return b

    def crockery_stack(self, pos, count=6, r=0.11, floor=0):
        """A stack of plates as ONE rigid body.

        Stacks of three or more free plates are numerically unstable under
        sub-threshold shaking -- measured: four free plates on a fixed shelf
        walk 37 cm in a minute at 15% g, where one plate does not move at
        all -- and a real stack of glazed plates slides and tips as a unit
        anyway, since plate-on-plate friction is no lower than plate-on-
        shelf. So the stack is one compound of plate-shaped boxes, drawn
        as plates, and it goes over or off the shelf as a stack.
        """
        side = 2.0 * r * 0.85
        parts = []
        for i in range(count):
            parts.append(((side, side, 0.022), (0, 0, 0.011 + i * 0.024)))
        mass = 0.34 * count
        b = self.phys.add_compound(parts, (pos[0], pos[1], pos[2] + 0.002), "ceramic",
                                   mass=mass, floor=floor, kind="contents",
                                   label=f"stack of {count} plates")
        cx, cy, cz = b.com_offset
        for size, off in parts:
            v = geom.make_cylinder(r, 0.022, 16, 1.0, colour=(0.93, 0.92, 0.89))
            geom.apply_material(v, None, (0.93, 0.92, 0.89), 1.0, True, 0.25, 0.0)
            v.reparentTo(b.path)
            v.setPos(off[0] - cx, off[1] - cy, off[2] - cz)
        self.groups.setdefault("contents", []).append(b)
        return [b]

    PALLET_H = 0.144

    def pallet(self, pos, floor=0, label="pallet"):
        """A EUR pallet: 1200 x 800 x 144 mm, about 25 kg.

        `pos[2]` is the surface it stands on. The whole pallet is one rigid
        body of its true overall height -- the deck boards and bearers are
        welded together in practice, and modelling them separately used to let
        loads settle through the gap between them.
        """
        x, y, z = pos
        h = self.PALLET_H
        deck = self.phys.add_box((1.20, 0.80, h), (x, y, z + h / 2), "wood",
                                 mass=25.0, floor=floor, kind="pallet",
                                 label=label)
        from . import geom as _g
        top = _g.make_box(1.20, 0.80, 0.022, 1.0 / 0.3)
        _g.apply_material(top, "wood")
        top.reparentTo(deck.path)
        top.setZ(h / 2 - 0.011)
        for dy in (-0.34, 0.0, 0.34):
            v = _g.make_box(1.20, 0.10, 0.10, 1.0 / 0.3)
            _g.apply_material(v, "wood")
            v.reparentTo(deck.path)
            v.setPos(0, dy, -h / 2 + 0.05)
        self.groups.setdefault("pallet", []).append(deck)
        return deck

    def pallet_top(self, pos) -> float:
        """Height of the load surface of a pallet standing at `pos`."""
        return pos[2] + self.PALLET_H + 0.002

    def drum(self, pos, floor=0, full=True, label="205 L drum"):
        """A standard 205 litre steel drum: 0.585 m dia, 0.88 m tall."""
        mass = 205.0 if full else 18.0
        return self.cyl(0.2925, 0.88, (pos[0], pos[1], pos[2] + 0.44),
                        "steel", "steel", mass=mass, floor=floor,
                        kind="contents", label=label, metallic=0.85,
                        roughness=0.45, colour=(0.20, 0.35, 0.55))
