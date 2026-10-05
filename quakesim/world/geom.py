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

"""Procedural geometry.

Boxes, cylinders, planes and spheres with correct normals, world-scaled UVs
(so a texture tiles at a fixed physical size) and *vertex* colours.

Why vertex colours matter here: the renderer batches geometry by render
state. Tinting an object with a colour attribute on its node makes every
distinct colour a distinct state and therefore a separate draw call, and a
supermarket with three hundred differently coloured bottles becomes three
hundred draw calls. Baking the tint into the vertices instead lets every
object that shares a texture share one draw call, which is the difference
between 3,500 draw calls a frame and about 30.
"""

from __future__ import annotations

import math
from functools import lru_cache

from panda3d.core import (Geom, GeomNode, GeomTriangles, GeomVertexData,
                          GeomVertexFormat, GeomVertexWriter, Material,
                          NodePath, TextureStage, TransparencyAttrib, Vec3,
                          Vec4)

from . import textures as tex

_FORMAT = GeomVertexFormat.getV3n3c4t2()
_WHITE = (1.0, 1.0, 1.0, 1.0)


def _rgba(colour) -> tuple:
    if colour is None:
        return _WHITE
    if len(colour) == 3:
        return (colour[0], colour[1], colour[2], 1.0)
    return tuple(colour)


def _writers(vdata):
    return (GeomVertexWriter(vdata, "vertex"),
            GeomVertexWriter(vdata, "normal"),
            GeomVertexWriter(vdata, "color"),
            GeomVertexWriter(vdata, "texcoord"))


def make_box(sx: float, sy: float, sz: float, uv_scale: float = 1.0,
             name: str = "box", colour=None) -> NodePath:
    """An axis-aligned box centred on the origin, UVs scaled to world size."""
    hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
    c = _rgba(colour)
    vdata = GeomVertexData(name, _FORMAT, Geom.UHStatic)
    vw, nw, cw, tw = _writers(vdata)
    tris = GeomTriangles(Geom.UHStatic)

    faces = [
        ((0, 0, 1), (-hx, -hy, hz), (1, 0, 0), (0, 1, 0), sx, sy),
        ((0, 0, -1), (-hx, hy, -hz), (1, 0, 0), (0, -1, 0), sx, sy),
        ((0, -1, 0), (-hx, -hy, -hz), (1, 0, 0), (0, 0, 1), sx, sz),
        ((0, 1, 0), (hx, hy, -hz), (-1, 0, 0), (0, 0, 1), sx, sz),
        ((-1, 0, 0), (-hx, hy, -hz), (0, -1, 0), (0, 0, 1), sy, sz),
        ((1, 0, 0), (hx, -hy, -hz), (0, 1, 0), (0, 0, 1), sy, sz),
    ]
    idx = 0
    for nrm, org, ua, va, ul, vl in faces:
        for (u, v) in ((0, 0), (1, 0), (1, 1), (0, 1)):
            vw.addData3(org[0] + ua[0] * ul * u + va[0] * vl * v,
                        org[1] + ua[1] * ul * u + va[1] * vl * v,
                        org[2] + ua[2] * ul * u + va[2] * vl * v)
            nw.addData3(*nrm)
            cw.addData4(*c)
            tw.addData2(u * ul * uv_scale, v * vl * uv_scale)
        tris.addVertices(idx, idx + 1, idx + 2)
        tris.addVertices(idx, idx + 2, idx + 3)
        idx += 4

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode(name)
    node.addGeom(geom)
    return NodePath(node)


def make_plane(sx: float, sy: float, uv_scale: float = 1.0,
               name: str = "plane", colour=None) -> NodePath:
    c = _rgba(colour)
    vdata = GeomVertexData(name, _FORMAT, Geom.UHStatic)
    vw, nw, cw, tw = _writers(vdata)
    tris = GeomTriangles(Geom.UHStatic)
    hx, hy = sx * 0.5, sy * 0.5
    for (x, y, u, v) in ((-hx, -hy, 0, 0), (hx, -hy, 1, 0),
                         (hx, hy, 1, 1), (-hx, hy, 0, 1)):
        vw.addData3(x, y, 0)
        nw.addData3(0, 0, 1)
        cw.addData4(*c)
        tw.addData2(u * sx * uv_scale, v * sy * uv_scale)
    tris.addVertices(0, 1, 2)
    tris.addVertices(0, 2, 3)
    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode(name)
    node.addGeom(geom)
    return NodePath(node)


def make_cylinder(radius: float, height: float, segments: int = 16,
                  uv_scale: float = 1.0, cap: bool = True,
                  name: str = "cyl", colour=None) -> NodePath:
    c = _rgba(colour)
    vdata = GeomVertexData(name, _FORMAT, Geom.UHStatic)
    vw, nw, cw, tw = _writers(vdata)
    tris = GeomTriangles(Geom.UHStatic)
    hz = height * 0.5
    circ = 2 * math.pi * radius

    idx = 0
    for i in range(segments):
        a0 = 2 * math.pi * i / segments
        a1 = 2 * math.pi * (i + 1) / segments
        for (a, zz, u, v) in ((a0, -hz, i / segments, 0), (a1, -hz, (i + 1) / segments, 0),
                              (a1, hz, (i + 1) / segments, 1), (a0, hz, i / segments, 1)):
            vw.addData3(math.cos(a) * radius, math.sin(a) * radius, zz)
            nw.addData3(math.cos(a), math.sin(a), 0)
            cw.addData4(*c)
            tw.addData2(u * circ * uv_scale, v * height * uv_scale)
        tris.addVertices(idx, idx + 1, idx + 2)
        tris.addVertices(idx, idx + 2, idx + 3)
        idx += 4

    if cap:
        for sign in (1, -1):
            centre = idx
            vw.addData3(0, 0, hz * sign)
            nw.addData3(0, 0, sign)
            cw.addData4(*c)
            tw.addData2(0.5, 0.5)
            idx += 1
            start = idx
            for i in range(segments + 1):
                a = 2 * math.pi * i / segments
                vw.addData3(math.cos(a) * radius, math.sin(a) * radius, hz * sign)
                nw.addData3(0, 0, sign)
                cw.addData4(*c)
                tw.addData2(math.cos(a) * radius * uv_scale,
                            math.sin(a) * radius * uv_scale)
                idx += 1
            for i in range(segments):
                if sign > 0:
                    tris.addVertices(centre, start + i, start + i + 1)
                else:
                    tris.addVertices(centre, start + i + 1, start + i)

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode(name)
    node.addGeom(geom)
    return NodePath(node)


def make_sphere(radius: float, rings: int = 10, sectors: int = 14,
                name: str = "sph", colour=None) -> NodePath:
    c = _rgba(colour)
    vdata = GeomVertexData(name, _FORMAT, Geom.UHStatic)
    vw, nw, cw, tw = _writers(vdata)
    tris = GeomTriangles(Geom.UHStatic)
    for r in range(rings + 1):
        phi = math.pi * r / rings
        for s in range(sectors + 1):
            th = 2 * math.pi * s / sectors
            x = math.sin(phi) * math.cos(th)
            y = math.sin(phi) * math.sin(th)
            z = math.cos(phi)
            vw.addData3(x * radius, y * radius, z * radius)
            nw.addData3(x, y, z)
            cw.addData4(*c)
            tw.addData2(s / sectors, 1 - r / rings)
    for r in range(rings):
        for s in range(sectors):
            a = r * (sectors + 1) + s
            b = a + sectors + 1
            tris.addVertices(a, b, a + 1)
            tris.addVertices(a + 1, b, b + 1)
    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode(name)
    node.addGeom(geom)
    return NodePath(node)


# ---------------------------------------------------------------------------
# Shared render states.  One Material object per (roughness, metallic) pair and
# one TextureStage for normal maps, so identical surfaces really are identical
# to the renderer and can be batched together.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=32)
def _material(roughness: float, metallic: float) -> Material:
    m = Material(f"pbr_r{roughness:.2f}_m{metallic:.2f}")
    m.setBaseColor(Vec4(1, 1, 1, 1))
    m.setRoughness(roughness)
    m.setMetallic(metallic)
    return m


@lru_cache(maxsize=1)
def _normal_stage() -> TextureStage:
    ts = TextureStage("normal")
    ts.setMode(TextureStage.MNormal)
    return ts


def _quantise(x: float, step: float = 0.1) -> float:
    return round(x / step) * step


def apply_material(np_: NodePath, texture: str | None = None,
                   colour: tuple | None = None, uv_scale: float = 1.0,
                   normal: bool = True, roughness: float | None = None,
                   metallic: float | None = None,
                   transparent: float | None = None) -> NodePath:
    """Texture and shade a node.

    Colour is expected to have been baked into the vertices already (every
    make_* function takes a `colour`); it is only applied as a state here for
    transparency, which genuinely needs its own render pass.
    """
    if texture:
        np_.setTexture(tex.get(texture), 1)
        if normal:
            nrm = tex.get_normal(texture)
            if nrm is not None:
                np_.setTexture(_normal_stage(), nrm)
    if transparent is not None:
        np_.setTransparency(TransparencyAttrib.MAlpha)
        c = _rgba(colour)
        np_.setColorScale(c[0], c[1], c[2], transparent)
    if roughness is not None or metallic is not None:
        np_.setMaterial(_material(_quantise(0.6 if roughness is None else roughness),
                                  _quantise(0.0 if metallic is None else metallic)), 1)
    return np_


def visual_for_body(body, parent: NodePath, texture: str | None = None,
                    colour: tuple | None = None, uv_scale: float = 1.0,
                    shape: str = "box", roughness: float | None = None,
                    metallic: float | None = None,
                    static_root: NodePath | None = None,
                    visual_size: tuple | None = None) -> NodePath:
    """Build and attach the renderable geometry for a physics body.

    Moving bodies get their geometry as a child, so it follows them. Bodies
    that never move get theirs placed once in a separate static tree that is
    later flattened into a handful of draw calls -- walls, floors, shelving
    and racking are most of the geometry in a scene and none of it moves.
    """
    sx, sy, sz = visual_size or body.size
    if shape == "cylinder":
        vis = make_cylinder(sx * 0.5, sz, 16, uv_scale, colour=colour)
    elif shape == "sphere":
        vis = make_sphere(sx * 0.5, colour=colour)
    else:
        vis = make_box(sx, sy, sz, uv_scale, colour=colour)
    apply_material(vis, texture, colour, uv_scale, True, roughness, metallic)

    if static_root is not None and body.anchored and not body.node.isKinematic():
        vis.reparentTo(static_root)
        vis.setPos(body.path.getPos())
        vis.setHpr(body.path.getHpr())
    else:
        vis.reparentTo(body.path)
    return vis
