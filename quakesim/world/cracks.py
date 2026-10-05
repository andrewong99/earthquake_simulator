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

"""Cracks.

Structural damage that stops short of collapse is mostly invisible in a
rigid-body world -- a frame at 1% drift looks exactly like a frame at 0%.
Real buildings tell you: diagonal shear cracks across infill walls, cracks
radiating from column heads, floor screed splitting along construction
joints. This draws them.

Crack textures are generated procedurally (a branching random walk), packed
into one atlas so every crack in a scene is a single draw call, and laid down
as thin decals on the surfaces a scene registers. How many appear, and where,
follows the storey's damage state from the structural solver: a few hairline
cracks at "slight", a web of them at "extensive".
"""

from __future__ import annotations

import math
import random

import numpy as np
from panda3d.core import (Geom, GeomNode, GeomTriangles, GeomVertexData,
                          GeomVertexFormat, GeomVertexWriter, NodePath,
                          SamplerState, Texture, TransparencyAttrib)

ATLAS_N = 2            # 2x2 variants
TILE = 256


def _walk(img, x, y, ang, steps, width, rng, depth=0):
    h, w = img.shape
    for _ in range(steps):
        ang += rng.uniform(-0.55, 0.55)
        x += math.cos(ang)
        y += math.sin(ang)
        xi, yi = int(x), int(y)
        if not (2 <= xi < w - 2 and 2 <= yi < h - 2):
            return
        r = max(1, int(width))
        img[yi - r:yi + r + 1, xi - r:xi + r + 1] = np.maximum(
            img[yi - r:yi + r + 1, xi - r:xi + r + 1], 1.0)
        if depth < 3 and rng.random() < 0.045:
            _walk(img, x, y, ang + rng.choice((-1, 1)) * rng.uniform(0.5, 1.2),
                  int(steps * 0.45), max(1.0, width * 0.7), rng, depth + 1)


def _crack_tile(seed: int) -> np.ndarray:
    rng = random.Random(seed)
    img = np.zeros((TILE, TILE), dtype=np.float32)
    # main fracture across the tile, a couple of branches, some fine spurs
    _walk(img, 10, rng.uniform(60, 200), rng.uniform(-0.4, 0.4), 240, 1.6, rng)
    for _ in range(rng.randint(1, 3)):
        _walk(img, rng.uniform(40, 220), rng.uniform(40, 220),
              rng.uniform(0, 2 * math.pi), 90, 1.0, rng, depth=1)
    # soften: dark core with a faint halo of crushed plaster
    from numpy import roll
    halo = np.zeros_like(img)
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            halo = np.maximum(halo, roll(roll(img, dx, 0), dy, 1) * 0.35)
    return np.clip(img + halo, 0.0, 1.0)


def crack_atlas() -> Texture:
    n = ATLAS_N
    size = TILE * n
    rgba = np.zeros((size, size, 4), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            a = _crack_tile(100 + i * n + j)
            core = (a > 0.9).astype(np.float32)
            y0, x0 = i * TILE, j * TILE
            rgba[y0:y0 + TILE, x0:x0 + TILE, 0] = 0.08 + 0.30 * (1 - core)
            rgba[y0:y0 + TILE, x0:x0 + TILE, 1] = 0.07 + 0.28 * (1 - core)
            rgba[y0:y0 + TILE, x0:x0 + TILE, 2] = 0.06 + 0.26 * (1 - core)
            rgba[y0:y0 + TILE, x0:x0 + TILE, 3] = a * 0.92
    bgra = np.empty((size, size, 4), dtype=np.uint8)
    bgra[..., 0] = (rgba[..., 2] * 255).astype(np.uint8)
    bgra[..., 1] = (rgba[..., 1] * 255).astype(np.uint8)
    bgra[..., 2] = (rgba[..., 0] * 255).astype(np.uint8)
    bgra[..., 3] = (rgba[..., 3] * 255).astype(np.uint8)
    tex = Texture("cracks")
    tex.setup2dTexture(size, size, Texture.TUnsignedByte, Texture.FRgba)
    tex.setRamImage(bgra[::-1].tobytes())
    tex.setMinfilter(SamplerState.FTLinearMipmapLinear)
    tex.setMagfilter(SamplerState.FTLinear)
    tex.setWrapU(SamplerState.WMClamp)
    tex.setWrapV(SamplerState.WMClamp)
    tex.setFormat(Texture.FSrgbAlpha)
    return tex


_ATLAS = None


def _atlas() -> Texture:
    global _ATLAS
    if _ATLAS is None:
        _ATLAS = crack_atlas()
    return _ATLAS


class Surface:
    """A rectangle cracks may appear on. `normal` is +z (floor), -z (ceiling),
    +x/-x/+y/-y (walls); `centre` and `size` describe the rectangle."""

    def __init__(self, centre, size, normal, story: int = 0, group: int = 0,
                 weight: float = 1.0, body=None):
        self.centre = tuple(centre)
        self.size = tuple(size)
        self.normal = normal
        self.story = story
        self.group = group
        self.weight = weight
        self.body = body            # the element this surface belongs to, if any

    @property
    def gone(self) -> bool:
        """The element carrying this surface has been released: its cracks
        must not hang in the air where the wall used to be."""
        b = self.body
        return b is not None and (getattr(b, "released", False) or getattr(b, "crushed", False))


class CrackLayer:
    LEVELS = (0, 3, 10, 24, 40)      # cracks per registered surface-weight unit

    def __init__(self, render: NodePath, seed: int = 4):
        self.rng = random.Random(seed)
        self.root = render.attachNewNode("cracks")
        self.root.setTransparency(TransparencyAttrib.MAlpha)
        self.root.setDepthOffset(2)
        self.root.setTexture(_atlas(), 1)
        self.root.setTwoSided(True)
        self.surfaces: list[Surface] = []
        self.placed: list[tuple] = []
        self.level_by_story: dict[tuple, int] = {}
        self._node: NodePath | None = None

    def register(self, surface: Surface) -> None:
        self.surfaces.append(surface)

    def reset(self) -> None:
        self.placed = []
        self.level_by_story = {}
        self._rebuild()

    def set_damage(self, group: int, story: int, level: int) -> bool:
        """Raise the damage level for a storey; returns True if cracks were added."""
        level = max(0, min(level, 4))
        key = (group, story)
        old = self.level_by_story.get(key, 0)
        if level <= old:
            return False
        self.level_by_story[key] = level
        added = 0
        for s in self.surfaces:
            if s.group != group or s.story != story:
                continue
            want = int(round(self.LEVELS[level] * s.weight))
            have = sum(1 for p in self.placed if p[0] is s)
            for _ in range(max(0, want - have)):
                self._place(s)
                added += 1
        if added:
            self._rebuild()
        return added > 0

    def _place(self, s: Surface) -> None:
        cx, cy, cz = s.centre
        sx, sy = s.size
        u = self.rng.uniform(-0.45, 0.45) * sx
        v = self.rng.uniform(-0.45, 0.45) * sy
        size = self.rng.uniform(0.35, 1.1) * min(1.6, max(sx, sy) * 0.35)
        rot = self.rng.uniform(0, math.pi)
        variant = self.rng.randrange(ATLAS_N * ATLAS_N)
        self.placed.append((s, u, v, size, rot, variant))

    def on_release(self) -> None:
        """Called when structural elements have been released: drop the
        decals that were on them."""
        if any(s.gone for s, *_ in self.placed):
            self._rebuild()

    def _rebuild(self) -> None:
        if self._node is not None:
            self._node.removeNode()
            self._node = None
        if not self.placed:
            return
        fmt = GeomVertexFormat.getV3n3c4t2()
        vdata = GeomVertexData("cracks", fmt, Geom.UHStatic)
        vw = GeomVertexWriter(vdata, "vertex")
        nw = GeomVertexWriter(vdata, "normal")
        cw = GeomVertexWriter(vdata, "color")
        tw = GeomVertexWriter(vdata, "texcoord")
        tris = GeomTriangles(Geom.UHStatic)
        idx = 0
        for s, u, v, size, rot, variant in self.placed:
            if s.gone:
                continue
            cx, cy, cz = s.centre
            n = s.normal
            # local axes on the surface
            if n in ("+z", "-z"):
                a, b = (1, 0, 0), (0, 1, 0)
                nrm = (0, 0, 1 if n == "+z" else -1)
                origin = (cx + u, cy + v, cz + (0.004 if n == "+z" else -0.004))
            elif n in ("+y", "-y"):
                a, b = (1, 0, 0), (0, 0, 1)
                nrm = (0, 1 if n == "+y" else -1, 0)
                origin = (cx + u, cy + (0.004 if n == "+y" else -0.004), cz + v)
            else:
                a, b = (0, 1, 0), (0, 0, 1)
                nrm = (1 if n == "+x" else -1, 0, 0)
                origin = (cx + (0.004 if n == "+x" else -0.004), cy + u, cz + v)
            ca, sa = math.cos(rot), math.sin(rot)
            ti, tj = divmod(variant, ATLAS_N)
            u0, v0 = tj / ATLAS_N, 1.0 - (ti + 1) / ATLAS_N
            u1, v1 = (tj + 1) / ATLAS_N, 1.0 - ti / ATLAS_N
            for (px, py, tu, tv) in ((-1, -1, u0, v0), (1, -1, u1, v0),
                                     (1, 1, u1, v1), (-1, 1, u0, v1)):
                rx = (px * ca - py * sa) * size * 0.5
                ry = (px * sa + py * ca) * size * 0.5
                vw.addData3(origin[0] + a[0] * rx + b[0] * ry,
                            origin[1] + a[1] * rx + b[1] * ry,
                            origin[2] + a[2] * rx + b[2] * ry)
                nw.addData3(*nrm)
                cw.addData4(1, 1, 1, 1)
                tw.addData2(tu, tv)
            tris.addVertices(idx, idx + 1, idx + 2)
            tris.addVertices(idx, idx + 2, idx + 3)
            idx += 4
        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode("crack-decals")
        node.addGeom(geom)
        self._node = self.root.attachNewNode(node)

    def count(self) -> int:
        return sum(1 for s, *_ in self.placed if not s.gone)
