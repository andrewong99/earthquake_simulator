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

"""Procedurally generated textures.

Everything the simulator looks like is generated in code at start-up -- there
are no image files to ship or download. Each surface produces an albedo map
and a matching normal map derived from the same height field, so lighting
reacts to the actual relief of the material rather than a flat colour.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from panda3d.core import SamplerState, Texture

RNG = np.random.default_rng(20260910)


# ---------------------------------------------------------------------------
# Noise helpers
# ---------------------------------------------------------------------------
def _value_noise(n: int, cells: int, rng=None) -> np.ndarray:
    """Tileable smooth value noise on an n x n grid."""
    rng = rng or RNG
    g = rng.random((cells, cells))
    g = np.vstack([g, g[:1]])
    g = np.hstack([g, g[:, :1]])
    ys = np.linspace(0, cells, n, endpoint=False)
    xs = np.linspace(0, cells, n, endpoint=False)
    y0 = ys.astype(int); x0 = xs.astype(int)
    fy = (ys - y0)[:, None]; fx = (xs - x0)[None, :]
    sy = fy * fy * (3 - 2 * fy); sx = fx * fx * (3 - 2 * fx)
    a = g[np.ix_(y0, x0)]; b = g[np.ix_(y0, x0 + 1)]
    c = g[np.ix_(y0 + 1, x0)]; d = g[np.ix_(y0 + 1, x0 + 1)]
    return (a * (1 - sx) * (1 - sy) + b * sx * (1 - sy)
            + c * (1 - sx) * sy + d * sx * sy)


def _fbm(n: int, octaves: int = 5, base: int = 4, gain: float = 0.5,
         rng=None) -> np.ndarray:
    out = np.zeros((n, n)); amp = 1.0; tot = 0.0; cells = base
    for _ in range(octaves):
        out += amp * _value_noise(n, cells, rng)
        tot += amp; amp *= gain; cells *= 2
        if cells > n:
            break
    return out / max(tot, 1e-9)


def _normal_from_height(h: np.ndarray, strength: float = 2.0) -> np.ndarray:
    dx = np.roll(h, -1, 1) - np.roll(h, 1, 1)
    dy = np.roll(h, -1, 0) - np.roll(h, 1, 0)
    nx = -dx * strength; ny = -dy * strength
    nz = np.ones_like(h)
    ln = np.sqrt(nx * nx + ny * ny + nz * nz)
    out = np.stack([nx / ln, ny / ln, nz / ln], axis=-1)
    return (out * 0.5 + 0.5)


def _to_texture(rgb: np.ndarray, name: str, srgb: bool = True,
                alpha: np.ndarray | None = None) -> Texture:
    """numpy HxWx3 float 0..1 -> Panda3D texture (Panda wants BGRA bytes)."""
    h, w = rgb.shape[:2]
    a = np.ones((h, w), dtype=np.float32) if alpha is None else alpha
    bgra = np.empty((h, w, 4), dtype=np.uint8)
    c = np.clip(rgb, 0.0, 1.0)
    bgra[..., 0] = (c[..., 2] * 255).astype(np.uint8)
    bgra[..., 1] = (c[..., 1] * 255).astype(np.uint8)
    bgra[..., 2] = (c[..., 0] * 255).astype(np.uint8)
    bgra[..., 3] = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    tex = Texture(name)
    tex.setup2dTexture(w, h, Texture.TUnsignedByte, Texture.FRgba)
    tex.setRamImage(bgra[::-1].tobytes())
    tex.generateRamMipmapImages()
    tex.setMinfilter(SamplerState.FTLinearMipmapLinear)
    tex.setMagfilter(SamplerState.FTLinear)
    tex.setAnisotropicDegree(8)
    tex.setWrapU(SamplerState.WMRepeat)
    tex.setWrapV(SamplerState.WMRepeat)
    if srgb:
        tex.setFormat(Texture.FSrgbAlpha)
    return tex


def _tint(h: np.ndarray, base, variation=0.10) -> np.ndarray:
    base = np.asarray(base, dtype=float)
    return np.clip(base[None, None, :] * (1.0 + (h[..., None] - 0.5) * 2 * variation),
                   0, 1)


# ---------------------------------------------------------------------------
# Individual materials -- each returns (albedo HxWx3, height HxW)
# ---------------------------------------------------------------------------
def _concrete(n):
    rng = np.random.default_rng(1)
    h = _fbm(n, 6, 6, 0.55, rng)
    speck = (rng.random((n, n)) > 0.995).astype(float)
    h = h * 0.8 + speck * 0.2
    alb = _tint(h, (0.60, 0.595, 0.575), 0.14)
    stain = _fbm(n, 3, 2, 0.6, np.random.default_rng(11))
    alb *= (0.86 + 0.14 * stain)[..., None]
    return alb, h * 0.35


def _screed(n):
    rng = np.random.default_rng(2)
    h = _fbm(n, 5, 8, 0.5, rng)
    alb = _tint(h, (0.50, 0.50, 0.49), 0.10)
    return alb, h * 0.2


def _plaster(n, colour=(0.90, 0.885, 0.855)):
    rng = np.random.default_rng(3)
    h = _fbm(n, 5, 10, 0.45, rng)
    alb = _tint(h, colour, 0.05)
    return alb, h * 0.12


def _floor_tile(n, tiles=4, grout=(0.42, 0.41, 0.39), colour=(0.80, 0.775, 0.73)):
    rng = np.random.default_rng(4)
    u = (np.arange(n) / n * tiles) % 1.0
    gx = np.minimum(u, 1 - u)[None, :].repeat(n, 0)
    gy = np.minimum(u, 1 - u)[:, None].repeat(n, 1)
    line = np.minimum(gx, gy)
    grout_mask = (line < 0.012).astype(float)
    grout_mask = np.clip(grout_mask + (line < 0.02) * 0.4, 0, 1)
    marble = _fbm(n, 5, 5, 0.55, rng)
    per_tile = rng.random((tiles, tiles))
    ix = (np.arange(n) / n * tiles).astype(int)
    shade = per_tile[np.ix_(ix, ix)] * 0.10 + 0.95
    alb = _tint(marble, colour, 0.09) * shade[..., None]
    alb = alb * (1 - grout_mask[..., None]) + np.asarray(grout)[None, None, :] * grout_mask[..., None]
    return alb, (1 - grout_mask) * 0.5 + marble * 0.1


def _wood(n, colour=(0.50, 0.34, 0.20)):
    rng = np.random.default_rng(5)
    y = np.linspace(0, 1, n)[:, None].repeat(n, 1)
    warp = _fbm(n, 4, 3, 0.6, rng) * 0.10
    rings = np.sin((y * 34 + warp * 26) * math.pi) * 0.5 + 0.5
    rings = rings ** 1.6
    grain = _fbm(n, 5, 22, 0.5, np.random.default_rng(15)) * 0.35
    h = rings * 0.7 + grain
    planks = 5
    pid = (np.arange(n) / n * planks).astype(int)
    pshade = (np.random.default_rng(25).random(planks) * 0.14 + 0.93)[pid][None, :]
    seam = (((np.arange(n) / n * planks) % 1.0) < 0.006).astype(float)[None, :]
    alb = _tint(h, colour, 0.30) * pshade[..., None]
    alb *= (1 - seam * 0.55)[..., None]
    return alb, h * 0.3


def _brushed_metal(n, colour=(0.56, 0.57, 0.60)):
    rng = np.random.default_rng(6)
    streak = rng.normal(0, 1, (1, n)).repeat(n, 0)
    streak = np.convolve(streak[0], np.ones(3) / 3, "same")[None, :].repeat(n, 0)
    h = 0.5 + streak * 0.06 + _fbm(n, 3, 16, 0.5, rng) * 0.12
    return _tint(h, colour, 0.10), h * 0.08


def _galvanised(n):
    rng = np.random.default_rng(7)
    spangle = _fbm(n, 3, 26, 0.4, rng)
    h = (spangle > 0.5).astype(float) * 0.3 + spangle * 0.7
    return _tint(h, (0.70, 0.715, 0.735), 0.13), h * 0.1


def _cardboard(n):
    rng = np.random.default_rng(8)
    flute = (np.sin(np.linspace(0, 1, n) * 120 * math.pi) * 0.5 + 0.5)[None, :].repeat(n, 0)
    fib = _fbm(n, 4, 30, 0.5, rng)
    h = flute * 0.25 + fib * 0.75
    return _tint(h, (0.66, 0.51, 0.33), 0.16), h * 0.25


def _brick(n, rows=8):
    rng = np.random.default_rng(9)
    yy = (np.arange(n) / n * rows)
    row = yy.astype(int)
    offset = (row % 2) * 0.5
    xx = (np.arange(n) / n * (rows / 2))[None, :] + offset[:, None]
    fy = yy % 1.0
    fx = xx % 1.0
    mortar = ((np.minimum(fy, 1 - fy)[:, None] < 0.055)
              | (np.minimum(fx, 1 - fx) < 0.030)).astype(float)
    tone = rng.random((rows, int(rows / 2) + 2))
    ti = np.clip(xx.astype(int), 0, tone.shape[1] - 1)
    shade = tone[np.ix_(np.clip(row, 0, rows - 1), np.arange(tone.shape[1]))]
    shade = np.take_along_axis(shade, ti, axis=1) * 0.30 + 0.85
    grit = _fbm(n, 4, 24, 0.5, rng)
    alb = _tint(grit, (0.55, 0.30, 0.24), 0.20) * shade[..., None]
    mcol = np.asarray((0.72, 0.71, 0.68))
    alb = alb * (1 - mortar[..., None]) + mcol[None, None, :] * mortar[..., None]
    return alb, (1 - mortar) * 0.8 + grit * 0.2


def _ceiling_tile(n, tiles=2):
    rng = np.random.default_rng(10)
    pit = (rng.random((n, n)) > 0.986).astype(float)
    fib = _fbm(n, 4, 30, 0.5, rng)
    h = fib * 0.7 + pit * 0.3
    u = (np.arange(n) / n * tiles) % 1.0
    g = np.minimum(u, 1 - u)
    grid = ((g[None, :] < 0.010) | (g[:, None] < 0.010)).astype(float)
    alb = _tint(h, (0.90, 0.895, 0.875), 0.06)
    alb = alb * (1 - grid[..., None]) + np.asarray((0.78, 0.79, 0.80))[None, None, :] * grid[..., None]
    return alb, (1 - grid) * 0.4 + h * 0.2


def _carpet(n, colour=(0.30, 0.32, 0.38)):
    rng = np.random.default_rng(12)
    h = _fbm(n, 3, 60, 0.45, rng)
    return _tint(h, colour, 0.16), h * 0.35


def _asphalt(n):
    rng = np.random.default_rng(13)
    agg = _fbm(n, 5, 40, 0.5, rng)
    h = agg ** 1.4
    return _tint(h, (0.20, 0.20, 0.21), 0.35), h * 0.5


def _grass(n):
    rng = np.random.default_rng(14)
    h = _fbm(n, 5, 26, 0.5, rng)
    patch = _fbm(n, 3, 4, 0.6, np.random.default_rng(24))
    col = np.stack([0.22 + 0.12 * patch, 0.36 + 0.16 * patch, 0.14 + 0.08 * patch], -1)
    return np.clip(col * (0.75 + 0.5 * h[..., None]), 0, 1), h * 0.4


def _soil(n):
    rng = np.random.default_rng(16)
    h = _fbm(n, 5, 18, 0.52, rng)
    return _tint(h, (0.36, 0.27, 0.19), 0.28), h * 0.45


def _corrugated(n, colour=(0.62, 0.63, 0.64)):
    x = np.linspace(0, 1, n)
    wave = (np.sin(x * 28 * math.pi) * 0.5 + 0.5)[None, :].repeat(n, 0)
    rust = _fbm(n, 4, 12, 0.5, np.random.default_rng(17))
    h = wave * 0.8 + rust * 0.2
    alb = _tint(h, colour, 0.22)
    rustmask = np.clip((rust - 0.62) * 3, 0, 1)
    alb = alb * (1 - rustmask[..., None]) + np.asarray((0.45, 0.26, 0.15))[None, None, :] * rustmask[..., None]
    return alb, wave * 0.9


def _painted(n, colour):
    rng = np.random.default_rng(18)
    h = _fbm(n, 4, 14, 0.45, rng)
    return _tint(h, colour, 0.04), h * 0.05


_BUILDERS = {
    "concrete": _concrete,
    "screed": _screed,
    "plaster": _plaster,
    "plaster_cream": lambda n: _plaster(n, (0.92, 0.88, 0.78)),
    "plaster_blue": lambda n: _plaster(n, (0.78, 0.84, 0.88)),
    "tile": _floor_tile,
    "tile_grey": lambda n: _floor_tile(n, 4, (0.35, 0.35, 0.34), (0.62, 0.63, 0.64)),
    "wood": _wood,
    "wood_dark": lambda n: _wood(n, (0.31, 0.20, 0.12)),
    "wood_light": lambda n: _wood(n, (0.70, 0.55, 0.36)),
    "steel": _brushed_metal,
    "galv_steel": _galvanised,
    "alu": lambda n: _brushed_metal(n, (0.80, 0.81, 0.83)),
    "rack_orange": lambda n: _painted(n, (0.78, 0.34, 0.08)),
    "rack_blue": lambda n: _painted(n, (0.13, 0.29, 0.55)),
    "cardboard": _cardboard,
    "brick": _brick,
    "ceiling_tile": _ceiling_tile,
    "carpet": _carpet,
    "carpet_red": lambda n: _carpet(n, (0.42, 0.20, 0.20)),
    "asphalt": _asphalt,
    "grass": _grass,
    "soil": _soil,
    "corrugated": _corrugated,
    "corrugated_red": lambda n: _corrugated(n, (0.52, 0.26, 0.20)),
    "white": lambda n: _painted(n, (0.92, 0.92, 0.92)),
    "shelf_white": lambda n: _painted(n, (0.88, 0.89, 0.90)),
}


@lru_cache(maxsize=64)
def get(name: str, size: int = 512) -> Texture:
    builder = _BUILDERS.get(name)
    if builder is None:
        builder = lambda n: _painted(n, (0.7, 0.7, 0.7))
    alb, _ = builder(size)
    return _to_texture(alb, f"{name}_albedo")


@lru_cache(maxsize=64)
def get_normal(name: str, size: int = 512, strength: float = 3.0) -> Texture:
    builder = _BUILDERS.get(name)
    if builder is None:
        return None
    _, h = builder(size)
    nrm = _normal_from_height(h, strength)
    return _to_texture(nrm, f"{name}_normal", srgb=False)


def available() -> list[str]:
    return sorted(_BUILDERS.keys())
