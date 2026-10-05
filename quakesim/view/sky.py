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

"""Sky dome, sun and stars.

You asked to be able to look from the ground all the way up to the sky and
back down again, so there has to be a real sky up there rather than a grey
void. The dome is shaded with a cheap analytic atmosphere: Rayleigh scattering
makes the zenith blue and the horizon pale, and both roll toward orange and
then black as the sun goes down. Stars come out when it does.
"""

from __future__ import annotations

import math

import numpy as np
from panda3d.core import (AmbientLight, CardMaker, CullFaceAttrib,
                          DirectionalLight, Fog, Geom, GeomNode, LVector3,
                          NodePath, Texture, TransparencyAttrib, Vec3, Vec4)

from ..world import geom as g
from ..world.textures import _to_texture


def _sun_elevation(hour: float, latitude: float = 3.14,
                   declination: float = 0.0) -> tuple[float, float]:
    """Solar elevation and azimuth in degrees.

    Default latitude is Kuala Lumpur, which is near enough the equator that
    the sun goes almost straight up and straight down -- short dawns, short
    dusks, and near-vertical light at midday.
    """
    h = math.radians((hour - 12.0) * 15.0)
    lat = math.radians(latitude)
    dec = math.radians(declination)
    sin_alt = (math.sin(lat) * math.sin(dec)
               + math.cos(lat) * math.cos(dec) * math.cos(h))
    alt = math.degrees(math.asin(max(-1.0, min(1.0, sin_alt))))
    cos_az = ((math.sin(dec) - math.sin(math.radians(alt)) * math.sin(lat))
              / max(math.cos(math.radians(alt)) * math.cos(lat), 1e-6))
    az = math.degrees(math.acos(max(-1.0, min(1.0, cos_az))))
    if h > 0:
        az = 360.0 - az
    return alt, az


def _sky_texture(hour: float, size: int = 256) -> Texture:
    """Vertical gradient from horizon to zenith for this time of day."""
    alt, _ = _sun_elevation(hour)
    day = max(0.0, min(1.0, (alt + 6.0) / 18.0))          # 0 at night, 1 by day
    dusk = math.exp(-((alt - 0.0) / 7.0) ** 2)            # peaks at the horizon

    v = np.linspace(0.0, 1.0, size)[:, None]              # 0 horizon, 1 zenith
    zen_day = np.array([0.24, 0.44, 0.78])
    hor_day = np.array([0.72, 0.82, 0.90])
    zen_night = np.array([0.015, 0.022, 0.055])
    hor_night = np.array([0.05, 0.06, 0.11])
    zen_dusk = np.array([0.20, 0.22, 0.44])
    hor_dusk = np.array([0.95, 0.48, 0.20])

    zen = zen_night + (zen_day - zen_night) * day
    hor = hor_night + (hor_day - hor_night) * day
    zen = zen * (1 - dusk) + zen_dusk * dusk
    hor = hor * (1 - dusk) + hor_dusk * dusk

    t = v ** 0.62
    col = hor[None, :] * (1 - t) + zen[None, :] * t
    img = np.repeat(col[:, None, :], 4, axis=1)

    if day < 0.45:
        rng = np.random.default_rng(7)
        stars = rng.random((size, 4))
        mask = stars > 0.9975
        bright = (1.0 - day / 0.45)
        img[mask] = np.minimum(img[mask] + 0.85 * bright, 1.0)
    return _to_texture(img, f"sky_{hour:.1f}")


class Sky:
    """Sky dome, sun, sunlight and fog for one scene."""

    def __init__(self, render: NodePath, hour: float = 14.0,
                 radius: float = 4000.0, shadow_size: int = 1024):
        self.render = render
        self.hour = hour
        self.root = render.attachNewNode("sky")
        self.radius = radius
        self.shadow_size = int(shadow_size)

        dome = g.make_sphere(radius, rings=32, sectors=48, name="skydome")
        dome.setTwoSided(True)
        dome.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullCounterClockwise))
        dome.setBin("background", 0)
        dome.setDepthWrite(False)
        dome.setLightOff(1)
        dome.setTextureOff(0)
        dome.reparentTo(self.root)
        self.dome = dome

        self.sun_np = None
        self.sun_light = None
        self.amb_light = None
        self._build_lights()
        self.set_hour(hour)

    # ------------------------------------------------------------------
    def _build_lights(self) -> None:
        dl = DirectionalLight("sun")
        dl.setColor(Vec4(1.0, 0.97, 0.92, 1))
        if self.shadow_size > 0:
            # The shadow pass re-draws the scene from the sun, so it doubles
            # the draw calls; that is affordable now the scene is batched.
            dl.setShadowCaster(True, self.shadow_size, self.shadow_size)
        lens = dl.getLens()
        lens.setFilmSize(90, 90)
        lens.setNearFar(1, 400)
        self.sun_light = self.render.attachNewNode(dl)
        self.render.setLight(self.sun_light)

        al = AmbientLight("ambient")
        al.setColor(Vec4(0.30, 0.33, 0.40, 1))
        self.amb_light = self.render.attachNewNode(al)
        self.render.setLight(self.amb_light)

        # A second, dimmer light from the opposite side keeps interiors from
        # going pitch black on the shadowed side.
        fill = DirectionalLight("fill")
        fill.setColor(Vec4(0.25, 0.27, 0.32, 1))
        self.fill_light = self.render.attachNewNode(fill)
        self.fill_light.setHpr(140, -30, 0)
        self.render.setLight(self.fill_light)

    def set_hour(self, hour: float) -> None:
        self.hour = hour % 24.0
        alt, az = _sun_elevation(self.hour)
        self.dome.setTexture(_sky_texture(self.hour), 1)

        day = max(0.0, min(1.0, (alt + 6.0) / 18.0))
        warm = math.exp(-((alt - 2.0) / 9.0) ** 2)
        r = 1.0
        gc = 0.97 - 0.30 * warm
        b = 0.92 - 0.55 * warm
        strength = 0.15 + 1.15 * day
        self.sun_light.node().setColor(Vec4(r * strength, gc * strength,
                                            b * strength, 1))
        self.sun_light.setHpr(az, -max(alt, -12.0), 0)

        amb = 0.06 + 0.30 * day
        tint = Vec4(amb * (1.0 + 0.25 * warm), amb, amb * (1.25 - 0.35 * warm), 1)
        self.amb_light.node().setColor(tint)
        self.fill_light.node().setColor(Vec4(amb * 0.7, amb * 0.75, amb * 0.9, 1))

        fog = Fog("haze")
        hz = 0.60 + 0.30 * day
        fog.setColor(hz * (0.72 + 0.2 * warm), hz * 0.78, hz * 0.88)
        fog.setExpDensity(0.00022 + 0.00018 * (1.0 - day))
        self.render.setFog(fog)
        self.render.setShaderAuto()

    def destroy(self) -> None:
        for np_ in (self.sun_light, self.amb_light, self.fill_light):
            if np_ is not None:
                self.render.clearLight(np_)
                np_.removeNode()
        self.root.removeNode()

    def follow(self, cam: NodePath) -> None:
        """Keep the dome centred on the camera so it never gets closer."""
        p = cam.getPos(self.render)
        self.root.setPos(p)

    def sun_direction(self) -> Vec3:
        alt, az = _sun_elevation(self.hour)
        a, e = math.radians(az), math.radians(alt)
        return Vec3(math.sin(a) * math.cos(e), math.cos(a) * math.cos(e),
                    math.sin(e))
