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

"""Site response: Vs30 profiles, quarter-wavelength amplification, kappa,
soil nonlinearity, and basin effects.

Site condition is the single biggest reason two places at the same distance
from the same earthquake shake completely differently.  Mexico City sits on
an old lake bed and amplifies 2-second motion by a factor of ten; hard rock
a few km away barely moves.  This module makes that difference real instead
of a fudge factor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import constants as K

# ---------------------------------------------------------------------------
# NEHRP / ASCE 7 site classes
# ---------------------------------------------------------------------------
NEHRP = {
    "A": dict(name="Hard rock", vs30=1620.0, kappa=0.010),
    "B": dict(name="Rock", vs30=1050.0, kappa=0.020),
    "C": dict(name="Very dense soil / soft rock",
              vs30=560.0, kappa=0.032),
    "D": dict(name="Stiff soil", vs30=270.0, kappa=0.048),
    "E": dict(name="Soft clay soil", vs30=150.0, kappa=0.065),
    "F": dict(name="Liquefiable / organic / peat",
              vs30=105.0, kappa=0.085),
}
NEHRP_ORDER = ["A", "B", "C", "D", "E", "F"]


# ---------------------------------------------------------------------------
# Velocity profile
# ---------------------------------------------------------------------------
_PROFILE_EXPONENT = 0.28    # Vs(z) ~ z^p, typical of real shallow profiles


def _surface_vs(vs30: float, p: float = _PROFILE_EXPONENT) -> float:
    """Surface shear velocity such that the travel-time average over the top
    30 m equals `vs30` for the power-law profile Vs(z) = vs_s * (1+z)^p."""
    integral = ((31.0 ** (1.0 - p)) - 1.0) / (1.0 - p)   # int_0^30 (1+z)^-p dz
    return vs30 * integral / 30.0


def velocity_profile(vs30: float, n: int = 400,
                     z_max: float = 4000.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (depth m, Vs m/s, density kg/m^3) for a generic profile anchored
    at `vs30` and merging into crustal rock at depth."""
    z = np.geomspace(0.5, z_max, n)
    vs_s = _surface_vs(vs30)
    vs = vs_s * (1.0 + z) ** _PROFILE_EXPONENT
    vs = np.minimum(vs, K.VS_CRUST)

    # Gardner et al. (1974): rho = 1.74 * Vp^0.25 (g/cm^3, Vp in km/s).
    # Use a depth-varying Vp/Vs that tends to 1.73 (Poisson solid) in rock.
    vp_vs = np.clip(3.4 - 1.7 * (vs / K.VS_CRUST), 1.73, 3.4)
    vp_kms = vs * vp_vs / 1000.0
    rho = 1740.0 * np.maximum(vp_kms, 0.3) ** 0.25          # kg/m^3
    rho = np.clip(rho, 1500.0, K.RHO_CRUST)
    return z, vs, rho


def quarter_wavelength_amplification(freqs: np.ndarray, vs30: float,
                                     rho_src: float = K.RHO_CRUST,
                                     vs_src: float = K.VS_CRUST) -> np.ndarray:
    """Joyner, Warrick & Fumal (1981) quarter-wavelength site amplification.

    At each frequency, find the depth z where z = V_avg(z) / (4 f); the
    amplification is sqrt(impedance at the source / impedance averaged down
    to that depth).
    """
    z, vs, rho = velocity_profile(vs30)

    # Travel-time-averaged velocity and density from the surface to depth z.
    dz = np.diff(np.concatenate(([0.0], z)))
    slowness_cum = np.cumsum(dz / vs)
    vavg = z / np.maximum(slowness_cum, 1e-12)
    rho_avg = np.cumsum(rho * dz) / np.maximum(np.cumsum(dz), 1e-12)

    f_of_z = vavg / (4.0 * z)                      # frequency matched to depth z
    order = np.argsort(f_of_z)
    f_sorted = f_of_z[order]

    imped_site = rho_avg[order] * vavg[order]
    imped_src = rho_src * vs_src
    amp_sorted = np.sqrt(imped_src / np.maximum(imped_site, 1e-9))

    out = np.interp(np.asarray(freqs, dtype=float), f_sorted, amp_sorted,
                    left=amp_sorted[0], right=amp_sorted[-1])
    return np.maximum(out, 1.0e-3)


def nonlinear_soil_factor(vs30: float, pga_rock_g: float) -> float:
    """Boore, Stewart, Seyhan & Atkinson (2014) NGA-West2 nonlinear site term.

    Strong shaking makes soft soil yield: its shear modulus falls, damping
    rises, and it stops amplifying -- above roughly 0.3 g soft sites can
    de-amplify relative to rock.
    """
    f3, f4, f5 = 0.1, -0.12, -0.0083
    v = min(vs30, 760.0)
    f2 = f4 * (math.exp(f5 * (v - 360.0)) - math.exp(f5 * (760.0 - 360.0)))
    return math.exp(f2 * math.log((max(pga_rock_g, 0.0) + f3) / f3))


# ---------------------------------------------------------------------------
# Site object
# ---------------------------------------------------------------------------
@dataclass
class Site:
    """Everything about the ground under the observer."""

    name: str = "Generic rock"
    nehrp: str = "B"
    vs30: float = 1050.0
    kappa0: float = 0.020
    basin_depth_km: float = 0.0       # depth to Vs = 1 km/s (Z1.0)
    basin_gain: float = 1.0           # extra long-period gain from a basin
    liquefaction_susceptible: bool = False
    note: str = ""

    @classmethod
    def from_class(cls, nehrp: str, name: str | None = None, **kw) -> "Site":
        spec = NEHRP[nehrp]
        return cls(name=name or spec["name"], nehrp=nehrp,
                   vs30=spec["vs30"], kappa0=spec["kappa"], **kw)

    def amplification(self, freqs: np.ndarray) -> np.ndarray:
        amp = quarter_wavelength_amplification(freqs, self.vs30)
        if self.basin_depth_km > 0.02 and self.basin_gain > 1.0:
            # Sediment basins trap and reverberate surface waves; the gain is
            # concentrated near the basin's fundamental resonance,
            # f0 ~ Vs30 / (4 * H).
            f0 = self.vs30 / (4.0 * self.basin_depth_km * 1000.0)
            width = 1.4
            bump = 1.0 + (self.basin_gain - 1.0) * np.exp(
                -0.5 * (np.log(np.maximum(freqs, 1e-4) / max(f0, 1e-4)) / width) ** 2)
            amp = amp * bump
        return amp

    def kappa_filter(self, freqs: np.ndarray) -> np.ndarray:
        return np.exp(-math.pi * self.kappa0 * np.asarray(freqs, dtype=float))

    def fundamental_period(self) -> float:
        """Site period T0 = 4H/Vs, the period this ground rings at."""
        if self.basin_depth_km > 0.0:
            return 4.0 * self.basin_depth_km * 1000.0 / self.vs30
        return 4.0 * 30.0 / self.vs30

    def describe(self) -> dict:
        return {
            "site": self.name,
            "NEHRP": f"{self.nehrp} - {NEHRP[self.nehrp]['name']}",
            "Vs30_m_s": self.vs30,
            "kappa0_s": self.kappa0,
            "Z1.0_km": self.basin_depth_km,
            "site_period_s": round(self.fundamental_period(), 2),
            "liquefaction": self.liquefaction_susceptible,
        }


# Ready-made sites used by the scenes ---------------------------------------
SITES = {
    "hard_rock": Site.from_class("A", "Granite outcrop"),
    "rock": Site.from_class("B", "Competent rock"),
    "soft_rock": Site.from_class("C", "Weathered sedimentary rock"),
    "stiff_soil": Site.from_class("D", "Stiff alluvial soil"),
    "soft_soil": Site.from_class("E", "Soft clay", basin_depth_km=0.35,
                                 basin_gain=2.2, liquefaction_susceptible=True),
    "reclaimed": Site.from_class("F", "Reclaimed / hydraulic fill",
                                 basin_depth_km=0.25, basin_gain=2.6,
                                 liquefaction_susceptible=True,
                                 note="Loose saturated fill - liquefaction likely above ~0.2 g"),
    "kl_alluvium": Site(name="Klang Valley alluvium over limestone", nehrp="D",
                        vs30=245.0, kappa0=0.050, basin_depth_km=0.30,
                        basin_gain=2.4,
                        note="Soft alluvium over karstic limestone; amplifies "
                             "the 0.5-2 s band that tall buildings live in"),
    "sabah_residual": Site(name="Crocker Range residual soil", nehrp="C",
                           vs30=430.0, kappa0=0.038, basin_depth_km=0.05,
                           basin_gain=1.3,
                           note="Thin residual soil over sandstone/shale on "
                                "steep terrain - strong topographic effects"),
}
