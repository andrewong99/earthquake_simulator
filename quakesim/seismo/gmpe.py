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

"""Ground Motion Prediction Equations, and anchoring the stochastic model to them.

Why both a physics model and a GMPE?

The stochastic model in `spectrum.py` gets the *character* of shaking right --
frequency content, duration, how a deep slab event rattles and a megathrust
rolls.  What a point-source model cannot do well is reproduce the absolute
amplitude that a regression over ten thousand real records gives you, because
a real rupture is a finite surface with directivity, asperities and hanging-wall
geometry that a point cannot represent.

So we do what engineering practice does: take the waveform shape from the
physics, and anchor its amplitude to a published GMPE (ASCE 7-16 Ch.16 style
target-spectrum matching).  For shallow crustal earthquakes that anchor is
Boore, Stewart, Seyhan & Atkinson (2014), NGA-West2, Earthquake Spectra 30(3).

Outside the GMPE's domain -- volcanic, collapse, explosion, deep-focus,
subduction -- no calibrated regression applies, so the model runs on physics
alone with literature-typical source parameters. `anchor_mode()` reports which
regime is in force so the UI can be honest about it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import constants as K
from .site import Site
from .source import Source

# ---------------------------------------------------------------------------
# BSSA14 -- PGA and PGV rows
# ---------------------------------------------------------------------------
_BSSA14 = {
    "pga": dict(e0=0.4473, e1=0.4856, e2=0.2459, e3=0.4539,
                e4=1.431, e5=0.05053, e6=-0.1662, Mh=5.5,
                c1=-1.134, c2=0.1917, c3=-0.008088, Mref=4.5, h=4.5,
                c=-0.60, Vc=1500.0, f4=-0.1, f5=-0.00844),
    "pgv": dict(e0=5.037, e1=5.078, e2=4.849, e3=5.033,
                e4=1.073, e5=-0.1536, e6=0.2252, Mh=6.2,
                c1=-1.243, c2=0.1489, c3=-0.00344, Mref=4.5, h=5.3,
                c=-0.84, Vc=1300.0, f4=-0.1, f5=-0.00844),
}
_VREF = 760.0

# Which mechanism column of BSSA14 each of our quake types maps onto
_MECH_COL = {"strike_slip": "e1", "normal": "e2", "reverse": "e3",
             "oblique": "e0", "isotropic": "e0", "implosive": "e0",
             "clvd": "e0"}

# Quake types for which a shallow-crustal GMPE is a legitimate anchor
CRUSTAL_ANCHORED = {"strike_slip", "normal", "reverse", "oblique",
                    "blind_thrust", "aftershock", "reservoir"}


def _bssa14_base(mw: float, rjb: float, key: str, mech: str) -> float:
    """ln of the median on reference rock (Vs30 = 760 m/s)."""
    p = _BSSA14[key]
    mh = p["Mh"]
    ei = p[_MECH_COL.get(mech, "e0")]
    if mw <= mh:
        fe = ei + p["e4"] * (mw - mh) + p["e5"] * (mw - mh) ** 2
    else:
        fe = ei + p["e6"] * (mw - mh)
    r = math.hypot(rjb, p["h"])
    fp = (p["c1"] + p["c2"] * (mw - p["Mref"])) * math.log(r) + p["c3"] * (r - 1.0)
    return fe + fp


def bssa14(mw: float, rjb: float, vs30: float, key: str = "pga",
           mech: str = "strike_slip") -> float:
    """BSSA14 median. Returns g for 'pga', cm/s for 'pgv'."""
    p = _BSSA14[key]
    ln_rock = _bssa14_base(mw, rjb, key, mech)
    pga_rock = math.exp(_bssa14_base(mw, rjb, "pga", mech))

    f_lin = p["c"] * math.log(min(vs30, p["Vc"]) / _VREF)
    f2 = p["f4"] * (math.exp(p["f5"] * (min(vs30, 760.0) - 360.0))
                    - math.exp(p["f5"] * 400.0))
    f_nl = f2 * math.log((pga_rock + 0.1) / 0.1)
    return math.exp(ln_rock + f_lin + f_nl)


# ---------------------------------------------------------------------------
# Anchoring
# ---------------------------------------------------------------------------
@dataclass
class Anchor:
    """A smooth log-log correction applied to the stochastic FAS."""
    level: float = 1.0      # multiplicative level at f_ref
    tilt: float = 0.0       # d(log10 amp) / d(log10 f)
    f_ref: float = 1.0
    mode: str = "physics"   # "gmpe" or "physics"
    target_pga_g: float = 0.0
    target_pgv_cms: float = 0.0

    def apply(self, freqs: np.ndarray, fas: np.ndarray) -> np.ndarray:
        if self.mode != "gmpe":
            return fas
        f = np.maximum(np.asarray(freqs, dtype=float), 1e-6)
        return fas * self.level * (f / self.f_ref) ** self.tilt


def anchor_mode(src: Source) -> str:
    # Above M8.5 the anchor stays on: bssa14() itself holds the magnitude at
    # 8.5, which is how its authors bound it, so a crustal M10 gets M8.5
    # shaking with an M10 source duration and rupture -- saturated, not
    # the hundreds of g the unanchored spectrum would scale to.
    if src.qtype.key in CRUSTAL_ANCHORED and src.depth_km <= 35.0 \
            and src.mw >= 3.0:
        return "gmpe"
    return "physics"


def solve_anchor(src: Source, site: Site, r_epi_km: float,
                 component: str = "h") -> Anchor:
    """Find the (level, tilt) pair that makes RVT-PGA and RVT-PGV match BSSA14.

    Two unknowns, two targets.  Because RVT peaks are very nearly log-linear in
    a log-linear spectral perturbation, a 2x2 Newton solve converges in a
    couple of iterations.
    """
    from . import spectrum as sp

    if anchor_mode(src) != "gmpe":
        return Anchor(mode="physics")

    rjb = max(math.sqrt(max(r_epi_km ** 2 - 0.0, 0.0)), 0.0)
    mech = src.qtype.mechanism
    tgt_pga = bssa14(src.mw, rjb, site.vs30, "pga", mech)          # g
    tgt_pgv = bssa14(src.mw, rjb, site.vs30, "pgv", mech)          # cm/s

    # Evaluate the raw model at the *reference* stress drop for this type, so
    # that the user moving the stress-drop slider still changes the answer
    # physically instead of being silently normalised away.
    from dataclasses import replace
    ref_src = replace(src, stress_drop_bar=src.qtype.stress_drop_default)

    f = np.geomspace(0.005, 100.0, 900)
    w = 2.0 * math.pi * f
    dur = sp.ground_motion_duration(ref_src, r_epi_km)
    base = sp.acceleration_fas(f, ref_src, site, r_epi_km, component)

    a = Anchor(mode="gmpe", target_pga_g=tgt_pga, target_pgv_cms=tgt_pgv)

    def peaks(level: float, tilt: float) -> tuple[float, float]:
        corr = level * (f / a.f_ref) ** tilt
        fas = base * corr
        return (sp.rvt_peak(f, fas, dur) / K.G0,
                sp.rvt_peak(f, fas / w, dur) * 100.0)

    x = np.array([0.0, 0.0])       # [log10 level, tilt]
    for _ in range(6):
        lvl = 10.0 ** x[0]
        pga, pgv = peaks(lvl, x[1])
        if pga <= 0 or pgv <= 0:
            break
        r = np.array([math.log10(pga / tgt_pga), math.log10(pgv / tgt_pgv)])
        if np.max(np.abs(r)) < 1e-4:
            break
        # numerical Jacobian
        j = np.zeros((2, 2))
        for k, d in enumerate((0.05, 0.05)):
            xp = x.copy(); xp[k] += d
            p1, v1 = peaks(10.0 ** xp[0], xp[1])
            j[0, k] = (math.log10(p1 / tgt_pga) - r[0]) / d
            j[1, k] = (math.log10(v1 / tgt_pgv) - r[1]) / d
        try:
            step = np.linalg.solve(j, -r)
        except np.linalg.LinAlgError:
            break
        x = x + np.clip(step, -1.0, 1.0)
        x[0] = float(np.clip(x[0], -2.0, 2.0))
        x[1] = float(np.clip(x[1], -1.5, 1.5))

    a.level = 10.0 ** x[0]
    a.tilt = float(x[1])
    return a


# ---------------------------------------------------------------------------
# Subduction: Atkinson & Boore (2003) style check value, used for reporting
# only (the waveform itself stays physics-driven).
# ---------------------------------------------------------------------------
def subduction_pga_check(mw: float, r_km: float, depth_km: float,
                         interface: bool = True) -> float:
    """Rough AB2003-form median PGA in g, for cross-checking megathrust cases.

    Not used to scale the simulation; shown in the UI as an independent second
    opinion so the physics-mode numbers can be sanity-checked.
    """
    mw = min(mw, 8.5 if interface else 8.0)
    if interface:
        c1, c2, c3, c4 = 2.991, 0.03525, 0.00759, -0.00206
    else:
        c1, c2, c3, c4 = -0.04713, 0.6909, 0.01130, -0.00202
    h = min(depth_km, 100.0)
    delta = 0.00724 * 10.0 ** (0.507 * mw)
    r = math.sqrt(r_km ** 2 + delta ** 2)
    g = 10.0 ** (0.301 - 0.01 * (10.0 - mw) if not interface else 0.0)
    log_y = (c1 + c2 * mw + c3 * h + c4 * r - math.log10(r)
             + (0.0 if interface else 0.0))
    return max(10.0 ** log_y / 981.0, 1e-6)   # cm/s^2 -> g
