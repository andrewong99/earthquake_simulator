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

"""Seismic phases: travel times, surface-wave dispersion, particle motion.

A real seismogram is not one blob of shaking.  It arrives in order:

    P   compressional, fastest, small, mostly vertical -- the "jolt" or bang
    S   shear, ~1.7x slower, the largest body-wave phase, mostly horizontal
    Love      transverse horizontal surface wave
    Rayleigh  retrograde elliptical, radial + vertical, dispersive and slow
    coda      scattered energy decaying over minutes

The gap between P and S is what lets you count seconds to estimate distance,
and it is why people feel a sharp bump and then, a moment later, the real
shaking.  Getting this sequence right is most of what makes a simulated quake
feel like an earthquake instead of a rumble.
"""

from __future__ import annotations

import math

import numpy as np

from . import constants as K


# ---------------------------------------------------------------------------
# Body-wave travel times
# ---------------------------------------------------------------------------
def body_wave_speeds(depth_km: float) -> tuple[float, float]:
    """Path-averaged (Vp, Vs) in m/s for a ray from this depth to the surface."""
    if depth_km <= 35.0:
        return K.VP_CRUST, K.VS_CRUST
    # Deeper rays spend most of their path in the faster mantle
    frac = min((depth_km - 35.0) / 300.0, 1.0)
    vp = K.VP_CRUST + frac * (K.VP_MANTLE - K.VP_CRUST)
    vs = K.VS_CRUST + frac * (K.VS_MANTLE - K.VS_CRUST)
    return vp, vs


def arrival_times(r_epi_km: float, depth_km: float) -> dict[str, float]:
    """Arrival time of each phase, seconds after origin."""
    r_hyp = math.hypot(r_epi_km, depth_km)
    vp, vs = body_wave_speeds(depth_km)
    t_p = r_hyp * K.KM / vp
    t_s = r_hyp * K.KM / vs
    # Surface waves travel along the surface, not from the hypocentre
    t_love = r_epi_km * K.KM / (K.VS_CRUST * K.LOVE_FACTOR) if r_epi_km > 0 else t_s
    t_rayleigh = r_epi_km * K.KM / (K.VS_CRUST * K.RAYLEIGH_FACTOR) if r_epi_km > 0 else t_s
    return {
        "P": t_p, "S": t_s,
        "Love": max(t_love, t_s * 0.98),
        "Rayleigh": max(t_rayleigh, t_s * 1.0),
        "S_minus_P": t_s - t_p,
    }


def distance_from_sp_interval(sp_seconds: float) -> float:
    """The classic field estimate: distance (km) ~ 8 * (S-P) seconds."""
    vp, vs = K.VP_CRUST, K.VS_CRUST
    return sp_seconds / (1.0 / vs - 1.0 / vp) / K.KM


# ---------------------------------------------------------------------------
# Surface-wave dispersion
# ---------------------------------------------------------------------------
def group_velocity(freqs: np.ndarray, kind: str = "rayleigh",
                   crust_vs: float = K.VS_CRUST) -> np.ndarray:
    """Group velocity U(f) in m/s for a normally dispersive crust.

    Long periods sample deeper, faster rock and therefore arrive first; short
    periods are trapped in the slow near-surface layers and straggle in behind.
    That spread-out, whistling-down arrival is the signature of a distant
    large earthquake.
    """
    f = np.maximum(np.asarray(freqs, dtype=float), 1e-4)
    t = 1.0 / f
    if kind == "love":
        u_short, u_long, t_c = 0.55 * crust_vs, 1.02 * crust_vs, 22.0
    else:
        u_short, u_long, t_c = 0.48 * crust_vs, 0.98 * crust_vs, 28.0
    return u_short + (u_long - u_short) * (1.0 - np.exp(-t / t_c))


def dispersive_delay(freqs: np.ndarray, r_epi_km: float,
                     kind: str = "rayleigh") -> np.ndarray:
    """Frequency-dependent arrival time (s) for a dispersed surface wave."""
    u = group_velocity(freqs, kind)
    return (r_epi_km * K.KM) / u


# ---------------------------------------------------------------------------
# Coda
# ---------------------------------------------------------------------------
def coda_envelope(t: np.ndarray, t_s: float, f_hz: float = 3.0,
                  q_coda: float = 250.0) -> np.ndarray:
    """Aki & Chouet (1975) single-scattering coda decay, t^-1 * exp(-w t / Qc)."""
    tt = np.maximum(np.asarray(t, dtype=float), t_s + 1e-3)
    return (tt / max(t_s, 1e-3)) ** -1.0 * np.exp(
        -math.pi * f_hz * (tt - t_s) / q_coda)


# ---------------------------------------------------------------------------
# Particle motion / component partitioning
# ---------------------------------------------------------------------------
def incidence_angle(r_epi_km: float, depth_km: float) -> float:
    """Angle of the incoming ray from vertical, at the surface (degrees).

    Directly above the hypocentre the ray comes straight up (0 deg) and the
    motion is nearly all vertical for P; far away it arrives almost
    horizontally and P becomes a horizontal push.
    """
    if depth_km <= 0.0:
        return 90.0
    return math.degrees(math.atan2(r_epi_km, depth_km))


def phase_partition(phase: str, r_epi_km: float, depth_km: float,
                    qtype=None) -> tuple[float, float, float]:
    """Return (radial, transverse, vertical) amplitude weights for a phase.

    Weights are amplitude fractions; energy is preserved approximately by
    normalising the vector length to 1.
    """
    inc = math.radians(incidence_angle(r_epi_km, depth_km))

    if phase == "P":
        # P moves along the ray: mostly vertical near, mostly radial far.
        rad, tra, ver = math.sin(inc), 0.0, math.cos(inc)
    elif phase == "S":
        # S moves perpendicular to the ray. SV in the radial-vertical plane,
        # SH purely transverse; a shear source radiates both.
        sv_r, sv_z = math.cos(inc), -math.sin(inc)
        rad, tra, ver = 0.707 * sv_r, 0.707, 0.707 * sv_z
    elif phase == "Love":
        rad, tra, ver = 0.0, 1.0, 0.0
    elif phase == "Rayleigh":
        # Retrograde ellipse: vertical about 1.5x the radial, 90 deg out of phase
        rad, tra, ver = 0.62, 0.0, 0.93
    else:
        rad, tra, ver = 0.577, 0.577, 0.577

    if qtype is not None:
        ver *= qtype.vertical_ratio / 0.6

    n = math.sqrt(rad * rad + tra * tra + ver * ver) or 1.0
    return rad / n, tra / n, ver / n


def rotate_to_en(radial: np.ndarray, transverse: np.ndarray,
                 azimuth_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate radial/transverse traces into East/North.

    `azimuth_deg` is the back-azimuth from the site to the epicentre, measured
    clockwise from north. Radial points away from the epicentre.
    """
    a = math.radians(azimuth_deg)
    # radial unit vector (pointing from epicentre to site) in (E, N)
    re, rn = math.sin(a), math.cos(a)
    te, tn = math.cos(a), -math.sin(a)
    east = radial * re + transverse * te
    north = radial * rn + transverse * tn
    return east, north
