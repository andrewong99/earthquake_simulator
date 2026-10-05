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

"""Fourier amplitude spectrum of ground motion, and Random Vibration Theory.

This is the analytical backbone. It follows the stochastic method as set out
in Boore, D.M. (2003) "Simulation of ground motion using the stochastic
method", Pure appl. geophys. 160, 635-676, and its SMSIM implementation.

The Fourier acceleration spectrum at a site is a product of four independent
physical effects:

    A(f) = C . S(f) . D(f,R) . P(f) . V(f)

    C     scaling constant  (radiation pattern, free surface, partition)
    S(f)  source           (Brune omega-squared, or two-corner)
    D(f)  path             (geometric spreading x anelastic attenuation)
    P(f)  near-site decay  (kappa)
    V(f)  site             (impedance amplification, basin, nonlinearity)

Random Vibration Theory then converts that spectrum into expected peak values
(PGA, PGV, PGD, response spectra) without needing a time history, using
Cartwright & Longuet-Higgins (1956) peak-factor statistics.
"""

from __future__ import annotations

import math

import numpy as np

from . import constants as K
from .site import Site
from .source import Source


# ---------------------------------------------------------------------------
# Source spectrum
# ---------------------------------------------------------------------------
def source_spectrum(freqs: np.ndarray, src: Source) -> np.ndarray:
    """Displacement source spectrum S(f) in N.m (moment-rate normalised)."""
    f = np.asarray(freqs, dtype=float)
    m0 = src.moment

    if src.qtype.two_corner:
        # Atkinson & Silva (2000) two-corner source: a big rupture is not one
        # smooth pulse but a chain of subevents, which fills in the spectrum
        # between a low corner (whole rupture) and a high one (asperities).
        mw = src.mw
        fa = 10.0 ** (2.181 - 0.496 * mw)
        fb = 10.0 ** (2.410 - 0.408 * mw)
        eps = 10.0 ** (0.605 - 0.255 * mw)
        eps = min(max(eps, 0.0), 1.0)
        # Rescale the corners so the model still honours the chosen stress drop
        scale = src.fc / (10.0 ** (2.181 - 0.496 * mw)) if fa > 0 else 1.0
        scale = max(0.2, min(5.0, scale * 0.55))
        fa, fb = fa * scale, fb * scale
        shape = ((1.0 - eps) / (1.0 + (f / fa) ** 2)
                 + eps / (1.0 + (f / fb) ** 2))
    else:
        n = src.qtype.spectral_n
        shape = 1.0 / (1.0 + (f / src.fc) ** n)

    return m0 * shape


# ---------------------------------------------------------------------------
# Path
# ---------------------------------------------------------------------------
def geometric_spreading(r_km: float, deep: bool = False) -> float:
    """Amplitude decay with distance, 1/metres."""
    r = max(r_km, 1.0)
    if deep:
        z = 1.0 / r                    # body waves through the mantle
    elif r <= K.R_HINGE1:
        z = 1.0 / r
    elif r <= K.R_HINGE2:
        # Post-critical Moho reflections hold the amplitude up over this band
        z = 1.0 / K.R_HINGE1
    else:
        z = (1.0 / K.R_HINGE1) * math.sqrt(K.R_HINGE2 / r)
    return z / K.KM                    # convert 1/km -> 1/m


def anelastic_attenuation(freqs: np.ndarray, r_km: float,
                          q0: float, eta: float,
                          beta: float = K.VS_CRUST) -> np.ndarray:
    """exp(-pi f R / (Q(f) beta)) -- energy actually absorbed by the rock."""
    f = np.maximum(np.asarray(freqs, dtype=float), 1e-6)
    q = np.maximum(q0 * f ** eta, K.Q_MIN)
    return np.exp(-math.pi * f * (r_km * K.KM) / (q * beta))


# ---------------------------------------------------------------------------
# Full spectrum
# ---------------------------------------------------------------------------
def scaling_constant(src: Source, component: str = "h") -> float:
    """C = R_theta_phi . F . V / (4 pi rho beta^3), units s^3/kg."""
    if component == "p":
        rad = K.RADIATION_P
        speed = K.VP_CRUST
    else:
        rad = K.RADIATION_S
        speed = K.VS_CRUST
    part = K.PARTITION if component == "h" else 1.0
    return rad * K.FREE_SURFACE * part / (4.0 * math.pi * K.RHO_CRUST * speed ** 3)


def acceleration_fas(freqs: np.ndarray, src: Source, site: Site,
                     r_epi_km: float, component: str = "h",
                     include_site: bool = True,
                     nonlinear_pga_g: float | None = None) -> np.ndarray:
    """Fourier amplitude spectrum of acceleration, in m/s (i.e. (m/s^2).s)."""
    f = np.maximum(np.asarray(freqs, dtype=float), 1e-6)
    r_hyp = src.hypocentral_distance(r_epi_km)
    deep = src.depth_km > 50.0

    c = scaling_constant(src, component)
    s = source_spectrum(f, src)
    omega2 = (2.0 * math.pi * f) ** 2          # displacement -> acceleration
    g = geometric_spreading(r_hyp, deep=deep)
    speed = K.VP_CRUST if component == "p" else K.VS_CRUST
    an = anelastic_attenuation(f, r_hyp, src.qtype.q0, src.qtype.q_eta, speed)

    kappa = math.sqrt(max(site.kappa0, 1e-4) ** 2 + src.qtype.kappa0 ** 2)
    p = np.exp(-math.pi * kappa * f)

    a = c * s * omega2 * g * an * p

    if include_site:
        a = a * site.amplification(f)
        if nonlinear_pga_g is not None:
            from .site import nonlinear_soil_factor
            a = a * nonlinear_soil_factor(site.vs30, nonlinear_pga_g)

    # Component partitioning and source-type radiation character
    if component == "v":
        a = a * src.qtype.vertical_ratio
    elif component == "p":
        a = a * src.qtype.p_s_ratio

    a = a * src.directivity_factor()
    return a


# ---------------------------------------------------------------------------
# Duration
# ---------------------------------------------------------------------------
def ground_motion_duration(src: Source, r_epi_km: float) -> float:
    """Source + path duration, seconds (Boore & Thompson style hinged path)."""
    r = src.hypocentral_distance(r_epi_km)
    if r < 10.0:
        t_path = 0.0
    elif r < 70.0:
        t_path = 0.16 * (r - 10.0)
    elif r < 130.0:
        t_path = 9.6 - 0.03 * (r - 70.0)
    else:
        t_path = 7.8 + 0.04 * (r - 130.0)
    t_src = src.source_duration
    # Rupture directivity compresses or stretches the observed duration
    t = t_src / src.directivity_factor() + t_path
    return max(t, 0.15)


# ---------------------------------------------------------------------------
# Random Vibration Theory
# ---------------------------------------------------------------------------
def spectral_moment(freqs: np.ndarray, fas: np.ndarray, order: int) -> float:
    """m_k = 2 * integral (2 pi f)^k |A(f)|^2 df."""
    w = (2.0 * math.pi * np.maximum(freqs, 1e-9)) ** order
    return 2.0 * float(np.trapezoid(w * fas ** 2, freqs))


def _clh_peak_factor(n_extrema: float, xi: float) -> float:
    """Cartwright & Longuet-Higgins (1956) expected peak / rms.

    pf = sqrt(2) * integral_0^inf { 1 - [1 - xi*exp(-z^2)]^N } dz
    """
    n = max(n_extrema, 1.5)
    xi = min(max(xi, 1e-6), 1.0)
    z = np.linspace(0.0, 12.0, 3000)
    inner = np.clip(1.0 - xi * np.exp(-z ** 2), 0.0, 1.0)
    with np.errstate(divide="ignore"):
        integrand = 1.0 - np.exp(n * np.log(np.maximum(inner, 1e-300)))
    return math.sqrt(2.0) * float(np.trapezoid(integrand, z))


def rvt_peak(freqs: np.ndarray, fas: np.ndarray, duration: float) -> float:
    """Expected peak value of the time series whose FAS is `fas`."""
    m0 = spectral_moment(freqs, fas, 0)
    m2 = spectral_moment(freqs, fas, 2)
    m4 = spectral_moment(freqs, fas, 4)
    if m0 <= 0.0 or m2 <= 0.0 or duration <= 0.0:
        return 0.0

    rms = math.sqrt(m0 / duration)
    n_zero = max(duration / math.pi * math.sqrt(m2 / m0), 1.33)
    n_extr = max(duration / math.pi * math.sqrt(max(m4, 1e-30) / m2), n_zero)
    xi = n_zero / n_extr
    return _clh_peak_factor(n_extr, xi) * rms


def _default_freqs(fmin: float = 0.005, fmax: float = 100.0,
                   n: int = 900) -> np.ndarray:
    return np.geomspace(fmin, fmax, n)


def calibrated_fas(freqs: np.ndarray, src: Source, site: Site,
                   r_epi_km: float, component: str = "h",
                   anchor=None) -> np.ndarray:
    """Acceleration FAS with the GMPE anchor applied (when one is in force)."""
    from .gmpe import solve_anchor
    if anchor is None:
        anchor = solve_anchor(src, site, r_epi_km, component)

    rock = Site.from_class("B")
    dur = ground_motion_duration(src, r_epi_km)
    pga_rock_g = rvt_peak(freqs,
                          acceleration_fas(freqs, src, rock, r_epi_km, component),
                          dur) / K.G0
    fas = acceleration_fas(freqs, src, site, r_epi_km, component,
                           nonlinear_pga_g=pga_rock_g)
    return anchor.apply(freqs, fas)


def peak_ground_motion(src: Source, site: Site, r_epi_km: float,
                       component: str = "h", anchor=None) -> dict:
    """PGA / PGV / PGD and the intensity measures derived from them."""
    f = _default_freqs()
    dur = ground_motion_duration(src, r_epi_km)

    # First pass on rock to get the reference PGA that drives soil nonlinearity
    rock = Site.from_class("B")
    a_rock = acceleration_fas(f, src, rock, r_epi_km, component)
    pga_rock_g = rvt_peak(f, a_rock, dur) / K.G0

    a = calibrated_fas(f, src, site, r_epi_km, component, anchor)
    w = 2.0 * math.pi * f
    v = a / w
    d = a / w ** 2

    pga = rvt_peak(f, a, dur)                 # m/s^2
    pgv = rvt_peak(f, v, dur)                 # m/s
    pgd = rvt_peak(f, d, dur)                 # m

    return {
        "duration_s": dur,
        "pga_ms2": pga,
        "pga_g": pga / K.G0,
        "pga_cms2": pga * 100.0,
        "pgv_ms": pgv,
        "pgv_cms": pgv * 100.0,
        "pgd_m": pgd,
        "pgd_cm": pgd * 100.0,
        "pga_rock_g": pga_rock_g,
        "site_amp_pga": (pga / K.G0) / max(pga_rock_g, 1e-9),
    }


def response_spectrum(src: Source, site: Site, r_epi_km: float,
                      periods: np.ndarray | None = None,
                      damping: float = 0.05,
                      component: str = "h",
                      anchor=None) -> tuple[np.ndarray, np.ndarray]:
    """5%-damped pseudo-acceleration response spectrum Sa(T), in g.

    Uses RVT on the oscillator transfer function -- the same operation a
    structural engineer does with a real record, done analytically.
    """
    if periods is None:
        periods = np.geomspace(0.02, 10.0, 60)
    f = _default_freqs()
    dur_gm = ground_motion_duration(src, r_epi_km)

    a = calibrated_fas(f, src, site, r_epi_km, component, anchor)

    sa = np.zeros_like(periods, dtype=float)
    for i, t in enumerate(periods):
        fn = 1.0 / t
        r = f / fn
        # SDOF oscillator relative-displacement transfer, times omega_n^2
        h = 1.0 / np.sqrt((1.0 - r ** 2) ** 2 + (2.0 * damping * r) ** 2)
        osc_fas = a * h
        # Boore & Joyner (1984) r.m.s. duration correction for the oscillator's
        # own ringing, which outlasts the ground motion at long periods.
        t_osc = t / (2.0 * math.pi * damping)
        z = (dur_gm / t) ** 3
        dur_rms = dur_gm + t_osc * (z / (z + 1.0 / 3.0))
        sa[i] = rvt_peak(f, osc_fas, dur_rms) / K.G0
    return periods, sa
