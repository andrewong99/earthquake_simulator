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

"""Time-domain ground-motion synthesis.

Turns the analytical spectrum into an actual three-component accelerogram --
the thing the physics engine shakes the world with.

Method: Boore's stochastic simulation. Filtered, windowed Gaussian noise whose
Fourier amplitude spectrum is forced to equal the target spectrum. Random
phase gives a different but statistically identical record every time, exactly
as two real earthquakes of the same size and distance produce different
squiggles with the same character.

On top of that, the record is decomposed into its real phases -- P, S, Love,
Rayleigh -- each with its own arrival time, duration, frequency content and
particle motion, then recombined onto East / North / Vertical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import constants as K
from . import phases as ph
from . import spectrum as sp
from .site import Site
from .source import Source


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------
# The most violent ground shaking on record is 3-4 g; nothing synthesised
# here goes above this, whatever the source (see synthesise()).
PGA_CEILING_G = 3.0


def saragoni_hart_window(t: np.ndarray, duration: float,
                         eps: float = 0.2, eta: float = 0.05) -> np.ndarray:
    """Boore's form of the Saragoni & Hart (1974) envelope.

    Rises quickly to a peak at `eps` of the window length, then decays to
    `eta` of the peak by the end -- the asymmetric shape every strong-motion
    record has.
    """
    tn = 2.0 * max(duration, 1e-3)
    b = -eps * math.log(eta) / (1.0 + eps * (math.log(eps) - 1.0))
    c = b / eps
    a = (math.e / eps) ** b
    x = np.clip(np.asarray(t, dtype=float) / tn, 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = a * np.power(x, b) * np.exp(-c * x)
    w[~np.isfinite(w)] = 0.0
    w[np.asarray(t) < 0.0] = 0.0
    return w


# ---------------------------------------------------------------------------
# Phase energy partition
# ---------------------------------------------------------------------------
def phase_weights(freqs: np.ndarray, src: Source, r_epi_km: float
                  ) -> dict[str, np.ndarray]:
    """Split the total spectrum into P / S / Love / Rayleigh, preserving energy.

    Surface waves need a horizontal path to develop, so they are suppressed
    directly above a deep source and grow as the epicentral distance exceeds
    a few source depths. They also live at long period; above ~1 Hz the
    record is all body waves.
    """
    f = np.maximum(np.asarray(freqs, dtype=float), 1e-6)
    qt = src.qtype

    # Geometric availability of surface waves
    geom = r_epi_km / (r_epi_km + 2.5 * max(src.depth_km, 0.2))
    depth_kill = math.exp(-max(src.depth_km - 40.0, 0.0) / 45.0)
    surf_amp = qt.surface_wave_gain * geom * depth_kill

    # Long-period dominance: surface waves take over below f_surf
    f_surf = 0.45
    surf_shape = 1.0 / (1.0 + (f / f_surf) ** 2.2)
    w_surf = np.clip(surf_amp * surf_shape, 0.0, 0.93)

    # P wave: flat fraction of S, rolling off at long period
    w_p = qt.p_s_ratio * (1.0 / (1.0 + (0.25 / f) ** 2)) ** 0.5
    w_p = np.clip(w_p, 0.0, 0.90)

    remaining = np.clip(1.0 - w_surf ** 2 - w_p ** 2, 1e-6, None)
    w_s = np.sqrt(remaining)

    # Love vs Rayleigh split (Love slightly stronger for strike-slip)
    love_frac = 0.62 if qt.mechanism == "strike_slip" else 0.48
    return {
        "P": w_p,
        "S": w_s,
        "Love": w_surf * love_frac,
        "Rayleigh": w_surf * math.sqrt(max(1.0 - love_frac ** 2, 0.0)),
    }


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class GroundMotion:
    """A synthesized three-component ground motion."""

    dt: float
    acc: np.ndarray                 # (3, N) East, North, Up  [m/s^2]
    vel: np.ndarray                 # [m/s]
    disp: np.ndarray                # [m]
    arrivals: dict[str, float] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.acc.shape[1]

    @property
    def duration(self) -> float:
        return self.n * self.dt

    @property
    def t(self) -> np.ndarray:
        return np.arange(self.n) * self.dt

    def pga_g(self) -> float:
        h = np.hypot(self.acc[0], self.acc[1])
        return float(np.max(h)) / K.G0

    def pgv_cms(self) -> float:
        h = np.hypot(self.vel[0], self.vel[1])
        return float(np.max(h)) * 100.0

    def pgd_cm(self) -> float:
        h = np.hypot(self.disp[0], self.disp[1])
        return float(np.max(h)) * 100.0

    def arias_intensity(self) -> float:
        """Arias intensity in m/s: pi/(2g) * integral a^2 dt, both horizontals."""
        s = np.sum(self.acc[0] ** 2 + self.acc[1] ** 2) * self.dt
        return math.pi / (2.0 * K.G0) * s

    def significant_duration(self, lo: float = 0.05, hi: float = 0.95) -> float:
        """Trifunac & Brady D5-95: the span holding the middle 90% of energy."""
        e = np.cumsum(self.acc[0] ** 2 + self.acc[1] ** 2)
        if e[-1] <= 0:
            return 0.0
        e = e / e[-1]
        i0 = int(np.searchsorted(e, lo))
        i1 = int(np.searchsorted(e, hi))
        return (i1 - i0) * self.dt

    def sample(self, time_s: float) -> np.ndarray:
        """Linearly interpolated acceleration vector [E, N, Up] at a time."""
        x = time_s / self.dt
        i = int(math.floor(x))
        if i < 0 or i >= self.n - 1:
            return np.zeros(3)
        a = x - i
        return self.acc[:, i] * (1.0 - a) + self.acc[:, i + 1] * a


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------
def _highpass_taper(f: np.ndarray, f_hp: float) -> np.ndarray:
    """Smooth 4th-order Butterworth-like high-pass, to keep integration stable."""
    x = np.maximum(f, 1e-9) / max(f_hp, 1e-9)
    return (x ** 4 / (1.0 + x ** 4)) ** 0.5


def synthesize(src: Source, site: Site, r_epi_km: float,
               azimuth_deg: float = 0.0,
               dt: float = 1.0 / 200.0,
               seed: int | None = None,
               normalize_to_target: bool = True,
               max_duration: float = 900.0) -> GroundMotion:
    """Generate a three-component accelerogram at a site."""
    rng = np.random.default_rng(seed)

    arr = ph.arrival_times(r_epi_km, src.depth_km)
    dur_gm = sp.ground_motion_duration(src, r_epi_km)

    # Total record: from before P until the dispersed surface train has passed
    t_end = max(arr["Rayleigh"] + 3.0 * dur_gm + 12.0,
                arr["S"] + 3.5 * dur_gm + 8.0)
    if r_epi_km > 100.0:
        slow_surface = r_epi_km * K.KM / (0.45 * K.VS_CRUST)
        t_end = max(t_end, slow_surface + 2.0 * dur_gm)
    t_end = min(t_end, max_duration)

    n = int(2 ** math.ceil(math.log2(max(t_end / dt, 256))))
    n = min(n, 1 << 21)
    t = np.arange(n) * dt
    f = np.fft.rfftfreq(n, dt)
    f_safe = np.maximum(f, 1e-9)

    # Target spectrum, GMPE-anchored where a GMPE applies
    from .gmpe import solve_anchor
    anchor = solve_anchor(src, site, r_epi_km)
    total_fas = sp.calibrated_fas(f_safe, src, site, r_epi_km, "h", anchor)
    total_fas[0] = 0.0

    f_hp = min(0.6 * src.fc, 0.02)
    total_fas = total_fas * _highpass_taper(f_safe, max(f_hp, 1.0 / (n * dt) * 2.0))

    weights = phase_weights(f_safe, src, r_epi_km)
    vp, vs = ph.body_wave_speeds(src.depth_km)

    acc_rtz = np.zeros((3, n))      # radial, transverse, vertical

    phase_specs = [
        ("P", arr["P"], max(0.35 * dur_gm, 0.4), None),
        ("S", arr["S"], dur_gm, None),
        ("Love", arr["Love"], 1.6 * dur_gm + 0.02 * r_epi_km, "love"),
        ("Rayleigh", arr["Rayleigh"], 1.8 * dur_gm + 0.03 * r_epi_km, "rayleigh"),
    ]

    for name, t_arr, pdur, disp_kind in phase_specs:
        w = weights[name]
        if float(np.max(w)) < 1e-4 or t_arr > t_end:
            continue

        target = total_fas * w
        if name == "P":
            # P is richer in high frequency: its corner sits above the S corner
            # and it attenuates along a faster, less lossy path.
            shift = 1.6
            target = target * (1.0 + (f_safe / (src.fc * shift)) ** -2) ** -0.0
            q = np.maximum(src.qtype.q0 * f_safe ** src.qtype.q_eta, K.Q_MIN)
            r_hyp = src.hypocentral_distance(r_epi_km)
            target = target * np.exp(
                -math.pi * f_safe * r_hyp * K.KM * (1.0 / (q * vp) - 1.0 / (q * vs)))

        # Windowed Gaussian noise -> spectrum forced to the target
        win = saragoni_hart_window(t - t_arr, pdur)
        if float(np.max(win)) <= 0.0:
            continue
        x = rng.standard_normal(n) * win
        X = np.fft.rfft(x)
        mag = np.abs(X)
        norm = math.sqrt(float(np.mean(mag ** 2))) or 1.0
        X = X / norm * target

        if disp_kind is not None:
            # Dispersion: shift each frequency by its own travel time, which
            # smears the packet out into the long-period-first surface train.
            delay = ph.dispersive_delay(f_safe, r_epi_km, disp_kind) - t_arr
            X = X * np.exp(-2j * math.pi * f * delay)

        a = np.fft.irfft(X, n=n)

        rad, tra, ver = ph.phase_partition(name, r_epi_km, src.depth_km, src.qtype)
        if name == "Rayleigh":
            # Retrograde elliptical motion: vertical leads radial by 90 degrees,
            # which is a Hilbert transform in the time domain.
            a_h = np.fft.irfft(X * (-1j) * np.sign(f_safe), n=n)
            acc_rtz[0] += rad * a_h
            acc_rtz[2] += ver * a
        else:
            acc_rtz[0] += rad * a
            acc_rtz[2] += ver * a
        acc_rtz[1] += tra * a if name != "Rayleigh" else 0.0

    # Rotate radial/transverse into East/North
    east, north = ph.rotate_to_en(acc_rtz[0], acc_rtz[1], azimuth_deg)
    acc = np.vstack([east, north, acc_rtz[2]])

    # Nothing arrives before P. Forcing the spectrum onto windowed noise
    # spreads a little energy over the whole record, wrap-around included,
    # so the record is eased in from a second before the first arrival.
    t_first = max(min(arr["P"], arr["S"]) - 1.0, 0.0)
    ramp = np.clip((t - t_first) / 1.0, 0.0, 1.0)
    acc *= (0.5 - 0.5 * np.cos(math.pi * ramp))[None, :]

    # Baseline: remove any DC and integrate in the frequency domain, which is
    # exact and avoids the drift that trapezoidal integration accumulates.
    vel = np.zeros_like(acc)
    disp = np.zeros_like(acc)
    hp = _highpass_taper(f_safe, max(f_hp, 2.0 / (n * dt)))
    for i in range(3):
        A = np.fft.rfft(acc[i])
        A[0] = 0.0
        A = A * hp
        acc[i] = np.fft.irfft(A, n=n)
        V = A / (2j * math.pi * f_safe)
        V[0] = 0.0
        V = V * hp
        vel[i] = np.fft.irfft(V, n=n)
        D = V / (2j * math.pi * f_safe)
        D[0] = 0.0
        D = D * hp
        disp[i] = np.fft.irfft(D, n=n)

    # The FFT length is a power of two, so the record can run well past
    # t_end (a 420 s cap becomes 655 s of mostly silence). Keep the record
    # to t_end, easing the last few seconds down where the cap has cut into
    # a long coda so the cut is not a step.
    n_keep = min(n, int(math.ceil(t_end / dt)))
    if n_keep < n:
        taper_n = max(2, min(n_keep // 4, int(8.0 / dt)))
        w = np.ones(n_keep)
        w[-taper_n:] = 0.5 * (1.0 + np.cos(np.linspace(0.0, math.pi, taper_n)))
        acc = acc[:, :n_keep] * w
        vel = vel[:, :n_keep] * w
        disp = disp[:, :n_keep] * w

    gm = GroundMotion(dt=dt, acc=acc, vel=vel, disp=disp,
                      arrivals={k: float(v) for k, v in arr.items()})

    if normalize_to_target:
        target_pk = sp.peak_ground_motion(src, site, r_epi_km, "h", anchor)
        realized = gm.pga_g()
        if realized > 1e-9:
            s = target_pk["pga_g"] / realized
            gm.acc *= s
            gm.vel *= s
            gm.disp *= s

    # Ground acceleration has a ceiling. The strongest motions ever recorded
    # are 3-4 g (Tohoku 2011, Christchurch 2011); a source spectrum scaled to
    # an M10 impact a kilometre away has none, and gave 550 g. Records that
    # exceed it are scaled down to it, whole, and say so.
    saturated = False
    if gm.pga_g() > PGA_CEILING_G:
        s = PGA_CEILING_G / gm.pga_g()
        gm.acc *= s
        gm.vel *= s
        gm.disp *= s
        saturated = True

    pk = sp.peak_ground_motion(src, site, r_epi_km, "h", anchor)
    from .intensity import mmi_from_pga_pgv, jma_from_motion, describe_mmi
    mmi = mmi_from_pga_pgv(gm.pga_g() * 981.0, gm.pgv_cms())
    gm.meta = {
        "anchor_mode": anchor.mode,
        "pga_g": gm.pga_g(),
        "pgv_cms": gm.pgv_cms(),
        "pgd_cm": gm.pgd_cm(),
        "predicted_pga_g": pk["pga_g"],
        "predicted_pgv_cms": pk["pgv_cms"],
        "arias_ms": gm.arias_intensity(),
        "d5_95_s": gm.significant_duration(),
        "mmi": mmi,
        "mmi_text": describe_mmi(mmi),
        "jma": jma_from_motion(gm),
        "site_amp": pk["site_amp_pga"],
        "record_length_s": gm.duration,
        "saturated": saturated,
    }
    return gm
