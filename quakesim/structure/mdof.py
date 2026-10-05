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

"""Multi-degree-of-freedom building response.

The room you are standing in does not move the way the ground moves.  A tall
building is a filter: it takes the ground motion, amplifies whatever is near
its own natural period, and delivers something slower and larger to the upper
floors.  That is why a distant megathrust barely felt at street level makes
level 25 of a KL condominium sway for four minutes, while a sharp local quake
that wrecks a bungalow leaves the high-rise almost unbothered.

Model: a lumped-mass shear building.  While every story stays elastic the
response is solved exactly by modal superposition in the frequency domain
(instant).  Once any story yields, the solver switches to Newmark's average-
acceleration method (gamma=1/2, beta=1/4) with bilinear kinematic-hardening
story springs, using a tridiagonal solve because a shear building's stiffness
matrix is tridiagonal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

from ..seismo import constants as K

# ---------------------------------------------------------------------------
# ASCE 7-16 Table 12.8-2 approximate period parameters (SI, height in metres)
# ---------------------------------------------------------------------------
PERIOD_COEFF = {
    "steel_moment_frame": (0.0724, 0.80),
    "concrete_moment_frame": (0.0466, 0.90),
    "steel_braced_frame": (0.0731, 0.75),
    "concrete_shear_wall": (0.0488, 0.75),
    "masonry": (0.0488, 0.75),
    "timber": (0.0488, 0.75),
    "steel_rack": (0.0724, 0.80),
    "nonductile_rc": (0.0466, 0.90),
}

# Yield drift ratio -- the inter-story drift at which the lateral system first
# yields. This is a property of the material and member proportions, not of the
# code load level (Priestley 1998, "Displacement-based approach to rational
# seismic design"), and it is what actually decides when damage begins. Note
# each value sits just above the corresponding "Slight" damage threshold below,
# which is exactly the relationship observed in real buildings.
YIELD_DRIFT = {
    "concrete_moment_frame": 0.0060,
    "steel_moment_frame": 0.0090,
    "concrete_shear_wall": 0.0025,
    "masonry": 0.0005,
    "timber": 0.0060,
    "steel_rack": 0.0120,
    "nonductile_rc": 0.0040,
}

# Observed lateral capacity, as base shear at yield divided by seismic weight,
# including overstrength. A stiff structure can reach its yield *drift* at an
# implausibly large force if stiffness and drift are set independently, so this
# caps the result at what the material can actually deliver. Unreinforced
# masonry in particular is stiff and weak, which is exactly why it fails.
CAPACITY_LIMITS = {
    "concrete_moment_frame": (0.05, 0.30),
    "steel_moment_frame": (0.05, 0.30),
    "concrete_shear_wall": (0.06, 0.40),
    "masonry": (0.06, 0.32),
    "timber": (0.06, 0.45),
    "steel_rack": (0.08, 0.45),
    "nonductile_rc": (0.04, 0.15),
}

# HAZUS-style inter-story drift ratio thresholds by construction type
DAMAGE_DRIFT = {
    "concrete_moment_frame": (0.0033, 0.0058, 0.0156, 0.0400),
    "steel_moment_frame": (0.0060, 0.0104, 0.0235, 0.0600),
    "concrete_shear_wall": (0.0020, 0.0038, 0.0100, 0.0250),
    "masonry": (0.0020, 0.0032, 0.0080, 0.0200),
    "timber": (0.0040, 0.0099, 0.0306, 0.0750),
    "steel_rack": (0.0080, 0.0150, 0.0350, 0.0700),
    # pre-code concrete frame with brick infill (HAZUS C3L, low code):
    # brittle, and a soft storey collapses at half the drift of a ductile one
    "nonductile_rc": (0.0020, 0.0040, 0.0100, 0.0200),
}
DAMAGE_NAMES = ["None", "Slight", "Moderate", "Extensive", "Complete"]


# ---------------------------------------------------------------------------
@dataclass
class Building:
    """A shear-building idealisation of a real structure."""

    name: str = "Generic building"
    stories: int = 10
    story_height: float = 3.2               # m
    floor_mass: float = 400_000.0           # kg per floor
    system: str = "concrete_moment_frame"
    damping: float = 0.05                   # first-mode fraction of critical
    base_shear_coeff: float = 0.10          # yield base shear / weight
    post_yield_ratio: float = 0.05
    ductile: bool = True
    period_override: float | None = None
    # Relative storey stiffnesses, bottom first, overriding the default
    # taper. A soft ground storey -- an open shopfront under infilled upper
    # floors -- is (0.35, 1, 1, 1): the frame alone below, frame plus brick
    # above. That storey then takes most of the drift, which is the whole
    # mechanism.
    stiffness_shape: tuple | None = None

    @property
    def height(self) -> float:
        return self.stories * self.story_height

    @property
    def total_mass(self) -> float:
        return float(np.sum(self.mass_vector()))

    def approximate_period(self) -> float:
        if self.period_override:
            return self.period_override
        ct, x = PERIOD_COEFF.get(self.system, (0.0488, 0.75))
        return ct * self.height ** x

    # --- matrices ---------------------------------------------------------
    def mass_vector(self) -> np.ndarray:
        m = np.full(self.stories, float(self.floor_mass))
        m[-1] *= 0.7
        return m

    def _shape(self) -> np.ndarray:
        n = self.stories
        if self.stiffness_shape is not None and len(self.stiffness_shape) == n:
            return np.array(self.stiffness_shape, dtype=float)
        return np.linspace(1.0, 0.45, n) if n > 1 else np.array([1.0])

    def stiffness_vector(self) -> np.ndarray:
        k = self._shape()
        t1 = self._period_of(k)
        return k * (t1 / self.approximate_period()) ** 2

    def _assemble(self, kvec: np.ndarray) -> np.ndarray:
        n = self.stories
        A = np.zeros((n, n))
        for i in range(n):
            A[i, i] += kvec[i]
            if i + 1 < n:
                A[i, i] += kvec[i + 1]
                A[i, i + 1] -= kvec[i + 1]
                A[i + 1, i] -= kvec[i + 1]
        return A

    def _period_of(self, kvec: np.ndarray) -> float:
        m = self.mass_vector()
        s = np.diag(1.0 / np.sqrt(m))
        w2 = np.linalg.eigvalsh(s @ self._assemble(kvec) @ s)
        return 2.0 * math.pi / math.sqrt(max(float(np.min(w2)), 1e-12))

    def modes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(periods, mode shapes as columns, modal participation factors)."""
        m = self.mass_vector()
        s = np.diag(1.0 / np.sqrt(m))
        w2, vecs = np.linalg.eigh(s @ self._assemble(self.stiffness_vector()) @ s)
        order = np.argsort(np.maximum(w2, 1e-12))
        w = np.sqrt(np.maximum(w2[order], 1e-12))
        phi = s @ vecs[:, order]
        for j in range(phi.shape[1]):
            if phi[-1, j] < 0:
                phi[:, j] *= -1.0
            phi[:, j] /= (np.max(np.abs(phi[:, j])) or 1.0)
        mm = np.array([float(phi[:, j] @ (m * phi[:, j])) for j in range(phi.shape[1])])
        gamma = np.array([float(phi[:, j] @ m) / mm[j] for j in range(phi.shape[1])])
        return 2.0 * math.pi / w, phi, gamma

    def yield_drift_ratio(self) -> float:
        return YIELD_DRIFT.get(self.system, 0.006)

    def yield_shears(self) -> np.ndarray:
        """Story yield strengths.

        Taken as the shear at which each story reaches its yield drift, so the
        elastic-plastic transition is consistent with the drift-based damage
        thresholds. The code force pattern is used only to redistribute a
        little strength toward the base, where real buildings are stronger.
        """
        n = self.stories
        k = self.stiffness_vector()
        dy = self.yield_drift_ratio() * self.story_height
        v_drift = k * dy

        w = self.mass_vector() * K.G0
        h = np.arange(1, n + 1) * self.story_height
        cvx = (w * h) / float(np.sum(w * h))
        forces = cvx * (self.base_shear_coeff * float(np.sum(w)))
        v_code = np.cumsum(forces[::-1])[::-1]

        # Blend: the drift criterion sets the level, the code pattern the shape.
        shape = v_code / max(float(v_code[0]), 1e-9)
        vy = 0.5 * v_drift + 0.5 * (float(v_drift[0]) * shape)

        # Clamp to the material's real capacity
        weight = float(np.sum(self.mass_vector())) * K.G0
        lo, hi = CAPACITY_LIMITS.get(self.system, (0.05, 0.35))
        cy = float(vy[0]) / weight
        if cy > hi:
            vy = vy * (hi / cy)
        elif cy < lo:
            vy = vy * (lo / cy)
        return vy

    def implied_base_shear_coeff(self) -> float:
        return float(self.yield_shears()[0]) / (float(np.sum(self.mass_vector())) * K.G0)

    def effective_yield_drift(self) -> float:
        """Story drift ratio at first yield, after the capacity clamp."""
        k = self.stiffness_vector()
        return float(np.min(self.yield_shears() / k)) / self.story_height

    def modal_mass_ratios(self) -> np.ndarray:
        m = self.mass_vector()
        _, phi, gamma = self.modes()
        mm = np.array([float(phi[:, j] @ (m * phi[:, j])) for j in range(phi.shape[1])])
        eff = gamma ** 2 * mm
        return eff / float(np.sum(m))


@dataclass
class Response:
    dt: float
    floor_acc: np.ndarray      # (stories, N) absolute acceleration, m/s^2
    floor_disp: np.ndarray     # (stories, N) displacement relative to ground, m
    drift: np.ndarray          # (stories, N) inter-story drift ratio
    peak_drift: np.ndarray
    residual_drift: np.ndarray
    yielded: np.ndarray
    periods: np.ndarray
    modes: np.ndarray
    damage_state: int = 0
    damage_name: str = "None"
    collapsed: bool = False
    nonlinear_used: bool = False
    # The orthogonal horizontal component, solved separately. A building has
    # two horizontal directions and an earthquake shakes both at once; scaling
    # one by the other's ratio (as a single-component solution forces you to)
    # produces sign flips and spikes wherever the reference component passes
    # through zero.
    floor_acc_y: np.ndarray | None = None
    floor_disp_y: np.ndarray | None = None
    drift_y: np.ndarray | None = None

    @property
    def stories(self) -> int:
        return self.floor_acc.shape[0]

    def acc_at(self, story: int, t: float) -> float:
        i = int(t / self.dt)
        if i < 0 or i >= self.floor_acc.shape[1]:
            return 0.0
        return float(self.floor_acc[min(story, self.stories - 1), i])

    def disp_at(self, story: int, t: float) -> float:
        i = int(t / self.dt)
        if i < 0 or i >= self.floor_disp.shape[1]:
            return 0.0
        return float(self.floor_disp[min(story, self.stories - 1), i])


# ---------------------------------------------------------------------------
def _decimate(x: np.ndarray, dt: float, target_fs: float = 50.0):
    """Anti-aliased decimation.

    Buildings do not respond meaningfully above about 15 Hz, so integrating at
    50 Hz is exact for our purposes. Very long records (a megathrust runs for
    minutes) drop to 25 Hz, which is still four times the highest structural
    mode of interest and keeps the nonlinear solve interactive.
    """
    fs = 1.0 / dt
    if x.size * dt > 120.0:
        target_fs = min(target_fs, 25.0)
    if fs <= target_fs * 1.2:
        return x, dt, 1
    q = max(int(round(fs / target_fs)), 1)
    n = x.size
    X = np.fft.rfft(x)
    X[n // (2 * q):] = 0.0
    return np.fft.irfft(X, n=n)[::q], dt * q, q


class _BandedSolver:
    """Cached LU factorisations of the effective tangent matrix.

    A bilinear building spends most of its time in one yield state, so the
    effective matrix only changes when a story crosses yield. Factorising once
    per distinct state instead of once per timestep is the difference between
    twenty seconds and one.
    """

    def __init__(self):
        from scipy.linalg import lapack
        self._trf = lapack.dgbtrf
        self._trs = lapack.dgbtrs
        self._cache: dict[bytes, tuple] = {}

    def solve(self, key: bytes, build_ab, rhs: np.ndarray) -> np.ndarray:
        hit = self._cache.get(key)
        if hit is None:
            ab = build_ab()
            n = rhs.size
            # LAPACK banded storage needs kl extra rows on top for the LU fill-in
            abf = np.zeros((2 * 1 + 1 + 1, n))
            abf[1:, :] = ab
            lu, ipiv, info = self._trf(abf, 1, 1)
            if info != 0:
                from scipy.linalg import solve_banded
                return solve_banded((1, 1), ab, rhs)
            hit = (lu, ipiv)
            if len(self._cache) < 256:
                self._cache[key] = hit
        lu, ipiv = hit
        x, info = self._trs(lu, 1, 1, rhs, ipiv)
        return x if info == 0 else rhs * 0.0


def _modal_linear(b: Building, ag: np.ndarray, dt: float):
    """Exact elastic response by modal superposition in the frequency domain.

    The record is zero-padded by enough free-vibration time for the response
    to die away before the FFT wraps round; without that, a long record whose
    coda is cut by the duration cap has its ringing tail folded onto t = 0,
    and the building is already swaying before the P wave arrives.
    """
    n = b.stories
    periods, phi, gamma = b.modes()
    nt = ag.size
    zeta = max(b.damping, 0.01)
    # time for the fundamental mode to decay to 1% (ln 100 / (zeta * w1))
    settle = math.log(100.0) * periods[0] / (2.0 * math.pi * zeta)
    pad = int(min(max(settle, 20.0), 300.0) / dt)
    ag_p = np.concatenate([ag, np.zeros(pad)])
    ntp = ag_p.size
    f = np.fft.rfftfreq(ntp, dt)
    w = 2.0 * math.pi * f
    AG = np.fft.rfft(ag_p)

    u = np.zeros((n, nt))
    acc_rel = np.zeros((n, nt))
    for j in range(n):
        wj = 2.0 * math.pi / periods[j]
        zj = b.damping * (1.0 + 0.25 * j / max(n - 1, 1))   # higher modes damp more
        h = 1.0 / (wj ** 2 - w ** 2 + 2j * zj * wj * w)
        Q = -gamma[j] * AG * h
        q = np.fft.irfft(Q, n=ntp)[:nt]
        qdd = np.fft.irfft(Q * (-(w ** 2)), n=ntp)[:nt]
        u += np.outer(phi[:, j], q)
        acc_rel += np.outer(phi[:, j], qdd)
    return u, acc_rel, periods, phi


def _tridiag_bands(kt: np.ndarray, mdiag: np.ndarray, c_scale: float,
                   k_damp: np.ndarray, coef_m: float, coef_c: float
                   ) -> np.ndarray:
    """Build the banded (3, n) representation of Keff for solve_banded."""
    n = kt.size
    main = np.empty(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    for i in range(n):
        kk = kt[i] + (kt[i + 1] if i + 1 < n else 0.0)
        kd = k_damp[i] + (k_damp[i + 1] if i + 1 < n else 0.0)
        main[i] = coef_m * mdiag[i] + coef_c * (c_scale * mdiag[i] + kd) + kk
    for i in range(n - 1):
        off = -kt[i + 1]
        offd = -k_damp[i + 1]
        upper[i + 1] = off + coef_c * offd
        lower[i] = off + coef_c * offd
    return np.vstack([upper, main, lower])


def solve_2d(building: Building, ground_acc_x: np.ndarray,
             ground_acc_y: np.ndarray, dt: float,
             nonlinear: bool = True, progress=None) -> Response:
    """Solve both horizontal directions and return one combined Response.

    `progress(fraction)`, if given, is called as the solution advances
    (the loading screen)."""
    px = (lambda f: progress(0.5 * f)) if progress else None
    py = (lambda f: progress(0.5 + 0.5 * f)) if progress else None
    rx = solve(building, ground_acc_x, dt, nonlinear, progress=px)
    ry = solve(building, ground_acc_y, dt, nonlinear, progress=py)
    rx.floor_acc_y = ry.floor_acc
    rx.floor_disp_y = ry.floor_disp
    rx.drift_y = ry.drift
    # Damage is governed by the worse of the two directions.
    rx.peak_drift = np.maximum(rx.peak_drift, ry.peak_drift)
    rx.residual_drift = np.maximum(rx.residual_drift, ry.residual_drift)
    rx.yielded = rx.yielded | ry.yielded
    thr = DAMAGE_DRIFT.get(building.system, (0.003, 0.006, 0.016, 0.04))
    pk = float(np.max(rx.peak_drift))
    ds = sum(1 for t in thr if pk >= t)
    rx.damage_state = ds
    rx.damage_name = DAMAGE_NAMES[ds]
    rx.collapsed = ds >= 4
    rx.nonlinear_used = rx.nonlinear_used or ry.nonlinear_used
    return rx


def solve(building: Building, ground_acc: np.ndarray, dt: float,
          nonlinear: bool = True, progress=None) -> Response:
    """Response of the building to a ground accelerogram (one direction)."""
    ag_full = np.asarray(ground_acc, dtype=float)
    ag, dts, q = _decimate(ag_full, dt)
    n = building.stories
    nt = ag.size

    m = building.mass_vector()
    k0 = building.stiffness_vector()
    vy = building.yield_shears()
    alpha = building.post_yield_ratio
    dy = vy / k0

    # --- elastic pass -----------------------------------------------------
    u, acc_rel, periods, phi = _modal_linear(building, ag, dts)
    drift = np.empty_like(u)
    drift[0] = u[0]
    if n > 1:
        drift[1:] = u[1:] - u[:-1]
    peak_force = np.max(np.abs(k0[:, None] * drift), axis=1)
    will_yield = bool(np.any(peak_force > vy)) and nonlinear

    if not will_yield:
        if progress:
            progress(1.0)
        acc_abs = acc_rel + ag[None, :]
        return _finish(building, dt, dts, q, len(ag_full), acc_abs, u,
                       drift / building.story_height,
                       np.zeros(n, dtype=bool), periods, phi, False)

    # --- nonlinear Newmark ------------------------------------------------
    gamma_n, beta = 0.5, 0.25
    w1 = 2.0 * math.pi / periods[0]
    wj = 2.0 * math.pi / periods[min(2, n - 1)]
    if abs(wj - w1) < 1e-9:
        a0c, a1c = 2.0 * building.damping * w1, 0.0
    else:
        a0c = 2.0 * building.damping * w1 * wj / (w1 + wj)
        a1c = 2.0 * building.damping / (w1 + wj)

    uu = np.zeros(n); vv = np.zeros(n); aa = np.zeros(n)
    up = np.zeros(n)
    yielded = np.zeros(n, dtype=bool)
    kcur = k0.copy(); vycur = vy.copy()

    out_acc = np.zeros((n, nt)); out_u = np.zeros((n, nt)); out_d = np.zeros((n, nt))
    coef_m = 1.0 / (beta * dts ** 2)
    coef_c = gamma_n / (beta * dts)
    dlimit = DAMAGE_DRIFT.get(building.system, (0.003, 0.006, 0.016, 0.04))[2]
    degraded = False

    def drift_of(x):
        d = np.empty(n); d[0] = x[0]
        if n > 1:
            d[1:] = x[1:] - x[:-1]
        return d

    # A negative post-yield ratio is strength degradation: unreinforced masonry
    # and other brittle systems shed capacity as they crack, rather than
    # holding load like a ductile frame. Capacity never falls below the
    # residual friction the rubble still provides.
    resid_floor = 0.15

    def story_force(d):
        el = d - up
        f = kcur * el
        lim = vycur * (1.0 + alpha * np.abs(up) / np.maximum(dy, 1e-12))
        lim = np.maximum(lim, resid_floor * vycur)
        over = np.abs(f) > lim
        f = np.where(over,
                     np.sign(f) * lim + alpha * kcur * (np.abs(el) - lim / kcur) * np.sign(f),
                     f)
        kt = np.where(over, np.maximum(alpha, 0.02) * kcur, kcur)
        return f, kt, over

    def internal(x):
        d = drift_of(x)
        fs, kt, over = story_force(d)
        r = np.empty(n)
        if n > 1:
            r[:-1] = fs[:-1] - fs[1:]
        r[-1] = fs[-1]
        return r, kt, d, over

    r0, kt0, _, _ = internal(uu)
    cv = a0c * m * vv + a1c * (_bandmul(kt0, vv))
    aa = (-m * ag[0] - cv - r0) / m

    bs = _BandedSolver()
    kd = k0 * a1c

    report_every = max(1, nt // 40)
    for it in range(nt):
        if progress and it % report_every == 0:
            progress(it / nt)
        p = -m * ag[it]
        x = uu + dts * vv + (0.5 - beta) * dts ** 2 * aa   # predictor
        for _ in range(8):
            r, kt, d, over = internal(x)
            du = x - uu
            v_new = coef_c * du + (1 - gamma_n / beta) * vv \
                + dts * (1 - gamma_n / (2 * beta)) * aa
            a_new = coef_m * du - vv / (beta * dts) - (1 / (2 * beta) - 1) * aa
            cvv = a0c * m * v_new + a1c * _bandmul(k0, v_new)
            resid = p - m * a_new - cvv - r
            if float(np.max(np.abs(resid))) < 1e-4 * max(float(np.max(np.abs(p))), 1.0):
                break
            key = np.packbits(over).tobytes()
            x = x + bs.solve(key,
                             lambda kt=kt: _tridiag_bands(kt, m, a0c, kd,
                                                          coef_m, coef_c),
                             resid)

        r, kt, d, over = internal(x)
        el = d - up
        lim = vycur * (1.0 + alpha * np.abs(up) / np.maximum(dy, 1e-12))
        f_lin = kcur * el
        ov = np.abs(f_lin) > lim
        up = np.where(ov, up + (np.abs(f_lin) - lim) / kcur * np.sign(f_lin), up)
        yielded |= ov

        du = x - uu
        vv = coef_c * du + (1 - gamma_n / beta) * vv \
            + dts * (1 - gamma_n / (2 * beta)) * aa
        # Acceleration straight from dynamic equilibrium, which keeps the
        # absolute floor acceleration consistent with the forces that produced it.
        cvv = a0c * m * vv + a1c * _bandmul(k0, vv)
        aa = (p - cvv - r) / m
        uu = x

        out_u[:, it] = uu
        out_d[:, it] = d / building.story_height
        out_acc[:, it] = aa + ag[it]

        if (not degraded) and (not building.ductile) and \
                float(np.max(np.abs(d))) / building.story_height > dlimit:
            kcur = kcur * 0.05
            vycur = vycur * 0.05
            degraded = True

    if progress:
        progress(1.0)
    return _finish(building, dt, dts, q, len(ag_full), out_acc, out_u, out_d,
                   yielded, periods, phi, True)


def _bandmul(kvec: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Multiply the tridiagonal shear-building stiffness matrix by a vector."""
    n = kvec.size
    y = np.empty(n)
    kk = kvec + np.append(kvec[1:], 0.0)
    y[:] = kk * x
    if n > 1:
        y[:-1] -= kvec[1:] * x[1:]
        y[1:] -= kvec[1:] * x[:-1]
    return y


def _finish(b, dt, dts, q, n_full, acc, u, drift, yielded, periods, phi, nl):
    if q > 1:
        acc = np.repeat(acc, q, axis=1)[:, :n_full]
        u = np.repeat(u, q, axis=1)[:, :n_full]
        drift = np.repeat(drift, q, axis=1)[:, :n_full]
    peak = np.max(np.abs(drift), axis=1)
    residual = np.abs(drift[:, -1])
    thr = DAMAGE_DRIFT.get(b.system, (0.003, 0.006, 0.016, 0.04))
    pk = float(np.max(peak))
    ds = sum(1 for t in thr if pk >= t)
    return Response(dt=dt, floor_acc=acc, floor_disp=u, drift=drift,
                    peak_drift=peak, residual_drift=residual, yielded=yielded,
                    periods=periods, modes=phi, damage_state=ds,
                    damage_name=DAMAGE_NAMES[ds], collapsed=(ds >= 4),
                    nonlinear_used=nl)
