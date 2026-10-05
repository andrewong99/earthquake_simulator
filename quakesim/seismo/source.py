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

"""Earthquake source: types, focal mechanism, rupture scaling, corner frequency.

The type catalogue below is not decoration -- each entry carries the physical
parameters (stress drop, depth band, attenuation, radiation partitioning,
spectral shape) that make that class of earthquake *sound and feel* different
in the simulation.  A deep intraslab event is jagged and high-frequency; a
megathrust is a slow three-minute roll; a mine collapse is a single downward
thump with almost no S wave.  All of that falls out of these numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Literal

from . import constants as K

Mechanism = Literal["strike_slip", "normal", "reverse", "oblique",
                    "isotropic", "implosive", "clvd"]


# The magnitude slider runs to this for every type. Each type also carries
# the band nature has actually produced (mw_range), which the interface shows
# as guidance; above it the source scaling extrapolates and the GMPE anchor
# saturates (it is capped at M8.5 inside the GMPE, as its authors did), so
# M10 on a crustal fault is a what-if with finite, saturated shaking.
MW_MAX = 10.0


# ---------------------------------------------------------------------------
# Type catalogue
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QuakeType:
    """A class of earthquake with its characteristic source physics."""

    key: str
    name: str
    mechanism: Mechanism

    depth_km: tuple[float, float]        # plausible hypocentral depth band
    depth_default: float
    stress_drop_bar: tuple[float, float]  # plausible Brune stress-drop band
    stress_drop_default: float

    mw_range: tuple[float, float]        # physically plausible magnitude range
                                         # (guidance only: see MW_MAX)
    q0: float = K.Q0_DEFAULT             # anelastic Q at 1 Hz along the path
    q_eta: float = K.Q_ETA_DEFAULT
    kappa0: float = 0.035                # near-site high-frequency decay (s)

    duration_factor: float = 1.0         # multiplies the source duration 1/fc
    spectral_n: float = 2.0              # omega^-n high-frequency falloff
    two_corner: bool = False             # use a two-corner (Atkinson) source

    p_s_ratio: float = 0.30              # P amplitude relative to S
    vertical_ratio: float = 0.60         # vertical / horizontal amplitude
    surface_wave_gain: float = 1.0       # Love/Rayleigh excitation efficiency

    blurb: str = ""

    def clamp_depth(self, d: float) -> float:
        return min(max(d, self.depth_km[0]), self.depth_km[1])

    def clamp_mw(self, m: float) -> float:
        """The model accepts any magnitude from the type's plausible minimum
        up to MW_MAX, whatever the type: the plausible band is guidance,
        not a limit, and the what-if beyond it is the point of a simulator."""
        return min(max(m, self.mw_range[0]), MW_MAX)


TYPES: dict[str, QuakeType] = {}


def _reg(t: QuakeType) -> QuakeType:
    TYPES[t.key] = t
    return t


# --- Shallow crustal tectonic ---------------------------------------------
_reg(QuakeType(
    key="strike_slip", name="Tectonic - strike-slip",
    mechanism="strike_slip",
    depth_km=(2.0, 20.0), depth_default=10.0,
    stress_drop_bar=(20.0, 200.0), stress_drop_default=70.0,
    mw_range=(2.0, 8.3), two_corner=True,
    q0=180.0, q_eta=0.45, kappa0=0.035,
    p_s_ratio=0.28, vertical_ratio=0.55, surface_wave_gain=1.0,
    blurb="Lateral slip on a near-vertical fault (San Andreas, North "
          "Anatolian). Strong horizontal shaking, modest vertical."))

_reg(QuakeType(
    key="normal", name="Tectonic - normal",
    mechanism="normal",
    depth_km=(2.0, 20.0), depth_default=9.0,
    stress_drop_bar=(10.0, 100.0), stress_drop_default=40.0,
    mw_range=(2.0, 7.8), two_corner=True,
    q0=200.0, q_eta=0.45, kappa0=0.038,
    p_s_ratio=0.30, vertical_ratio=0.68, surface_wave_gain=1.05,
    blurb="Crustal extension, hanging wall drops (Basin and Range, Apennines). "
          "Lower stress drop, noticeably more vertical motion."))

_reg(QuakeType(
    key="reverse", name="Tectonic - reverse / thrust",
    mechanism="reverse",
    depth_km=(2.0, 30.0), depth_default=12.0,
    stress_drop_bar=(30.0, 300.0), stress_drop_default=100.0,
    mw_range=(2.0, 8.2), two_corner=True,
    q0=170.0, q_eta=0.45, kappa0=0.033,
    p_s_ratio=0.32, vertical_ratio=0.75, surface_wave_gain=1.0,
    blurb="Compressional shortening, hanging wall thrust upward. Highest "
          "stress drops of the crustal families; violent vertical throw "
          "on the hanging wall."))

_reg(QuakeType(
    key="oblique", name="Tectonic - oblique slip",
    mechanism="oblique",
    depth_km=(2.0, 25.0), depth_default=11.0,
    stress_drop_bar=(20.0, 200.0), stress_drop_default=80.0,
    mw_range=(2.0, 8.0), two_corner=True,
    q0=180.0, q_eta=0.45, kappa0=0.035,
    p_s_ratio=0.30, vertical_ratio=0.65,
    blurb="Mixed lateral and dip slip - the most common real-world case."))

_reg(QuakeType(
    key="blind_thrust", name="Blind thrust (buried)",
    mechanism="reverse",
    depth_km=(4.0, 25.0), depth_default=15.0,
    stress_drop_bar=(50.0, 400.0), stress_drop_default=150.0,
    mw_range=(3.0, 7.5), two_corner=True,
    q0=160.0, q_eta=0.45, kappa0=0.030,
    p_s_ratio=0.33, vertical_ratio=0.80, surface_wave_gain=0.85,
    blurb="Fault never reaches the surface (Northridge 1994). Concentrated, "
          "high stress drop, savage vertical accelerations directly above."))

# --- Subduction ------------------------------------------------------------
_reg(QuakeType(
    key="megathrust", name="Subduction megathrust (interface)",
    mechanism="reverse",
    depth_km=(5.0, 60.0), depth_default=25.0,
    stress_drop_bar=(10.0, 60.0), stress_drop_default=30.0,
    mw_range=(6.0, 9.6), q0=380.0, q_eta=0.39, kappa0=0.045,
    duration_factor=2.2, two_corner=True,
    p_s_ratio=0.25, vertical_ratio=0.55, surface_wave_gain=1.6,
    blurb="Plate interface rupture (Sumatra 2004, Tohoku 2011, Chile 1960). "
          "Low stress drop but enormous area: minutes of long-period rolling "
          "that high-rises hundreds of km away feel as slow sway."))

_reg(QuakeType(
    key="intraslab", name="Subduction intraslab (in-slab)",
    mechanism="normal",
    depth_km=(40.0, 300.0), depth_default=90.0,
    stress_drop_bar=(100.0, 700.0), stress_drop_default=250.0,
    mw_range=(3.0, 8.0), q0=600.0, q_eta=0.35, kappa0=0.025,
    duration_factor=0.8,
    p_s_ratio=0.42, vertical_ratio=0.72, surface_wave_gain=0.35,
    blurb="Inside the descending slab (Nisqually 2001, Puebla 2017). Very "
          "high stress drop and an efficient deep path: sharp, rattling, "
          "high-frequency shaking felt over a wide area."))

_reg(QuakeType(
    key="outer_rise", name="Outer-rise normal",
    mechanism="normal",
    depth_km=(5.0, 40.0), depth_default=15.0,
    stress_drop_bar=(20.0, 120.0), stress_drop_default=50.0,
    mw_range=(5.0, 8.4), q0=400.0, q_eta=0.40, kappa0=0.045,
    p_s_ratio=0.28, vertical_ratio=0.70, surface_wave_gain=1.4,
    blurb="Plate bending seaward of the trench. Offshore, strongly "
          "tsunamigenic, felt on land as long-period motion."))

_reg(QuakeType(
    key="deep_focus", name="Deep-focus",
    mechanism="clvd",
    depth_km=(300.0, 700.0), depth_default=450.0,
    stress_drop_bar=(200.0, 1500.0), stress_drop_default=500.0,
    mw_range=(4.0, 8.3), q0=1200.0, q_eta=0.30, kappa0=0.020,
    duration_factor=0.7,
    p_s_ratio=0.55, vertical_ratio=0.85, surface_wave_gain=0.05,
    blurb="Deeper than 300 km, in the transition zone (Bolivia 1994 Mw 8.2). "
          "Almost no surface waves and little damage despite the size - the "
          "energy spreads through the mantle rather than along the surface."))

# --- Volcanic --------------------------------------------------------------
_reg(QuakeType(
    key="volcano_tectonic", name="Volcano-tectonic",
    mechanism="strike_slip",
    depth_km=(0.2, 10.0), depth_default=2.0,
    stress_drop_bar=(5.0, 80.0), stress_drop_default=25.0,
    mw_range=(0.5, 6.2), q0=90.0, q_eta=0.55, kappa0=0.050,
    p_s_ratio=0.35, vertical_ratio=0.62, surface_wave_gain=1.2,
    blurb="Brittle failure around a pressurising magma body. Shallow, so "
          "even a small one is sharply felt locally."))

_reg(QuakeType(
    key="volcano_lp", name="Volcanic long-period (LP)",
    mechanism="clvd",
    depth_km=(0.1, 8.0), depth_default=1.5,
    stress_drop_bar=(0.5, 15.0), stress_drop_default=4.0,
    mw_range=(0.5, 4.5), q0=45.0, q_eta=0.70, kappa0=0.080,
    duration_factor=3.5, spectral_n=3.0,
    p_s_ratio=0.20, vertical_ratio=0.45, surface_wave_gain=1.3,
    blurb="Resonance of a fluid-filled crack - magma or gas, not rock "
          "breaking. Narrowband, ringing, emergent onset with no sharp P."))

_reg(QuakeType(
    key="harmonic_tremor", name="Volcanic harmonic tremor",
    mechanism="clvd",
    depth_km=(0.1, 6.0), depth_default=1.0,
    stress_drop_bar=(0.2, 8.0), stress_drop_default=1.5,
    mw_range=(0.5, 3.5), q0=40.0, q_eta=0.75, kappa0=0.090,
    duration_factor=12.0, spectral_n=3.5,
    p_s_ratio=0.15, vertical_ratio=0.40, surface_wave_gain=1.2,
    blurb="Sustained banded shaking from continuous fluid movement. Can run "
          "for minutes to hours - a hum rather than a shock."))

# --- Anthropogenic and collapse -------------------------------------------
_reg(QuakeType(
    key="collapse", name="Mine / cavity collapse",
    mechanism="implosive",
    depth_km=(0.05, 2.0), depth_default=0.4,
    stress_drop_bar=(2.0, 60.0), stress_drop_default=15.0,
    mw_range=(0.5, 5.2), q0=70.0, q_eta=0.60, kappa0=0.060,
    duration_factor=0.45,
    p_s_ratio=0.85, vertical_ratio=1.25, surface_wave_gain=1.5,
    blurb="Roof falls into a void: the source implodes rather than shears. "
          "First motion is *downward* everywhere - a single hard thump with "
          "an unusually weak S wave."))

_reg(QuakeType(
    key="induced", name="Induced (injection / fracking)",
    mechanism="strike_slip",
    depth_km=(0.5, 8.0), depth_default=3.0,
    stress_drop_bar=(3.0, 90.0), stress_drop_default=20.0,
    mw_range=(0.5, 5.9), q0=130.0, q_eta=0.50, kappa0=0.045,
    p_s_ratio=0.30, vertical_ratio=0.60, surface_wave_gain=1.25,
    blurb="Fluid pressure unclamps a pre-existing fault. Small magnitude but "
          "very shallow, so shaking at the surface is disproportionate."))

_reg(QuakeType(
    key="reservoir", name="Reservoir-triggered",
    mechanism="normal",
    depth_km=(1.0, 15.0), depth_default=5.0,
    stress_drop_bar=(5.0, 120.0), stress_drop_default=35.0,
    mw_range=(1.0, 6.5), two_corner=True,
    q0=150.0, q_eta=0.48, kappa0=0.040,
    p_s_ratio=0.30, vertical_ratio=0.63, surface_wave_gain=1.15,
    blurb="Impounded water loads and lubricates the crust (Koyna 1967)."))

_reg(QuakeType(
    key="geothermal", name="Geothermal / EGS",
    mechanism="strike_slip",
    depth_km=(0.5, 6.0), depth_default=2.5,
    stress_drop_bar=(2.0, 50.0), stress_drop_default=12.0,
    mw_range=(0.5, 5.0), q0=110.0, q_eta=0.55, kappa0=0.050,
    duration_factor=0.8,
    p_s_ratio=0.32, vertical_ratio=0.58, surface_wave_gain=1.2,
    blurb="Enhanced-geothermal stimulation swarms (Basel 2006, Pohang 2017)."))

_reg(QuakeType(
    key="explosion", name="Explosion (chemical / nuclear)",
    mechanism="isotropic",
    depth_km=(0.02, 2.5), depth_default=0.6,
    stress_drop_bar=(50.0, 3000.0), stress_drop_default=600.0,
    mw_range=(0.5, 7.0), q0=100.0, q_eta=0.55, kappa0=0.040,
    duration_factor=0.30, spectral_n=2.0,
    p_s_ratio=3.20, vertical_ratio=1.45, surface_wave_gain=0.55,
    blurb="Isotropic pressure pulse: P wave dominates and S is nearly "
          "absent, the opposite of a tectonic quake. This P/S ratio is how "
          "test-ban monitoring tells bombs from earthquakes."))

_reg(QuakeType(
    key="landslide", name="Landslide / rockfall source",
    mechanism="clvd",
    depth_km=(0.01, 1.0), depth_default=0.1,
    stress_drop_bar=(0.2, 10.0), stress_drop_default=2.0,
    mw_range=(0.5, 5.5), q0=60.0, q_eta=0.70, kappa0=0.085,
    duration_factor=5.0, spectral_n=3.0,
    p_s_ratio=0.22, vertical_ratio=0.55, surface_wave_gain=1.5,
    blurb="A mass sliding and stopping loads the crust over tens of seconds. "
          "Very long-period, emergent, no impulsive arrival at all."))

_reg(QuakeType(
    key="icequake", name="Glacial / icequake",
    mechanism="clvd",
    depth_km=(0.01, 3.0), depth_default=0.2,
    stress_drop_bar=(0.5, 20.0), stress_drop_default=5.0,
    mw_range=(0.5, 5.2), q0=55.0, q_eta=0.65, kappa0=0.070,
    duration_factor=4.0, spectral_n=2.6,
    p_s_ratio=0.25, vertical_ratio=0.50, surface_wave_gain=1.45,
    blurb="Glacier calving or basal stick-slip. Long-period, surface-wave "
          "rich, and essentially invisible on short-period instruments."))

_reg(QuakeType(
    key="impact", name="Meteorite impact",
    mechanism="isotropic",
    depth_km=(0.0, 0.5), depth_default=0.02,
    stress_drop_bar=(100.0, 5000.0), stress_drop_default=1200.0,
    mw_range=(0.5, 8.0), q0=90.0, q_eta=0.60, kappa0=0.045,
    duration_factor=0.20,
    p_s_ratio=2.60, vertical_ratio=1.55, surface_wave_gain=0.85,
    blurb="Surface pressure source coupled with an air blast. Impulsive, "
          "P-dominated, with strong ground-coupled airwave arriving later."))

_reg(QuakeType(
    key="aftershock", name="Aftershock",
    mechanism="oblique",
    depth_km=(1.0, 30.0), depth_default=8.0,
    stress_drop_bar=(10.0, 150.0), stress_drop_default=45.0,
    mw_range=(0.5, 8.0), two_corner=True,
    q0=180.0, q_eta=0.45, kappa0=0.038,
    duration_factor=0.85,
    p_s_ratio=0.30, vertical_ratio=0.62,
    blurb="Stress readjustment after a mainshock. Lower average stress drop "
          "and, in a sequence, obeys Omori decay in rate and "
          "Bath's law in size (about 1.2 magnitude below the mainshock)."))

TYPE_ORDER = list(TYPES.keys())


# ---------------------------------------------------------------------------
# Rupture scaling
# ---------------------------------------------------------------------------
# Wells & Coppersmith (1994), BSSA 84(4), Table 2A: log10(Y) = a + b*Mw
_WC94 = {
    #                 length(km)        width(km)        area(km^2)      avg slip(m)
    "strike_slip":  ((-2.57, 0.62), (-0.76, 0.27), (-3.42, 0.90), (-6.32, 1.03)),
    "reverse":      ((-2.42, 0.58), (-1.61, 0.41), (-3.99, 0.98), (-0.74, 0.08)),
    "normal":       ((-1.88, 0.50), (-1.14, 0.35), (-2.87, 0.82), (-4.45, 0.63)),
    "all":          ((-2.44, 0.59), (-1.01, 0.32), (-3.49, 0.91), (-4.80, 0.69)),
}

# Strasser, Arango & Bommer (2010), SRL 81(6) - subduction-specific
_STRASSER_INTERFACE = ((-2.477, 0.585), (-0.882, 0.351), (-3.476, 0.952))
_STRASSER_INSLAB = ((-2.350, 0.562), (-1.058, 0.356), (-3.225, 0.890))


@dataclass
class Rupture:
    """Geometry and slip budget of the rupture."""
    length_km: float
    width_km: float
    area_km2: float
    avg_slip_m: float
    max_slip_m: float
    rise_time_s: float
    rupture_velocity_kms: float
    rupture_duration_s: float


def rupture_geometry(mw: float, qtype: QuakeType) -> Rupture:
    """Empirical rupture dimensions for a given magnitude and quake class."""
    def _p(ab: tuple[float, float]) -> float:
        return 10.0 ** (ab[0] + ab[1] * mw)

    if qtype.key == "megathrust":
        L = _p(_STRASSER_INTERFACE[0])
        W = _p(_STRASSER_INTERFACE[1])
        A = _p(_STRASSER_INTERFACE[2])
    elif qtype.key in ("intraslab", "outer_rise", "deep_focus"):
        L = _p(_STRASSER_INSLAB[0])
        W = _p(_STRASSER_INSLAB[1])
        A = _p(_STRASSER_INSLAB[2])
    else:
        mech = qtype.mechanism if qtype.mechanism in _WC94 else "all"
        c = _WC94[mech]
        L, W, A = _p(c[0]), _p(c[1]), _p(c[2])

    # Slip from the moment itself: M0 = mu * A * D -- self-consistent, unlike
    # taking the empirical displacement relation independently.
    mu = K.RHO_CRUST * K.VS_CRUST ** 2                      # rigidity, Pa
    m0 = moment_from_mw(mw)
    d_avg = m0 / (mu * A * 1.0e6) if A > 0 else 0.0          # m
    d_max = d_avg * 2.2                                      # typical peak/avg

    vr = 0.8 * K.VS_CRUST / 1000.0                           # km/s
    t_rup = L / vr if vr > 0 else 0.0
    rise = max(0.1, 0.05 * math.sqrt(max(A, 1.0)))

    return Rupture(L, W, A, d_avg, d_max, rise, vr, t_rup)


# ---------------------------------------------------------------------------
# Moment / magnitude / energy
# ---------------------------------------------------------------------------
def moment_from_mw(mw: float) -> float:
    """Seismic moment M0 in N.m from moment magnitude (Hanks & Kanamori 1979)."""
    return 10.0 ** (1.5 * mw + K.MW_MOMENT_OFFSET)


def mw_from_moment(m0: float) -> float:
    return (math.log10(max(m0, 1e-30)) - K.MW_MOMENT_OFFSET) / 1.5


def radiated_energy(mw: float) -> float:
    """Radiated seismic energy in joules (Gutenberg-Richter, Es/M0 ~ 5e-5)."""
    return 10.0 ** (1.5 * mw + 4.8)


def energy_tnt_tons(mw: float) -> float:
    return radiated_energy(mw) / K.JOULE_PER_TON_TNT


def corner_frequency(m0: float, stress_drop_pa: float,
                     beta: float = K.VS_CRUST) -> float:
    """Brune (1970) corner frequency in Hz.

    fc = 0.49 * beta * (dsigma / M0)^(1/3)  with beta in m/s, dsigma in Pa,
    M0 in N.m -- the SI form of the familiar 4.9e6*beta*(dsigma/M0)^(1/3).
    """
    if m0 <= 0.0:
        return 10.0
    return 0.4906 * beta * (stress_drop_pa / m0) ** (1.0 / 3.0)


# ---------------------------------------------------------------------------
# The source object the rest of the simulator consumes
# ---------------------------------------------------------------------------
@dataclass
class Source:
    """A fully specified earthquake source."""

    mw: float = 6.0
    qtype: QuakeType = field(default_factory=lambda: TYPES["strike_slip"])
    depth_km: float = 10.0
    stress_drop_bar: float = 70.0
    strike_deg: float = 0.0
    dip_deg: float = 90.0
    rake_deg: float = 0.0
    directivity: float = 0.0    # -1 away from site, +1 rupturing toward it

    # -- derived -----------------------------------------------------------
    @property
    def moment(self) -> float:
        return moment_from_mw(self.mw)

    @property
    def stress_drop_pa(self) -> float:
        return self.stress_drop_bar * K.BAR

    @property
    def fc(self) -> float:
        return corner_frequency(self.moment, self.stress_drop_pa)

    @property
    def rupture(self) -> Rupture:
        return rupture_geometry(self.mw, self.qtype)

    @property
    def source_duration(self) -> float:
        """Duration of the source-time function, seconds."""
        return self.qtype.duration_factor / max(self.fc, 1e-4)

    @property
    def energy_j(self) -> float:
        return radiated_energy(self.mw)

    def hypocentral_distance(self, epicentral_km: float) -> float:
        return math.hypot(epicentral_km, self.depth_km)

    def directivity_factor(self) -> float:
        """Simple Ben-Menahem Doppler-like factor on amplitude and duration.

        Rupture running toward the site compresses the radiation into a
        shorter, larger pulse; running away stretches and weakens it.
        """
        vr_over_beta = 0.8
        d = max(-1.0, min(1.0, self.directivity))
        return 1.0 / (1.0 - vr_over_beta * d * 0.75)

    def with_defaults_for_type(self, qtype: QuakeType) -> "Source":
        return replace(self, qtype=qtype,
                       depth_km=qtype.depth_default,
                       stress_drop_bar=qtype.stress_drop_default,
                       mw=qtype.clamp_mw(self.mw))

    def describe(self) -> dict:
        r = self.rupture
        return {
            "type": self.qtype.name,
            "mechanism": self.qtype.mechanism,
            "Mw": round(self.mw, 2),
            "M0_Nm": self.moment,
            "energy_J": self.energy_j,
            "energy_tons_TNT": energy_tnt_tons(self.mw),
            "depth_km": self.depth_km,
            "stress_drop_bar": self.stress_drop_bar,
            "corner_freq_Hz": self.fc,
            "source_duration_s": self.source_duration,
            "rupture_length_km": r.length_km,
            "rupture_width_km": r.width_km,
            "rupture_area_km2": r.area_km2,
            "avg_slip_m": r.avg_slip_m,
            "max_slip_m": r.max_slip_m,
            "rupture_velocity_kms": r.rupture_velocity_kms,
        }
