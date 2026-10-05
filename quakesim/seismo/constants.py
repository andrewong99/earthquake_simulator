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

"""Physical constants and crustal model parameters.

Units convention used throughout `quakesim.seismo`:
    moment          N.m
    stress drop     Pa internally, bar in user-facing API (1 bar = 1e5 Pa)
    distance        km in the public API, metres internally where noted
    density         kg/m^3
    wave speed      m/s internally, km/s in user-facing display
    acceleration    m/s^2 internally, reported in g and cm/s^2
    frequency       Hz
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------
G0 = 9.80665            # m/s^2, standard gravity
BAR = 1.0e5             # Pa per bar
KM = 1.0e3              # m per km
CM = 1.0e-2             # m per cm
JOULE_PER_TON_TNT = 4.184e9

# ---------------------------------------------------------------------------
# Generic continental crust (used as the default path model)
# ---------------------------------------------------------------------------
RHO_CRUST = 2800.0      # kg/m^3
VS_CRUST = 3500.0       # m/s, shear-wave speed in the source region
VP_CRUST = 6000.0       # m/s, compressional speed
VS_MANTLE = 4500.0
VP_MANTLE = 8100.0

# Rayleigh waves travel at ~0.92*Vs in a Poisson solid half-space; Love waves
# are bounded between the layer and half-space shear speeds.
RAYLEIGH_FACTOR = 0.9194
LOVE_FACTOR = 0.97

# ---------------------------------------------------------------------------
# Source radiation terms (Boore 2003, "Simulation of ground motion using the
# stochastic method", Pure appl. geophys. 160, 635-676)
# ---------------------------------------------------------------------------
RADIATION_S = 0.55      # RMS S-wave radiation pattern over the focal sphere
RADIATION_P = 0.52      # RMS P-wave radiation pattern
FREE_SURFACE = 2.0      # free-surface amplification
PARTITION = 1.0 / math.sqrt(2.0)   # energy split onto two horizontal components

# ---------------------------------------------------------------------------
# Anelastic attenuation, generic western-US-like crust
# ---------------------------------------------------------------------------
Q0_DEFAULT = 180.0
Q_ETA_DEFAULT = 0.45
Q_MIN = 60.0

# Hinged trilinear geometric spreading distances (Atkinson & Boore 1995)
R_HINGE1 = 70.0         # km
R_HINGE2 = 130.0        # km

# ---------------------------------------------------------------------------
# Magnitude scale definitions
# ---------------------------------------------------------------------------
# Hanks & Kanamori (1979), M0 in N.m
MW_MOMENT_OFFSET = 9.05

# Wood-Anderson torsion seismometer, as re-determined by Uhrhammer & Collins
# (1990). Richter's original nominal gain was 2800; 2080 is the measured value
# and is what modern ML practice uses.
WA_GAIN = 2080.0
WA_PERIOD = 0.8         # s
WA_DAMPING = 0.8        # fraction of critical

# Richter's -log10(A0) at 100 km is 3.0 by definition of the scale, i.e. an
# ML 3.0 event writes 1 mm peak on a Wood-Anderson at 100 km epicentral range.
ML_REF_DISTANCE = 100.0     # km
ML_REF_LOG_A0 = 3.0

__all__ = [n for n in dir() if not n.startswith("_")]
