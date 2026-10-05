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

"""Seismology checks.

Three things worth pinning down:

  1. the magnitude / moment / energy relations reproduce their textbook values
  2. the corner frequency and rupture scaling land where the literature says
  3. where a GMPE legitimately applies, the simulated PGA and PGV *are* its
     medians rather than merely being near them

Number 3 is the important one. A point-source stochastic model cannot
reproduce a regression over ten thousand records on its own, so the model
takes its waveform shape from the physics and its amplitude from BSSA14. This
test is what proves the anchoring works.
"""

from __future__ import annotations

import math
import sys

import numpy as np

from quakesim.seismo import constants as K
from quakesim.seismo import gmpe, spectrum
from quakesim.seismo.intensity import (jma_class, mmi_from_pga_pgv)
from quakesim.seismo.phases import arrival_times, distance_from_sp_interval
from quakesim.seismo.site import Site
from quakesim.seismo.source import (TYPES, corner_frequency, energy_tnt_tons,
                                    moment_from_mw, mw_from_moment,
                                    radiated_energy, rupture_geometry, Source)
from quakesim.seismo.synth import synthesize

fails = 0


def check(name, got, want, tol, unit=""):
    global fails
    rel = abs(got - want) / abs(want) if want else abs(got - want)
    ok = rel <= tol
    fails += not ok
    print(f"   {name:44s} {got:12.4g}{unit}  want {want:10.4g}{unit}"
          f"  {rel*100:5.1f}%  {'ok' if ok else 'FAIL'}")


def main() -> int:
    print("\nMOMENT AND ENERGY")
    # Hanks & Kanamori: Mw = (2/3)(log10 M0 - 9.05), M0 in N.m
    check("M0 for Mw 6.0 (N.m)", moment_from_mw(6.0), 1.122e18, 0.01)
    check("M0 for Mw 9.0 (N.m)", moment_from_mw(9.0), 3.548e22, 0.01)
    check("round trip Mw(M0(7.3))", mw_from_moment(moment_from_mw(7.3)), 7.3, 1e-9)
    # Gutenberg-Richter: log10 Es = 1.5 M + 4.8
    check("radiated energy Mw 6.0 (J)", radiated_energy(6.0), 10 ** 13.8, 0.01)
    check("Es/M0 for Mw 7.0", radiated_energy(7.0) / moment_from_mw(7.0),
          5.6e-5, 0.10)
    # 1 Mt = 4.184e15 J
    check("Mw 8.0 in megatons TNT", energy_tnt_tons(8.0) / 1e6, 15.1, 0.05)

    print("\nSOURCE SCALING")
    # Brune: fc = 0.49 * beta * (dsigma/M0)^(1/3); M6 at 100 bar ~ 0.35 Hz
    fc = corner_frequency(moment_from_mw(6.0), 100 * K.BAR)
    check("corner frequency Mw 6.0, 100 bar (Hz)", fc, 0.355, 0.05)
    fc9 = corner_frequency(moment_from_mw(9.0), 30 * K.BAR)
    check("corner frequency Mw 9.0, 30 bar (Hz)", fc9, 0.0075, 0.10)
    check("  -> source duration Mw 9.0 (s)", 1.0 / fc9, 133.0, 0.10, " s")
    # Wells & Coppersmith strike-slip: log10 RLD = -2.57 + 0.62 M
    #   M7.0 -> 10^1.770 = 58.9 km, which brackets the observed 40-70 km range
    #   for real M7 strike-slip ruptures (Landers 1992 was about 85 km at M7.3,
    #   and the relation gives 90 km for that magnitude).
    r = rupture_geometry(7.0, TYPES["strike_slip"])
    check("rupture length Mw 7.0 strike-slip (km)", r.length_km, 58.9, 0.03)
    check("average slip Mw 7.0 (m)", r.avg_slip_m, 1.2, 0.35, " m")
    # Strasser et al. (2010) interface: log10 L = -2.477 + 0.585 M
    r9 = rupture_geometry(9.0, TYPES["megathrust"])
    check("megathrust Mw 9.0 rupture length (km)", r9.length_km, 614.0, 0.03)

    print("\nPHASE ARRIVALS")
    a = arrival_times(100.0, 10.0)
    check("S-P interval at 100 km (s)", a["S_minus_P"], 12.0, 0.10, " s")
    check("distance recovered from S-P (km)",
          distance_from_sp_interval(a["S_minus_P"]),
          math.hypot(100.0, 10.0), 0.02)

    print("\nGMPE ANCHORING  (BSSA14 medians, Vs30 = 760 m/s, strike-slip)")
    site = Site(name="ref", nehrp="B", vs30=760.0, kappa0=0.030)
    worst_a = worst_v = 0.0
    for mw in (4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0):
        for r_km in (5, 10, 20, 40, 80, 150, 300):
            src = Source(mw=mw, qtype=TYPES["strike_slip"], depth_km=8.0,
                         stress_drop_bar=70.0)
            anc = gmpe.solve_anchor(src, site, r_km)
            pk = spectrum.peak_ground_motion(src, site, r_km, anchor=anc)
            ta = gmpe.bssa14(mw, r_km, 760.0, "pga")
            tv = gmpe.bssa14(mw, r_km, 760.0, "pgv")
            worst_a = max(worst_a, abs(pk["pga_g"] / ta - 1.0))
            worst_v = max(worst_v, abs(pk["pgv_cms"] / tv - 1.0))
    check("worst PGA error over M4.5-8, R5-300 km", worst_a, 0.0, 0.02)
    check("worst PGV error over the same grid", worst_v, 0.0, 0.02)

    print("\nSITE RESPONSE  (Mw 6.0 at 20 km; softer ground must amplify)")
    prev = 0.0
    ok = True
    for cls in ("A", "B", "C", "D", "E"):
        s = Site.from_class(cls)
        src = Source(mw=6.0, qtype=TYPES["strike_slip"], depth_km=8.0)
        pk = spectrum.peak_ground_motion(src, s, 20.0,
                                         anchor=gmpe.solve_anchor(src, s, 20.0))
        print(f"   NEHRP {cls}  Vs30 {s.vs30:6.0f} m/s   "
              f"PGA {pk['pga_g']*100:6.2f} %g   PGV {pk['pgv_cms']:6.2f} cm/s")
        ok = ok and pk["pgv_cms"] > prev
        prev = pk["pgv_cms"]
    global fails
    fails += not ok
    print(f"   PGV increases monotonically as the ground softens: "
          f"{'ok' if ok else 'FAIL'}")

    print("\nSYNTHESISED RECORDS  (the time series must match its own spectrum)")
    for label, mw, qk, r_km, sk in (
            ("M6.0 crustal, 15 km, rock", 6.0, "strike_slip", 15.0, "B"),
            ("M7.2 thrust, 12 km, stiff soil", 7.2, "reverse", 12.0, "D"),
            ("M9.0 megathrust, 500 km, soft soil", 9.0, "megathrust", 500.0, "E")):
        s = Site.from_class(sk)
        src = Source(mw=mw, qtype=TYPES[qk], depth_km=TYPES[qk].depth_default,
                     stress_drop_bar=TYPES[qk].stress_drop_default)
        gm = synthesize(src, s, r_km, seed=4)
        m = gm.meta
        err = abs(m["pga_g"] / m["predicted_pga_g"] - 1.0)
        fails += err > 0.02
        print(f"   {label:38s} PGA {m['pga_g']*100:6.2f} %g  "
              f"MMI {m['mmi_text']['roman']:5s} JMA {m['jma']['class']:>2s}  "
              f"D5-95 {m['d5_95_s']:6.1f} s  "
              f"realised/predicted {m['pga_g']/m['predicted_pga_g']:.3f}  "
              f"{'ok' if err <= 0.02 else 'FAIL'}")

    print("\nINTENSITY SCALES")
    check("MMI at 100 cm/s2, 10 cm/s", mmi_from_pga_pgv(100.0, 10.0), 6.0, 0.12)
    check("MMI at 10 cm/s2, 1 cm/s", mmi_from_pga_pgv(10.0, 1.0), 3.3, 0.15)
    print(f"   JMA class from instrumental 5.2                      "
          f"{jma_class(5.2):>4s}  want   5+  "
          f"{'ok' if jma_class(5.2) == '5+' else 'FAIL'}")
    fails += jma_class(5.2) != "5+"

    print()
    if fails:
        print(f"FAIL: {fails} check(s) outside tolerance")
        return 1
    print("PASS: all seismology checks within tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
