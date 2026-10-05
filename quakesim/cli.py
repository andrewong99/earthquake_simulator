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

"""Headless analysis, for when you want the numbers rather than the picture."""

from __future__ import annotations

import math
import sys

import numpy as np

from .seismo import constants as K
from .seismo import gmpe, spectrum
from .seismo.intensity import (china_intensity, object_response_summary)
from .seismo.phases import arrival_times, distance_from_sp_interval
from .seismo.site import NEHRP, SITES, Site
from .seismo.source import TYPE_ORDER, TYPES, Source
from .seismo.synth import synthesize


def print_catalogue() -> None:
    from .world.scenes import SCENES, SCENE_ORDER
    print("\nSCENES")
    print("-" * 78)
    for k in SCENE_ORDER:
        sp = SCENES[k].spec
        print(f"  {k:14s} {sp.name}")
        print(f"                 default: M{sp.default_mw} "
              f"{TYPES[sp.default_quake].name}, {sp.default_distance_km:.0f} km, "
              f"{sp.site.name}")
        print(f"                 {len(sp.viewpoints)} viewpoints")

    print("\nEARTHQUAKE TYPES")
    print("-" * 78)
    for k in TYPE_ORDER:
        t = TYPES[k]
        print(f"  {k:18s} {t.name}")
        print(f"                     M {t.mw_range[0]:.1f}-{t.mw_range[1]:.1f}   "
              f"depth {t.depth_km[0]:.0f}-{t.depth_km[1]:.0f} km   "
              f"stress drop {t.stress_drop_bar[0]:.0f}-"
              f"{t.stress_drop_bar[1]:.0f} bar")

    print("\nSITE CONDITIONS")
    print("-" * 78)
    for k, s in SITES.items():
        print(f"  {k:16s} {s.name:44s} Vs30 {s.vs30:6.0f} m/s  class {s.nehrp}")
    print()


def report(args) -> int:
    qkey = args.qtype or "strike_slip"
    if qkey not in TYPES:
        print(f"Unknown quake type {qkey!r}. Try --list.", file=sys.stderr)
        return 2
    qt = TYPES[qkey]
    site_key = args.site or "rock"
    site = SITES.get(site_key) or Site.from_class(site_key.upper())

    src = Source(mw=args.mw if args.mw is not None else 6.5, qtype=qt,
                 depth_km=args.depth if args.depth is not None else qt.depth_default,
                 stress_drop_bar=qt.stress_drop_default)
    r = args.distance if args.distance is not None else 20.0

    gm = synthesize(src, site, r, azimuth_deg=35.0, seed=4)
    d = src.describe()
    m = gm.meta
    arr = gm.arrivals

    w = 74
    print()
    print("=" * w)
    print(f"  {d['type']}")
    print("=" * w)
    print(f"  {qt.blurb}")
    print()
    print("  SOURCE")
    print(f"    Magnitude               M {d['Mw']:.2f}")
    print(f"    Seismic moment          {d['M0_Nm']:.3e} N·m")
    print(f"    Radiated energy         {d['energy_J']:.3e} J"
          f"   ({d['energy_tons_TNT']/1e3:,.4g} kilotons TNT)")
    print(f"    Focal depth             {d['depth_km']:.1f} km")
    print(f"    Stress drop             {d['stress_drop_bar']:.0f} bar")
    print(f"    Corner frequency        {d['corner_freq_Hz']:.4f} Hz"
          f"   (source duration {d['source_duration_s']:.1f} s)")
    print(f"    Rupture                 {d['rupture_length_km']:.1f} km long × "
          f"{d['rupture_width_km']:.1f} km deep"
          f"   ({d['rupture_area_km2']:,.0f} km²)")
    print(f"    Average slip            {d['avg_slip_m']:.2f} m"
          f"   (peak about {d['max_slip_m']:.2f} m)")
    print(f"    Rupture velocity        {d['rupture_velocity_kms']:.2f} km/s")
    print()
    print("  PATH AND SITE")
    print(f"    Epicentral distance     {r:.1f} km"
          f"   (hypocentral {src.hypocentral_distance(r):.1f} km)")
    for k2, v in site.describe().items():
        print(f"    {k2:23s} {v}")
    print()
    print("  ARRIVALS")
    print(f"    P wave                  {arr['P']:.2f} s after origin")
    print(f"    S wave                  {arr['S']:.2f} s")
    print(f"    S minus P               {arr['S_minus_P']:.2f} s"
          f"   (implies {distance_from_sp_interval(arr['S_minus_P']):.0f} km)")
    print(f"    Love wave               {arr['Love']:.2f} s")
    print(f"    Rayleigh wave           {arr['Rayleigh']:.2f} s")
    print()
    print("  GROUND MOTION HERE")
    print(f"    Peak acceleration       {m['pga_g']*100:.2f} %g"
          f"   ({m['pga_g']*981:.1f} cm/s²)")
    print(f"    Peak velocity           {m['pgv_cms']:.2f} cm/s")
    print(f"    Peak displacement       {m['pgd_cm']:.2f} cm")
    print(f"    Arias intensity         {m['arias_ms']:.4f} m/s")
    print(f"    Significant duration    {m['d5_95_s']:.1f} s  (D5-95)")
    print(f"    Record length           {m['record_length_s']:.0f} s")
    print(f"    Site amplification      × {m['site_amp']:.2f} on PGA")
    print(f"    Amplitude basis         "
          f"{'anchored to BSSA14 (NGA-West2)' if m['anchor_mode']=='gmpe' else 'stochastic physics model'}")
    print()
    print("  INTENSITY")
    mmi = m["mmi_text"]
    print(f"    Modified Mercalli       {mmi['roman']}  ({m['mmi']:.2f})  {mmi['word']}")
    print(f"    JMA shindo              {m['jma']['class']}  "
          f"(instrumental {m['jma']['instrumental']:.2f})")
    print(f"    CSIS (GB/T 17742)       {china_intensity(m['pga_g']*K.G0, m['pgv_cms']/100):.1f}")
    print()
    print(f"    {mmi['text']}")
    print()
    print("  WHAT MOVES")
    for line in object_response_summary(m["pga_g"], m["pgv_cms"]):
        print(f"    · {line}")
    print()

    per, sa = spectrum.response_spectrum(src, site, r)
    print("  RESPONSE SPECTRUM  (5% damped, horizontal)")
    print("    period s :  " + " ".join(f"{p:6.2f}" for p in
                                        (0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0)))
    vals = np.interp([0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0], per, sa)
    print("    Sa (g)   :  " + " ".join(f"{v:6.3f}" for v in vals))
    print()

    if getattr(args, "export", None):
        path = args.export
        t = gm.t
        step = max(1, int(round((1.0 / 100.0) / gm.dt)))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("# Synthesised by quakesim (stochastic method, "
                     f"Boore 2003); {d['type']} M{d['Mw']:.2f} at {r:.0f} km, "
                     f"{site.name}\n")
            fh.write("time_s,acc_east_ms2,acc_north_ms2,acc_up_ms2,"
                     "vel_east_ms,vel_north_ms,vel_up_ms\n")
            for i in range(0, gm.n, step):
                fh.write(f"{t[i]:.4f},{gm.acc[0,i]:.6f},{gm.acc[1,i]:.6f},"
                         f"{gm.acc[2,i]:.6f},{gm.vel[0,i]:.6f},"
                         f"{gm.vel[1,i]:.6f},{gm.vel[2,i]:.6f}\n")
        print(f"  Accelerogram written to {path} "
              f"({gm.n // step:,} rows at 100 Hz)\n")
    return 0
