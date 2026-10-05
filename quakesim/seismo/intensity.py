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

"""Intensity scales: Modified Mercalli, JMA, and the China Seismic Intensity Scale (CSIS).

Magnitude describes the earthquake; intensity describes what happened *here*.
One earthquake has one magnitude and a different intensity at every address.
"""

from __future__ import annotations

import math

import numpy as np

from . import constants as K

# ---------------------------------------------------------------------------
# Modified Mercalli -- Worden et al. (2012), BSSA 102(1), the ShakeMap standard
# ---------------------------------------------------------------------------
def mmi_from_pga(pga_cms2: float) -> float:
    if pga_cms2 <= 0.0:
        return 1.0
    x = math.log10(pga_cms2)
    mmi = 1.78 + 1.55 * x if x <= 1.57 else -1.60 + 3.70 * x
    return min(max(mmi, 1.0), 10.0)


def mmi_from_pgv(pgv_cms: float) -> float:
    if pgv_cms <= 0.0:
        return 1.0
    x = math.log10(pgv_cms)
    mmi = 3.78 + 1.47 * x if x <= 0.53 else 2.89 + 3.16 * x
    return min(max(mmi, 1.0), 10.0)


def mmi_from_pga_pgv(pga_cms2: float, pgv_cms: float) -> float:
    """ShakeMap's combined estimate.

    Acceleration predicts low intensities well (you feel a small jolt long
    before anything moves); velocity predicts damage-level intensities better,
    because damage tracks energy, not peak spike. So the weighting slides from
    one to the other.
    """
    ia, iv = mmi_from_pga(pga_cms2), mmi_from_pgv(pgv_cms)
    w = min(max((max(ia, iv) - 5.0) / 2.0, 0.0), 1.0)     # 0 at MMI<=5, 1 at >=7
    return (1.0 - w) * ia + w * iv


MMI_LEVELS = {
    1: ("I", "Not felt",
        "Detected only by instruments."),
    2: ("II", "Weak",
        "Felt by a few people at rest, especially on upper floors. Hanging "
        "objects may swing slightly."),
    3: ("III", "Weak",
        "Felt indoors; many do not recognise it as an earthquake. Standing "
        "cars rock. Vibration like a passing lorry."),
    4: ("IV", "Light",
        "Felt indoors by many, outdoors by few. Dishes and windows rattle, "
        "doors swing, walls creak. Parked cars rock noticeably."),
    5: ("V", "Moderate",
        "Felt by nearly everyone. Small unstable objects overturn, pictures "
        "swing, some crockery breaks, liquids spill. Sleepers wake."),
    6: ("VI", "Strong",
        "Felt by all, many frightened and run outdoors. Heavy furniture "
        "moves. Plaster cracks, chimneys and weak masonry damaged. Books "
        "come off shelves."),
    7: ("VII", "Very strong",
        "Difficult to stand. Furniture broken, unbraced parapets and "
        "cornices fall. Damage slight in good design, considerable in poorly "
        "built structures. Drivers notice it."),
    8: ("VIII", "Severe",
        "Steering of cars affected. Partial collapse of ordinary masonry, "
        "chimneys and towers fall, heavy furniture overturns. Water table "
        "changes; sand and mud ejected."),
    9: ("IX", "Violent",
        "General panic. Well-designed frames thrown out of plumb, "
        "substantial buildings shifted off foundations. Ground cracks "
        "conspicuously; underground pipes broken."),
    10: ("X", "Extreme",
         "Most masonry and frame structures destroyed with their "
         "foundations. Ground badly cracked, rails bent, landslides on "
         "riverbanks and steep slopes, water thrown out of channels."),
}


def describe_mmi(mmi: float) -> dict:
    lvl = int(min(max(round(mmi), 1), 10))
    roman, word, text = MMI_LEVELS[lvl]
    return {"value": round(mmi, 2), "level": lvl, "roman": roman,
            "word": word, "text": text}


# ---------------------------------------------------------------------------
# JMA seismic intensity (shindo), Japan Meteorological Agency
# ---------------------------------------------------------------------------
def _jma_filter(f: np.ndarray) -> np.ndarray:
    """The JMA weighting filter applied to acceleration before the peak search.

    Three factors: a 1/sqrt(f) period weighting that approximates how the body
    responds, a low-cut that removes motion too slow to be felt, and a high-cut
    that removes spikes too fast to do damage.
    """
    f = np.maximum(np.asarray(f, dtype=float), 1e-9)
    period_weight = np.sqrt(1.0 / f)

    y = f / 10.0
    high_cut = (1.0 + 0.694 * y ** 2 + 0.241 * y ** 4 + 0.0557 * y ** 6
                + 0.009664 * y ** 8 + 0.00134 * y ** 10
                + 0.000155 * y ** 12) ** -0.5

    low_cut = np.sqrt(1.0 - np.exp(-((f / 0.5) ** 3)))
    return period_weight * high_cut * low_cut


def jma_intensity(acc_ms2: np.ndarray, dt: float) -> float:
    """Compute the JMA instrumental intensity from a 3-component record.

    Official algorithm: filter all three components, combine into a vector
    magnitude, then find the level a0 that the vector exceeds for a *total*
    of 0.3 seconds. I = 2*log10(a0) + 0.94, with a0 in cm/s^2.
    """
    a = np.atleast_2d(np.asarray(acc_ms2, dtype=float))
    n = a.shape[1]
    f = np.fft.rfftfreq(n, dt)
    w = _jma_filter(f)

    filt = np.zeros_like(a)
    for i in range(a.shape[0]):
        filt[i] = np.fft.irfft(np.fft.rfft(a[i]) * w, n=n)

    vec = np.sqrt(np.sum(filt ** 2, axis=0)) * 100.0        # cm/s^2
    if float(np.max(vec)) <= 0.0:
        return 0.0

    # a0 such that total time above a0 equals 0.3 s
    need = 0.3
    ordered = np.sort(vec)[::-1]
    k = int(round(need / dt))
    if k < 1:
        k = 1
    if k >= ordered.size:
        a0 = ordered[-1]
    else:
        a0 = ordered[k - 1]
    if a0 <= 0.0:
        return 0.0

    i = 2.0 * math.log10(a0) + 0.94
    return math.floor(i * 100.0 + 0.5) / 100.0


def jma_class(i: float) -> str:
    """Map instrumental intensity onto the 10-step JMA scale."""
    if i < 0.5:
        return "0"
    if i < 1.5:
        return "1"
    if i < 2.5:
        return "2"
    if i < 3.5:
        return "3"
    if i < 4.5:
        return "4"
    if i < 5.0:
        return "5-"
    if i < 5.5:
        return "5+"
    if i < 6.0:
        return "6-"
    if i < 6.5:
        return "6+"
    return "7"


JMA_TEXT = {
    "0": "Imperceptible.",
    "1": "Felt by some people sitting quietly indoors.",
    "2": "Felt by many indoors; hanging lamps swing slightly.",
    "3": "Felt by most indoors; crockery rattles.",
    "4": "Startling. Hanging objects swing considerably, unstable ornaments fall.",
    "5-": "Most people try to move to safety. Books fall, unsecured furniture moves.",
    "5+": "Hard to move without holding on. Unsecured furniture topples, "
          "windows break, unreinforced block walls collapse.",
    "6-": "Difficult to stand. Many unsecured items topple. Tiles and windows "
          "fall; weak wooden houses lean or collapse.",
    "6+": "Impossible to stand or move without crawling. Most unsecured "
          "furniture topples. Large cracks in walls, ground fissures.",
    "7": "Thrown by the shaking. Reinforced-concrete buildings can collapse "
         "even where well built. Ground deforms grossly.",
}


def jma_from_motion(gm) -> dict:
    i = jma_intensity(gm.acc, gm.dt)
    c = jma_class(i)
    return {"instrumental": i, "class": c, "text": JMA_TEXT[c]}


# ---------------------------------------------------------------------------
# China seismic intensity scale, GB/T 17742
# ---------------------------------------------------------------------------
def china_intensity(pga_ms2: float, pgv_ms: float) -> float:
    """GB/T 17742 instrumental intensity, the mean of the PGA and PGV forms."""
    ia = 3.17 * math.log10(max(pga_ms2, 1e-6)) + 6.59
    iv = 3.00 * math.log10(max(pgv_ms, 1e-8)) + 9.77
    return min(max(0.5 * (ia + iv), 1.0), 12.0)


# ---------------------------------------------------------------------------
# Perceptual helpers used by the viewer
# ---------------------------------------------------------------------------
def shaking_colour(mmi: float) -> tuple[float, float, float]:
    """USGS ShakeMap intensity colour ramp, as linear RGB in 0..1."""
    stops = [
        (1.0, (1.00, 1.00, 1.00)), (2.0, (0.75, 0.87, 0.94)),
        (3.0, (0.53, 0.82, 0.94)), (4.0, (0.50, 1.00, 1.00)),
        (5.0, (0.47, 1.00, 0.47)), (6.0, (1.00, 1.00, 0.00)),
        (7.0, (1.00, 0.85, 0.00)), (8.0, (1.00, 0.60, 0.00)),
        (9.0, (1.00, 0.00, 0.00)), (10.0, (0.78, 0.00, 0.00)),
    ]
    m = min(max(mmi, 1.0), 10.0)
    for i in range(len(stops) - 1):
        a, ca = stops[i]
        b, cb = stops[i + 1]
        if a <= m <= b:
            u = (m - a) / (b - a)
            return tuple(ca[j] + u * (cb[j] - ca[j]) for j in range(3))
    return stops[-1][1]


def object_response_summary(pga_g: float, pgv_cms: float) -> list[str]:
    """What actually happens to loose objects at this level of shaking.

    Sliding starts when horizontal acceleration exceeds the static friction
    coefficient; a rigid block of width b and height h begins to rock when
    a/g > b/h. Those two inequalities decide almost everything you see in a
    room during an earthquake.
    """
    out = []
    if pga_g >= 0.02:
        out.append("Hanging lamps and blinds swing.")
    if pga_g >= 0.05:
        out.append("Loose crockery and glassware rattle audibly.")
    if pga_g >= 0.10:
        out.append("Books slide out of shelves; tall bottles rock.")
    if pga_g >= 0.15:
        out.append("Slender objects (b/h < 0.15) topple; unsecured TVs walk.")
    if pga_g >= 0.25:
        out.append("Most shelved goods are thrown clear; light furniture slides.")
    if pga_g >= 0.35:
        out.append("Filing cabinets and unanchored racking overturn.")
    if pga_g >= 0.50:
        out.append("Heavy furniture overturns; unanchored equipment airborne.")
    if pga_g >= 1.00:
        out.append("Objects leave the floor -- vertical acceleration exceeds gravity.")
    if pgv_cms >= 30.0:
        out.append("Long-period sway strong enough to make walking impossible.")
    return out or ["Nothing visibly moves."]
