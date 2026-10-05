# Earthquake Simulator

A physically-grounded earthquake simulator in Python. You stand somewhere —
inside a warehouse, on the 25th floor of a KL condominium, in a village house
below the Crocker Range, in a supermarket aisle, in a shophouse on a small-town
street — set an earthquake, and watch what it does. You can look anywhere from
the ground to the zenith and back down again, walk around, orbit the building
from outside, or rise above it and look straight down.

Nothing in it is animated. Every bottle, carton, plate, pallet and ceiling tile
is a rigid body with its real mass and its real friction, and whether it stays
put, slides, rocks or goes over is decided by the physics at the moment it
happens. The buildings are built the same way: every wall panel, column, slab,
purlin, roof sheet, rack frame and tower storey is a structural element that
holds until the structural solution says it has failed, and then falls as a
free body onto whatever is beneath it. Push the magnitude up and the racking
goes over, the gables come off, the soft storey folds, the slope behind the
house lets go, and the tower pancakes — each for its own stated reason.

---

## Running it

```
pip install -r requirements.txt
python earthquake.py
```

or on Windows just double-click **`earthquake.bat`**.

```
python earthquake.py                         opens the warehouse; pick any scene from
                                             the Scene menu at the top of the window
python earthquake.py --scene kl_highrise     or start in a particular scene
python earthquake.py --detail high           more objects (needs more CPU)
python earthquake.py --list                  scenes, quake types, site classes
python earthquake.py --report --mw 7.2 --type megathrust --distance 400 --site kl_alluvium
python earthquake.py --report --mw 6.0 --export ranau.csv     write the accelerogram
```

### Controls

| | |
|---|---|
| **mouse drag** or **Tab** | look around — all the way up to the zenith and down to the nadir, Stellarium style |
| **W A S D** | move (walking, flying and plan views) |
| **Space / Ctrl** | up / down &nbsp;&nbsp; **Shift** move faster |
| **mouse wheel** | zoom (narrows the field of view) |
| **Scene** / **Viewpoint** menus (top) | choose the scene and where to stand in it |
| **1**–**9**, **[** **]** | jump between the scene's viewpoints |
| **V** | cycle camera mode &nbsp;&nbsp; **N** next scene |
| **R** | run the earthquake &nbsp;&nbsp; **P** pause &nbsp;&nbsp; **F** reset objects &nbsp;&nbsp; **K** dial in this scene's collapse case |
| **PLAY / PAUSE**, **−10 s**, **+10 s**, scrubber (seismogram panel) | transport controls — see *Seeking* below |
| **T** | advance time of day &nbsp;&nbsp; **H** hide panels &nbsp;&nbsp; **G** collision shapes |
| **drag a title bar** | move a panel &nbsp;&nbsp; **L** put the panels back |
| **F1** | the key list (a card over the scene; F1, Esc, its button or a click outside closes it) &nbsp;&nbsp; **Esc** quit |

Every list in the panels (earthquake type, ground, scene, viewpoint) opens
below its field in the panels' own style, at most twelve rows tall, with a
scroll bar when there are more — drag it, click its arrows, or roll the wheel
over the list. The wheel zooms the view only when the pointer is over the
scene. The current choice is shown in blue in the list, and a click anywhere
else closes it.

**Pause.** PAUSE (or **P**) freezes the world: the record, the shaking and
every falling object stop together and the picture holds until PLAY. (It
used to stop only the record while the physics went on under the
acceleration of the paused instant — that was not a pause.) PLAY at t = 0 or
after the record has ended runs the earthquake again.

**Seeking.** The seismogram panel has PLAY/PAUSE, −10 s, +10 s and a
scrubber. Drop the scrubber anywhere and the scene *jumps* there: the last
frame stays on screen as a still picture with a progress card — *Moving to
113.9 s … 37 %* — and the scene reappears at that second. Nothing is shown
speeded up. Behind the card is honest physics, helped by **keyframes**:
while the record plays the simulator saves the complete state of every
object every few seconds (a few hundred kilobytes each, at most 150 of
them). A jump backward, or to any second already simulated, restores the
last keyframe before the target and catches up from there — a few seconds
of physics at most, so it is close to instant. A jump forward past what has
been simulated has to simulate the gap, because rigid bodies cannot be
rewound or skipped; in the small scenes that is quick, in the supermarket
with six thousand objects it runs at about the speed the quake itself
plays, and the card says how far it has got. **CANCEL** on the card (or
Esc) abandons the jump and puts everything back exactly where it was
before you touched the scrubber; STOP on the transport bar holds the scene
wherever the jump has reached. Keyframes are discarded when the scene is
reset, the record changes or another scene loads.

Rendering is on the GPU; the physics is Bullet, which in Panda3D runs on
one CPU core and has no GPU path, so a seek is as fast as the CPU can
simulate the gap — the keyframes are what make most seeks instant.

**Loading.** The window opens at most of the display with a splash while
the first scene builds; scene changes show the same splash, and building
an earthquake shows a card over the scene, both with the percentage:
*Building the scene: 4 120 objects*, *Solving the building response*, and
so on. A quake that builds in under a quarter of a second shows nothing.

Five panels, laid out so none overlaps another on a 16:10 or wider
screen: **Earthquake** (left) is what you set — magnitude (Mw, up to 10
for every type; the line under the type name shows the band nature has
actually produced, e.g. M2.0–8.2 for a crustal reverse fault, and above it the
source scaling extrapolates while the GMPE anchor saturates at M8.5, as its
authors capped it), type, depth, distance, stress drop, ground conditions. **At this site** (right) is what
that produces where you are standing. **Intensity** (right, below it) is the
MMI / JMA / CSIS reading and what breaks. **Seismogram** (bottom) is the actual
three-component accelerogram with the P, S and surface-wave arrivals marked
and a cursor. **Status** (top) carries the **Scene** and **Viewpoint** menus
and shows the camera, clock, frame rate, physics time per frame, which GPU is
rendering, and — once the shaking has started — how many structural elements
are down and what the scene reports (a collapsed rack bay, a landslide, a
crushed storey). Every scene opens on an establishing view — down the aisle,
across the living room — and the other viewpoints are a click or a number key
away. All dock to their
edges by default, computed from your actual screen aspect, and re-dock when
the window is resized or maximised; grab any title bar to move one, and it
stays where you put it across runs (`layout.json` next to `earthquake.py`) —
remembered by its distance from the nearest edges, so a panel you parked at
the left stays at the left when the window grows. **L** resets.

**SET COLLAPSE CASE** (or **K**) dials in the earthquake each scene's author
found brings its structure down; press **R** to run it. Every scene also
says, in its notes, what it takes.

---

## The five scenes

**Distribution warehouse** — 46 × 30 m portal-frame shed with six runs of
loaded pallet racking 7.2 m high. The racking is solved as its own structure
(1.05 s down-aisle against the shed's 0.55 s), so it sways on its own period,
and anything on a beam level feels the rack's motion rather than the floor's.
That mismatch is why racks fail in earthquakes that leave the building itself
untouched — and here they do: a bay is released when any level drifts 5%, and
its load comes down with it while the shed stands. The portal frame itself
goes at 6%, and takes the roof and wall sheeting with it.

**KL high-rise apartment, level 25** — the Malaysian far-field case. A great
Sumatran rupture several hundred kilometres away arrives stripped of all high
frequency; what is left is 2–6 second energy, which is exactly the band soft
Klang Valley alluvium amplifies and exactly the band a 25-storey concrete tower
is tuned to. No bang, no rattle: water moving in a glass and a slow sway that
goes on for minutes, while street level barely notices. Try a megathrust at
500–700 km and watch the pendant lamp, then jump to the street-level viewpoint.
The tower is built storey by storey; a storey that drifts 4% is crushed and
everything above it comes down through the gap, storey by storey, and you
ride the apartment floor down.

**Village house, Ranau (Sabah)** — the near-field case, built around the real
Mw 6.0 of 5 June 2015. Unreinforced block walls, timber above, a corrugated
roof, and gable walls braced by nothing at all. Sharp, violent, over in
seconds. The walls are courses of panels released by the ground acceleration
(top course first), the roof rides on the walls, the slope behind the house
is a rock field at a factor of safety of 1.08 that slides by the Newmark
integral and runs once it has moved, and a strip of the front yard subsides
into a fissure at 0.35 g. The real M6.0 at 15 km cracks the walls and rattles
the kitchen; M7 at 5 km takes the house down and brings the hill with it.

**Supermarket sales floor** — eight gondola runs, chillers, a suspended
ceiling, and a couple of thousand individually simulated items. The scene that
looks like the CCTV footage, because nothing in it is choreographed. The
gondolas and chillers are not rigid blocks: each is solved as its own
one-storey frame (0.35 s and 0.30 s, 5% damping) standing on the floor, so
the shelves swing harder than the ground — the amplification is what the
CCTV footage shows, and what a rigid shelf would never give (at M8.3 11 km
away, 0.30 g on the floor, 1.6× that on the shelves, and half the stock
ends up on the floor; a rigid model kept it at 6%). The gondolas are bolted
to nothing and go over at about 0.35 g of ground acceleration; the wall
panels tear off their roof anchorage at 0.45 g of the shell's own response,
as tilt-up shells did at Northridge, and the roof follows the walls.

**Shophouse row, small-town Malaysia** — three four-storey shophouses:
concrete frame, brick infill, a sundry shop on the ground floor and the
family above, a covered five-foot way along the front. An open shopfront
under three infilled floors is a soft storey, the building type that kills
more people in earthquakes than any other. The ground storey takes nearly all
the drift and its non-ductile columns fail at 2%; the slabs above lose their
support and come down onto the shop as a stack. Stand in the shop, on the
walkway, or in the living room upstairs and ride it down.

---

## What is actually modelled

The chain runs **source → path → site → building → room → object**, and every
link is a published model rather than a tuning knob.

**Source.** Sixteen earthquake types, each with its own depth band, stress
drop, attenuation, radiation pattern and spectral character: shallow crustal
(strike-slip, normal, reverse, oblique, blind thrust), subduction (megathrust,
intraslab, outer-rise, deep-focus), volcanic (volcano-tectonic, long-period,
harmonic tremor), anthropogenic (mine collapse, injection-induced,
reservoir-triggered, geothermal, explosion), plus landslide, icequake,
meteorite impact and aftershock. Brune ω² source spectrum, two-corner
(Atkinson & Silva 2000) for large crustal and subduction events; rupture
dimensions from Wells & Coppersmith (1994) and Strasser et al. (2010), with
slip made self-consistent with the moment.

Setting the moment magnitude sets everything downstream — seismic moment,
radiated energy, corner frequency, rupture length, duration.

**Path and site.** Hinged geometric spreading, frequency-dependent anelastic
attenuation, near-site kappa. Site amplification by the quarter-wavelength
method (Joyner, Warrick & Fumal 1981) over a Vs30-anchored velocity profile,
with soil nonlinearity from BSSA14 — so soft ground amplifies weak shaking and
stops amplifying strong shaking, as it really does.

**Ground motion.** Boore's stochastic method (2003) produces a three-component
accelerogram whose Fourier spectrum matches the target, then it is decomposed
into real phases — P, S, Love, Rayleigh — each with its own arrival time,
duration, frequency content and particle motion, including surface-wave
dispersion. So the record arrives in the right order, with the right gap
between P and S, and Rayleigh waves that whistle down from long period to
short.

**The vertical component is real motion, not decoration.** Each phase is
partitioned into radial, transverse and vertical motion by its particle
motion and the angle the ray arrives at: P is almost entirely vertical
straight above the hypocentre — the jolt that lifts the floor — and
becomes a horizontal push far away; S splits into SV (radial–vertical) and
SH (transverse); Rayleigh is the retrograde ellipse, vertical about 1.5× the
radial and 90° ahead of it (a Hilbert transform in the time domain). In the
room the vertical acceleration acts on every object as a time-varying
effective gravity: an upward floor acceleration presses things down, a
downward one unweights them — friction is proportional to the normal force,
so a jar slides at a horizontal acceleration well below μg in those
instants, and the simulator wakes objects on that basis — and past 1 g
downward the floor drops away and loose things leave it. Your own view
bounces too: the camera rides a three-axis model of the observer's body.
What is *not* modelled is the floor's own vertical structural response
(slabs on their vertical period): the floors and racks move laterally with
the building's shear response and vertically with the ground.

**Amplitude** is anchored to Boore, Stewart, Seyhan & Atkinson (2014) —
NGA-West2 — wherever a shallow-crustal GMPE legitimately applies. PGA and PGV
match its medians exactly. Outside that domain (subduction, volcanic,
collapse, explosion, deep-focus) no calibrated regression exists, so the model
runs on physics alone with literature-typical parameters, and the interface
says which regime is in force.

**Structure.** A lumped-mass shear building, solved by exact modal
superposition while elastic and by Newmark integration with bilinear
hysteretic story springs once it yields — both horizontal directions,
independently. Periods from ASCE 7-16, yield drift from material properties,
capacity clamped to what the material can really deliver. Unreinforced masonry
sheds strength as it cracks rather than holding load.

Non-structural damage is tracked separately and by the right demand: glazing,
partitions, cladding and stairs by drift; ceilings, parapets, out-of-plane
walls, sprinklers and equipment by floor acceleration. That distinction is why
a masonry gable is dangerous at 0.1 g while the frame below it is untouched.

**Collapse.** The structure is built from breakable elements — static bodies
that carry everything until a rule says they have failed, and are then
released to the physics engine with their real mass. Columns, infill and
cladding fail by storey drift at the HAZUS damage-state thresholds; parapets,
gables, out-of-plane masonry and unanchored shelving by acceleration; slabs,
purlins and roof sheets when enough of what they stand on has gone, cascading
upward. A storey that fails in the tower is crushed and the stack above drops
through it, storey by storey, which is the pancake mechanism. Limits are
scattered ±12% per element so walls come apart panel by panel. Nothing about
the fall is scripted. Damage short of collapse is drawn as cracks, in numbers
that follow each storey's running peak drift.

**Landslide.** The Ranau slope is a Newmark sliding-block problem solved by
the engine itself: rocks on a 33° surface at friction angle 35° (factor of
safety 1.08, yield acceleration 0.035 g) are still below the yield
acceleration and slide above it by the correct integral; once a rock has moved
8 cm its friction drops to the residual value, below the slope angle, and it
runs. Jibson's regression and the simulation agree on about half a metre of
Newmark displacement at 0.4 g before the strength loss takes over.

**Objects.** Bullet rigid-body dynamics in the frame of the floor, with each
body feeling the inertial force `F = −m·a(t)` of the floor supporting it. The
solver reproduces the analytic thresholds to within 1%: a block slides when
`a/g > μ` and tips when `a/g > b/h` (see `tests/`). Collision margins are
scaled to object size, because Bullet's 40 mm default is fine for vehicles and
catastrophic for groceries.

**Intensity** on three scales: Modified Mercalli (Worden et al. 2012, the
ShakeMap standard), JMA instrumental shindo computed by the official
filter-and-0.3-second algorithm, and the China Seismic Intensity Scale (CSIS, GB/T 17742).

---

## Verification

```
python tests/test_physics_thresholds.py    friction and overturning vs theory
python tests/test_seismo.py                magnitude, energy, GMPE agreement
python tests/test_scene_integrity.py       nothing moves without an earthquake;
                                           no two structural elements overlap
python tests/test_smoke.py <scene>         full application, end to end
python tests/test_stress.py [scene]        every scene through a ladder of
                                           earthquakes up to M9.5 beneath it
python tests/test_stress_app.py [scene]    everything the interface can do:
                                           every type at both ends of its
                                           range, live changes mid-quake,
                                           pause, speed, gain, collapse and
                                           reset repeated, scene switching
python tests/test_snapshots.py [quick]     keyframe restore is exact; seeks land
                                           and cancel back, scene by scene
python tests/stress_pictures.py            twenty-one captioned frames of the
                                           stress cases, in stress_report/
```

`test_scene_integrity.py` reads the collision engine's own contact
manifolds after one substep (Panda's `contactTest` reports a stale distance
for every pair but the last), with the static bodies woken for that one
step so that two *fixed* elements built through each other are found too:
they cost nothing while both stand and become a shove the moment the
collapse model releases one.

`tests/test_snapshots.py` checks the keyframes: in the middle of every
scene's collapse a snapshot is taken, the run carried on, the snapshot
restored — every pose, velocity, sleep state, floor group, released or
crushed flag, structural failure record and damage report must come back
to float32 precision — and the run carried on again from there; then, in
the application, a seek backward must land by way of a keyframe and a
cancelled seek must return to the pre-seek state exactly.

`docs/PHYSICS.md` has the model chain in full, with every reference and the
validation results.

---

## Performance

Rendering is on the GPU (OpenGL, physically-based shading). Physics is Bullet,
which is CPU-only in Panda3D — there is no GPU rigid-body path — so the two
budgets are separate and both are managed:

**Draw calls.** Every object used to be its own draw call, doubled again by
the shadow pass; the warehouse was 3,570 a frame and the supermarket 6,100,
which is what floors a GPU. Now everything that never moves is flattened
into about ten batches, and everything that can move is drawn through a
`RigidBodyCombiner` that re-transforms the vertices itself, so the warehouse
is ~30 draw calls and the supermarket ~250. Colours are baked into vertices
rather than applied as node attributes for the same reason: a colour
attribute is a distinct render state, and three hundred differently-coloured
bottles would be three hundred batches.

**Physics.** 120 Hz substeps (was 240), at most four per frame, and the frame
never asks the physics to catch up more than 1/30 s at once — so under load
time slows slightly rather than the frame rate collapsing. Every body sleeps
when at rest and is woken when the shaking crosses its own sliding or tipping
threshold, including the unweighting from vertical acceleration; racking is
moved as static bodies rather than Bullet kinematic ones, because nothing
resting on a kinematic body is ever allowed to sleep (that one detail was
most of the difference between 16 fps and 60 with a thousand cartons on the
racks), and its load is carried along with it -- each body shifted by the
rack's displacement, with its own dynamics run in the rack's frame -- since
a static body moved by transform has no surface velocity for friction to
act on. Static bodies themselves are put to sleep once the scene is built:
Bullet only skips a pair when neither body is active, and a wall left
"active" was being collided with every sleeping item against it each
substep (a third of the supermarket's at-rest cost). The status panel shows
the physics time per frame.
`--physics-hz 240` restores the finer step if you want it and have the CPU
to spare.

**Quality.** `--quality auto` (default) reads which GPU it got: a discrete
card gets 2×MSAA, 2048 shadows and normal maps; an integrated one gets 1024
shadows and no normal maps. `low` turns shadows and MSAA off. The status
panel names the renderer — if it says *Intel* on a machine that also has an
NVIDIA or AMD card, Windows handed Python the integrated GPU: Settings →
System → Display → Graphics → add `python.exe` → **High performance**.

Object counts by detail level:

| detail | warehouse | KL | Ranau | supermarket | shophouse |
|---|---|---|---|---|---|
| `low` | 900 | 90 | 90 | 1 900 | 150 |
| `medium` | 1 500 | 90 | 115 | 6 000 | 190 |
| `high` | 5 600 | 90 | 160 | 11 900 | 290 |
| `ultra` | 6 500 | 90 | 215 | 18 000 | 390 |

(Free bodies at rest. Every scene also carries its structural elements —
57 to 561 of them — which cost nothing until they are released.)

Start at `medium`. During strong shaking every body is awake and the physics
cost is what it is — about 20 ms a frame for 1,500 bodies on a mid-range
desktop CPU — so if the supermarket drops below real time mid-quake, use
`low` for that scene.

## Notes

- Needs Python 3.10+ and a GPU that can compile GLSL. Without one it falls back
  to basic shading automatically.
- The interface is English only. It uses the system UI font (Segoe UI on
  Windows, Helvetica on macOS, DejaVu Sans on Linux); drop any `.ttf` into a
  `fonts/` folder next to `earthquake.py` to use a different one.

## License

Earthquake Simulator is free software, released under the
[GNU General Public License, version 3 or later](LICENSE) —
Copyright (C) 2026 Earthquake Simulator contributors. You may run, study,
share and change it; anything you distribute built from it must carry the
same freedom. Every source file carries the notice; the full text is in
`LICENSE`.

It builds on Panda3D (with Bullet), panda3d-simplepbr, NumPy and SciPy,
all under BSD- or zlib-style licences compatible with the GPL. The seismological
models it implements are cited in `docs/PHYSICS.md`.
