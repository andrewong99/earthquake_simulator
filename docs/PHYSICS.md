# The model chain

Everything the simulator shows follows one chain, and each link is a published
model rather than a tuning knob:

```
   source  ->  path  ->  site  ->  building  ->  room  ->  object
```

Set the moment magnitude and every downstream quantity follows from it —
seismic moment, radiated energy, corner frequency, rupture dimensions,
duration, and ultimately whether a particular bottle on a particular shelf
goes over.

---

## 1. Source — `quakesim/seismo/source.py`

**Moment and energy.** Hanks & Kanamori (1979): `Mw = ⅔(log₁₀ M₀ − 9.05)`,
with `M₀` in N·m. Radiated energy from Gutenberg–Richter,
`log₁₀ Eₛ = 1.5 M + 4.8`, which gives `Eₛ/M₀ ≈ 5×10⁻⁵` — Kanamori's value.

**Corner frequency.** Brune (1970, 1971):

```
fc = 0.4906 · β · (Δσ / M₀)^(1/3)          β in m/s, Δσ in Pa, M₀ in N·m
```

the SI form of the familiar `4.9×10⁶·β·(Δσ/M₀)^⅓`. An Mw 6 at 100 bar gives
0.356 Hz; an Mw 9 at 30 bar gives 0.0075 Hz, i.e. a 133-second source — which
is why a great subduction earthquake shakes for minutes.

**Spectral shape.** Brune ω² for small events. For large crustal and
subduction events the source is not one smooth pulse but a chain of subevents,
so a two-corner model (Atkinson & Silva 2000) fills in the intermediate
frequencies. Without it the model over-predicts how fast PGA grows with
magnitude; with it, the observed near-saturation above about M 5.5 falls out
on its own.

**Rupture geometry.** Wells & Coppersmith (1994) by mechanism, Strasser,
Arango & Bommer (2010) for subduction. Slip is taken from the moment itself
(`M₀ = μ·A·D`) rather than from an independent empirical relation, so the
geometry and the moment cannot disagree with each other.

**Sixteen types**, each carrying its own depth band, stress-drop range,
attenuation, radiation partitioning, duration factor and spectral falloff.
The differences are physical, not cosmetic:

| | why it feels different |
|---|---|
| megathrust | very low stress drop, enormous area — minutes of long-period roll |
| intraslab | very high stress drop, efficient deep path — sharp and rattling |
| deep-focus | almost no surface waves; huge magnitude, little damage |
| mine collapse | *implosive*: first motion downward everywhere, weak S wave |
| explosion | isotropic, P/S ratio inverted — this is how test-ban monitoring tells bombs from earthquakes |
| volcanic LP | a fluid-filled crack resonating, not rock breaking: narrowband, emergent, no sharp P |

---

## 2. Path — `quakesim/seismo/spectrum.py`

Geometric spreading is hinged (Atkinson & Boore 1995): `1/R` to 70 km, a
plateau to 130 km where post-critical Moho reflections hold the amplitude up,
then `R^-½`. Deep sources use plain `1/R` through the mantle.

Anelastic attenuation is `exp(−πfR / Q(f)β)` with `Q(f) = Q₀f^η`, `Q₀` and `η`
set per quake type — a slab path is far less lossy than a crustal one.

Near-site high-frequency decay is `exp(−πκf)`, with the source and site
contributions combined in quadrature.

---

## 3. Site — `quakesim/seismo/site.py`

Site condition is the single biggest reason two places at the same distance
from the same earthquake shake completely differently.

A generic velocity profile is built from the site's Vs30 as a power law
`Vs(z) ∝ (1+z)^0.28`, with the surface value solved so the travel-time average
over the top 30 m equals Vs30 exactly, and density from Gardner et al. (1974).
Amplification then comes from the quarter-wavelength method (Joyner, Warrick &
Fumal 1981): at each frequency, find the depth where `z = V̄(z)/4f` and take
the square root of the impedance ratio.

Soil nonlinearity uses the BSSA14 site term: strong shaking makes soft soil
yield, so its shear modulus falls and it stops amplifying. Above roughly 0.3 g
a soft site can de-amplify relative to rock — which the model reproduces
(see the NEHRP table in `tests/test_seismo.py`, where class E overtakes D in
velocity but falls behind it in acceleration).

Basins get an extra resonance bump near `f₀ = Vs30/4H`.

---

## 4. Ground motion — `quakesim/seismo/synth.py`, `phases.py`

**Method.** Boore's stochastic simulation (2003): Gaussian noise, windowed
with the Saragoni & Hart (1974) envelope, FFT'd, its spectrum forced to the
target, and transformed back. Random phase means every run is a different but
statistically identical record — exactly as two real earthquakes of the same
size and distance produce different squiggles with the same character.

**Phases.** The total spectrum is *partitioned* (energy-preserving) into P, S,
Love and Rayleigh, each given its own arrival time, duration, frequency
content and particle motion, then recombined onto East / North / Vertical.
Surface waves get a frequency-dependent delay from a dispersion curve, so long
periods arrive first and the packet spreads into the whistling-down train that
distant large earthquakes actually produce. Rayleigh motion is retrograde
elliptical, with the vertical leading the radial by 90° (a Hilbert transform).

Surface waves are suppressed for deep sources and grow with epicentral
distance, because they need a horizontal path to develop.

**Peak values** come from Random Vibration Theory rather than from reading the
maximum off one realisation: spectral moments, then Cartwright &
Longuet-Higgins (1956) peak-factor statistics. This gives the *expected* peak,
which is what a GMPE predicts and what makes the simulator repeatable.

---

## 5. Amplitude anchoring — `quakesim/seismo/gmpe.py`

A point-source stochastic model gets the *character* of shaking right but
cannot reproduce the absolute amplitude of a regression over ten thousand real
records, because a real rupture is a finite surface with directivity,
asperities and hanging-wall geometry that a point cannot represent.

So the simulator does what engineering practice does (ASCE 7-16 Ch. 16 target-
spectrum matching): waveform shape from the physics, amplitude from a
published GMPE. A smooth two-parameter correction (level and log-log tilt) is
solved by a 2×2 Newton iteration so that the RVT-predicted PGA *and* PGV both
equal the Boore, Stewart, Seyhan & Atkinson (2014) NGA-West2 medians.

Measured agreement over M 4.5–8.0 and 5–300 km: **worst-case error 0.02% in
both PGA and PGV**.

The correction is solved at each type's *reference* stress drop, so moving the
stress-drop slider still changes the answer physically instead of being
normalised away.

Where no shallow-crustal GMPE legitimately applies — subduction, volcanic,
collapse, explosion, deep-focus — the model runs on physics alone with
literature-typical parameters, and the interface reports which regime is in
force rather than pretending to a calibration it does not have.

**Magnitude range.** The slider goes to M 10 for every type. Each type's
line in the panel gives the band nature has actually produced (M 2.0–8.3 for
a strike-slip fault, M 7.0–9.5 for a megathrust, and so on); above it the
source model keeps scaling — moment, corner frequency, rupture area, duration
— because the physics does not know where the catalogue ends. For the
crustal types the BSSA14 anchor is used up to its authors' own cap of M 8.5
and held there beyond it; the source terms alone carry the last step to
M 10. One hard limit remains: **peak ground acceleration is capped at 3 g**.
Peaks near 3 g have been recorded on the ground (Tsukidate, Tohoku 2011) and
about 4 g once, on a hilltop station (IWTH25, Iwate–Miyagi 2008); an
unbounded synthetic at M 10 and 1 km would otherwise reach hundreds of g and
mean nothing. A record scaled to the cap says so in the panel
(*Amplitude basis: … (capped at 3 g)*).

---

## 6. Structure — `quakesim/structure/mdof.py`

A lumped-mass shear building. While every story stays elastic the response is
exact, by modal superposition in the frequency domain. Once any story yields
the solver switches to Newmark's average-acceleration method (γ=½, β=¼,
unconditionally stable) with bilinear kinematic-hardening story springs,
solved on the tridiagonal band with cached LU factorisations keyed on the
yield pattern — a building spends most of its time in one state, so
factorising per state rather than per timestep is the difference between
twenty seconds and one.

Both horizontal directions are solved independently. An earthquake shakes both
at once, and scaling one by the other's ratio produces sign flips wherever the
reference component crosses zero.

Periods from ASCE 7-16 Table 12.8-2. Yield drift is a material and
proportioning property (Priestley 1998), and each value sits just above the
corresponding HAZUS "Slight" damage threshold — the relationship observed in
real buildings. Yield strength is then clamped to the material's real capacity,
because a stiff structure reaches its yield *drift* at an implausible force if
stiffness and drift are set independently. Unreinforced masonry takes a
negative post-yield stiffness: it sheds capacity as it cracks rather than
holding load, down to a residual friction floor.

**Non-structural damage** is tracked separately, and by the right demand
parameter:

- **drift** breaks anything spanning between floors — glazing, partitions,
  cladding, stair flights
- **floor acceleration** throws anything supported by one floor — ceilings,
  parapets, out-of-plane walls, sprinklers, equipment, contents

That distinction is the whole story of unreinforced masonry. A URM wall is
strong in its own plane and nearly helpless across it, so it fails by
acceleration long before the building drifts enough to matter — which is why a
gable is dangerous at 0.1 g while the frame below it is untouched, and why the
Ranau scene models its gable as free blocks resting on the wall head.

---

## 7. Objects — `quakesim/sim/physics.py`

**The non-inertial frame.** Moving the floor and letting contact drag
everything along works badly: the camera lurches, kinematic bodies tunnel at
high frequency, and the world translates metres away during a big event.
Instead the simulation runs *in the frame of the floor*. The room stays put
and every free body feels the d'Alembert force

```
F = −m · a_floor(t)
```

which is exactly equivalent by the equivalence principle. Because that force
is proportional to mass, it is applied as a per-body gravity vector rather
than a force, at no cost:

```
g_effective = (−aₓ, −a_y, −g − a_z)
```

**Grouping.** Bodies carry a `floor` id, and each group can be driven by a
different structure. In the warehouse the racking is solved as its own 5-level
structure (1.05 s down-aisle against the shed's 0.55 s), and everything on a
beam level feels the rack's motion rather than the floor's — the rack top
sees about 2.2× the floor acceleration.

**Validation.** The engine reproduces both analytic thresholds
(`tests/test_physics_thresholds.py`):

| | theory | measured | error |
|---|---|---|---|
| sliding, μ = 0.20 … 0.70 | `a/g = μ` | 0.200 … 0.700 | ≤ 0.1% |
| overturning, b/h = 0.10 … 0.80 | `a/g = b/h` | 0.099 … 0.800 | ≤ 0.8% |
| free-body inertial forcing | −3.0000 m/s after 1 s | −3.0000 | exact |

Thresholds are found by bisection under a *held* acceleration. Ramping the
load lets the body's inertia lag behind the ramp and reports a threshold too
high by however fast you ramped — an easy way to conclude, wrongly, that the
engine is broken.

**Collision margin** is scaled to object size. Bullet's default is 40 mm,
which is fine for vehicles and catastrophic for groceries: a 74 mm drinks can
is almost entirely margin, so two cans standing 1 mm apart read as deeply
interpenetrating and get fired across the room.

**Sleeping.** Every body may sleep, and is woken the moment the shaking
crosses its own sliding or tipping threshold. The threshold accounts for
vertical acceleration: an upward-negative excursion unweights everything in
the room, and friction is proportional to normal force, so a jar slides at a
horizontal acceleration well below μg during those moments. Real records
have vertical peaks around 60% of horizontal, so this is not a small
correction — ignoring it let objects sleep straight through the instants when
they would actually have moved. Rocks on a slope wake at half their Newmark
yield acceleration instead, which is far below the level-ground figure.

Sleeping is not only a speed matter. Measured, for the record: the
sequential-impulse solver leaves a *stack* of bodies slightly compliant — a
pallet with twelve cartons on it moves at `v = −a × 47 ms` under a base
acceleration `a`, at any iteration count — so a stack kept awake under
shaking well below its sliding threshold jitters at tens of mm/s and walks a
few centimetres a minute. Six solver iterations made it far worse (a
four-plate stack at 36 mm/s, walking 8 cm in twenty seconds); ten, Bullet's
default, gives 2 mm/s. With the sleep thresholds at 0.10 m/s and 0.5 rad/s
that jitter is below the threshold, the stack sleeps, and a sleeping body is
exactly still, which is the physically right answer. Two related findings
became rules: stacked *cylinders* meet through the general convex algorithm
and get one contact point a step, so a plate stack spins and walks off its
shelf — plates are now collided as boxes under a cylinder visual; and a
free-standing carcass whose base, back and sides all touched the floor rocked
at 0.07 rad/s from five contact patches fighting in the solver — the sides now
stand on the base. Anything resting on a Bullet *kinematic* body is never
allowed to sleep, so driven structure (the rack frames) is built as static
bodies moved by transform.

**Driven supports and their loads.** A static body moved by transform has no
surface velocity, so Bullet's friction pins whatever rests on it to the
*world*, not to the support: with only the beams moved, a pallet that
should have ridden a 2.5 cm rack sway sat still and appeared to slide 3 cm
the other way (measured, M5.6 at 25 km; kinematic beams were measured too
and did no better through Panda, and their load never sleeps). So the load
is carried: whenever a level has moved 4 mm from where its load was last
put, every body on it is shifted by that displacement, and its own dynamics
run in the level's frame under the level's *absolute* acceleration —
`x_world = x_relative + d(t)`, the frame transformation and nothing more.
Two facts about the Panda binding shape the shift. Panda hands a body's
transform to Bullet only when that node's own transform is set (a moved
parent node moves the picture, not the physics), and the transform Panda
reads back lags the true state by one substep, because Bullet interpolates
motion states with latency: setting a moving body's transform pulls it
back by `v·h`, and a sliding box covered half its distance. Each awake body
is therefore set to `p + d + v·h` and rotated on by `ω·h`, which reproduces
the untouched trajectory to a micrometre; a sleeping body is put straight
back to sleep after the move and wakes, like everything else, when the
level's acceleration crosses its own threshold.

**Compound bodies and their centre of mass.** A Panda Bullet body has its
centre of mass at the node's origin, wherever the shapes are. Every compound
here was described from its base — a table at floor level, a gondola on the
slab, a cabinet, a stack of plates — so its mass sat on the floor and it
could not tip: measured, a 1 m wide, 2 m tall box built with its origin at
the base slid 3.5 m under 0.7 g and stayed upright, while the same box built
about its centre went over. Compounds are now placed at the volume-weighted
centroid of their parts with the parts offset back, and tip at `b/h` of that
centroid like everything else. The overturning *rule* for a gondola or a
chiller (0.35 / 0.38 g, ±12%) is the fragility of the loaded unit — far below
the rigid `b/h` of its empty carcass, whose centroid is 0.6 m up — so when
the rule fires the block is set rotating about its base edge with enough
energy to carry its centroid over the edge, in the direction the inertial
force is pushing, and gravity and the stock do the rest. Under strong
shaking the next half-cycle can still rock it back, which is real.

**Shelves are structures too.** A gondola is a 1.85 m steel frame carrying
some 300 kg of stock on cantilevered shelves, and a chiller a 2 m cabinet on
levelling feet: neither is rigid. Modelled as a rigid block that moves with
the slab, a shelf gives its contents exactly the ground motion, and at
0.30 g (M 8.3 strike-slip 11 km away, the GMPE median) that is enough to
slide 6% of the stock and drop 0.2% — much less than the CCTV footage of any
comparable event shows, and the reason the first stress test looked wrong.
Each gondola run and each chiller is now solved as its own one-storey shear
frame on the slab (`GONDOLA`: T = 0.35 s, `CHILLER`: T = 0.30 s, 5% damping,
steel-rack hysteresis) and its stock rides *that* motion, carried the same
way rack contents ride the racking in the warehouse. The periods are a
modelling assumption, checked by hand rather than measured: a 1.85 m
gondola upright of 50 × 30 × 1.5 mm tube bends across the aisle on its weak
axis (I ≈ 3.5 × 10⁻⁸ m⁴, k ≈ 3.3 kN/m per upright as a cantilever), a 17 m
run has about fourteen of them carrying 1–1.5 t of stock centred a metre up,
which gives 0.4–0.5 s as plain cantilevers fixed at the slab and less once
the shelves and back panels act as a frame, so 0.35 s is a stiff but
plausible fit-out. A 0.3–0.35 s
oscillator on a broadband crustal record sits near the peak of the response
spectrum, so the shelf sees 1.5–2× the ground: measured here, 0.48 g on the
shelves for 0.30 g on the floor. At full detail the same M 8.3 then shifts
roughly two items in five, topples a third and drops one in ten within the
first fifteen seconds, and the unanchored units go over by their own
overturning rule as before. The response is capped only by the steel-rack
model's own yielding; the shelf periods are constants at the top of
`scenes/supermarket.py` if a stiffer or softer fit-out is wanted.

**Static bodies sleep.** Bullet runs the narrowphase for a pair when either
body is active, and a static body built with deactivation disabled is
permanently active: every wall against a floor, every ceiling tee crossing
another, every shelf under a sleeping item was being collided afresh each
substep. Putting the statics into `ISLAND_SLEEPING` once the scene is built
removed every static–static manifold (1,419 in the supermarket) and took
its at-rest step from 64 to 44 ms; contacts with active bodies are
unaffected, and an element is woken when the collapse model releases it.

**Keyframes.** Bullet has no rewind, and a rigid-body state cannot be
interpolated, so seeking is built on snapshots: every few seconds of the
record `PhysicsWorld.snapshot` reads back every body's pose, velocity,
sleep state, floor group and released / crushed flags (velocities only for
awake bodies — a sleeping body's are zero by Bullet's definition, which
halves the cost of the read-back in a scene that is mostly asleep), and
`StructuralModel.snapshot` and `Scene.snapshot_state` keep the failure
record, the crack decals and the scene's own flags. `restore` puts a body
whose released or crushed state differs through the same release / refreeze
path the collapse model uses, so Bullet holds it exactly as it would have,
and re-poses the rest in place; static bodies are put back to sleep, since
setting a transform wakes them. A restore is exact to float32 and takes
~10 ms for the warehouse, ~130 ms for the supermarket. Continuing from a
restored state is a valid simulation but not a bit-copy of the
uninterrupted one — Bullet's island order changes with body order — so a
collapse re-run from a keyframe reaches the same structural outcome with
object counts a few per cent different (`tests/test_snapshots.py`).

**The observer.** Standing on a shaking floor, your feet go with the floor and
your head does not: you are an inverted pendulum with a period near ⅔ s and
heavy damping. The camera is offset by head-minus-floor, so the view lurches
the way a real one does instead of being given an arbitrary jitter.

---

## 7b. Collapse, landslide and cracks — `quakesim/structure/collapse.py`

A rigid-body world cannot bend a column, so structural failure is modelled
where it actually happens — at the moment an element stops carrying load —
and everything after that is rigid-body dynamics. Every scene is built with
its structure as *breakable elements*: static bodies that hold everything up
at no cost until a rule says otherwise, and are then **released** — given
their real mass and handed to the physics engine, which decides how they
fall, what they land on, and what that sets off. Nothing about the fall is
scripted.

**Rules.** Each element fails by one of

| rule | demand | typical use |
|---|---|---|
| `drift` | inter-storey drift ratio of its storey, from the shear-building solution | columns, infill, cladding, rack frames |
| `acc` | peak horizontal acceleration of its storey (or of the ground, for the walls of a single-storey masonry house — which *are* the structure, and whose solved roof acceleration is capped by their own yielding) | parapets, gables, out-of-plane masonry, unanchored gondolas and cabinets |
| `support` | enough of the elements it stands on have failed | slabs on columns, roof on purlins, purlins on the wall head, the storey above on the storey below |

An element may carry a `drift_limit` on top of its own rule, and any element
with supports goes when its supports go, whatever its rule. Support failures
cascade to a fixed point within a frame. Demands are the *peak over the
frame*, not the value at the frame's instant. Limits are scattered ±12% per
element so a wall comes apart panel by panel.

Limits follow HAZUS damage-state drifts (Complete: concrete frame 4%,
shear wall 2.5%, steel frame 6%, URM 2%, pre-code concrete frame 2%, storage
rack 7% — racks are released between Extensive and Complete, at 5%) and the
out-of-plane literature for masonry (top course 0.30 g, middle 0.42 g, bottom
0.60 g; gable courses 0.42 → 0.27 g toward the apex; parapets 0.35 g) and
the overturning fragility of unanchored shelving (median 0.35 g of ground
acceleration, in the FEMA P-58 / HAZUS range; the gondolas stand on the
slab, so their demand is the ground's, not the shed roof's, which sees
about twice as much).

**Crushing.** A storey that fails by its own rule in the KL tower is not
released as a block: it is *crushed* — taken out of the world — and the
storeys above lose their support and drop through it. The lowest standing
storey of the falling stack is crushed in turn when it lands (it has fallen
most of a storey height and its downward velocity has been arrested), which
is the pancake mechanism: the impact of everything above is beyond what any
storey carries. Each crushed storey becomes rubble where it was crushed.

**Soft storey.** The shophouse row is solved with a `stiffness_shape` of
(0.35, 1, 1, 1): an open shopfront under three infilled floors. The ground
storey then takes almost all the drift (0.9% there against 0.08% above at
M6.9), which is the mechanism, and its non-ductile columns go at 2%. What
fails in the ground storey is *crushed* -- columns, the infill panels
cracked through at 1%, the shop's shelving and chiller -- because in a
rigid-body world a failed column left standing is an intact block, and a
frame of released blocks under 39 t of slab stood there upright (measured:
the row did not come down at all until they were removed). The three
floors above are released as blocks and come down as a stack. Every
released element is also handed its storey's lateral velocity at the
instant of release; a frame lets go while it is moving.

**Landslide.** The Ranau slope is a field of rocks on a 33° surface at
friction angle 35°: static factor of safety 1.08, Newmark yield acceleration
`k_y = tan(35° − 33°) = 0.035 g`. Below `k_y` they are still; above it they
slide by exactly the Newmark integral, which the engine reproduces (Jibson's
regression gives about half a metre at 0.4 g; a firmer slope at `k_y =
0.15 g` moves a few centimetres, which is correct and invisible). A rock
that has slid 8 cm has its friction dropped to residual (φ 35° → 31°, below
the slope angle), and runs — the peak-to-residual strength loss that turns
a creep into a landslide.

**Cracks.** Structural damage short of collapse is invisible in a rigid
world, so it is drawn: procedurally generated crack textures laid as decals
on registered surfaces, in numbers that follow the storey's running peak
drift through the HAZUS damage states (a few hairlines at Slight, a web at
Extensive). Each surface belongs to the panel that carries it and goes with
it.

**Bullet details that mattered.** A body added to the world as static is
never integrated again no matter what its mass is later set to — Bullet
sorts bodies at `addRigidBody` time — so release removes and re-adds it.
Two static elements may overlap harmlessly right up to the frame both are
released, when the overlap becomes an impulse; the integrity test wakes the
static bodies for one substep so Bullet reports those pairs too (it skips a
pair only when neither body is active). The `RigidBodyCombiner` bakes
geometry at collect time and follows only transforms afterwards, so a
crushed storey is parked three kilometres underground rather than hidden.

---

## 8. Intensity — `quakesim/seismo/intensity.py`

- **Modified Mercalli** — Worden et al. (2012), the current ShakeMap
  relations, with the standard weighted crossover from acceleration-based at
  low intensity to velocity-based at damage levels.
- **JMA instrumental shindo** — the official algorithm: the three components
  are filtered (a `1/√f` period weighting, a low cut, and a high cut),
  combined into a vector magnitude, and the level `a₀` exceeded for a *total*
  of 0.3 seconds is found; `I = 2 log₁₀ a₀ + 0.94`, mapped to the 10-step
  scale.
- **China GB/T 17742** — the mean of the PGA-based and PGV-based forms.

---

## Scene integrity

Two failure modes ruin a physics scene before the earthquake starts:
interpenetration (bodies built overlapping, which Bullet resolves by firing
them apart) and unsupported placement (an object a few millimetres past the
edge of its shelf). Both look like "the earthquake did it" and neither is
physics.

`tests/test_scene_integrity.py` settles every scene with no shaking at all and
asserts nothing moves, reporting overlapping pairs and, for anything that does
move, what surface is under it and by how much. Overlaps are read from
Bullet's own persistent manifolds after one real substep, with every static
body woken for that step: an overlap between two fixed elements costs nothing
while both stand and becomes a shove the moment the collapse model releases
one, so those count as defects; overlaps between two things that can never
move (a ceiling grid's tees crossing, the ground box around the rock slope)
are reported and ignored. Two earlier versions of this probe were wrong in
instructive ways. Panda's `contactTest()` result holds a reference to a
manifold point that has gone out of scope, so every contact of a body
reported the distance of whichever pair was computed last (a 15 cm overlap
read as −0.0). And the probe used to be a one-microsecond fixed step: Bullet's
position correction is `erp · penetration / dt`, so 0.1 mm of contact noise —
normal for a bottle on an 18 m shelf — became an 18 m/s kick, and the audit
reported items "launched" that the simulator itself never moves. The real
probe found, and the scenes were fixed for: rack uprights built twice at the
same place (adjacent bays now share a frame), a ceiling tee run through a
column, the Ranau wall courses sharing their corners and its roof sheets
built 18 mm into the purlins, a mezzanine and a staircase built into the
columns beside them, and the longest gondola run built into the chillers.
All five scenes pass at every detail level.

`tests/test_stress.py` then runs every scene through a ladder of earthquakes
— from one that must do nothing (M4 at 20 km) through each scene's own
collapse case to M9.5 directly beneath, a 700 km deep-focus event, a 1500-bar
stress drop at 2 km and a 500 km megathrust with a full-length record —
checking for exceptions, NaNs, bodies flung beyond a kilometre, structure
that fails when it should not or stands when it should not, records past
the duration cap, and that a reset after a collapse restores every element
and object. The long records are simulated to the time 95% of their Arias
intensity has arrived. Two bugs found by it are now fixed: the modal FFT
solution wrapped a long record's ringing tail onto `t = 0` (the building was
swaying before P arrived), and the power-of-two FFT length let a 420 s cap
produce a 655 s record.

---

## References

- Atkinson, G.M. & Boore, D.M. (1995) *BSSA* **85**, 17–30 — hinged spreading
- Atkinson, G.M. & Silva, W. (2000) *BSSA* **90**, 255–274 — two-corner source
- Boore, D.M. (2003) *Pure appl. geophys.* **160**, 635–676 — stochastic method
- Boore, D.M., Stewart, J.P., Seyhan, E. & Atkinson, G.M. (2014)
  *Earthquake Spectra* **30**(3), 1057–1085 — NGA-West2 GMPE
- Brune, J.N. (1970) *JGR* **75**, 4997–5009 — source spectrum
- Cartwright, D.E. & Longuet-Higgins, M.S. (1956) *Proc. R. Soc. A* **237** — peak factors
- Gardner, G.H.F. et al. (1974) *Geophysics* **39**, 770–780 — density–velocity
- Hanks, T.C. & Kanamori, H. (1979) *JGR* **84**, 2348–2350 — moment magnitude
- Housner, G.W. (1963) *BSSA* **53**, 403–417 — rocking of rigid blocks
- Joyner, W.B., Warrick, R.E. & Fumal, T.E. (1981) *BSSA* **71** — quarter-wavelength
- Priestley, M.J.N. (1998) *Bull. NZ Soc. Earthq. Eng.* — displacement-based design
- Saragoni, G.R. & Hart, G.C. (1974) *Earthq. Eng. Struct. Dyn.* **2** — envelope
- Strasser, F.O., Arango, M.C. & Bommer, J.J. (2010) *SRL* **81**(6) — subduction scaling
- Wells, D.L. & Coppersmith, K.J. (1994) *BSSA* **84**(4) — rupture scaling
- Worden, C.B. et al. (2012) *BSSA* **102**(1), 204–221 — MMI relations
- ASCE 7-16, HAZUS-MH technical manual — periods and damage-state drifts
- Newmark, N.M. (1965) *Géotechnique* **15**(2), 139–160 — sliding-block slope displacement
- Jibson, R.W. (2007) *Eng. Geol.* **91**, 209–218 — Newmark displacement regressions
- Doherty, K. et al. (2002) *Earthq. Eng. Struct. Dyn.* **31** — out-of-plane URM walls
