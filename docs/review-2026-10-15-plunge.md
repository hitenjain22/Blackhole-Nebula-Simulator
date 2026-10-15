# Why gas falls into the hole too early: review of the Phase 4 findings (15 Oct 2026)

This is for Hiten first, then for Claude Code. I checked the repo at commit `60a72eb` (Phase 4, Steps 4.1–4.5) and read DECISIONS.md, AI_LOG.md and the Phase 4 breakdown. Then I reran everything with the repo's own code. Only settings were changed (through `dataclasses.replace`); nothing in `src/` was edited. One test uses a prototype installed on top, `scripts/recon.py`, and it is labelled wherever it appears. The scripts are in `scripts/` next to this file.

**Status:** final, updated the evening of 15 Oct with the smaller-cloud results (D4) and a better measure of how hot the gas should really get (§2d). Two runs were stopped before the end, with their counts still rising: radius 100 with the fix, and radius 25 without it (§2d).

---

## 1. The short version (for Hiten)

**It is not a bug in your code, and it is not real physics. It is the simulation method reaching its limits.** Three things are stacked on top of each other.

**1. With 1,000 particles, the "neighbour bubbles" near the hole are bigger than the hole's neighbourhood.**
- Each SPH particle feels its neighbours inside a bubble that must hold about 137 particles.
- Near the hole, the 1k cloud's bubbles are 3–10 times wider than their distance from the hole.
- So a particle counts gas on the *far side* of the hole as a neighbour, even though that gas is moving the opposite way at up to a quarter of the speed of light.
- The shock friction (artificial viscosity) treats that as a head-on crash. It brakes the particle and heats it to about 100 billion K, and the hole captures it.
- The bubbles span the hole even with viscosity switched off. So **1k can never be used near the hole.**

**2. On the long fall inward, the viscosity heats and stirs gas that should stay cold.**
- As the cloud falls, it is squeezed sideways faster than its sound speed. The viscosity mistakes that smooth squeeze for a shock.
- By the time the gas arrives it is 25–34 times hotter than it should be.
- Its angular momentum is smeared by 4–6% rms, against about 1% from real pressure.
- So some particles arrive hot and swing in deeper than they should.

**3. Right at closest approach, the gas's path bends so sharply that the code sees collisions that aren't there.**
- At 10,000 particles the current viscosity overheats the hottest gas there by hundreds of times more than the real squeeze can produce. With the fix below it is still 20–30 times too hot.
- The hot gas swells, its bubbles grow across the hole, and some of it is braked into the hole.

**What fixes what:**
- **Problem 2 has a published fix, and I tested it:** velocity reconstruction, from Frontiere et al. 2017 and Rosswog's MAGMA2 code. Rosswog is one of the authors of the gravity method you already use.
  - How it works: the viscosity only acts when the velocities around a particle genuinely disagree, as in a shock, and stays off when they change smoothly.
  - With it, the fall inward is exactly as clean as with no viscosity at all.
  - It still passes the plan's shock-tube test.
  - At 10k it cut the particles swallowed by t = 388,348 from **32 to 0**.
- **Problem 3 needs finer resolution near the hole.** The cheapest way is a smaller, denser cloud (D4).
  - With the fix and a radius-25 cloud, the 10k run takes the whole cloud past closest approach with **nothing swallowed**, as the gravity-only run says it should.
  - Neither change alone does it: the radius-25 cloud without the fix lost 30 particles, and the radius-100 cloud with the fix lost 17 and counting.
  - The gas at closest approach is still warmer than it should be. The 50k run will show that more accurately.

**How hot should the gas really get?** I measured this with a 50,000-particle gravity-only run (§2d). Even if every bit of the squeeze at closest approach turned into heat, typical gas would reach about 3 × 10⁻⁷ in energy units (about **1 million K**), and the hottest gas about 10⁻⁵ (about **40 million K**). The runs as in the repo reach 10⁻³ to 10⁻², which is hundreds to thousands of times too much.

**Your Claude's instinct was right.** Its Step 5.0 tests (gravity only, splitting the heating, 10× particles, the 13% drift) were exactly the right ones. I've run all of them (§2), so your Mac only needs to confirm a few.

---

## 2. The evidence

All runs use the cloud from `cloud_ic()` on block steps with the repo's defaults, unless a table says otherwise.
- **Lowest L** is the smallest angular momentum of any particle; below 4 the hole captures it.
- **u** is internal energy: 2.3 × 10⁻⁹ is 10⁴ K, and 10⁻² is about 4 × 10¹⁰ K.

### 2a. What makes the 1k particle fall in

| Run | What changed | Falls in? | Lowest L | Hottest u |
|---|---|---|---|---|
| A | gravity only | no (to t = 390,000) | 7.213 (exactly conserved) | 2.3 × 10⁻⁹ |
| B | viscosity off | no (to t = 390,000) | 7.077 | 2.3 × 10⁻⁸ |
| C | **as in the repo** | **yes, t = 386,646** (particle 146, matching DECISIONS.md) | 0.26 | 2.6 × 10⁻² |
| D | conduction off | yes, t = 387,017 | 0.12 | 6.4 × 10⁻² |
| E | pressure-based conduction | yes, t = 386,869 | 0.36 | 2.5 × 10⁻² |
| F | viscosity, β part only (α = 0) | yes, t = 386,867 | 0.81 | 1.8 × 10⁻² |
| G | viscosity, α part only (β = 0) | not by 390,000, but 15 particles already doomed (L < 4) | 0.66 | 2.4 × 10⁻³ |
| H | cubic-spline kernel (smaller bubbles) | yes, t = 386,251 (earlier) | 0.23 | 2.2 × 10⁻² |

What these runs show:
- **Gravity, pressure, conduction and block steps are not the cause.**
- **Total angular momentum is conserved** to 10⁻⁹ in every run. So the viscosity *moves* angular momentum between particles rather than destroying it.
- **Real stream crossing is gentle.** With gravity only, the gas paths do fold over each other after closest approach, but the crossing flows differ in speed by less than 0.003c (§2d). So the head-on crashes the viscosity reacts to are not real.
- **Both parts of Phantom's viscosity contribute**, so tuning the switch alone can't fix it.
- **Your code matches Phantom here.** Phantom's quadratic β term acts on every approaching pair whatever the switch says (`force.F90`: `vsigavi = max(alphai*vwavei - beta*projv, 0.)`), and your code copies this correctly.

### 2b. Do the bubbles span the hole?

Measured for particles at r < 60. "Across the hole" means a neighbour pair more than 90° apart as seen from the hole.

| Run | Time | Typical 2h/r | Neighbour pairs across the hole |
|---|---|---|---|
| 1k, as in repo | 384,624 → 386,424 | 9.1 → 3.0 | 90% → 35% |
| 1k, viscosity off | 384,613 → 389,513 | 10.2 → 1.7 | 87% → 21% (1k is too coarse even without viscosity) |
| **10k, viscosity off** | 385,280 → 389,005 | 0.7 → 0.2 | **0%** |
| 10k, as in repo | 383,520 → 384,717 | 8.2 → 1.3 | 78% → 16% (viscous heating makes the bubbles swell) |

### 2c. The fall inward (problem 2), at t ≈ 384,600

| Run | Spread of L change (rms) | Worst 1% | Heating since start |
|---|---|---|---|
| 1k, viscosity on | 5.7% | −11.7% | 34× |
| 1k, viscosity off | 1.1% | −2.7% | 2.5× (just squeezing) |
| 10k, viscosity on | 3.6% | −8.9% | 25× |
| 10k, viscosity off | 1.2% | −3.6% | 3× |
| 1k / 10k with the **reconstruction fix** | as with viscosity off (lowest L 7.078 at t = 359,936; 6.964 at 383,352) | | as with viscosity off |

The viscosity switch is fully on (α = 1 for every particle from t ≈ 150,000) even though nothing is shocking:
- The gas is so cold (sound speed 5 × 10⁻⁵) that any squeeze trips the switch.
- The switch's limiter only calms it in *swirling* flow, and this flow has no swirl.
- Ten times more particles barely helps (5.7% → 3.6%).

### 2d. Closest approach (problem 3)

**Particles swallowed with swallowing switched on** (r_acc = 4; script `run_cloud_acc.py`):

| Run | Swallowed | By time |
|---|---|---|
| 1k, as in repo | 37 of 993 | t = 394,985 |
| 1k with fix | 56 of 993 | t = 399,247 (1k is too coarse either way) |
| 10k, as in repo | 32 of 9,992 | t = 388,348 |
| 10k, viscosity off until t = 380,000, then on | 27 | t = 388,478 |
| 10k with fix | **0 at t = 388,348**; 3 by t = 390,065; 17 by t = 398,575 | the cloud is about halfway past closest approach, and the count is still rising |
| 10k, radius 25, as in repo | 30 (the first at t = 392,666) | t = 396,370, still rising |
| **10k, radius 25, with fix** | **0**, with the whole cloud past closest approach | t = 400,000 |

**What the gas looks like at closest approach.** Take the cleanest possible arrival: 10k with viscosity off.
- Within one bubble near the hole, pairs approach each other at up to 0.02–0.04c, about 400–800 times the sound speed. This comes mostly from the orbits bending, not from real collisions.
- Reconstruction cuts the typical approach speed by 10–40×. But the fastest 1% of pairs still approach at 0.007–0.019c, and at those speeds any viscosity heats the gas to 10⁻⁴–10⁻³ (the real ceiling is about 10⁻⁵; see below).
- The cure is a bubble that is small compared with the distance to the hole. With reconstruction the leftover error falls about as h⁴, so halving h cuts it 16×. Without reconstruction it falls only as h².

**How hot the gas should really get.** I ran the cloud with gravity only at 50,000 particles (`ballistic.py`) and measured how hard the gas is squeezed, and how fast its paths cross. The first version of this review gave 2 × 10⁻⁶; that counted only the up-and-down squeeze.
- **Paths do cross, gently.** After closest approach the gas folds over itself (about half the gas near the hole by t = 392,000). But once the smooth flow is taken out, the two sides of each fold differ in speed by less than 0.003c. That is worth at most about 10⁻⁶ of heat (`folds.py`).
- **The squeeze matters more.** The up-and-down part is exact: no particle's ½ v_z² goes above 6 × 10⁻⁶, and half stay below 4 × 10⁻⁷ (`vz_budget.py`). The in-plane squeeze is about the same size: 90% of the gas below 2–6 × 10⁻⁶ from t = 386,000 on (`conv_budget.py`). That estimate is blurred by bending for the gas right at the hole, so it is a ceiling.
- **So the real ceiling,** if every bit of that motion became heat, is about 3 × 10⁻⁷ for typical gas and about 10⁻⁵ for the hottest.
- **For the radius-25 cloud (D4) it is about 16 times lower,** because a smaller cloud is squeezed less: typical about 2 × 10⁻⁸, hottest 1% about 1–3 × 10⁻⁶ (up-and-down part at most 4 × 10⁻⁷).

| Gas near the hole (r < 300), internal energy u | Typical (median) | Hottest 1% | Hottest |
|---|---|---|---|
| Real ceiling (gravity only, 50k) | about 3 × 10⁻⁷ | about 10⁻⁵ | about 10⁻⁵ |
| 10k as in the repo, t = 384,717 (its first plunge) | 3.7 × 10⁻⁷ | 3.5 × 10⁻³ (350×) | 3.9 × 10⁻² |
| 10k with the fix, t = 386,980 | 8.8 × 10⁻⁶ (30×) | 2.1 × 10⁻⁴ (20×) | 2.0 × 10⁻³ |
| Real ceiling, radius-25 cloud (gravity only, 50k) | about 2 × 10⁻⁸ | 1–3 × 10⁻⁶ | about 5 × 10⁻⁶ |
| 10k, radius 25, with the fix, t = 396,976 | 1.4 × 10⁻⁶ (70×) | 8.6 × 10⁻⁶ (3–8×) | 4.6 × 10⁻⁴ |

**What the fix does at closest approach** (10k with the fix, followed in detail to t = 387,430):
- **The bubbles never reach across the hole** (0% of pairs, 2h/r about 0.5–1).
- **The gas still overheats,** about 20–30 times above the real ceiling (table above). By t = 387,430, 761 particles are above u = 10⁻⁵ and 9 are above 10⁻³.
- **This leads to a new way of losing angular momentum.** The hot gas swells as it heads back out. At r ≈ 110–130 its bubbles (h ≈ 50) overlap the gas still falling in, which is moving the opposite way. The viscosity brakes the two against each other.
- **One particle shows the whole chain.** Particle 1353 kept L = 6.47 through closest approach, then lost it on the way out (6.47 → 0.62, r ≈ 110–130).

So the fix removes problem 2 and the reaching across the hole. The overheating at closest approach is pure resolution, and it is what D3/D4 have to solve.

### 2e. The hot-blob test (Phase 4)

I reran the u = 4 × 10⁻⁸ blob past t = 18,000. It fell in at t = 22,720, as in DECISIONS.md.

| Phase | 2h/r near the hole | Pairs across the hole |
|---|---|---|
| First pass, t = 13,000–14,000 | ≤ 0.5 | **0%** (clean) |
| Debris returning, t = 20,000–22,700 | 5–18 | 32–100% |

With viscosity off, the same blob runs to t = 26,000 with every particle's L above 7.03. So the Phase 4 ledger test stopped at t = 18,000 is valid. The debris plunge after that is this numerical effect, not "real physics (the stream collisions of Phase 6)".

---

## 3. Decisions for Hiten

Each comes with a recommendation. Rule 7 of your CLAUDE.md says method changes need your OK.

**D1. What the 1k cloud is for.** *Recommend:* a smoke test only (does it run, do the ledgers stay flat). Never use it for anything near the hole.

**D2. Fix the viscosity on the way in: slope-limited velocity reconstruction.** *Recommend: yes.* This is a method change, so it needs your OK. It follows Frontiere, Raskin & Owen 2017 (J. Comput. Phys. 332, 160) and Rosswog 2020 (MNRAS 498, 4230, MAGMA2, Eqs. 17–18).

How it works:
- Before the viscosity compares two particles, each particle's velocity is projected to the point halfway between them, using its own velocity gradient.
- In smooth flow the two projections agree, so there is no friction.
- In a shock they disagree, and a limiter switches the viscosity back to normal.

What it costs:
- About 20–30% slower in my test.
- A bit more ripple behind shocks: 1.5% instead of 0.4% (the plan's limit is 5%).

How the prototype (`scripts/recon.py`) does against your repo's own test criteria:

| Test (plan / repo criterion) | Current viscosity | With the fix |
|---|---|---|
| Sod, full resolution: plateaus within 3% | within 1.1% | within 1.0% |
| Sod: shock within 2 h | 0.18 h | 0.27 h |
| Sod: post-shock ripple below 5% | 0.4% | 1.5% |
| Gresho vortex (low), switch on: L1 error, lower is better | 0.143 | **0.123** (14% better) |
| Sound wave, 32 per wavelength: phase error below 0.01 | 0.0032 | 0.0032 |
| Hot blob, shared step: energy below 10⁻⁵, ratio 3–5 | (repo: passes) | 4.7 × 10⁻⁶, ratio 4.0 |
| Hot blob, shared step: angular momentum below 10⁻¹⁰ | (repo: 2 × 10⁻¹⁵) | 3 × 10⁻¹⁵ |
| Hot blob, block steps: energy below 10⁻⁵, ratio 3–5 | 4.06 × 10⁻⁶, ratio 4.0 | 4.06 × 10⁻⁶, ratio 4.0 |
| Hot blob, block steps: angular-momentum drift falls ≥ 3× per halving | 3.1 × 10⁻¹⁰, 3.5× | **2.4 × 10⁻⁹, 1.9× (fails)** |
| Blob: lowest particle L (higher is better) | 6.67 | 6.99 |

**The one failure, and where it comes from.** It is the block-step angular-momentum check. I found the cause by switching pieces of the prototype off:
- Without Rosswog's extra "closeness" factor (it switches reconstruction off for very close pairs), the drift is 4.1 × 10⁻¹⁰ with a 2.6× fall per halving. That is almost the current code's result.
- Without the limiter as well, it is 7.9 × 10⁻¹¹ with a 6.7× fall per halving, better than now. But a limiter is needed for shocks.
- Making the limiter smoother made it worse: 2.3 × 10⁻⁹, a 1.4× fall. So the trouble isn't the limiter's sharp corner. It is how strongly the limiter reacts to the neighbours' velocity gradients, which on block steps are slightly out of step.
- So for the real version: leave out the closeness factor, and find a limiter that reacts less to the neighbours' gradients on block steps (one idea: fix each particle's limiter once per step). This is an open question, not a known recipe.
- On block steps, the neighbours' velocity gradients must be recomputed at the current time, just as your code already does for their densities. With stale gradients the drift was 1.8 × 10⁻⁸.
- If the 3× can't be reached, it comes back to you as a decision. Never loosen the test.

In the "current viscosity" column, the Sod, Gresho and sound-wave values are my reruns; the blob values come from DECISIONS.md. The Sedov blast (110k particles, many hours) hasn't been run with the fix yet.

Your Claude Code should write it properly: a reference version and a Numba version, like the rest of `sph.py`. Then it reruns the Sod, Sedov, Gresho and sound-wave tests and the blob ledger with unchanged tolerances.

**D3. Resolution near the hole.** *Recommend:* after D2 (and D4 if you pick it), run 50k overnight and check the gas temperature at closest approach against the gravity-only ceiling in §2d.
- With radius 25 and the fix, 10k already gets the orbits right (D4). The 50k run is for the temperatures.
- With radius 100, 10k loses gas even with the fix (17 particles by t = 398,575, and rising). It needs bubbles at the hole about 3× smaller, very roughly 100k–300k particles. That is many nights on your laptop.

**D4. Scenario option: a smaller, denser cloud** (radius 25 instead of 100). This changes the scenario, so it is your call. *Recommend: yes, together with D2.*

A 4× smaller cloud makes every bubble about 4× smaller at the same particle count. That is as good as roughly 64× more particles, with no extra computer time. Here is what it did (10k particles, swallowing on):

| Cloud | Viscosity | Swallowed | Lowest L afterwards | Bubbles across the hole |
|---|---|---|---|---|
| Radius 100 | as in the repo | 32 by t = 388,348 | 0.9 | 16–78% of pairs |
| Radius 100 | with the fix | 17 by t = 398,575, still rising | 0.4 | 0% |
| Radius 25 | as in the repo | 30 by t = 396,370, still rising | 0.3 | not measured |
| **Radius 25** | **with the fix** | **0**, with the whole cloud past closest approach (t = 400,000) | **5.2** (no particle below 4) | **0%** (2h/r about 0.1–0.3) |

- **Only the two together work.** They take the cloud past closest approach the way the gravity-only run says it should go: nothing captured. The spread in angular momentum is 3% rms, and the worst 1% lost 8%.
- **What remains: the gas is still too warm.** Typical gas reaches about 1.5 × 10⁻⁶ (about 6 million K), against a real ceiling of about 2 × 10⁻⁸ for this cloud. The hottest 1% reach about 10⁻⁵, against 1–3 × 10⁻⁶. That is far too little heat to change the orbits (the gas's sound speed stays under 1% of its orbital speed). But temperatures at closest approach can't be trusted at 10k, so check them in the 50k run (D3).
- **The trade-off:** the stream is narrower, the disk forms from less spread-out gas, and the pictures change a little. The real G2 cloud is far bigger than either size, so neither is "more realistic".
- **If you'd rather keep radius 100,** it needs roughly 100k–300k particles with the fix (D3).

**D5. A new check in every run: "do the bubbles span the hole?"** *Recommend: yes.* For particles at r < 60, record at each snapshot:
- the median and largest 2h/r;
- the percentage of neighbour pairs more than 90° apart around the hole;
- the typical and hottest-1% gas at closest approach, compared with the gravity-only ceiling in §2d (radius 100: about 3 × 10⁻⁷ typical, 10⁻⁵ hottest; radius 25: about 16 times lower).

Print a warning when the across-the-hole percentage is above 1%. This turns today's problem into numbers in every report, and it makes a strong portfolio figure.

**D6. The swallowing radius r_acc = 4, now.** *Recommend: yes* (it is already in Phase 5's table). Removed particles get full ledger entries (mass, H, u, x × p), so runs stop crashing. Keep the plan's alarm: any particle swallowed on the *first* pass is listed with its history.

**D7. Conduction: velocity-based.** *Recommend: approve.* It is Phantom's own rule whenever gravity acts, and it has nothing to do with this problem (runs C, D, E).

**D8. The blob test.** *Recommend:* keep the cooler blob stopped at t = 18,000 as the Phase 4 ledger test, because its pass is fully resolved. Correct the "real physics" sentence in DECISIONS.md. The "warm blob over a full orbit with swallowing at r = 4" variant isn't worth running: that blob swells to about 100 across and would only measure this artifact.

**D9. Kernel.** *Recommend: keep Wendland C4, η = 1.6.* The cubic spline's smaller bubbles still span the hole, and its particle fell in even earlier (run H).

**D10. The speed-up is only 1.5×.** *Not a problem yet.* During the long approach, the whole cloud sits at about the same distance from the hole, so nearly everyone shares one step size. Block steps pay off at and after closest approach. Re-measure on the 10k run.

**D11. A small bug found on the way.** Setting `h_max` to 10 or 20 crashes the gradient pass (`SystemError` in `_gradients_kernel`) once a capped particle has too few neighbours, because the correction matrix becomes singular. It isn't needed for anything above. Log it, and add a fallback: use the uncorrected gradient when the matrix is near-singular. `scripts/recon.py` shows one way.

---

## 4. What I got wrong in the plan

- **The warm blob** (u = 4 × 10⁻⁶). Your Claude caught and fixed it correctly.
- **Phase 5 assumed "1k, then 10k" would get through closest approach.** It also said a crash before t = 500,000 meant "a bug or a pressure effect". I didn't foresee how badly SPH viscosity handles a cold, sharply bending stream near the hole. The plan's own alarm ("particles reaching r_acc on the first pass: stop and investigate") is what caught it.
- **Phase 3 copied Phantom's viscosity without its known weakness** in cold supersonic flow. D2 fixes that.

---

## 5. Message for your Claude Code (copy and paste)

> First move the `nebula-review` folder I put in the repo root: `plunge-review-2026-10-15.md` to `docs/review-2026-10-15-plunge.md`, and its `scripts/` folder to `scripts/review-2026-10-15/`. Then read the review fully before Step 5.0; it replaces the investigation you planned. The plunge, the 13% angular-momentum drift and the blob's late plunge are numerical: Phantom's viscosity heats the cold gas on the way in, and near the hole the cloud is under-resolved. Gravity, pressure, conduction and block steps are fine.
>
> 1. Confirm on this Mac with the review's scripts: runs A, B and C, and the across-the-hole check. Record the results in DECISIONS.md and AI_LOG.md, and correct the "real physics" sentence in the blob entry.
> 2. Hiten approves D1, D5, D6, D7, D8, D9 and D11 as written.
> 3. D2 (velocity reconstruction in the viscosity) is approved as a method change. Implement it properly (a reference version and a Numba version, with its equation and sources in the docstring) and rerun every Phase 3 and 4 test with unchanged tolerances. If the block-step angular-momentum check can't reach its 3× fall (review D2), stop and ask Hiten.
> 4. D4 (cloud radius 25 instead of 100) is approved as a scenario change. Record it in DECISIONS.md. Then run the 10k radius-25 cloud with D2 and swallowing on, and compare it with the review's own run, where nothing was swallowed by t = 400,000. Report the swallowed fraction and the typical and hottest-1% gas at closest approach against the gravity-only ceiling in the review's §2d.
> 5. Then run 50k overnight (radius 25, with D2) and compare its temperatures at closest approach with the same ceiling.
>
> Whenever you compare with the review, put your numbers next to the review's in a table and say in plain words which match and which don't.
>
> Never change an expected value or a tolerance.

(Hiten: if you don't approve D2 or D4, delete its item from that message. Without D4, item 4 becomes "run the 10k radius-100 cloud with D2" and item 5 needs far more particles; see D3.)

---

## 6. How to reproduce (for Claude Code)

Run everything from the repo root:

`uv run python scripts/review-2026-10-15/<script> <out.npz> key=value ...`

Times are single-core on my machine, at about 20,000 particle-steps per second with four runs sharing it. DECISIONS.md measured about 30,000 per second.

| Script | What it does | Time |
|---|---|---|
| `run_cloud.py OUT.npz [n=] [hydro=off] [visc=off] [cond=pressure\|off] [alpha=0] [beta=0] [kernel=cubic] [hmax=]` | the cloud to t = 390,000, no swallowing; stops at the first plunge and saves | 20 s (gravity only) to 2 min (1k); 10k about 21 min to its plunge |
| `run_cloud_acc.py OUT.npz [n=] [radius=] [recon=on] [visc=off] [von=T] [tend=] [tdense=] [dtdense=]` | the same with swallowing at r_acc = 4 (logged); `recon=on` installs the prototype fix; `von=T` turns viscosity on only from time T | 1k with fix: 30 min to t = 399,000; 10k: about 45 min to t = 388,000; 10k radius 25 with fix: about 2 h to t = 400,000 |
| `run_blob.py OUT.npz [visc=off]` | the u = 4 × 10⁻⁸ blob to t = 26,000 | about 2.5 min |
| `across.py RUN.npz [times]` | the across-the-hole check | seconds |
| `lstats.py RUN.npz` | angular-momentum spread and heating over time | seconds |
| `doomed.py RUN.npz ...` | per run: stop time, particles with L < 4, lowest L | seconds |
| `pairspeeds.py RUN.npz times` | approach speeds of neighbour pairs near the hole, raw and reconstructed | seconds |
| `recon.py` | the **prototype** fix (monkey-patches `forces_numba`); not repo code | – |
| `test_sod.py plain\|recon [quick\|full]` | the Sod tube with and without the fix | full: about 5–6 min |
| `suite_recon.py blob\|gresho [plain]` | the blob ledger, Gresho vortex and sound wave, with or without the fix | blob: about 10 min |
| `blob_mode.py MODE` | the block-step blob check with parts of the fix switched off (modes listed at the top of `recon.py`) | about 6 min |
| `ballistic.py OUT.npz [n=50000] [radius=100] [t0=] [t1=] [dt=]` | the cloud with gravity only, at many particles, saving positions and momenta around closest approach | 50k: 20–30 min |
| `vz_budget.py BALL.npz` | the exact up-and-down squeeze ceiling, ½ v_z², per particle | about 1 min |
| `conv_budget.py BALL.npz times [k=64]` | the squeeze ceiling in every direction | a few min |
| `folds.py BALL.npz times` | where the gas folds over itself, and how fast the two sides meet | a few min |
| `ustats.py RUN.npz times` | internal energy of the gas near the hole: typical, 90%, 99%, hottest | seconds |
| `lend.py RUN.npz times` | angular-momentum change of the gas that has passed closest approach | seconds |

Runs A, B and C are `run_cloud.py` with `hydro=off`, `visc=off`, and no options, respectively.

---

## 7. Addendum (evening of 15 Oct): cloud size and the Phase 6 stream crash

Your Claude Code spotted something I missed when recommending D4. Phase 6's first stream crash comes from the spread in orbital energy across the cloud, and that spread scales with the cloud's radius. A smaller cloud stays together longer, so its stream takes more passages to grow long enough to run into itself. I reran the plan's own script 12 for other radii (`scripts/crash_by_radius.py`; same test-particle model, pressure ignored). Its radius-100 result matches expected-values §2b exactly.

| Cloud radius | First crash | Time (1 M ≈ 21.2 s) | Share of the stream still incoming at first contact |
|---|---|---|---|
| 25 | passage 8 | t ≈ 5.9 million (about 4 years) | 3% |
| 50 | passage 4, barely; clearly at passage 5 | t ≈ 2.8 million / 3.6 million | 1% |
| 60 | passage 4 | t ≈ 2.8 million (about 1.9 years) | 10% |
| 75 | passage 3, barely | t ≈ 2.0 million | 3% |
| 100 | passage 3 | t ≈ 2.0 million (about 1.3 years) | 24% |

Phase 6 plans to run to t = 3 million. So radius 25 gets a clean first pass at 10k but pushes the crash to about twice the planned run length. Radius 100 crashes on time, but needs roughly 100k–300k particles near the hole (D3). Radius 50–60 might do both, and a radius-50 test at 10k with the fix is running now. Nothing changes for the 10k radius-25 run: it is still the check that the real D2 matches the prototype. Decide the Phase 6 cloud size once the radius-50 result is in.
