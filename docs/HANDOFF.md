# HANDOFF: read this first (for any Claude session working on this project)

**Purpose.** Hiten switches between Claude accounts when one runs out of usage. This file carries
everything a new session needs to continue seamlessly, as if it were the same assistant. It is
written for Claude, not for Hiten.

**Your two standing jobs with this file:**
1. **Read it fully before doing anything** (plus `CLAUDE.md`, `PLAN.md`, the current phase file,
   and the latest entries of `docs/DECISIONS.md` and `AI_LOG.md`).
2. **Keep it current.** Whenever something meaningful happens (a decision, a finished step or run,
   a new problem, a cloud change, anything Hiten says that matters), update §3 "Current state",
   §4 "Waiting on Hiten", §7 "Runs inventory" and add a line to §11 "Session log". Do it as you go,
   not only at the end: the session can end abruptly when usage runs out. Commit and push it.
   The next session (possibly the original account, after its usage resets) relies on it.

Last updated: 2026-10-19 (night, Pacific).

---

## 1. Who Hiten is and how to work with him

- Hiten owns the project and **does not write code**. Explain in plain language, no jargon; when a
  term is needed, say what it means. Give him something concrete to look at after every step
  (a picture path, a number, a table). He checks results visually and makes the decisions.
- He values **correctness over speed** ("we can take weeks"; "I'd rather spend days fixing a bug
  than have it cause problems later"). Investigate odd results to the root before building on them.
  Cross-check against papers and open-source codes (Phantom, PySPH/MAGMA2, SWIFT, GADGET). Licenses
  matter: Phantom is GPL, so never copy its code; re-derive from papers and compare.
- He likes honest status: say plainly when something failed, when you were wrong, and when an
  estimate is uncertain. He asks "how's it looking?" often; answer with a short status and the key
  numbers. He likes side-by-side tables.
- **Workflow (CLAUDE.md):** one step at a time; before coding, say in 2–5 lines what you'll build
  and how you'll test it; after each step run the suite, report expected/actual/PASS, say what to
  look at, then **stop and wait** for his OK; commit on confirmation (never with failing tests);
  log notable slips in `AI_LOG.md` and decisions in `docs/DECISIONS.md`.
- **When he says he is away or asleep:** keep going step by step without waiting, verify extra
  thoroughly, decide with the option you'd recommend and record it in DECISIONS, and leave a
  morning summary. Per-night scope limits he sets are binding (e.g. "don't start Phase 7").
- Commit attribution line: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Pushing to
  GitHub (origin main) after a passing suite is approved standing practice.
- Never change an expected value or loosen a tolerance to make a test pass (CLAUDE.md rule 3). If a
  *new* test's tolerance is being set, derive it from the method's own accuracy and say so.
- Method changes need his approval (rule 7) and an entry in DECISIONS.
- He has asked: no multiple free-trial accounts on Google Cloud (told him it breaks Google's terms);
  don't use his brother's Microsoft employee Azure credits (personal-use only terms).

## 2. The project in one paragraph

Simulate a 0.1 M☉ gas cloud (G2-like, 10⁴ K) falling onto a 4.3 × 10⁶ M☉ non-spinning black hole
from r = 5,000 (code units G = c = M = 1; time unit 21.18 s; length 6.35 × 10⁶ km), pericentre 25:
tidal stretching into a stream, relativistic precession (24.04°/orbit), stream self-crash, shock
heating, cooling, circularization into a torus near r ≈ 51, later viscous drain (Phase 7), then a
GPU-rendered, lensed, true-colour video (Phases 8–10), spin (11), science report (12), portfolio
(13). SPH (Wendland C4, η = 1.6, grad-h), GN gravity (Tejeda & Rosswog 2013), generalized
leapfrog, block time steps, Phantom-style viscosity with Cullen–Dehnen switch + velocity
reconstruction (D2, Frontiere 2017 / MAGMA2) with a min-mod gate, gas + radiation EOS, β-cooling,
swallowing at r_acc = 4 with a removed-particle ledger. See `PLAN.md` and `phases/`.

## 3. Current state (2026-10-19 evening)

- **Phases 0–5 done.** Phase 6 code steps 6.1–6.7 built and all Phase 6 tests 1–9 pass. Phase 6
  science runs are the open work. **Phase 7 not started** (needs a breakdown approved first).
- **Phase 6 cloud radius = 60** (Hiten, D4 update), end time **t = 3.8 × 10⁶** (Hiten approved).
  Expected first stream crash: passage 4, first contact t = 2,765,139 (test-particle; passage 3
  misses by only 8,700, so hydro may crash at passage 3, t ≈ 2.0 × 10⁶). expected-values §2c.
- **Resolution:** 10k runs are dominated by numerical overheating at pericentre (unbound 11%,
  energy spread ±43%); 50k is much better (0.75%, ±13%). **Phase 6 main run R6.1 must be 50k.**
- **R6.1 v2 at 50k is RUNNING on Google Cloud (`runs/R6.1_50k_v2`, started 2026-10-19 17:27 Pacific / 00:27 UTC,
  commit ffcd410, α_u = 0.05, h_max = 200, t_end 3.8 × 10⁶).** Helpers armed: `~/resume_if_needed.sh` and
  `~/shutdown_when_done.sh` (both for R6.1_50k_v2, in crontab @reboot; the watcher is running). The old
  stalled `R6.1_50k` (α_u = 1, no cap) is kept on the VM as a record (paused at t ≈ 446,000).
- **Why it stalled (fully diagnosed, DECISIONS 2026-10-19):** over-heated gas sprays into a
  near-vacuum halo; halo particles' h grows to 2,000–2,800, neighbouring the whole cloud; the level
  limiter then held ~49,000 particles at tiny steps. Capping h (h_max) fixes it and does **not**
  change the physics (verified digit-for-digit at 10k through the first pass). A real bug found on
  the way is fixed: the neighbour search had no skin margin above h_max (forced a rebuild every step).
- **Overheating root cause (heat-budget instrumentation, DECISIONS 2026-10-18/19):** ~30
  front-edge particles are over-heated by artificial viscosity at the bend around the hole (u up
  to ~10⁻²); artificial **conductivity** (ours: Phantom's α_u = 1 with velocity signal) pumps that
  heat into the stream (~90% of the typical particle's excess heat). Experiments cond_x1–x4: weaker
  conductivity halves the unbound gas (11% → 5.8%) and cuts pre-pass heating ~10×, but the energy
  spread (±42% at 10k) does not depend on conductivity (resolution).
- **Speed work done (2.8× on the Mac, 13.6× on the cloud vs the original Mac code):** EOS only for
  active + neighbours; exact per-particle neighbour-list reuse; Numba copies of GN v(p) and the
  gas+radiation T solve (`sim/fastphys.py`, tested). All tests pass.
- **Test status:** 181 fast tests pass (`uv run pytest -q`). Slow suite (21 tests, ~3 h on the Mac)
  last passed 2026-10-18 on the speed-up code; **re-run it** before restarting R6.1 (the heat-budget
  and h_max-skin changes came after; both verified bit-identical, but re-run anyway).

## 4. Waiting on Hiten (ask him / remind him)

**Next decision (to prepare 2026-10-21):** how to handle the few hundred over-heated particles
near the hole that drive R6.1 v2's gas loss (4.8% at t = 1.14M, rising) and ledger drift
(6 × 10⁻⁴ > the 10⁻⁴ criterion). Diagnosis in DECISIONS 2026-10-20/21 (drift is real, not a
step-size problem). Claude promised options with evidence: physically justified cooling of hot
thin gas, a principled heat limit, more particles; measure what each removes before recommending.
Hiten agreed to let R6.1 v2 run overnight through its second pass (~t = 1.19M).

## 5. Key decisions to remember (details in docs/DECISIONS.md)

D1 1k is a smoke test only; Phase 6 runs 10k (now 50k for R6.1). D2 velocity reconstruction in the
viscosity with a min-mod gate (our own choice, approved). D4 radius 25 for Phase 5, **60 for Phase 6+**.
D5 hole-resolution diagnostic. D6 swallowing r_acc = 4 with ledger. Snapshots follow Phase 10.2a
(clamp(T_orb(r_med)/40, 10, 10⁴)). Shock marker = viscous heating rate. Conductivity signal with the
hole = velocity (Phantom Eq. 44). β-cooling β = 230, T floor 10⁴ K. Lost-gas share (swallowed +
L < 4) reported per run, flagged above 1%; resolution check 10k vs 50k at radius 60 PASSED (peak
0.13% → 0.016%). Tail/front-of-cloud particle losses are a known, accepted numerical effect
(Hiten agreed). Cloud machine: Google Cloud free trial (Hiten's choice).

## 6. Google Cloud (how to operate it)

- Account: Hiten's Google account, **free trial**: $300 credit, 90 days from 2026-10-18 (ends
  ~2027-01-16 or when credit runs out); **cannot bill** unless he upgrades (tell him never to click
  "Upgrade" without deciding to pay). Trial caps **8 vCPUs** at once. Check credit: console →
  Billing. Rough spend so far: tens of dollars (check; c4d-highcpu-8 ≈ $0.30–0.40/h).
- CLI on the Mac: `~/google-cloud-sdk/bin/gcloud` (logged in as Hiten; project
  `project-f6231423-0846-4084-823`). VM **`nebula-sim`**, zone **`us-central1-b`**,
  c4d-highcpu-8 (AMD EPYC 9B45 Turin, 4 cores / 8 threads, 15 GB), Ubuntu 24.04, 60 GB disk,
  standard (not spot), live-migrates during maintenance, auto-restart on.
  **Currently RUNNING R6.1_50k_v2** (it powers itself off ~10 min after the run ends).
- On the VM: repo at `~/nebula` (updated by `git bundle`: `git bundle create /tmp/nebula.bundle
  HEAD main` on the Mac, `gcloud compute scp` it, `git pull -q ~/nebula.bundle main` there), uv at
  `~/.local/bin` (`export PATH=$HOME/.local/bin:$PATH`). Same lock as the Mac (Python 3.12,
  NumPy 2.3.5, Numba 0.62.1). 168 non-GPU tests pass there; GPU/render tests cannot (no GPU).
  Benchmark: 32,304 particle-steps/s (8 threads) vs 6,640 on the Mac.
- Helpers on the VM (run name set at the top of each, now R6.1_50k_v2): `~/resume_if_needed.sh`
  (resume after a reboot unless finished or errored) and `~/shutdown_when_done.sh` (waits 5 min, then
  powers off ~10 min after the run process ends). Both are in crontab @reboot and the watcher is running.
  To work on the VM while it is idle, first stop the watcher by PID (`ps -eo pid,args | grep "[s]hutdown_when"`).
- Launch a run so it survives SSH: `cd ~/nebula; setsid nohup uv run python scripts/cloud_run.py
  NAME ... > outputs/recon/NAME.out 2>&1 < /dev/null &`. Repo must be clean (manifest records the
  commit; remove stray files first).
- Copy results back: `scripts/sync_cloud_run.sh RUN` (rsync, incremental, skips the checkpoint;
  refreshes the SSH config because the VM's IP changes on each start).
- **Pitfalls that bit before:**
  - Never `pgrep -f`/`pkill -f` a pattern inside an `ssh --command` string: it matches its own
    shell and kills the SSH session. Use `ps -eo pid,args | grep "[x]pattern"` and kill by PID.
  - Background jobs started with `&` ignore SIGINT: stop runs with `kill -TERM <python pid>`
    (checkpoints are written atomically; then trim records after the checkpoint, which
    `cloud_run.py --resume` now does itself).
  - zsh does not word-split `$VAR`: write options out or use arrays.
  - Ask Hiten before leaving the VM running idle; stop it when no run is going.

## 7. Runs inventory (`runs/` is git-ignored: data lives on the Mac and/or the VM)

| run | where | what | state |
|---|---|---|---|
| cloud10k_r25_D2 | Mac | Phase 5, 10k, radius 25, ideal gas | to t = 408,670 (paused extension; 4 tail particles swallowed) |
| cloud50k_r25_D2 | Mac | Phase 5, 50k, radius 25 | to t = 400,000; 0 swallowed |
| R6.1_r60 | Mac | 10k radius 60, Phase 6 settings | paused t = 403,901 (superseded) |
| R6.1_r60_50k | Mac | 50k radius 60 first pass (resolution check) | to t = 404,962 |
| R6.1_50k | VM (snapshots to ~437k synced) | R6.1 first attempt, 50k, α_u = 1, no cap | paused t ≈ 446,000 (stalled; superseded) |
| **R6.1_50k_v2** | VM | **R6.1 main run**, 50k, α_u = 0.05, h_max 200 | running since 2026-10-19 00:27 UTC |
| heat10k_r60 | Mac | 10k radius 60 with per-particle heat budgets | to t = 404,000 |
| cond_x1..x4 | VM + synced | conductivity experiments (DECISIONS 2026-10-19) | x2–x4 to 420,000; x1 stopped ~403,000 |

## 8. Tools and scripts worth knowing

`scripts/cloud_run.py NAME n= radius= tend= eos= beta= r_escape= gravity= seed= cond= alpha_u=
h_max=` (new run) / `NAME --resume [tend=]`; `scripts/plot_cloud_run.py RUN` (6 panels + edge-on);
`scripts/cloud_video.py RUN [hold=]`; `scripts/cloud_metrics.py RUN`; `scripts/torus_diagnostics.py
RUN` (Phase 6 diagnostics, cached per snapshot); `scripts/loss_check.py LOW HIGH [t=]`;
`scripts/investigations/compare_cond_runs.py RUN...`; `scripts/sync_cloud_run.sh RUN`.
Pictures for Hiten go to `outputs/checks/`; keepers are copied to `docs/figures/` and logged in
`docs/results-log.md`. Snapshot fields include heat budgets (heat_visc/cond/pdv/cool).
Phantom source excerpts and PySPH magma2.py were downloaded to a scratch folder for reading only.

## 9. Next steps after the pending decision

1. Implement the approved conductivity/cap settings; full fast + slow suites; DECISIONS entry.
2. Restart R6.1 at 50k on the VM (re-arm resume + shutdown helpers); check after the first pass
   (~2–4 h: speed, lost gas, halo, heating vs the old R6.1 numbers) and report to Hiten.
3. While it runs: Phase 6 analysis prep is done (sync, cached diagnostics, video). Remaining Phase 6
   items: R6.2–R6.5 (cooling β scan, Newtonian control R6.4, seeds) — sizes/order to agree with
   Hiten after R6.1's cost is known; the numerical-circularization check (6.7); results-log entries
   answering phase-06 §6.8; pictures reviewed by Hiten.
4. Then propose the Phase 7 breakdown (CLAUDE.md workflow 5) and wait for approval.

## 10. Hiten's broader questions already answered (so you stay consistent)

- The simulation is fully 3D; the disk forms in the original orbital plane (non-spinning hole) but
  is a thick torus; tilted/warped behaviour comes in Phase 11 (spin, a cut-list item).
- The final video is glowing, lensed gas (Phases 8–9), not dots; the dot plots are quick looks.
  The cloud starts as a smooth uniform ball (G2-like), stretched into a stream; a clumpier
  "nebula-looking" start would be a scenario change for him to decide.
- Overall estimate given: core video (Phases 6–10) ~5–8 weeks, everything ~2–3 months; faster with
  more paid compute (~$50–150 for 16–32 cores).
- Claude usage: simulations on the cloud don't use Claude usage; only check-ins do.

## 11. Session log (append one line per meaningful event; newest last)

- 2026-10-14–16: Phases 0–5; plunge review and decisions D1–D11; D2 implemented; 10k/50k radius-25 runs.
- 2026-10-16 night: Phase 6 steps 6.1–6.7, tests 1–9 pass; slow suite passes.
- 2026-10-16: Hiten chose radius 60, end time 3.8 × 10⁶; R6.1 (10k) started on the Mac, then paused
  for the 50k resolution check (10k results mostly numerical).
- 2026-10-17: 50k radius-60 first pass done on the Mac; lost-gas check PASS.
- 2026-10-18: speed work (2.8×); Google Cloud trial set up (VM nebula-sim, 13.6× vs old Mac code);
  R6.1 50k started on the VM; heat-budget instrumentation added.
- 2026-10-19: R6.1 stalled after the pass (halo + limiter), paused at t ≈ 446,000; h_max skin bug
  fixed; heat budget shows conductivity pumps the excess heat; experiments cond_x1–x4 done; awaiting
  Hiten's decision (§4). VM stopped. Hiten switching Claude accounts; this file written.
- 2026-10-19: continued on Hiten's second Claude account (same conversation carried over); nothing changed since the handover; recommendation (§4) re-stated, awaiting his answer.
- 2026-10-19: Hiten approved α_u 0.05 + h_max 200 + R6.1 restart. Implemented (commit 3d51dd9; 184 fast tests pass; MAGMA2's 0.05 confirmed in the paper text). Slow suite running on the VM before launching R6.1_50k_v2.
- 2026-10-19: full suite passes (184 fast; 19 slow sim tests on the VM + 2 GPU slow tests on the Mac). R6.1_50k_v2 launched on the VM 00:27 UTC with helpers armed. Next: check after the first pass (~2–4 h) against the old R6.1 numbers.
- 2026-10-19: gravity-only ceiling for radius 60 computed (method validated at radius 25): typical ~1e-7, hottest 1% ~5e-6. R6.1 v2 typical gas below it on approach, ~1.5–3.5× above at closest approach (first attempt 4–10×). Results log updated. Next: R6.1 v2 post-pass check (unbound gas, halo, lost gas) vs the first attempt.
- 2026-10-20/21: R6.1 v2 at t = 1.14M: 2.8% swallowed (4.8% lost), ledger drift 6e-4. Exact-ledger and half-step tests: drift real, not step size; cause = over-heated particles near the hole. Running overnight through the 2nd pass; options to prepare next.
