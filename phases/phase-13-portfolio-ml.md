# Phase 13: AI/ML extensions and the portfolio (in-depth)

**Goal:** package the project for job applications in AI: show the physics, the engineering, and **how AI was used responsibly** (plans, independent checks, tests catching AI mistakes). Add two small, honest machine-learning pieces that use the simulator's own data: a differentiable fit of the disk viscosity, and a fast emulator of the simulation's outcomes.

**Ends with:** a portfolio-ready repository (README with video, validation gallery, report, AI-workflow write-up), a fitted α with uncertainty, an emulator with held-out validation, and the videos hosted where they play well.

**Cut-list note:** the ML pieces (13.1–13.2) are the first thing to cut if time runs short (PLAN §12). The AI-workflow write-up (13.3) and the portfolio (13.4) are never cut.

---

## 13.1 Differentiable 1D disk: fit α by gradient descent

**Idea:** rewrite one Crank–Nicolson step of the Phase 7 1D model in an automatic-differentiation framework, run it forward, compare its accretion rate Ṁ(t) with the SPH run's Ṁ(t), and let gradients find the α (and H/r profile scale) that fits best. It's the same machinery used to train neural networks, applied to a physics model.

- **Framework check first (Intel Mac):** PyTorch publishes no macOS Intel wheels after version 2.2.2 (confirmed: PyTorch's January 2024 deprecation notice), so current PyTorch won't install natively. JAX's Intel-Mac status wasn't confirmed while planning: test it. Options in order: (1) whichever framework installs and passes a tiny gradient test via uv on this Mac; (2) Google Colab (free, runs in the browser; the 1D model is small); (3) hand-written adjoint (the CN step is linear in Σ, so its gradient is another linear solve: elegant and dependency-free, and a good showcase). Record the choice in DECISIONS.md.
- **Implementation:** the differentiable model must match `disk1d.py` to 10⁻¹⁰ in the forward direction (test), and its gradients must match finite differences to 10⁻⁵ (test).
- **Fit:** loss = mean squared difference of log Ṁ over the drain phase (log because Ṁ spans orders of magnitude), with time bins matching the SPH binning. Optimizer: Adam or L-BFGS.
- **Validation:** the gradient fit must agree with a brute-force grid search over α (scipy, 200 values) within the grid spacing. Fit synthetic data generated from the 1D model with a known α (+ noise): recover it within the noise-limited error.
- **Uncertainty:** bootstrap over time bins and over SPH seeds (Phase 12) → α ± σ.
- **What it means:** the fitted α_eff is the "effective viscosity" of the SPH run, physical + numerical. Compare with the input α and the measured α_num (Phase 7). If α_eff ≫ α, the run is numerically dominated: an important honest result.

---

## 13.2 Emulator: predicting outcomes without running the simulator

**Idea:** train a model on many cheap 1k-particle runs that predicts the headline quantities (t_crash, t_circ, Ṁ_peak, t_half, L_peak) from the inputs (β, α, r_p, R). Then explore parameter space instantly and show where the emulator is and isn't trustworthy.

- **Training data:** a design over (log β, log α, r_p, R) with a Latin hypercube (40–60 runs at 1k; ~10–30 min each → 2–3 nights). Store inputs and the `metrics.py` outputs in one CSV with run ids.
- **Model:** a Gaussian process (scikit-learn) first: small data, gives uncertainty for free. Optionally a small neural network for comparison (if a framework installs; otherwise skip).
- **Validation (non-negotiable):** hold out ~20% of runs **by parameter point** (never split one run's time samples between train and test). Report error on held-out runs vs the GP's own uncertainty (calibration plot). Add 5 new runs at points the emulator predicts confidently and check.
- **Honesty:** 1k-particle results aren't converged for some quantities (Phase 12). The emulator emulates the 1k simulator, not nature. Say exactly that.
- **Optional demo:** a small web page (static HTML/JS) with sliders showing the GP prediction and its uncertainty band.

---

## 13.3 How this was built with AI (the centrepiece for AI-job applications)

Source material: `AI_LOG.md` (kept since Phase 0: per step, what was asked, what Claude Code produced, what the tests found, what was fixed), `docs/DECISIONS.md`, the plan files, and git history.

Write `docs/ai-workflow.md`:
1. **The workflow:** plan → expected values computed independently → tests written first → implementation by Claude Code → tests → review → commit. One step at a time, a human checking each.
2. **Guardrails that mattered:** the CLAUDE.md rules (physics in one place, fast-code-matches-reference, never loosen tolerances, ledgers, reproducible manifests) and why each exists.
3. **Case studies (3–5):** real moments where a test caught an AI mistake (e.g., a sign error found by the Euler–Lagrange finite-difference test, a race condition found by the thread-determinism test, a Doppler sign found by the bright-side test). With the failing output, the fix and the commit.
4. **What AI was good and bad at** in this project, with evidence.
5. **Numbers:** lines of code, tests, test values computed independently, runs, compute hours, number of AI-introduced bugs caught by tests (from the log).

This is what distinguishes the project from "I made a pretty video with AI": evidence of judgement and verification.

---

## 13.4 The portfolio repository

**README.md** (first screen matters most):
1. Hero: a 10–20 s looping GIF/WebM of the torus with lensing, plus a link to the full video.
2. One paragraph: "A gas cloud like G2 falls into a black hole like Sgr A*. Relativistic gravity, shock heating, cooling and viscosity form and drain a glowing torus; a custom GPU ray tracer renders it with gravitational lensing, Doppler beaming and true blackbody colour."
3. The seven stages, one clip or still each, one line each.
4. "What's physically right, what's approximated" (short, honest, linking to the report).
5. Validation gallery: 6 small figures (orbit, Sod, ledger, ring spreading, shadow test, redshift test).
6. Results highlights (3–5 numbers with caveats).
7. How it was built (link to `docs/ai-workflow.md`) and architecture diagram.
8. Quick start (uv install, run a 1k preview, render a frame).
9. References and credits (CIE data, font, star data if any, papers).

**Also:** `LICENSE` (code: MIT or BSD-3; images/videos: CC BY 4.0, Hiten to choose), `CITATION.cff`, a short `docs/` site via GitHub Pages (optional).

**Videos:** GitHub blocks files over 100 MB in the repository. Put final videos in **GitHub Releases** (files up to 2 GB) and/or YouTube/Vimeo; embed a compressed preview in the README. Never commit `runs/`, `frames/`, `videos/` (they're in `.gitignore` since Phase 0).

**For the CV/interviews:** a one-page summary (problem, methods, verification, results, AI workflow) and a 2-minute talk track. Claude can draft both from the report.

---

## 13.5 What will go wrong, and the fix

| Problem | Fix |
|---|---|
| ML framework won't install on the Intel Mac | Colab or the hand-written adjoint (13.1); GP needs only scikit-learn |
| Overselling the ML | state what the emulator emulates (1k runs), show its errors, keep it secondary to the physics |
| Data leakage | split by parameter point; never by time sample |
| Fitted α absorbs numerical viscosity | compare with α_num from Phase 7; report both |
| Licences | record licences for CIE data, fonts, any star catalogue or map before publishing |
| Repo too big | large outputs in Releases or external hosting; `.gitignore` check before every push |
| Claims in the README drift from the report | README numbers copied from the report's tables, with links |
| "Nebula" wording | the object is a 0.1 M☉ gas cloud (like G2, but on a closer orbit); say "gas cloud" in technical text and explain the word choice once |

---

## Done when

- [ ] 13.1 and 13.2 done (or explicitly cut), with their validation results.
- [ ] `docs/ai-workflow.md` written with real case studies from AI_LOG.md.
- [ ] README, LICENSE, CITATION.cff done; videos hosted and embedded.
- [ ] Hiten and his mentor have read the README and report; final commit and a tagged release (v1.0).
