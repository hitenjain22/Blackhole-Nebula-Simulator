# How to use this plan (for Hiten)

This folder is the complete plan for the project: a physically accurate, cinematic simulation of a gas cloud being torn apart and swallowed by a black hole. Claude Code builds it; you steer, check and decide.

## What's in here

| File | What it is |
|---|---|
| `CLAUDE.md` | The rules Claude Code follows in every session. It reads this automatically. |
| `PLAN.md` | The master plan: goals, decisions already made, architecture, conventions, the list of phases, and project-wide risks. |
| `phases/phase-00-setup.md` … `phase-02-…` | **Detailed, step-by-step** plans for the first three phases. Ready to execute. |
| `phases/phase-03-…` … `phase-13-…` | **In-depth plans** for the rest: goals, design, tests with known answers, what will go wrong and how to fix it, and what "done" means. Review each one with Claude before starting it, as you planned. |
| `reference/expected-values.md` | Every number the tests check against, and how each was computed. |
| `reference/references.md` | Papers and codes, with what each one is used for, marked as checked online or from memory. |
| `reference/scripts/` | The 21 small scripts that computed the expected values, so anyone can re-check them. |

## How to start

1. Create a new empty GitHub repository (for example `nebula-blackhole`) and clone it to your Mac:
   ```
   git clone https://github.com/<your-username>/nebula-blackhole.git
   ```
2. Copy everything in this folder into the repository folder (keep the `phases/` and `reference/` subfolders).
3. Open the folder in VS Code, open Claude Code, and say:
   > Read CLAUDE.md, PLAN.md and phases/phase-00-setup.md. Then do Step 0.1 only, and stop when it's done.
4. After each step, Claude Code tells you what to look at (a test result, a plot, an image). Check it, then say "looks good, commit and do the next step", or describe what's wrong.

## Before each later phase

Phases 3–13 were written ahead of time. Before starting one, ask Claude Code:
> Read phases/phase-NN-….md. Check it against what we've actually built so far. List anything that no longer fits, anything missing, and any risks. Then propose the step-by-step breakdown for this phase, and wait for my OK.

That is where you "run it through Claude again", as you said. The plan is designed to be revised; what must never change without a written reason are the rules in `CLAUDE.md` and the expected values in the tests.

## What your job is

- **Decide.** When Claude Code asks a question, it's because the answer changes the result. Ask it to explain options in plain words if needed.
- **Look.** Every step ends with something to see. If a plot looks wrong to you, say so: you don't need to know why.
- **Understand the physics.** You'll present this. Ask Claude Code "explain what this step does physically" whenever it's unclear.
- **Keep the AI log.** `AI_LOG.md` (created in Phase 0) records what Claude got wrong and how tests caught it. This is your best evidence of using AI well.

## A correction to the earlier notes

The earlier guide `guide/nebula-part2-setup-colour-spin.md` (in the project files, not in this folder) suggested a cloud "about the mass of a few Earths". When I worked through the radiation physics for this plan, that turned out to be inconsistent with the glowing blue-white torus we want: a cloud that light would feed the hole far too slowly, and the gas would stay too thin to glow like a blackbody. **The plan uses 0.1 solar masses instead** (details in PLAN.md §3). Everything else in the earlier notes still holds.

## Things the planning calculations found

- **The crash comes late.** The stretched cloud goes round the hole (about 194 days per orbit), and only on its second or third pass is its stream long enough for the outgoing gas to run into the incoming gas, about 70 AU (1,664 M) from the hole, almost head-on at 16,800 km/s. The simulation will tell us exactly when.
- **The torus drains fast.** It's thick and hot, so it empties in days to weeks, not years. The "stage 6 (the drain) takes way longer" behaviour from your earlier notes belongs to thin, cool disks; the plan shows it with the 1D disk model (Phase 7).
- **True colour is blue-white, not orange.** Gas this hot (above 10,000 K) looks blue-white to the eye. The orange look is available as a labelled cinematic grade (Phase 9).
- **The hot torus is only partly opaque.** It blocks light well by scattering, but true absorption is weak at a million degrees, so treating gas and light as sharing one temperature is an approximation there. Phase 6 measures it and the report says so.
- **Spin is partly captured.** The renderer shows the true spinning-hole shadow; the gas feels frame dragging accurately (about 1%), but the inner edge of the disk stays where a non-spinning hole puts it. Phase 11 explains the trade-off.
