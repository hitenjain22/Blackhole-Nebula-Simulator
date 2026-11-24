# Nebula plan (Phase 14 source material)

Exported from Hiten's planning doc "Building a giant nebula for the black hole video" on 2026-11-23.
Hiten approved everything here on 2026-11-23, including the five decisions at the top of the build spec.

Read in this order:

1. [00-plan.md](00-plan.md): the plan, the techniques and the timeline
2. [02-build-spec.md](02-build-spec.md): steps N0 to N9 with settings, tests and dates (the source for `phases/phase-14-nebula.md`)
3. [01-realism-check.md](01-realism-check.md): what is real physics and what is a labelled cheat
4. [03-free-compute.md](03-free-compute.md): free compute, with terms checked
5. [04-open-source-tools.md](04-open-source-tools.md), [05-reading-list.md](05-reading-list.md), [06-look-references.md](06-look-references.md): tools, papers, visual references
6. [07-mapelli-trani-setup.md](07-mapelli-trani-setup.md): the exact setup of the Act 1 model paper, and the differences to decide in N2 (added 2026-11-23)

## How to start (paste into Claude Code)

> Read CLAUDE.md, PLAN.md, docs/HANDOFF.md and everything in docs/nebula-plan/, starting with 00-plan.md and 02-build-spec.md. This is Phase 14: the nebula. Hiten has approved all five decisions in the build spec, including self-gravity off in Act 1. Do step N0 only: write phases/phase-14-nebula.md from these docs, add the DECISIONS entries listed in N0, add a LICENSE file (the repo includes a Phantom GPL patch), and update HANDOFF §3 and §4, which are out of date. Then stop and show me the phase file before starting N1.

Numbers marked "my estimate" or *start* are planning guesses, not test values. Expected values for tests must still be computed by scripts in `reference/scripts/` (CLAUDE.md rule 3).
