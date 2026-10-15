# Phase 10: Production pipeline: unattended overnight runs (in-depth)

**Goal:** make long simulations and renders run unattended overnight on the MacBook Air and survive sleep, crashes, a full disk and Ctrl-C; resume exactly where they stopped; assemble videos with informative overlays; produce a morning report that Claude Code can check. A crashed overnight run costs a whole day, so this phase is about reliability, not features.

**Ends with:** a `nebula` command-line tool (run, resim, render, video, report, status); a kill-and-resume test that reproduces an uninterrupted run bit for bit; a full overnight run → overnight render → video done without touching the machine; a one-page overnight checklist for Hiten.

---

## 10.1 The `nebula` command

Installed as a console script in `pyproject.toml`; Hiten runs it as `uv run nebula …`.

| Command | Does |
|---|---|
| `nebula run configs/runs/<run>.toml [--resume runs/<id>]` | start or resume a simulation |
| `nebula resim --from runs/<id>/checkpoints/<ckpt> --times <file>` | re-run a segment writing snapshots at exact frame times (Phase 8 hero shots) |
| `nebula render configs/shots/<shot>.toml [--preview] [--resume]` | render frames; `--preview` = 360p, 1 sub-ray, every 4th frame |
| `nebula video configs/shots/<shot>.toml` | frames → MP4 with overlays |
| `nebula report runs/<id>` | morning report (10.8) |
| `nebula status` | what's running, progress, ETA |
| `nebula estimate configs/runs/<run>.toml` or `configs/shots/<shot>.toml` | predicted wall time and disk use before starting |

Every command prints what it's about to do, the estimate, and asks no questions once started (unattended).

---

## 10.2 Run folders and manifests

```
runs/<YYYYMMDD-HHMM>-<name>/
  manifest.json        resolved config, seed, git commit + dirty flag (and the diff saved as patch),
                       uv.lock hash, Python/NumPy/Numba/wgpu versions, machine info, CLI args, start time
  log.txt              progress lines (10.5) and warnings
  checkpoints/         float64 full state, atomic writes (10.3)
  snapshots/           float32 render fields + float64 time, one .npz each, index.json (schedule: 10.2a)
  diagnostics.csv      ledgers, Ṁ, L, timings
  report/              morning report output
```
Rule (CLAUDE.md): every result in the report traces back to a run folder with a manifest.

### 10.2a Snapshot schedule

Snapshots are what the renderer interpolates between (Phase 8.2), so the interval must be short compared with an orbit of the gas in view, but a fixed short interval over a ~2-year run would fill the disk.

- Default rule: **Δt_snap = clamp(T_orb(r_med) / 40, 10, 10⁴)**, where T_orb(r) = 2π r^{3/2} and r_med is the median radius of the gas inside r = 2,000 (all gas if none is inside yet). Far out the cap of 10⁴ (≈ 2.4 days) applies; around the torus at r ≈ 50 it is ≈ 55 (≈ 20 minutes); during a pericentre passage at r ≈ 25 it is ≈ 20, giving several snapshots across the ~100-long swing (the floor of 10 only guards against runaway counts).
- Plan-time estimate: **~4,000–5,000 snapshots per production run** (1–3 GB at 10k particles, 6–14 GB at 50k). `nebula estimate` must print the real count and size before the run starts.
- At 50k particles, prefer coarser stored snapshots (e.g. ×4 the interval) and `nebula resim` windows for the shots that are actually rendered (Phase 8.2): deterministic re-runs write exactly the frames' times, so nothing is lost.
- Extra snapshots at events (each pericentre passage, first crash, t_circ) are written regardless of the rule.

---

## 10.3 Checkpoints

- Every **15 minutes of wall time**, plus at events (end of each pericentre passage, t_circ).
- Content: everything needed to continue identically: particle state (float64), integer timeline counter and every particle's level and last full state (Phase 4 prediction needs them), α_AV values, ledger accumulators, RNG state, snapshot schedule position, the manifest hash.
- **Atomic write:** write `ckpt.tmp`, flush and fsync, then rename. A crash mid-write leaves the previous checkpoint intact.
- Keep the last 3 rolling checkpoints plus event checkpoints.
- **Resume is bit-identical** (test 10.9-1). If anything is recomputed on resume (neighbour lists, densities), it must be recomputed identically: the code must not depend on state that isn't checkpointed (e.g., tree build order).

---

## 10.4 Signals, sleep and power

- SIGINT (Ctrl-C) and SIGTERM: write a checkpoint, flush logs, exit with a clear message. A second Ctrl-C exits immediately.
- **Mac sleep:** wrap long jobs in `caffeinate -i -s` (prevents idle and system sleep while on AC power) via `scripts/overnight.sh`. The display may sleep.
- **The lid must stay open** (closing it sleeps a MacBook Air regardless) and **the charger plugged in**. This goes on Hiten's checklist.
- Sim and render are never run at the same time (they compete for CPU, memory and heat).

---

## 10.5 Progress, ETA and logs

One line per minute in `log.txt` and the terminal:
```
[02:14:07] t = 1.234e6 (41.1%)  wall 3h12m  9.8e4 upd/s  levels 3–19  ΔE/E 2.1e-6  ΔL/L 3e-12  ETA 4h35m
```
ETA from the recent rate, not the run average (rates change a lot between stages). Warnings (h-iteration failures, EOS fallback used, NaN count, ledger drift above threshold) are logged with counts, never silently.

---

## 10.6 Throughput, heat and the real budget

- The MacBook Air throttles under sustained load. Measure **sustained** throughput: run the Phase 3 benchmark for 60 minutes and record the rate after 10, 30, 60 minutes. Plan with the 60-minute number.
- `nebula estimate` uses measured rates (stored in `docs/DECISIONS.md` or a `machine_profile.json`) for both sim (particle-updates/s by stage) and render (s/frame by resolution and sub-rays).
- Refuse to start (with a clear message) when the estimate exceeds the time window given (`--hours 9`), unless `--force`.

---

## 10.7 Disk budget

| Item | Size (estimate; measure) |
|---|---|
| Snapshot, float32, ~14 fields | ~56 bytes/particle: 0.56 MB (10k), 2.8 MB (50k); ~4,000–5,000 snapshots per production run (10.2a) → 1–3 GB (10k), 6–14 GB (50k; use `resim` windows) |
| Checkpoint, float64 + integrator state | ~200 bytes/particle: 10 MB at 50k; 3 rolling + events |
| Frame, 1080p PNG | 3–5 MB; a 2-minute video at 30 fps (3,600 frames) ≈ 15 GB |
| Frame, 4K PNG | ~4× that |
| Video, 1080p H.264 CRF 18 | 100–400 MB for 2 minutes |

- `nebula estimate` checks free space and refuses to start if the job would leave < 10 GB free.
- After a video is encoded and checked (10.9), frames can be deleted or moved to an external drive (a command option, never automatic).
- Hiten's overnight checklist includes "≥ 30 GB free".

---

## 10.8 Video assembly and overlays

- ffmpeg from `imageio-ffmpeg` (verified in Phase 0, including that its binary is present without internet): H.264 (`libx264`), `-pix_fmt yuv420p`, CRF 18, preset slow, 30 fps, even width and height, colour tags `-colorspace bt709 -color_primaries bt709 -color_trc bt709` so players show the right colours.
- **Overlays drawn with Pillow** onto frames before encoding (portable; ffmpeg's drawtext depends on system fonts). Bundle an open-licence font (e.g., DejaVu Sans or Inter; record its licence).
  - simulation time since the start in days/hours (from the frame's t_sim via units.py);
  - **stage name**, computed from diagnostics by explicit rules (e.g., "crash" while mass in shocks > threshold; "drain" once t > t_circ), not typed by hand;
  - scale bar in M and AU;
  - colour-mode label (true colour / band / cinematic grade);
  - optional inset: light curve with a moving marker.
- Shot assembly: concatenate shots with short cross-fades; title and end cards; a 10–20 s loop as GIF/WebM for the README.

---

## 10.9 Tests

| # | Test | Requirement |
|---|---|---|
| 1 | **Kill and resume** | a 1k run killed at 3 random moments (SIGKILL, not just SIGINT) and resumed each time ends **bit-identical** to the uninterrupted run |
| 2 | Atomic checkpoint | simulate a crash during checkpoint write: the previous checkpoint loads; the partial file is ignored |
| 3 | SIGINT | Ctrl-C produces a checkpoint and a clean exit within 30 s |
| 4 | Estimator | predicted vs actual wall time for a 20-minute job within 30% |
| 5 | Disk guard | with a fake small disk quota, the job refuses to start |
| 6 | Render resume | kill a render at frame 10 of 20, resume: frames 1–9 untouched (checksums), 10–20 identical to an uninterrupted render |
| 7 | Video check | `ffprobe` reports the expected frame count, fps, codec, pixel format and colour tags |
| 8 | Overlay timing | frame k's printed time equals t_sim(k) from the shot's time warp |
| 9 | Manifest | every run folder has a manifest with a commit hash; a dirty tree's diff is saved |

---

## 10.10 Hiten's overnight checklist (plain words; put it in `docs/overnight.md`)

**Evening**
1. Plug in the charger. Keep the lid open. Close other apps.
2. In VS Code's terminal: `uv run nebula estimate …` (Claude gives the exact command). Check it says it will finish in time and there's enough disk space.
3. Start it with the command Claude gives (it uses `scripts/overnight.sh`, which keeps the Mac awake).
4. Wait until two progress lines appear. Then leave it.

**Morning**
1. Run `uv run nebula report runs/<id>`.
2. Open the report, look at the pictures, and paste the summary text into Claude Code with: "Check last night's run."
3. If the run stopped early, don't start it again from scratch: ask Claude to resume it.

---

## 10.11 The morning report (`nebula report`)

Writes `runs/<id>/report/report.md` + PNGs: progress and stop reason; ledger plots (should be flat); Ṁ(t), L(t); level histogram over time; wall-time breakdown by stage; all warnings with counts; a contact sheet of 12 snapshots (particle preview). Ends with a **checklist of automatic pass/fail checks** (ledger drift below thresholds, no NaNs, no unexpected floor hits, finished or resumable) so Claude Code can verify quickly.

---

## 10.12 What will go wrong, and the fix

| Problem | Symptom | Fix |
|---|---|---|
| Mac slept | run paused for hours | caffeinate wrapper; lid open; charger in; log timestamps reveal gaps |
| Disk filled mid-run | crash, corrupt files | estimator; graceful stop with checkpoint when free space < 5 GB |
| Corrupt checkpoint | resume fails | atomic writes; keep 3; test 2 |
| Library update changed results | rerun differs | uv.lock pinned; manifest records versions; never `uv lock --upgrade` mid-campaign |
| Stale Numba cache after code change | strange errors or old behaviour | clear `__pycache__`/Numba cache when in doubt; the manifest records the commit |
| A bug discovered at frame 2,000 | wasted night | always render a `--preview` of the whole shot first (cheap), and look at it |
| GPU driver reset mid-render | black or missing frames | catch the error, retry the tile once, log; frames are written only when complete |
| Frame order wrong in the video | jumps | zero-padded names (`frame_000123.png`); test 7 counts frames |
| Wrong colours in some players | washed-out video | bt709 tags; test 7 |
| Thermal throttling | slower than the estimate | plan with the 60-minute sustained rate; raise the ETA margin |

---

## Done when

- [ ] Tests 1–9 pass.
- [ ] One complete overnight cycle done unattended: run (night 1) → render (night 2) → video, with the morning reports reviewed.
- [ ] `docs/overnight.md` written and walked through with Hiten; committed.
