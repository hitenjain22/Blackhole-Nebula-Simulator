"""Phase 3-4 tests with the reconstruction prototype (review 2026-10-15). Usage: suite_recon.py blob|gresho [plain]"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
which = sys.argv[1]; plain = len(sys.argv) > 2 and sys.argv[2] == 'plain'
if not plain:
    import recon; recon.install()
tag = 'plain' if plain else 'recon'
w = time.perf_counter()
if which == 'blob':
    from nebula.sim.blob import hot_blob, U_BLOB
    for blocks_ in (False, True):
        fine = hot_blob(0.00125, 0.15, block_steps=blocks_); coarse = hot_blob(0.0025, 0.3, block_steps=blocks_)
        print(f"{tag} hot blob {'block' if blocks_ else 'shared'} steps: max dE fine {fine['max_energy_error']:.2e} (limit 1e-5), coarse {coarse['max_energy_error']:.2e}, ratio {coarse['max_energy_error']/fine['max_energy_error']:.2f} (3-5); "
              f"dL fine {fine['max_angmom_error']:.1e} coarse {coarse['max_angmom_error']:.1e}; u_max/U0 {fine['u_max']/U_BLOB:.0f}; lowest L {fine['min_particle_angmom']:.3f}; wall {time.perf_counter()-w:.0f}s", flush=True)
elif which == 'gresho':
    from nebula.sim.testproblems import gresho_l1, gresho_setup, run_problem, sound_setup, sound_measure, run_sound
    for sw in (True, False):
        l1 = gresho_l1(run_problem(gresho_setup('low', switch=sw)))[0]
        print(f"{tag} Gresho low switch={sw}: L1 {l1:.4f}  wall {time.perf_counter()-w:.0f}s", flush=True)
    prob = sound_setup('low'); m = sound_measure(prob, *run_sound(prob))
    print(f"{tag} sound wave 32/wavelength: phase error {m.phase_cycles:.4f} wavelengths (limit 0.01)  wall {time.perf_counter()-w:.0f}s", flush=True)
