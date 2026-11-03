"""Energy audit between two snapshots of a run (diagnosis of the R6.1 ledger drift, rule 9).

The run's ledger E = sum m (H + u) + removed + radiated is computed from the same state the
snapshots store, so recomputing it from snapshots would only repeat it. The audit instead splits
the change of E into two channels that can each be checked on their own, using the per-particle
heat budgets the block integrator books at every kick (sim/blocks.py ``_add_heat``):

- heat channel: r_heat = dU - (dQ - dC), the change of thermal energy not explained by the
  booked heating dQ (viscosity + conduction + disk viscosity + p dV) and cooling dC. Every
  change of u in the integrator is booked, so this is ~0 up to the part of a step not yet
  kicked at the snapshot time (bounded, not cumulative).
- motion channel: r_motion = dH + dQ, orbital energy H (GN or Newtonian Hamiltonian,
  physics.gravity) lost to the gas forces minus the heat those forces booked. Exact SPH moves
  energy between the two without loss, so this is ~0 in a consistent scheme; a growing value
  means the forces and the heating no longer match.

- removal channel: r_removed, for particles present in a but removed (swallowed or escaped)
  before b: the energy m (H + u) the integrator booked at removal minus their energy at a. Up
  to removal they still exchange energy with their neighbours through the gas forces, so this
  is not zero by itself; it closes the books with the other two.

r_heat + r_motion + r_removed is the ledger change from a to b, apart from what the removed
particles radiated before removal (not stored; zero without cooling). All sums are
mass-weighted, code units.
"""

from collections.abc import Callable, Mapping

import numpy as np

HEATING = ("heat_visc", "heat_cond", "heat_pdv", "heat_disk")


def _part(s: Mapping, key: str, rows) -> np.ndarray:
    """One cumulative heat-budget part per particle; a part a snapshot does not store counts 0
    (older runs have no heat_disk, and a run's first snapshot, at t = 0, has no budgets)."""
    return np.asarray(s[key], float)[rows] if key in s else np.zeros(len(rows))


def _booked_heating(s: Mapping, rows) -> np.ndarray:
    """Cumulative booked heating per particle."""
    return sum(_part(s, k, rows) for k in HEATING)


def energy_channels(
    a: Mapping,
    b: Mapping,
    mass: float | np.ndarray,
    hamiltonian: Callable,
    removed: Mapping[int, float] | None = None,
) -> dict:
    """Energy changes from snapshot a to snapshot b of the particles present in both.

    a, b: snapshot fields (ids, x, p, u, heat_visc, heat_cond, heat_pdv[, heat_disk],
    heat_cool). mass: one particle mass, or an array indexed by particle id.
    hamiltonian(x, p): specific orbital energy, e.g. physics.gravity.gn_hamiltonian.
    removed: specific energy H + u booked at removal, by particle id (the run's removed.json);
    needed when particles of a are missing from b.
    Returns the mass-weighted dH, dU, dQ (booked heating), dC (booked cooling), r_heat,
    r_motion, r_removed and the numbers of particles compared and removed."""
    ids_a, ids_b = np.asarray(a["ids"]), np.asarray(b["ids"])
    common, ia, ib = np.intersect1d(ids_a, ids_b, assume_unique=True, return_indices=True)
    m = np.asarray(mass, float)
    m = m[common] if m.ndim else np.full(len(common), float(m))

    def h(s, rows):
        return hamiltonian(np.asarray(s["x"], float)[rows], np.asarray(s["p"], float)[rows])

    d_h = float(np.sum(m * (h(b, ib) - h(a, ia))))
    d_u = float(np.sum(m * (np.asarray(b["u"], float)[ib] - np.asarray(a["u"], float)[ia])))
    d_q = float(np.sum(m * (_booked_heating(b, ib) - _booked_heating(a, ia))))
    d_c = float(np.sum(m * (_part(b, "heat_cool", ib) - _part(a, "heat_cool", ia))))
    gone = np.flatnonzero(~np.isin(ids_a, ids_b))
    r_removed = 0.0
    if len(gone):
        if removed is None or any(int(k) not in removed for k in ids_a[gone]):
            raise ValueError("particles left between the snapshots without a removal record")
        m_gone = np.asarray(mass, float)
        m_gone = m_gone[ids_a[gone]] if m_gone.ndim else np.full(len(gone), float(m_gone))
        e_rem = np.array([removed[int(k)] for k in ids_a[gone]])
        e_a = h(a, gone) + np.asarray(a["u"], float)[gone]
        r_removed = float(np.sum(m_gone * (e_rem - e_a)))
    return {
        "dH": d_h, "dU": d_u, "dQ": d_q, "dC": d_c,
        "r_heat": d_u - (d_q - d_c), "r_motion": d_h + d_q, "r_removed": r_removed,
        "n_common": len(common), "n_removed": len(gone),
    }  # fmt: skip
