"""Swallowing by the hole (phases/phase-05-cloud-infall.md 5.1: r_acc = 4; DECISIONS D6).

A particle is removed when, at the end of one of its own steps (its state is then true, not
predicted), it is inside r_acc. Its mass, energy m (H + u) and angular momentum m (x cross p)
go into the ledger of removed particles, which ``blocks.ledger`` adds to the totals of the
particles still present, so the conservation ledgers stay exact (CLAUDE.md rule 9; phase-04
§4.3 "ledger must record H, u, m and angular momentum of each removed particle at removal").

The plan's alarm (phase-05 §5.3): a particle swallowed on its *first* approach (it never
passed a pericentre) means something is wrong; such removals are flagged in the log, with the
particle's starting angular momentum and its state at removal.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RemovedLedger:
    """Totals and a log of particles removed from the simulation."""

    mass: float = 0.0
    energy: float = 0.0  # sum m (H + u) at removal
    angmom: np.ndarray = field(default_factory=lambda: np.zeros(3))  # sum m (x cross p)
    momentum: np.ndarray = field(default_factory=lambda: np.zeros(3))  # sum m p
    log: list = field(default_factory=list)  # one dict per removed particle

    @property
    def count(self) -> int:
        return len(self.log)

    def totals(self, reason: str) -> dict:
        """Mass, energy m (H + u) and count of the particles removed for ``reason``
        ("accreted" or "escaped"), from the log."""
        sel = [e for e in self.log if e["reason"] == reason]
        return {"count": len(sel), "mass": sum(e["m"] for e in sel),
                "energy": sum(e["m"] * (e["H"] + e["u"]) for e in sel)}  # fmt: skip

    def first_pass(self) -> list:
        """Removals of particles that had not yet passed a pericentre (the plan's alarm)."""
        return [e for e in self.log if e["first_pass"]]


def record_removal(
    ledger: RemovedLedger, t: float, ids, mass, hamiltonian, u, angmom, p, r, passed, L0, reason
) -> None:
    """Add the removed particles (arrays over them) to the ledger and its log."""
    for k in range(len(ids)):
        ledger.momentum = ledger.momentum + mass[k] * np.asarray(p[k], float)
        ledger.mass += float(mass[k])
        ledger.energy += float(mass[k] * (hamiltonian[k] + u[k]))
        ledger.angmom = ledger.angmom + mass[k] * np.asarray(angmom[k], float)
        ledger.log.append(
            {
                "t": float(t), "id": int(ids[k]), "reason": reason, "r": float(r[k]),
                "m": float(mass[k]),
                "H": float(hamiltonian[k]), "u": float(u[k]),
                "L": float(np.linalg.norm(angmom[k])), "L0": float(L0[k]),
                "first_pass": not bool(passed[k]),
            }
        )  # fmt: skip
