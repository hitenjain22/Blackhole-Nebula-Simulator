"""Gas in the hole's gravity: SPH coupled to the GN Hamiltonian through canonical momenta
(phases/phase-04-timestepping.md 4.1 "Per-particle step" and "Why hydro kicks go into p").

Each step of size dt is Strang-split:
    1. hydro half-kick   p += dt/2 a_hydro(start),   u += dt/2 du/dt(start)
    2. gravity flow      generalized leapfrog of H_GN over dt (Phase 1, sim/integrator.py)
    3. hydro forces at the new positions (with v = dH/dp), then the second half-kick
       p += dt/2 a_hydro(end),   u += dt/2 du/dt(end)
The SPH acceleration acts as a generalized force on p. Then dH/dt = v . a_hydro exactly
matches the work the SPH energy equation removes from u, so the ledger
    E = sum_i m_i (H_i + u_i)    (+ radiated, accreted ... in later phases)
is conserved by construction up to the time-stepping error, and the angular momentum
sum m (x cross p) is conserved because SPH pair forces are central and antisymmetric.
("Wrong coupling", for the report's comparison only: the kicks applied to v instead of p.)

Without the hole (``hole=False``) H = p^2/2, the flow is a straight drift, and the step
reduces to Phase 3's kick-drift-kick (sim/hydro.py).

Velocity-dependent terms (viscosity, conductivity) at the end of the step use the velocity
predicted with the start-of-step acceleration, as in sim/hydro.py.
"""

import math
from dataclasses import dataclass, field, replace

import numpy as np

from nebula.physics import viscosity as visc
from nebula.physics.cooling import T_FLOOR, beta_cool, cooling_time
from nebula.physics.gravity import (
    gn_angmom,
    gn_hamiltonian,
    gn_p_from_v,
    gn_v_from_p,
    newton_hamiltonian,
)
from nebula.sim.fastphys import gn_v_from_p_nb
from nebula.sim.hydro import HydroConfig, eos_state, floor_energy
from nebula.sim.integrator import glf_step_many, kdk_newton_many
from nebula.sim.neighbours import tighten
from nebula.sim.sph import (
    COND_VELOCITY,
    compute_density,
    forces_numba,
    gradients_numba,
    velocity_gradients_numba,
)


@dataclass(frozen=True)
class CoupledConfig:
    hydro: HydroConfig = HydroConfig(conductivity_signal=COND_VELOCITY)
    hole: bool = True
    eta_orb: float = 0.00125  # orbit step criterion dt <= eta r^(3/2) (DECISIONS, Step 1.4)
    wrong_coupling: bool = False  # kicks on v instead of p (report comparison only)
    gravity: str = "gn"  # "gn" (generalized Newtonian, the default) or "newton" (control R6.4)
    # beta-cooling toward the floor (physics/cooling.py, phase-06 6.2): beta = inf: no cooling
    cooling_beta: float = math.inf
    t_floor: float = T_FLOOR


@dataclass
class GasState:
    t: float
    x: np.ndarray
    p: np.ndarray  # canonical momentum per unit mass (= velocity without the hole)
    u: np.ndarray
    mass: np.ndarray
    h: np.ndarray
    alpha: np.ndarray
    a_hydro: np.ndarray = field(default=None)
    dudt: np.ndarray = field(default=None)
    rho: np.ndarray = field(default=None)
    c_s: np.ndarray = field(default=None)
    vsig_max: np.ndarray = field(default=None)
    neighbour_cache: object = field(default=None)
    e_radiated: float = 0.0  # energy radiated so far (cooling), code units
    visc_heating: np.ndarray = field(default=None)  # viscous part of du/dt (the shock marker)


def _gn(cfg: CoupledConfig) -> bool:
    if cfg.gravity not in ("gn", "newton"):
        raise ValueError(f"unknown gravity {cfg.gravity!r}")
    return cfg.hole and cfg.gravity == "gn"


def velocity(cfg: CoupledConfig, x, p):
    """v = dH/dp: GN (Phase 1) with the hole; p itself without it or with Newtonian gravity."""
    if not _gn(cfg):
        return p
    # Numba copy for particle arrays (sim/fastphys.py, tested against physics.gravity)
    return gn_v_from_p_nb(x, p) if np.ndim(x) == 2 else gn_v_from_p(x, p)


def p_from_v(cfg: CoupledConfig, x, v):
    """Canonical momentum from velocity (inverse of ``velocity``)."""
    return gn_p_from_v(x, v) if _gn(cfg) else np.array(v, float)


def hamiltonian(cfg: CoupledConfig, x, p):
    """Specific orbital energy H: GN, Newtonian point mass, or |p|^2/2 without the hole."""
    if _gn(cfg):
        return gn_hamiltonian(x, p)
    if cfg.hole:
        return newton_hamiltonian(x, p)
    return 0.5 * np.sum(p * p, axis=1)


def angmom(cfg: CoupledConfig, x, p):
    """Specific angular momentum x cross p per particle (gn_angmom is the same quantity)."""
    return gn_angmom(x, p) if _gn(cfg) else np.cross(x, p)


def gravity_flow(cfg: CoupledConfig, x, p, dt):
    """The gravity-only flow over dt (scalar or per particle): GN generalized leapfrog,
    Newtonian kick-drift-kick, or a straight drift without the hole."""
    if _gn(cfg):
        x1, p1, _ = glf_step_many(x, p, dt)
    elif cfg.hole:
        x1, p1 = kdk_newton_many(x, p, dt)
    else:
        x1, p1 = x + np.reshape(dt, (-1, 1)) * p, np.array(p, float)
    if cfg.hydro.box is not None:
        x1 = np.mod(x1, np.asarray(cfg.hydro.box, float))
    return x1, p1


def _hydro(st: GasState, cfg: CoupledConfig, v, dt_alpha: float, a_prev) -> GasState:
    hc = cfg.hydro
    box = None if hc.box is None else np.asarray(hc.box, float)
    d = compute_density(
        st.x, st.mass, st.h, box=box, kernel_name=hc.kernel, eta=hc.eta, h_max=hc.h_max,
        cache=st.neighbour_cache, skin=hc.skin,
    )  # fmt: skip
    if np.any(st.u <= 0):
        raise FloatingPointError(
            f"non-positive internal energy at {int(np.sum(st.u <= 0))} particles"
        )
    p_gas, cs = eos_state(hc, d.rho, st.u)
    nb = tighten(d.neighbours, st.x, d.h)
    alpha = st.alpha
    if hc.viscosity and hc.switch and dt_alpha > 0:
        g = gradients_numba(st.x, v, a_prev, st.mass, d.h, nb, hc.kernel)
        a_loc = visc.alpha_local(d.h, g.ddivv_dt, visc.balsara_xi(g.div_v, g.curl_v), cs)
        alpha = visc.alpha_update(alpha, a_loc, dt_alpha, visc.decay_time(d.h, cs))
    elif hc.viscosity and not hc.switch:
        alpha = np.full(len(st.x), hc.alpha_const)
    shock = hc.shock()
    grad_v = (
        velocity_gradients_numba(st.x, v, st.mass, d.h, nb, hc.kernel)
        if shock.reconstruction
        else None
    )
    f = forces_numba(
        st.x, v, st.mass, d.h, d.rho, d.omega, p_gas, nb, hc.kernel,
        u=st.u if hc.conductivity else None, c_s=cs if hc.viscosity else None,
        alpha=alpha if hc.viscosity else None, shock=shock, grad_v=grad_v,
    )  # fmt: skip
    return replace(
        st, h=d.h, alpha=alpha, a_hydro=f.acc, dudt=f.dudt, rho=d.rho, c_s=cs,
        vsig_max=f.vsig_max if hc.viscosity else cs, neighbour_cache=d.cache,
        visc_heating=f.dudt_visc,
    )  # fmt: skip


def initialise(x, v, u, mass, h_guess, cfg: CoupledConfig, t: float = 0.0) -> GasState:
    """Start from positions and velocities (p computed from v)."""
    x = np.array(x, float)
    n = len(x)
    v = np.array(v, float)
    p = p_from_v(cfg, x, v)
    hc = cfg.hydro
    st = GasState(
        t, x, p, np.array(u, float) * np.ones(n), np.array(mass, float) * np.ones(n),
        np.array(h_guess, float) * np.ones(n),
        np.full(n, hc.alpha_init if hc.switch else hc.alpha_const),
    )  # fmt: skip
    return _hydro(st, cfg, velocity(cfg, x, p), 0.0, None)


def stable_dt(st: GasState, cfg: CoupledConfig) -> float:
    """Shared step: min over particles of the orbit, Courant and force criteria
    (phase-04 4.1)."""
    hc = cfg.hydro
    dt = hc.courant * np.min(st.h / st.vsig_max)
    amag = np.linalg.norm(st.a_hydro, axis=1)
    if np.any(amag > 0):
        dt = min(dt, hc.c_force * np.min(np.sqrt(st.h[amag > 0] / amag[amag > 0])))
    if cfg.hole:
        r = np.linalg.norm(st.x, axis=1)
        dt = min(dt, cfg.eta_orb * np.min(r**1.5))
    return float(dt)


def _kick(cfg: CoupledConfig, x, p, a, dt_half):
    if not cfg.wrong_coupling or not cfg.hole:
        return p + dt_half * a
    if not _gn(cfg):
        raise ValueError("the wrong-coupling comparison is defined for GN gravity only")
    v = gn_v_from_p(x, p) + dt_half * a  # wrong: the hydro acceleration added to v
    return gn_p_from_v(x, v)


def step(st: GasState, cfg: CoupledConfig, dt: float) -> GasState:
    """One Strang-split step (module docstring)."""
    p_half = _kick(cfg, st.x, st.p, st.a_hydro, 0.5 * dt)
    u_half = st.u + 0.5 * dt * st.dudt
    x1, p_flow = gravity_flow(cfg, st.x, p_half, dt)
    e_rad = st.e_radiated
    if math.isfinite(cfg.cooling_beta):  # in the middle of the step (Strang split, phase-06 6.2)
        r_mid = 0.5 * (np.linalg.norm(st.x, axis=1) + np.linalg.norm(x1, axis=1))
        u_cool = beta_cool(
            u_half, floor_energy(cfg.hydro, st.rho, cfg.t_floor), dt,
            cooling_time(r_mid, cfg.cooling_beta),
        )  # fmt: skip
        e_rad += float(np.sum(st.mass * (u_half - u_cool)))
        u_half = u_cool
    p_pred = _kick(cfg, x1, p_flow, st.a_hydro, 0.5 * dt)  # predictor for velocity terms
    pred = replace(st, t=st.t + dt, x=x1, p=p_flow, u=u_half + 0.5 * dt * st.dudt)
    new = _hydro(pred, cfg, velocity(cfg, x1, p_pred), dt, st.a_hydro)
    return replace(
        new, p=_kick(cfg, x1, p_flow, new.a_hydro, 0.5 * dt), u=u_half + 0.5 * dt * new.dudt,
        e_radiated=e_rad,
    )  # fmt: skip


def ledger(st: GasState, cfg: CoupledConfig) -> dict:
    """Conserved totals: E = sum m (H + u) + E_radiated, angular momentum sum m (x cross p),
    and sum m |H| (the scale for relative energy errors)."""
    h = hamiltonian(cfg, st.x, st.p)
    return {
        "energy": float(np.sum(st.mass * (h + st.u))) + st.e_radiated,
        "e_radiated": st.e_radiated,
        "abs_h": float(np.sum(st.mass * np.abs(h))),
        "angmom": np.sum(st.mass[:, None] * angmom(cfg, st.x, st.p), axis=0),
    }
