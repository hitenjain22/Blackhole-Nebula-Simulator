"""Pure-hydrodynamics time stepping for the Phase 3 test problems (no black hole).

One global time step for all particles (Phase 4 replaces this with block steps and couples
the gas to the hole through canonical momenta; with no hole, p = v). Kick-drift-kick
leapfrog with a predicted end-of-step velocity for the velocity-dependent terms (artificial
viscosity, conductivity), as in Phantom (Price et al. 2018, §2.3):

    v_half = v + dt/2 a(t),          u_half = u + dt/2 du/dt(t)
    x(t+dt) = x + dt v_half
    v_pred = v_half + dt/2 a(t),     u_pred = u_half + dt/2 du/dt(t)        (predictor)
    evaluate density, gradients, switch, forces at (x(t+dt), v_pred, u_pred)
    v(t+dt) = v_half + dt/2 a(t+dt), u(t+dt) = u_half + dt/2 du/dt(t+dt)

Time step: dt = min over particles of C_cour h / v_sig,max and C_force sqrt(h / |a|)
(phase-04 4.1 criteria; C_cour = 0.3, C_force = 0.25), or a fixed dt when given.
"""

from dataclasses import dataclass, field, replace

import numpy as np

from nebula.physics import viscosity as visc
from nebula.physics.eos import ideal_gas_pressure, ideal_gas_sound_speed, isothermal_pressure
from nebula.physics.kernels import DEFAULT_ETA, DEFAULT_KERNEL
from nebula.sim.neighbours import tighten
from nebula.sim.sph import (
    COND_PRESSURE,
    ShockCapturing,
    compute_density,
    forces_numba,
    gradients_numba,
    velocity_gradients_numba,
)


@dataclass(frozen=True)
class HydroConfig:
    gamma: float = 5.0 / 3.0
    isothermal_cs: float | None = None  # if set: P = c_s^2 rho, u not evolved
    kernel: str = DEFAULT_KERNEL
    eta: float = DEFAULT_ETA
    box: tuple | None = None  # periodic box lengths, or None (open)
    h_max: float = np.inf
    courant: float = 0.3
    skin: float = 1.4  # neighbour-search margin; 1.4 was fastest of 1.25/1.4/1.6 (DECISIONS 3.8)
    c_force: float = 0.25
    viscosity: bool = True
    conductivity: bool = True
    switch: bool = True  # Cullen-Dehnen (Phantom variant); False: constant alpha_const
    alpha_const: float = 1.0
    alpha_init: float = 0.0
    beta: float = visc.BETA_AV
    alpha_u: float = visc.ALPHA_U
    conductivity_signal: int = COND_PRESSURE
    reconstruction: bool = True  # velocity reconstruction in the viscosity (DECISIONS D2)
    recon_closeness: bool = True  # its closeness factor (Frontiere et al. 2017 Eq. 51)
    recon_gate: int = 2  # sph.GATE_MINMOD (DECISIONS D2)

    def shock(self) -> ShockCapturing:
        """The force pass's shock-capturing settings for this configuration."""
        return ShockCapturing(
            self.beta, self.alpha_u, self.conductivity_signal,
            reconstruction=self.reconstruction and self.viscosity,
            closeness=self.recon_closeness, eta_crit=1.0 / self.eta, recon_gate=self.recon_gate,
        )  # fmt: skip


@dataclass
class HydroState:
    t: float
    x: np.ndarray
    v: np.ndarray
    u: np.ndarray
    mass: np.ndarray
    h: np.ndarray
    alpha: np.ndarray
    acc: np.ndarray = field(default=None)
    dudt: np.ndarray = field(default=None)
    rho: np.ndarray = field(default=None)
    pressure: np.ndarray = field(default=None)
    c_s: np.ndarray = field(default=None)
    vsig_max: np.ndarray = field(default=None)
    n_flagged: int = 0  # density flags (h at its cap) in the last evaluation
    neighbour_cache: object = field(default=None)  # sph.NeighbourCache, reused between steps


def _eos(cfg: HydroConfig, rho, u):
    if cfg.isothermal_cs is not None:
        return isothermal_pressure(rho, cfg.isothermal_cs), np.full_like(rho, cfg.isothermal_cs)
    if np.any(u <= 0):
        raise FloatingPointError(f"non-positive internal energy at {int(np.sum(u <= 0))} particles")
    return ideal_gas_pressure(rho, u, cfg.gamma), ideal_gas_sound_speed(u, cfg.gamma)


def evaluate(
    state: HydroState, cfg: HydroConfig, dt_alpha: float = 0.0, acc_prev=None
) -> HydroState:
    """Density, EOS, switch and forces at the state's positions/velocities. ``dt_alpha``: time
    step over which the viscosity switch is advanced (0 at start-up: alpha left as is)."""
    box = None if cfg.box is None else np.asarray(cfg.box, float)
    d = compute_density(
        state.x, state.mass, state.h, box=box, kernel_name=cfg.kernel, eta=cfg.eta,
        h_max=cfg.h_max, cache=state.neighbour_cache, skin=cfg.skin,
    )  # fmt: skip
    p, cs = _eos(cfg, d.rho, state.u)
    nb = tighten(d.neighbours, state.x, d.h)  # only interacting pairs (identical sums, less work)
    alpha = state.alpha
    if cfg.viscosity and cfg.switch and dt_alpha > 0:
        a_prev = np.zeros_like(state.x) if acc_prev is None else acc_prev
        g = gradients_numba(state.x, state.v, a_prev, state.mass, d.h, nb, cfg.kernel)
        xi = visc.balsara_xi(g.div_v, g.curl_v)
        a_loc = visc.alpha_local(d.h, g.ddivv_dt, xi, cs)
        alpha = visc.alpha_update(alpha, a_loc, dt_alpha, visc.decay_time(d.h, cs))
    elif cfg.viscosity and not cfg.switch:
        alpha = np.full(len(state.x), cfg.alpha_const)
    shock = cfg.shock()
    grad_v = (
        velocity_gradients_numba(state.x, state.v, state.mass, d.h, nb, cfg.kernel)
        if shock.reconstruction
        else None
    )
    f = forces_numba(
        state.x, state.v, state.mass, d.h, d.rho, d.omega, p, nb, cfg.kernel,
        u=state.u if (cfg.conductivity and cfg.isothermal_cs is None) else None,
        c_s=cs if cfg.viscosity else None,
        alpha=alpha if cfg.viscosity else None,
        shock=shock, grad_v=grad_v,
    )  # fmt: skip
    dudt = np.zeros(len(state.x)) if cfg.isothermal_cs is not None else f.dudt
    vsig = f.vsig_max if cfg.viscosity else np.maximum(cs, 0.0)
    return replace(
        state, h=d.h, alpha=alpha, acc=f.acc, dudt=dudt, rho=d.rho, pressure=p, c_s=cs,
        vsig_max=vsig, n_flagged=int(np.sum(d.flags != 0)), neighbour_cache=d.cache,
    )  # fmt: skip


def initialise(x, v, u, mass, h_guess, cfg: HydroConfig, t: float = 0.0) -> HydroState:
    n = len(x)
    st = HydroState(
        t, np.array(x, float), np.array(v, float), np.array(u, float) * np.ones(n),
        np.array(mass, float) * np.ones(n), np.array(h_guess, float) * np.ones(n),
        np.full(n, cfg.alpha_init if cfg.switch else cfg.alpha_const),
    )  # fmt: skip
    return evaluate(st, cfg)


def stable_dt(state: HydroState, cfg: HydroConfig) -> float:
    """Global step: min(C_cour h / v_sig,max, C_force sqrt(h / |a|))."""
    dt_cour = cfg.courant * np.min(state.h / state.vsig_max)
    amag = np.linalg.norm(state.acc, axis=1)
    with np.errstate(divide="ignore"):
        dt_force = cfg.c_force * np.min(np.sqrt(state.h / np.where(amag > 0, amag, np.nan)))
    return float(np.nanmin([dt_cour, dt_force]))


def step(state: HydroState, cfg: HydroConfig, dt: float) -> HydroState:
    """One kick-drift-kick step of size dt (see module docstring)."""
    v_half = state.v + 0.5 * dt * state.acc
    u_half = state.u + 0.5 * dt * state.dudt
    x1 = state.x + dt * v_half
    if cfg.box is not None:
        x1 = np.mod(x1, np.asarray(cfg.box, float))
    pred = replace(
        state,
        t=state.t + dt,
        x=x1,
        v=v_half + 0.5 * dt * state.acc,
        u=u_half + 0.5 * dt * state.dudt,
    )
    new = evaluate(pred, cfg, dt_alpha=dt, acc_prev=state.acc)
    return replace(new, v=v_half + 0.5 * dt * new.acc, u=u_half + 0.5 * dt * new.dudt)


def run(
    state: HydroState, cfg: HydroConfig, t_end: float, dt_fixed: float | None = None, callback=None
):
    """Advance to t_end (last step shortened to land on it). callback(state) after each step."""
    while state.t < t_end * (1 - 1e-14):
        dt = dt_fixed if dt_fixed is not None else stable_dt(state, cfg)
        dt = min(dt, t_end - state.t)
        state = step(state, cfg, dt)
        if callback is not None:
            callback(state)
    return state


def totals(state: HydroState) -> dict:
    """Conserved totals: energy sum m (v^2/2 + u), momentum, angular momentum (about 0)."""
    m = state.mass
    return {
        "energy": float(np.sum(m * (0.5 * np.sum(state.v**2, axis=1) + state.u))),
        "momentum": np.sum(m[:, None] * state.v, axis=0),
        "angmom": np.sum(m[:, None] * np.cross(state.x, state.v), axis=0),
    }
