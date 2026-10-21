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

from nebula import units
from nebula.physics import viscosity as visc
from nebula.physics.disc_viscosity import nu_av
from nebula.physics.eos import (
    gasrad_state,
    gasrad_u_si,
    ideal_gas_pressure,
    ideal_gas_sound_speed,
    ideal_gas_u_si,
    isothermal_pressure,
    locally_isothermal_cs,
    radiation_limit_pressure,
    radiation_limit_sound_speed,
)
from nebula.physics.kernels import DEFAULT_ETA, DEFAULT_KERNEL
from nebula.physics.navier_stokes import ss_viscosity, viscous_dt, viscous_heating, viscous_stress
from nebula.sim.fastphys import gasrad_state_nb
from nebula.sim.neighbours import tighten
from nebula.sim.sph import (
    COND_PRESSURE,
    ShockCapturing,
    compute_density,
    disc_forces_numba,
    forces_numba,
    gradients_numba,
    ns_forces_numba,
    ns_gradients_numba,
    velocity_gradients_numba,
)


@dataclass(frozen=True)
class HydroConfig:
    gamma: float = 5.0 / 3.0
    isothermal_cs: float | None = None  # if set: P = c_s^2 rho, u not evolved
    # locally isothermal (phase-07 7.4 ring test): with iso_r0 set, c_s = isothermal_cs at r =
    # iso_r0 and c_s ~ r^(-iso_q) with r the distance from the origin (the hole)
    iso_r0: float | None = None
    iso_q: float = 0.75
    eos: str = (
        "ideal"  # "ideal" (gamma), "gasrad" (gas + radiation, Phase 6), "radiation" (P = rho u/3)
    )
    gas_units: units.GasUnits = units.GasUnits()  # mass scale for "gasrad" (PLAN §3)
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
    # physical Navier-Stokes viscosity (Phase 7 Option A; physics/navier_stokes.py), off by
    # default: a constant kinematic viscosity, or Shakura-Sunyaev nu = alpha c_s^2 / Omega_K
    # with Omega_K = r^(-3/2) about the origin (the hole)
    ns_nu: float = 0.0
    ns_alpha: float = 0.0
    # disc viscosity through the AV form (Phase 7 Option B; physics/disc_viscosity.py): a fixed
    # alpha_AV on every pair, nu ~ alpha_AV c_s h / 10; off by default; not with Option A
    disc_alpha_av: float = 0.0

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
    nu: np.ndarray = field(default=None)  # physical kinematic viscosity per particle (or None)
    n_flagged: int = 0  # density flags (h at its cap) in the last evaluation
    neighbour_cache: object = field(default=None)  # sph.NeighbourCache, reused between steps


def eos_state(cfg: HydroConfig, rho, u, x=None):
    """Pressure and sound speed (code units) from the configured equation of state; positions
    ``x`` are needed only for the locally isothermal one."""
    if cfg.isothermal_cs is not None and cfg.iso_r0 is not None:
        cs = locally_isothermal_cs(np.linalg.norm(x, axis=1), cfg.isothermal_cs, cfg.iso_r0,
                                   cfg.iso_q)  # fmt: skip
        return isothermal_pressure(rho, cs), cs
    if cfg.isothermal_cs is not None:
        return isothermal_pressure(rho, cfg.isothermal_cs), np.full_like(rho, cfg.isothermal_cs)
    if np.any(u <= 0):
        raise FloatingPointError(f"non-positive internal energy at {int(np.sum(u <= 0))} particles")
    if cfg.eos == "ideal":
        return ideal_gas_pressure(rho, u, cfg.gamma), ideal_gas_sound_speed(u, cfg.gamma)
    if cfg.eos == "gasrad":  # Numba copy of physics.eos.gasrad_state (sim/fastphys.py)
        p, cs, _, _ = gasrad_state_nb(rho, u, cfg.gas_units)
        return p, cs
    if cfg.eos == "radiation":
        return radiation_limit_pressure(rho, u), radiation_limit_sound_speed(u)
    raise ValueError(f"unknown equation of state {cfg.eos!r}")


_eos = eos_state  # (old name)


def temperature(cfg: HydroConfig, rho, u):
    """Gas temperature (K) from code rho and u with the configured equation of state."""
    if cfg.eos == "gasrad":
        return gasrad_state(rho, u, cfg.gas_units)[2]
    if cfg.eos == "ideal":
        u_si = units.specific_energy_to_si(np.asarray(u, float))
        return u_si * (cfg.gamma - 1.0) * units.MU * units.M_P / units.K_B
    raise ValueError(f"no temperature for equation of state {cfg.eos!r}")


def floor_energy(cfg: HydroConfig, rho, t_floor: float):
    """Code specific energy at temperature t_floor (the cooling floor), per particle."""
    rho = np.asarray(rho, float)
    if cfg.eos == "gasrad":
        rho_si = cfg.gas_units.density_to_si(rho)
        return units.specific_energy_from_si(gasrad_u_si(rho_si, t_floor))
    if cfg.eos == "ideal":
        return np.full_like(rho, units.specific_energy_from_si(ideal_gas_u_si(t_floor, cfg.gamma)))
    raise ValueError(f"no temperature floor for equation of state {cfg.eos!r}")


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
    p, cs = eos_state(cfg, d.rho, state.u, state.x)
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
    acc, dudt_all = f.acc, f.dudt
    nu = physical_viscosity(cfg, state.x, cs)
    if nu is not None:
        g = ns_gradients_numba(state.x, state.v, state.mass, d.h, d.rho, d.omega, nb, cfg.kernel)
        stress = viscous_stress(g, d.rho, nu)
        acc = acc + ns_forces_numba(state.x, state.mass, d.h, d.rho, d.omega, stress, nb,
                                    cfg.kernel)  # fmt: skip
        dudt_all = dudt_all + viscous_heating(g, nu)
    if cfg.disc_alpha_av > 0:  # Option B (sim/sph.py disc_forces_numba)
        a_disc, u_disc = disc_forces_numba(state.x, state.v, state.mass, d.h, d.rho, d.omega, cs,
                                           cfg.disc_alpha_av, nb, cfg.kernel)  # fmt: skip
        acc, dudt_all = acc + a_disc, dudt_all + u_disc
        nu = nu_av(cfg.disc_alpha_av, cs, d.h)  # equivalent viscosity (time step, diagnostics)
    dudt = np.zeros(len(state.x)) if cfg.isothermal_cs is not None else dudt_all
    vsig = f.vsig_max if cfg.viscosity else np.maximum(cs, 0.0)
    return replace(
        state, h=d.h, alpha=alpha, acc=acc, dudt=dudt, rho=d.rho, pressure=p, c_s=cs,
        vsig_max=vsig, nu=nu, n_flagged=int(np.sum(d.flags != 0)), neighbour_cache=d.cache,
    )  # fmt: skip


def physical_viscosity(cfg: HydroConfig, x, c_s):
    """Kinematic viscosity per particle of Option A (None if off): ns_nu, or
    ss_viscosity(ns_alpha, c_s, r) with r the distance from the origin (the hole)."""
    if (cfg.ns_nu > 0) + (cfg.ns_alpha > 0) + (cfg.disc_alpha_av > 0) > 1:
        raise ValueError("set only one of ns_nu, ns_alpha, disc_alpha_av")
    if cfg.ns_nu > 0:
        return np.full(len(x), cfg.ns_nu)
    if cfg.ns_alpha > 0:
        return ss_viscosity(cfg.ns_alpha, c_s, np.linalg.norm(x, axis=1))
    return None


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
    dt_visc = np.min(viscous_dt(state.h, state.nu)) if state.nu is not None else np.inf
    return float(np.nanmin([dt_cour, dt_force, dt_visc]))


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
