"""Block (individual power-of-two) time steps for gas in the hole's gravity
(phases/phase-04-timestepping.md 4.1; Springel 2005 §5; Saitoh & Makino 2009; Durier & Dalla
Vecchia 2012 App. C).

Every particle i has its own step [n_beg_i, n_end_i] on the integer timeline (timesteps.py) and
goes through the same Strang-split kick-flow-kick as the shared-step scheme (coupled.py):

    at n_beg:   p_half = p + L/2 a_hydro,   u_half = u + L/2 du/dt          (opening kick)
    flow:       (x, p) = GLF flow of (x_beg, p_half) over the elapsed time s   (gravity only)
    at n_end:   density and forces for the active particles only, then
                p = p_flow + L/2 a_hydro(new),   u = u_half + L/2 du/dt(new)   (closing kick)

with L = (n_end - n_beg) tick. Between its sync points a particle is only *predicted*, for its
neighbours' force sums (plan 4.1 "Inactive neighbours"): the flow over the elapsed time s for
x, and p = p_flow + a (s - L/2), u = u_half + du/dt (s - L/2). Density, h and Omega are
computed afresh every system step for the active particles *and all their neighbours* (from
the predicted positions), because stale or extrapolated neighbour densities make the scheme
only first order in energy (DECISIONS, Step 4.3: measured on mixed levels, second order is
recovered only with exact neighbour densities and Omega). Forces are computed for the active
particles only; a neighbour's pressure and sound speed use its predicted u.

A system step goes to the earliest n_end; the particles ending there are the active set. The
new level of an active particle is the smallest-step criterion (orbit, Courant, force; plus an
optional forced level offset for tests), subject to the level rules of timesteps.py and to
landing exactly on the run's stop time.

Limiter and wake-up (``limiter=True``; Saitoh & Makino 2009, Durier & Dalla Vecchia 2012
App. C): levels of neighbours differ by at most 2. An active particle's new level is raised
until it is within 2 of every neighbour's; then any inactive neighbour j whose step is more
than 4x the new one is woken: its step is cut to end at the latest multiple of the allowed step
T' that is <= n + T' (Durier Eqs. C2-C4), and its opening kick is corrected to the new
length (Eq. C5): p_half += a (L' - L)/2, u_half += du/dt (L' - L)/2. Its step then is an
ordinary kick-flow-kick step of length L' (the flow restarts from x_beg with the corrected
p_half; positions predicted for it earlier were only predictions). A neighbour whose step
already ends by n + T' keeps its step but takes the finer level label.

With every particle on one level and no limiter this reproduces coupled.step result for
result (tests/sim/test_blocks.py).
"""

import time
from dataclasses import dataclass, field

import numba
import numpy as np

from nebula.physics import viscosity as visc
from nebula.physics.eos import ideal_gas_pressure, ideal_gas_sound_speed
from nebula.physics.gravity import gn_angmom, gn_hamiltonian, gn_p_from_v, gn_v_from_p
from nebula.sim.coupled import CoupledConfig
from nebula.sim.integrator import glf_step_many
from nebula.sim.neighbours import Neighbours, tighten
from nebula.sim.sph import ShockCapturing, compute_density, forces_numba, gradients_numba
from nebula.sim.timesteps import K_MAX, allowed_level, level_for_dt, ticks


@dataclass(frozen=True)
class BlockConfig:
    coupled: CoupledConfig = CoupledConfig()
    dt_max: float = 4096.0  # level-0 step; outputs at multiples of it are full sync points
    limiter: bool = True  # neighbour level limiter and wake-up (Step 4.4)
    max_gap: int = 2  # neighbouring levels differ by at most this (factor 4)
    hydro: bool = True  # False: gravity only (plan test 1; no density or forces)


@dataclass
class BlockState:
    n: int  # current tick; t = t0 + n tick
    t0: float
    tick: float
    level: np.ndarray  # (N,) int64
    n_beg: np.ndarray  # (N,) int64, start of the current step
    n_end: np.ndarray  # (N,) int64, end of the current step
    x_beg: np.ndarray  # position at n_beg
    p_half: np.ndarray  # p after the opening kick (input of the gravity flow)
    u_half: np.ndarray
    x: np.ndarray  # at tick n: true for particles active at n, predicted otherwise
    p: np.ndarray
    u: np.ndarray
    mass: np.ndarray
    h: np.ndarray  # from each particle's last active evaluation (also rho, omega, ...)
    alpha: np.ndarray
    a_hydro: np.ndarray
    dudt: np.ndarray
    rho: np.ndarray
    omega: np.ndarray
    vsig_max: np.ndarray
    level_offset: np.ndarray  # forced extra levels (tests); zeros normally
    n_stop: int = 0  # no step may end beyond this tick
    neighbour_cache: object = field(default=None)
    active_counts: list = field(default_factory=list)  # per system step
    particle_steps: int = 0
    woken: int = 0  # wake-ups so far
    last_pairs: object = field(default=None)  # interacting pairs of the last active set
    timers: dict = field(default_factory=lambda: {"flow": 0.0, "hydro": 0.0, "levels": 0.0})

    @property
    def t(self) -> float:
        return self.t0 + self.n * self.tick

    def synced(self) -> bool:
        """True when every particle is at a step boundary (all states are true, not predicted)."""
        return bool(np.all(self.n_beg == self.n))


def _velocity(cc: CoupledConfig, x, p):
    return gn_v_from_p(x, p) if cc.hole else p


def _flow(cc: CoupledConfig, x, p, s):
    """Gravity-only flow over per-particle times s (GLF with the hole, straight drift without)."""
    if cc.hole:
        x1, p1, _ = glf_step_many(x, p, s)
    else:
        x1, p1 = x + s[:, None] * p, p.copy()
    if cc.hydro.box is not None:
        x1 = np.mod(x1, np.asarray(cc.hydro.box, float))
    return x1, p1


@numba.njit(cache=True)
def _mark_neighbours(rows, offset, index, mark):
    for q in range(rows.shape[0]):
        i = rows[q]
        mark[i] = True
        for k in range(offset[i], offset[i + 1]):
            mark[index[k]] = True


def _with_neighbours(nb, rows, n: int) -> np.ndarray:
    """The rows together with every particle in their neighbour lists (sorted)."""
    mark = np.zeros(n, np.bool_)
    _mark_neighbours(np.asarray(rows, np.int64), nb.offset, nb.index, mark)
    return np.flatnonzero(mark)


def _density_touched(st: BlockState, hc, rows, box):
    """Density, h and Omega for the rows and all their interacting neighbours, at the current
    (predicted) positions. Neighbours are taken from the reused lists first; any interacting
    neighbour not yet computed (e.g. after a list rebuild, or as h changes) is added until
    none is missing. Updates st.h, st.rho, st.omega there; returns the rows' pair lists."""
    n = len(st.x)
    cache = st.neighbour_cache
    todo = np.arange(n) if cache is None else _with_neighbours(cache.neighbours, rows, n)
    done = np.zeros(n, np.bool_)
    for _ in range(8):
        d = compute_density(
            st.x, st.mass, st.h, box=box, kernel_name=hc.kernel, eta=hc.eta, h_max=hc.h_max,
            cache=st.neighbour_cache, skin=hc.skin, rows=todo,
        )  # fmt: skip
        st.neighbour_cache = d.cache
        st.h[todo], st.rho[todo], st.omega[todo] = d.h, d.rho, d.omega
        done[todo] = True
        nb = tighten(d.neighbours, st.x, st.h, rows=rows)
        need = _with_neighbours(nb, rows, n)
        todo = need[~done[need]]
        if len(todo) == 0:
            return nb
    raise RuntimeError("neighbour set of the active particles did not settle")


def _hydro_rows(st: BlockState, cfg: BlockConfig, rows, v, u, dt_alpha, a_prev):
    """Density (rows and their neighbours), switch and forces (rows only) at the current
    positions st.x, velocities v and internal energies u (all particles; inactive ones
    predicted). Updates st's per-particle arrays; returns the rows' interacting-pair lists."""
    if not cfg.hydro:
        return Neighbours(np.zeros(len(st.x) + 1, np.int64), np.zeros(0, np.int64), None)
    hc = cfg.coupled.hydro
    box = None if hc.box is None else np.asarray(hc.box, float)
    if np.any(u <= 0):
        raise FloatingPointError(f"non-positive internal energy at {int(np.sum(u <= 0))} particles")
    nb = _density_touched(st, hc, rows, box)
    h, rho, omega = st.h, st.rho, st.omega
    p_gas = ideal_gas_pressure(rho, u, hc.gamma)
    cs = ideal_gas_sound_speed(u, hc.gamma)
    alpha = st.alpha.copy()
    if hc.viscosity and hc.switch and np.any(dt_alpha > 0):
        g = gradients_numba(st.x, v, a_prev, st.mass, h, nb, hc.kernel, rows=rows)
        a_loc = visc.alpha_local(h[rows], g.ddivv_dt, visc.balsara_xi(g.div_v, g.curl_v), cs[rows])
        alpha[rows] = visc.alpha_update(
            alpha[rows], a_loc, dt_alpha, visc.decay_time(h[rows], cs[rows])
        )
    elif hc.viscosity and not hc.switch:
        alpha[rows] = hc.alpha_const
    f = forces_numba(
        st.x, v, st.mass, h, rho, omega, p_gas, nb, hc.kernel,
        u=u if hc.conductivity else None, c_s=cs if hc.viscosity else None,
        alpha=alpha if hc.viscosity else None,
        shock=ShockCapturing(hc.beta, hc.alpha_u, hc.conductivity_signal), rows=rows,
    )  # fmt: skip
    st.alpha = alpha
    st.a_hydro[rows] = f.acc
    st.dudt[rows] = f.dudt
    st.vsig_max[rows] = f.vsig_max if hc.viscosity else cs[rows]
    return nb


def wanted_dt(st: BlockState, cfg: BlockConfig, rows) -> np.ndarray:
    """Per-particle step criteria (plan 4.1): orbit eta r^(3/2), Courant C h / v_sig,max,
    force C_force sqrt(h / |a_hydro|); the smallest wins."""
    cc = cfg.coupled
    if not cfg.hydro:
        return cc.eta_orb * np.linalg.norm(st.x[rows], axis=1) ** 1.5
    hc = cc.hydro
    h = st.h[rows]
    dt = hc.courant * h / st.vsig_max[rows]
    amag = np.linalg.norm(st.a_hydro[rows], axis=1)
    with np.errstate(divide="ignore"):
        dt = np.minimum(dt, np.where(amag > 0, hc.c_force * np.sqrt(h / amag), np.inf))
    if cc.hole:
        dt = np.minimum(dt, cc.eta_orb * np.linalg.norm(st.x[rows], axis=1) ** 1.5)
    return dt


@numba.njit(cache=True)
def _limit_rows(level, rows, offset, index, max_gap):
    """Raise the levels of ``rows`` until each is within max_gap of all its neighbours'
    (levels of other particles are fixed constraints). Repeats until nothing changes."""
    changed = True
    while changed:
        changed = False
        for q in range(rows.shape[0]):
            i = rows[q]
            need = level[i]
            for k in range(offset[i], offset[i + 1]):
                j = index[k]
                if level[j] - max_gap > need:
                    need = level[j] - max_gap
            if need > level[i]:
                level[i] = need
                changed = True


@numba.njit(cache=True)
def _wake_targets(level, rows, offset, index, max_gap, need):
    """need[j] = max over active neighbours i of level[i] - max_gap (or -1)."""
    for q in range(rows.shape[0]):
        i = rows[q]
        for k in range(offset[i], offset[i + 1]):
            j = index[k]
            if level[i] - max_gap > need[j]:
                need[j] = level[i] - max_gap


def _choose_levels(st: BlockState, cfg: BlockConfig, rows, nb) -> None:
    """New levels for the active ``rows`` at tick st.n; limiter and wake-up if enabled."""
    k = level_for_dt(wanted_dt(st, cfg, rows), cfg.dt_max) + st.level_offset[rows]
    k = allowed_level(st.n, st.level[rows], k)
    level = st.level.copy()
    level[rows] = k
    if cfg.limiter:
        _limit_rows(level, rows, nb.offset, nb.index, cfg.max_gap)
    level[rows] = _land(st, level[rows])
    if np.any(level[rows] > K_MAX):
        raise ValueError(f"a particle needs a level beyond K_MAX = {K_MAX} (investigate it)")
    st.level = level
    if cfg.limiter:
        _wake(st, cfg, rows, nb)


def _land(st: BlockState, k: np.ndarray) -> np.ndarray:
    """Raise levels so that steps opened now end no later than the stop time (shorter steps are
    always allowed). At the stop time itself nothing is changed: ``run`` re-opens then."""
    if st.n >= st.n_stop:
        return k
    while True:
        over = st.n + ticks(np.minimum(k, K_MAX)) > st.n_stop
        if not np.any(over):
            break
        k = np.where(over, k + 1, k)
    return k


def _wake(st: BlockState, cfg: BlockConfig, rows, nb) -> None:
    """Durier & Dalla Vecchia (2012) App. C: shorten inactive neighbours' steps that are more
    than 2^max_gap times the step of an active neighbour (see module docstring)."""
    need = np.full(len(st.level), -1, np.int64)
    _wake_targets(st.level, rows, nb.offset, nb.index, cfg.max_gap, need)
    inactive = st.n_end > st.n
    j = np.flatnonzero(inactive & (need > st.level))
    if len(j) == 0:
        return
    t_new = ticks(need[j])
    end_new = (st.n // t_new + 1) * t_new  # latest multiple of T' that is <= n + T'
    # The level label always follows (the step's ends are on the finer grid too); the step
    # itself is cut only if it would end after n + T' (Durier Eq. C2).
    st.level[j] = need[j]
    cut = end_new < st.n_end[j]
    j, end_new = j[cut], end_new[cut]
    d_len = (end_new - st.n_end[j]) * st.tick  # negative: the step is shortened
    st.p_half[j] += 0.5 * d_len[:, None] * st.a_hydro[j]
    st.u_half[j] += 0.5 * d_len * st.dudt[j]
    st.n_end[j] = end_new
    st.woken += len(j)


def _open(st: BlockState, rows) -> None:
    """Opening kick for the active rows at tick st.n."""
    length = ticks(st.level[rows]) * st.tick
    st.n_beg[rows] = st.n
    st.n_end[rows] = st.n + ticks(st.level[rows])
    st.x_beg[rows] = st.x[rows]
    st.p_half[rows] = st.p[rows] + 0.5 * length[:, None] * st.a_hydro[rows]
    st.u_half[rows] = st.u[rows] + 0.5 * length * st.dudt[rows]


def initialise(
    x, v, u, mass, h_guess, cfg: BlockConfig, t_stop: float, t0: float = 0.0, level_offset=None
) -> BlockState:
    """All particles start active at t0; the first steps never run past t_stop (rounded to
    the tick; ``run`` moves the stop time to its t_end)."""
    cc = cfg.coupled
    hc = cc.hydro
    x = np.array(x, float)
    n = len(x)
    p = gn_p_from_v(x, np.asarray(v, float)) if cc.hole else np.array(v, float)
    tick = cfg.dt_max / 2.0**K_MAX
    zeros3 = np.zeros((n, 3))
    st = BlockState(
        n=0, t0=t0, tick=tick, level=np.zeros(n, np.int64), n_beg=np.zeros(n, np.int64),
        n_end=np.zeros(n, np.int64), x_beg=x.copy(), p_half=p.copy(),
        u_half=np.array(u, float) * np.ones(n), x=x, p=p, u=np.array(u, float) * np.ones(n),
        mass=np.array(mass, float) * np.ones(n), h=np.array(h_guess, float) * np.ones(n),
        alpha=np.full(n, hc.alpha_init if hc.switch else hc.alpha_const),
        a_hydro=zeros3.copy(), dudt=np.zeros(n), rho=np.zeros(n), omega=np.ones(n),
        vsig_max=np.zeros(n),
        level_offset=np.zeros(n, np.int64) if level_offset is None
        else np.asarray(level_offset, np.int64),
        n_stop=int(round((t_stop - t0) / tick)),
    )  # fmt: skip
    rows = np.arange(n)
    nb = _hydro_rows(st, cfg, rows, _velocity(cc, x, p), st.u, 0.0, None)
    st.level[:] = 0
    _choose_levels(st, cfg, rows, nb)
    _open(st, rows)
    return st


def advance(st: BlockState, cfg: BlockConfig) -> np.ndarray:
    """One system step: to the earliest step end; returns the indices of the active set."""
    cc = cfg.coupled
    t0 = time.perf_counter()
    n_next = int(np.min(st.n_end))
    s_ticks = n_next - st.n_beg
    s = s_ticks * st.tick
    half_len = 0.5 * (st.n_end - st.n_beg) * st.tick
    x, p_flow = _flow(cc, st.x_beg, st.p_half, s)
    t1 = time.perf_counter()
    rows = np.flatnonzero(st.n_end == n_next)
    lag = (s - half_len)[:, None]  # 0 at the midpoint of the step, +L/2 at its end
    p_pred = p_flow + lag * st.a_hydro
    u_pred = st.u_half + (s - half_len) * st.dudt
    a_prev = st.a_hydro.copy()
    st.n = n_next
    st.x = x
    st.u = u_pred
    nb = _hydro_rows(st, cfg, rows, _velocity(cc, x, p_pred), u_pred, 2.0 * half_len[rows], a_prev)
    t2 = time.perf_counter()
    p_pred[rows] = p_flow[rows] + half_len[rows, None] * st.a_hydro[rows]
    st.u[rows] = st.u_half[rows] + half_len[rows] * st.dudt[rows]
    st.p = p_pred
    _choose_levels(st, cfg, rows, nb)
    _open(st, rows)
    st.last_pairs = nb
    st.timers["flow"] += t1 - t0
    st.timers["hydro"] += t2 - t1
    st.timers["levels"] += time.perf_counter() - t2
    st.active_counts.append(len(rows))
    st.particle_steps += len(rows)
    return rows


def run(st: BlockState, cfg: BlockConfig, t_end: float, callback=None) -> BlockState:
    """Advance until every particle is exactly at t_end (a full sync point): from now on no
    step may end beyond it. callback(st) after each system step."""
    n_end = int(round((t_end - st.t0) / st.tick))
    st.n_stop = n_end
    fresh = np.flatnonzero((st.n_beg == st.n) & (st.n_end > n_end))
    if len(fresh):  # steps just opened (nothing done yet): re-open them to land on t_end
        st.level[fresh] = _land(st, st.level[fresh])
        if np.any(st.level[fresh] > K_MAX):
            raise ValueError(f"t_end needs a level beyond K_MAX = {K_MAX}")
        _open(st, fresh)
    if np.any(st.n_end > n_end):
        raise ValueError("some particles' current steps already run past t_end")
    while st.n < n_end:
        advance(st, cfg)
        if callback is not None:
            callback(st)
    return st


def ledger(st: BlockState, cfg: BlockConfig) -> dict:
    """Totals as in coupled.ledger (true values when st.synced(), predicted otherwise)."""
    cc = cfg.coupled
    h = gn_hamiltonian(st.x, st.p) if cc.hole else 0.5 * np.sum(st.p * st.p, axis=1)
    am = gn_angmom(st.x, st.p) if cc.hole else np.cross(st.x, st.p)
    return {
        "energy": float(np.sum(st.mass * (h + st.u))),
        "abs_h": float(np.sum(st.mass * np.abs(h))),
        "angmom": np.sum(st.mass[:, None] * am, axis=0),
        "momentum": np.sum(st.mass[:, None] * st.p, axis=0),
    }


def diagnostics(st: BlockState) -> dict:
    """Plan 4.4 diagnostics: level histogram, active counts, total particle-steps, wake-ups and
    wall time per stage (flow = gravity prediction of all particles; hydro = density, switch
    and forces of the active set; levels = criteria, limiter, wake-up and opening kicks)."""
    counts = np.asarray(st.active_counts)
    return {
        "t": st.t,
        "level_histogram": np.bincount(st.level, minlength=int(st.level.max()) + 1),
        "system_steps": len(counts),
        "particle_steps": st.particle_steps,
        "mean_active": float(counts.mean()) if len(counts) else 0.0,
        "woken": st.woken,
        "timers": dict(st.timers),
    }


def snapshot(st: BlockState, cfg: BlockConfig, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Positions and momenta of every particle at time t (between sync points), for output
    (plan 4.1 "Snapshots"): each particle's last opened state advanced by a gravity-only flow,
    p with the same predictor as for inactive neighbours. The real state is untouched."""
    n_t = (t - st.t0) / st.tick
    if np.any(n_t < st.n_beg - 1e-6) or np.any(n_t > st.n_end + 1e-6):
        raise ValueError("snapshot time outside some particle's current step")
    s = np.maximum((t - st.t0) - st.n_beg * st.tick, 0.0)
    x, p_flow = _flow(cfg.coupled, st.x_beg, st.p_half, s)
    lag = s - 0.5 * (st.n_end - st.n_beg) * st.tick
    return x, p_flow + lag[:, None] * st.a_hydro
