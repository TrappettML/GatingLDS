"""Experiments and figures for the neuronal split-gating note, `splitgating_neuronal.tex`.

Standalone: it imports nothing from `hadamard.py`, `lag_curves.py` or
`verify_neuronal.py` and writes no file either of those writes.  The model is the
note's, with gates that open and close whole neurons:

    F_t = diag(f_1..f_N),  B_t = diag(b_1..b_N),  b_i <= f_i,  both N x N
    yhat_t(x) = v^T F_t W x / sqrt(D)
    Wdot      = -B_t v (v^T F_t W - theta_t)                        eq. (6)

One gate pair per task serves the whole network, so the write mass c_t = v^T B_t v
is a scalar, the coupling Gamma_kt is a scalar, and the whole stream is one
triangular system of size K with the D columns of Theta as right-hand sides.

    python3 neuronal_curves.py            # self-test, then all four figures
    python3 neuronal_curves.py --test     # the self-test alone (~15 s)
    python3 neuronal_curves.py 1 3        # only figures 1 and 3

Figures written next to this file:

    fig1_flow.png       the stream as it actually runs: integrated gradient flow
                        against the two closed forms, eq. (16) and eq. (27)
    fig2_vs_tasks.png   metric against task index, one trace per density d_f = d_b,
                        mean-field theory dotted and simulation solid
    fig3_vs_density.png metric against read density, one trace per split ratio
    fig4_heatmaps.png   density x splitness, three snapshots of one run to T = 1000

Every figure carries both metrics: 1 - eps, normalised against the null state W = 0,
and 1 - M, normalised against a read gate that was never trained.  1 is a solved
task, 0 is the baseline, below 0 is worse than the baseline.
"""

import sys
from pathlib import Path

import numpy as np


# =============================================================== configuration

CONFIG = dict(
    # --- the network, shared by every figure -------------------------------
    N=60,            # N_h: neurons.  The gate opens and closes whole neurons, so
                     #      this is the only width the dynamics sees
    D=12,            # N_in: input dimension.  The D coordinates share one gate and
                     #      differ only in what drives them
    SEEDS=20,        # independent draws (readout, gates, teachers) per point
    N_CTRL=4,        # never-trained read gates averaged into M's denominator
    LAG_PAD=5,       # retention lag Delta = round(N / n_b) + LAG_PAD: the task is
                     #      read out just past the point where the write pool has
                     #      been turned over once

    # --- figure 1, the flow ------------------------------------------------
    K_FLOW=16,       # tasks drawn out in flow time
    DF_FLOW=0.5,     # read density
    RHO_FLOW=2.0,    # split ratio d_f / d_b for the flow figure
    FLOW_SEED=0,
    PER_TASK=240,    # points recorded inside each task's window
    DECAY=20.0,      # window width, in time constants 1/c_t
    SUBSTEPS=4,      # integrator steps between two recorded points, at least
    MAX_RATE_STEP=0.1,   # ceiling on c_t dtau

    # --- figure 2, metric against task index -------------------------------
    K_TASKS=5000,    # stream length
    DENSITIES=(0.5, 0.3, 0.2, 0.1),   # d_f = d_b, one trace each
    N_BINS_DECADE=12,  # logarithmic bins of anchors per decade.  Every anchor
                     #   of the stream is measured; a point on figure 2 is the
                     #   geometric mean over its bin and over the seeds

    # --- figure 3, metric against density ----------------------------------
    K_DENSITY=1500,  # stream length; the metric is averaged over the second half
    RHOS=(1.0, 2.0, 3.0),             # d_f / d_b, one trace each
    DF_STEP=0.05,    # read-density grid, from D/N to 1.0

    # --- figure 4, density against splitness -------------------------------
    K_HEAT=1000,     # one run per cell; the three columns are snapshots of it
    HEAT_EPS_FLOOR=-4.0,   # colour floors, in -log10(metric).  -4 means the cell
    HEAT_CTRL_FLOOR=-0.1,  #   only has to read as "eps is 10^4 times the baseline
                     #   or worse"; the .txt keeps the unclipped numbers
    SNAPSHOTS=(10, 100, 1000),
    HEAT_DF_STEP=0.1,                 # read density, from D/N to 1.0
    HEAT_RHOS=tuple(range(1, 11)),    # splitness 1..10
    HEAT_WINDOW=0.1, # anchors averaged into a snapshot, as a fraction of k

    # --- the self-test -----------------------------------------------------
    K_CHECK=600,     # stream length at which the streamed route is checked
                     #      against the exact triangular solve, eq. (27)
)


# ------------------------------------------------------------------ palette
# Categorical slots in the validated order; four is the most any panel uses and
# that set clears the adjacent colour-vision gates.  Aqua and yellow sit below 3:1
# on this surface, so every trace is also labelled at its right-hand end and
# identity is never carried by colour alone.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

# Diverging ramp for the heatmaps: the metric has a meaningful zero (no better than
# the baseline) with values either side of it, so one hue per sign and a neutral
# grey in the middle.  Blue is retained, red is worse than the baseline.
RAMP_BLUE = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
RAMP_RED = ("#fbd7d7", "#f3acac", "#e88484", "#e34948", "#c13232", "#992424", "#701919")
NEUTRAL = "#f0efec"


def diverging_cmap():
    """Red -> neutral grey -> blue, equal steps per arm, lightness monotone."""
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    stops = list(RAMP_RED[::-1]) + [NEUTRAL] + list(RAMP_BLUE)
    lum = [0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
           for c in (to_rgb(h) for h in stops)]
    rise, fall = lum[:len(RAMP_RED) + 1], lum[len(RAMP_RED):]
    if not (all(np.diff(rise) > 0) and all(np.diff(fall) < 0)):
        raise ValueError("the diverging ramp is not monotone in lightness per arm")
    return LinearSegmentedColormap.from_list("worse_better", stops, N=512)


# ==================================================================== the draw

def counts(N, d_f, rho):
    """Neurons read and written, as counts, with the nesting checked here.

    Densities are rounded onto the 1/N grid once, and everything downstream uses
    the realised pair.  The nesting (7) says a neuron may be written only if it is
    read, so n_b <= n_f; an empty write pool has no write mass and eq. (19) would
    divide by zero, so n_b >= 1.
    """
    n_f = int(round(d_f * N))
    n_b = int(round(n_f / rho))
    if not 1 <= n_f <= N:
        raise ValueError(f"d_f={d_f} gives n_f={n_f} with N={N}; need 1 <= n_f <= N")
    if n_b < 1:
        raise ValueError(
            f"d_f={d_f}, rho={rho} give n_b={n_b}: an empty write pool has no write "
            f"mass and the coupling (19) is undefined there.  Raise N or lower rho")
    if n_b > n_f:
        raise ValueError(f"n_b={n_b} > n_f={n_f}: the nesting (7) needs rho >= 1")
    return n_f, n_b


def draw(N, D, K, n_f, n_b, seed=0):
    """One realisation: readout, nested per-task neuron gates, teachers.

    One uniform permutation of the N neurons per task; the first n_f places are
    read and the first n_b written, so B_t sits inside F_t by construction and any
    two tasks' gates are independent uniform subsets -- the constant array
    d_fb^{kt} = n_f/N = d_f for k != t, which is the most decorrelated member of
    the family eq. (9) allows.  Teacher rows are unit vectors so that the null
    loss is 1 and eq. (32) needs no rescaling.
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(N) / np.sqrt(N)                     # eq. (1)
    rank = np.argsort(np.argsort(rng.random((K, N)), axis=1), axis=1)
    F = (rank < n_f).astype(float)
    B = (rank < n_b).astype(float)
    Theta = rng.standard_normal((K, D))
    Theta /= np.linalg.norm(Theta, axis=1, keepdims=True)
    return v, F, B, Theta


def draw_controls(N, D, n_f, n_ctrl, seed=0):
    """Read gates and teachers for tasks that are never trained: M's denominator.

    Eq. (22) has no special case for the reader, so a gate that never writes is
    read out of every state the stream visits exactly like any other.  Its own
    generator is keyed separately, so adding controls to a run leaves v, F, B and
    Theta bit-identical.
    """
    rng = np.random.default_rng((seed, 0xC0FFEE))
    rank = np.argsort(np.argsort(rng.random((n_ctrl, N)), axis=1), axis=1)
    Fc = (rank < n_f).astype(float)
    Tc = rng.standard_normal((n_ctrl, D))
    Tc /= np.linalg.norm(Tc, axis=1, keepdims=True)
    return Fc, Tc


# ============================================================= the stream

def run_stream(v, F, B, Theta, Delta, Fc=None, Tc=None):
    """Walk the stream once, reading the anchor back at two lags as it goes.

    The update is the endpoint of the within-task flow, eq. (16), so no integration
    is needed: the flow has the fixed left factor B_t v, the state moves on a line,
    and the endpoint is the line's end.  State W^s is the state after s tasks have
    trained, W^0 = 0, and task j (0-based) trains into W^{j+1}.

    Three readings come back, all normalised by the null loss of their own task:

        eps_tr[j]    Delta = -1: task j read out of W^j, the state just before it
                     trains -- what the stream already solves of it for free
        eps_ret[j]   task j read out of W^{j+1+Delta}, Delta tasks after it was
                     trained -- what is left of it
        eps_ctrl[s]  the never-trained controls read out of W^s, averaged

    Rows of eps_ret with no such state are left as NaN.  Only a rolling window of
    Delta+1 gates is needed, so the cost is O(K N D) in time and O(N D) in space
    whatever K is.
    """
    K, N = F.shape
    D = Theta.shape[1]
    w = v ** 2
    c = B @ w                                             # write masses, eq. (15)
    if (c <= 0).any():
        raise ValueError("a task has zero write mass; eq. (19) is undefined there")

    W = np.zeros((N, D))
    eps_tr = np.empty(K)
    eps_ret = np.full(K, np.nan)
    eps_ctrl = np.full(K + 1, np.nan)
    norm = (Theta ** 2).sum(1)

    if Fc is not None:
        Fcv = Fc * v[None, :]
        nc = (Tc ** 2).sum(1)
        eps_ctrl[0] = float((((Tc - Fcv @ W) ** 2).sum(1) / nc).mean())

    for t in range(K):
        r = Theta[t] - (F[t] * v) @ W                     # eq. (12)
        eps_tr[t] = (r ** 2).sum() / norm[t]
        W += np.outer(B[t] * v, r) / c[t]                 # eq. (16)
        s = t + 1
        if Fc is not None:
            eps_ctrl[s] = float((((Tc - Fcv @ W) ** 2).sum(1) / nc).mean())
        j = s - 1 - Delta
        if j >= 0:
            eps_ret[j] = ((Theta[j] - (F[j] * v) @ W) ** 2).sum() / norm[j]
    return eps_tr, eps_ret, eps_ctrl


def run_stream_snapshots(v, F, B, Theta, Delta, snapshots, window, Fc, Tc):
    """The same walk, but recording only at a few states: figure 4's three columns.

    A snapshot at k is the state W^k.  The transfer reading there is task k-1 read
    out of W^{k-1}, the retention reading is task k-1-Delta read out of W^k, and
    both are averaged over the `window` anchors ending there so that a single
    heavy-tailed draw does not set the cell.  A retention reading needs the anchor
    to exist, so cells with Delta >= k come back NaN rather than as a shorter lag.
    """
    K, N = F.shape
    w = v ** 2
    c = B @ w
    W = np.zeros((N, Theta.shape[1]))
    norm = (Theta ** 2).sum(1)
    Fcv = Fc * v[None, :]
    nc = (Tc ** 2).sum(1)

    snaps = sorted(int(s) for s in snapshots)
    want_tr, want_ret = {}, {}
    for k in snaps:
        nw = max(1, int(round(window * k)))
        for j in range(max(0, k - nw), k):                # transfer anchors
            want_tr.setdefault(j, []).append(k)
        for j in range(max(0, k - nw - Delta), k - Delta):   # retention anchors
            if j >= 0:
                want_ret.setdefault(j + 1 + Delta, []).append((j, k))
    out = {k: {"tr": [], "ret": [], "ctrl_tr": [], "ctrl_ret": []} for k in snaps}

    eps_ctrl = np.full(K + 1, np.nan)
    eps_ctrl[0] = float((((Tc - Fcv @ W) ** 2).sum(1) / nc).mean())
    for t in range(K):
        r = Theta[t] - (F[t] * v) @ W
        if t in want_tr:
            e = (r ** 2).sum() / norm[t]
            for k in want_tr[t]:
                out[k]["tr"].append(e)
                out[k]["ctrl_tr"].append(eps_ctrl[t])
        W += np.outer(B[t] * v, r) / c[t]
        s = t + 1
        eps_ctrl[s] = float((((Tc - Fcv @ W) ** 2).sum(1) / nc).mean())
        if s in want_ret:
            for j, k in want_ret[s]:
                e = ((Theta[j] - (F[j] * v) @ W) ** 2).sum() / norm[j]
                out[k]["ret"].append(e)
                out[k]["ctrl_ret"].append(eps_ctrl[s])
        if s > max(snaps):
            break
    return out


# ================================================= the exact route, eq. (27)

def coupling(v, F, B):
    """Gamma_kt = v^T F_k B_t v / v^T B_t v, eq. (19): one K x K array, no D axis."""
    w = v ** 2
    return (F @ (w[:, None] * B.T)) / (B @ w)[None, :]


def solve_stream(Gamma, Theta):
    """R = (I + L)^{-1} Theta by forward substitution, eq. (27).

    L is the strict lower triangle of Gamma.  One triangular system of size K with
    the D columns of Theta as right-hand sides -- one system and not D of them,
    because a neuron is open or closed on all D of its synapses at once.
    """
    K = Theta.shape[0]
    L = np.tril(Gamma, -1)
    R = np.empty_like(Theta)
    for t in range(K):
        R[t] = Theta[t] - L[t, :t] @ R[:t]
    return R


def lagged(Gamma, R, Delta):
    """R^(Delta) = -U(Delta) R, eq. (30): the first Delta superdiagonals of Gamma."""
    U = np.triu(Gamma, 1) - np.triu(Gamma, Delta + 1)
    return -U @ R


# ============================================== the mean field, Gamma -> d_f

def mf_eps(d_f, t, Delta):
    """The metric with the realised Gamma replaced by its mean, eq. (21).

    For gates drawn independently each task E[Gamma_kt] = d_f whatever the split
    ratio is, so the recursion eq. (24) collapses to a scalar one.  With
    S_t = sum_{s<=t} r_s it reads S_t = (1-d_f) S_{t-1} + theta_t, so with p = 1-d_f

        r_t          = theta_t - d_f S_{t-1}
        r_t^(Delta)  = -d_f (S_{t+Delta} - S_t)
                     = -d_f [ (p^Delta - 1) S_t + sum_{s=t+1}^{t+Delta} p^{t+Delta-s} theta_s ]

    and for teachers that are independent, isotropic and of unit norm the two
    pieces of the second line are uncorrelated, so

        eps^(-1)_t     = 1 + d_f (1 - p^{2(t-1)}) / (2 - d_f)
        eps^(Delta)_t  = d_f [ (1 - p^{2 Delta})
                               + (1 - p^Delta)^2 (1 - p^{2t}) ] / (2 - d_f)

    using 1 - p^2 = d_f (2 - d_f).  `t` is 1-based, as the note counts tasks.

    Two things this is not.  It is free of the split ratio, because E[Gamma] is:
    everything the split does to the metric lives in second moments and in the
    stability of eq. (17), neither of which survives replacing Gamma by its mean.
    And it is a floor, not an estimate -- eps is quadratic in Gamma and
    E[Gamma^2] > E[Gamma]^2, so the measured curve sits above this one.
    """
    p = 1.0 - d_f
    t = np.asarray(t, dtype=float)
    if Delta == -1:
        return 1.0 + d_f * (1.0 - p ** (2 * (t - 1))) / (2.0 - d_f)
    if Delta < 0:
        raise ValueError(f"Delta={Delta}: only -1 and Delta >= 0 are defined here")
    return d_f * ((1.0 - p ** (2 * Delta))
                  + (1.0 - p ** Delta) ** 2 * (1.0 - p ** (2 * t))) / (2.0 - d_f)


def mf_ctrl(d_f, s):
    """The same for a never-trained gate read out of state s: 1 + d_f^2 E||S_s||^2."""
    p = 1.0 - d_f
    s = np.asarray(s, dtype=float)
    return 1.0 + d_f * (1.0 - p ** (2 * s)) / (2.0 - d_f)


def mf_curves(d_f, t, Delta):
    """(1 - eps, 1 - M) for both arms, mean field, at 1-based anchor index t.

    The state the anchor is read from is t + Delta, and the control is read from
    the same state, so on the transfer arm the two coincide exactly: before a task
    is trained the mean field cannot tell it from a task never trained on, and
    1 - M is identically zero there.  Everything the transfer panel shows under M
    is therefore fluctuation, which is the honest reading of it.
    """
    e_tr, e_ret = mf_eps(d_f, t, -1), mf_eps(d_f, t, Delta)
    return dict(tr_eps=1.0 - e_tr,
                ret_eps=1.0 - e_ret,
                tr_ctrl=1.0 - e_tr / mf_ctrl(d_f, np.asarray(t, float) - 1.0),
                ret_ctrl=1.0 - e_ret / mf_ctrl(d_f, np.asarray(t, float) + Delta))


def retention_lag(N, n_b, pad):
    """Delta = round(N / n_b) + pad: one turnover of the write pool, plus a little."""
    return int(round(N / n_b)) + int(pad)


# ==================================================================== figure 1

def flow_case(cfg):
    """One short stream, three ways: integrated, closed form, closed form from (27).

    The integration is told nothing.  It steps

        dW/dtau = B_t v (theta_t - v^T F_t W)

    with a fourth-order explicit scheme and reads the loss off the state as it
    goes; the residual each task inherits is whatever the previous integration
    left behind.  The closed form is eq. (16) read in time: within task t the read
    error obeys de/dtau = -c_t e, so

        loss_t(tau) = 1/2 ||r_t||^2 exp(-2 c_t tau) ,

    and because the gate is neuronal there is one rate per task rather than one per
    input coordinate -- all D coordinates decay together, which is why a task draws
    as a single straight line on a logarithmic axis.  The two closed-form traces
    differ only in where r_t came from: the stream, or the triangular solve that
    never trained anything.
    """
    N, D = cfg["N"], cfg["D"]
    K = cfg["K_FLOW"]
    n_f, n_b = counts(N, cfg["DF_FLOW"], cfg["RHO_FLOW"])
    v, F, B, Theta = draw(N, D, K, n_f, n_b, seed=cfg["FLOW_SEED"])
    w = v ** 2
    c = B @ w
    per, decay = cfg["PER_TASK"], cfg["DECAY"]
    span = decay / c
    start = np.concatenate([[0.0], np.cumsum(span)[:-1]])
    frac = np.linspace(0.0, 1.0, per)
    tau = span[:, None] * frac[None, :]

    # route 1: the protocol, eq. (16)
    W = np.zeros((N, D))
    R_stream = np.empty((K, D))
    for t in range(K):
        r = Theta[t] - (F[t] * v) @ W
        R_stream[t] = r
        W += np.outer(B[t] * v, r) / c[t]

    # route 2: the triangular solve, eq. (27) -- no training at all
    R_direct = solve_stream(coupling(v, F, B), Theta)

    def closed(R):
        return 0.5 * (R ** 2).sum(1)[:, None] * np.exp(-2.0 * c[:, None] * tau)

    # route 3: integrate the flow and be told nothing
    W = np.zeros((N, D))
    loss_int = np.empty((K, per))
    R_int = np.empty((K, D))
    dt_rec = tau[:, 1] - tau[:, 0]
    for t in range(K):
        Ft, Bt, th = F[t] * v, B[t] * v, Theta[t]
        nsub = max(cfg["SUBSTEPS"],
                   int(np.ceil(c[t] * dt_rec[t] / cfg["MAX_RATE_STEP"])))
        h = dt_rec[t] / nsub

        def rhs(state):
            return np.outer(Bt, th - Ft @ state)

        for p in range(per):
            e = th - Ft @ W
            loss_int[t, p] = 0.5 * (e ** 2).sum()
            if p == 0:
                R_int[t] = e
            if p + 1 < per:
                for _ in range(nsub):
                    k1 = rhs(W)
                    k2 = rhs(W + 0.5 * h * k1)
                    k3 = rhs(W + 0.5 * h * k2)
                    k4 = rhs(W + h * k3)
                    W += (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    loss_stream, loss_direct = closed(R_stream), closed(R_direct)
    scale = loss_stream[:, :1]
    return dict(
        time=start[:, None] + tau, K=K, n_f=n_f, n_b=n_b, c=c,
        loss_int=loss_int, loss_stream=loss_stream, loss_direct=loss_direct,
        box=0.5 * (R_direct ** 2).sum(1),
        flow_gap=float(np.abs(loss_int - loss_stream).max() / scale.max()),
        box_gap=float(np.abs(R_int - R_direct).max() / np.abs(R_direct).max()),
        solve_gap=float(np.abs(R_stream - R_direct).max() / np.abs(R_direct).max()),
    )


def fig_flow(cfg, res, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13.0, 4.6))
    K = res["K"]
    raw = res["time"]
    # equal slots per task: the windows are 20 time constants of their own task and
    # a thin write gate makes those vary, so raw flow time would give one slow task
    # half the axis.  Within a slot the coordinate is the fraction of that window,
    # so every tooth descends at the same slope and the picture compares heights.
    time = np.arange(K)[:, None] + (raw - raw[:, :1]) / (raw[:, -1:] - raw[:, :1])
    for t in range(K):
        first = t == 0
        ax.semilogy(time[t], res["loss_int"][t], color=MUTED, linewidth=3.2,
                    alpha=0.5, solid_capstyle="round",
                    label="gradient flow, integrated" if first else None)
        ax.semilogy(time[t], res["loss_stream"][t], color=SERIES[0], linewidth=1.3,
                    label="closed form, stream residuals (16)" if first else None)
        ax.semilogy(time[t], res["loss_direct"][t], color=SERIES[1], linewidth=1.0,
                    linestyle=(0, (4, 2)),
                    label="closed form, direct residuals (27)" if first else None)
    ax.semilogy(np.arange(K), res["box"], linestyle="none", marker="s",
                markersize=11, markerfacecolor="none", markeredgecolor=SERIES[1],
                markeredgewidth=1.6,
                label=r"direct solve, $\frac{1}{2}\|r_t\|^2$")

    ax.set_xlabel("task, and the fraction of its window "
                  r"($20/c_t$ of flow time each)", color=INK2)
    ax.set_ylabel(r"$\frac{1}{2}\|\hat\theta_t-\theta_t\|^2$", color=INK2)
    ax.set_title("the stream as it actually runs   "
                 f"($N={cfg['N']}$, $N_{{in}}={cfg['D']}$, $n_f={res['n_f']}$, "
                 f"$n_b={res['n_b']}$, seed {cfg['FLOW_SEED']})",
                 pad=30, color=INK, fontsize=11)
    top, bottom = res["box"].max(), res["box"].min()
    ax.set_ylim(bottom * 1e-15, top * 3.0)
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="lower left",
              bbox_to_anchor=(0.0, 1.005), labelcolor=INK2)
    ax.annotate(
        f"integrated vs closed form: {res['flow_gap']:.1e} of each task's height "
        f"(the step, spent at the top of each tooth)\n"
        f"integrated residuals vs eq. (27): {res['box_gap']:.1e}   "
        f"($e^{{-{cfg['DECAY']:.0f}}}$ = {np.exp(-cfg['DECAY']):.0e}, the finite window)"
        f"      stream vs eq. (27): {res['solve_gap']:.1e}",
        xy=(0.008, 0.03), xycoords="axes fraction", fontsize=8, color=MUTED,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2.0))
    _chrome(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


# ==================================================================== figure 2

def geo(x):
    """Geometric mean of eps over draws and anchors, the way this file aggregates.

    eps is a ratio of squared norms and, once the stream is unstable, heavy-tailed
    across draws: at rho = 3 a single draw can sit twenty orders of magnitude above
    the rest, so an arithmetic mean reports that draw and nothing else.  log eps is
    the near-Gaussian variable, so the mean is taken there and exponentiated back.
    Every curve and every cell in this file is 1 - (geometric mean of eps).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        return np.nan
    return float(np.exp(np.log(x).mean()))


def log_bins(K, per_decade):
    """Logarithmic bins of 1-based anchor indices, as (lo, hi) half-open pairs."""
    n = max(2, int(round(np.log10(K) * per_decade)) + 1)
    edges = np.unique(np.round(np.logspace(0.0, np.log10(K + 1), n)).astype(int))
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def _end_labels(ax, items, gap=0.055):
    """Name each trace at its own right-hand end, pushed apart so none is hidden.

    Two of the four categorical slots sit below 3:1 on this surface, so identity
    cannot rest on colour: every trace carries its own label.  Labels are set in
    secondary ink rather than the series colour -- the line arriving at the label is
    what carries identity -- and are nudged apart in axis coordinates when they
    would otherwise overlap.
    """
    if not items:
        return
    lo, hi = ax.get_ylim()
    log = ax.get_yscale() != "linear"

    def to_frac(y):
        if not log:
            return (y - lo) / (hi - lo)
        return (ax.transScale.transform((0, y))[1]
                - ax.transScale.transform((0, lo))[1]) / max(
            ax.transScale.transform((0, hi))[1]
            - ax.transScale.transform((0, lo))[1], 1e-12)

    order = sorted(range(len(items)), key=lambda i: to_frac(items[i][1]))
    placed = []
    for i in order:
        f = min(max(to_frac(items[i][1]), 0.0), 1.0)
        if placed and f - placed[-1] < gap:
            f = placed[-1] + gap
        placed.append(f)
    for f, i in zip(placed, order):
        x, _, text = items[i]
        ax.annotate(text, xy=(x, f), xycoords=("data", "axes fraction"),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=8, color=INK2, va="center", clip_on=False)


# ==================================================================== figure 2

def case_vs_tasks(cfg, d_f):
    """One density, every anchor of a stream of K_TASKS, binned logarithmically.

    Every anchor is measured, so a bin near the end of the stream pools hundreds of
    them; that, and not a longer stream, is what makes the tail of the curve smooth.
    """
    N, D, K = cfg["N"], cfg["D"], cfg["K_TASKS"]
    n_f, n_b = counts(N, d_f, 1.0)                 # d_f = d_b on this figure
    Delta = retention_lag(N, n_b, cfg["LAG_PAD"])
    keys = ("tr_eps", "ret_eps", "tr_ctrl", "ret_ctrl")
    raw = {k: [] for k in keys}
    for s in range(cfg["SEEDS"]):
        v, F, B, Theta = draw(N, D, K, n_f, n_b, seed=100 + s)
        Fc, Tc = draw_controls(N, D, n_f, cfg["N_CTRL"], seed=100 + s)
        e_tr, e_ret, e_ctrl = run_stream(v, F, B, Theta, Delta, Fc, Tc)
        j = np.arange(K)
        raw["tr_eps"].append(e_tr)
        raw["ret_eps"].append(e_ret)
        raw["tr_ctrl"].append(e_tr / e_ctrl[j])
        raw["ret_ctrl"].append(e_ret / e_ctrl[np.minimum(j + 1 + Delta, K)])
    stack = {k: np.vstack(a) for k, a in raw.items()}        # (seeds, K)

    bins = log_bins(K, cfg["N_BINS_DECADE"])
    out = {k: [] for k in keys}
    centre = []
    for a, b in bins:
        idx = np.arange(a, b)                                # 1-based anchors
        centre.append(float(np.exp(np.log(idx).mean())))
        for k in keys:
            out[k].append(geo(stack[k][:, idx - 1]))
    out = {k: np.array(x) for k, x in out.items()}
    out["centre"] = np.array(centre)
    out["Delta"], out["n_b"] = Delta, n_b
    out["last"] = {k: float(np.array(x)[np.isfinite(x)][-1]) for k, x in out.items()
                   if k in keys}
    return out


def fig_vs_tasks(cfg, cases, path, solve_gap):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.2), sharex=True)
    arms = (("ret", "retention:  the task read back $\\Delta$ tasks after it trained"),
            ("tr", "transfer:  the task read out of the state just before it trains"))
    mets = (("eps", r"$1-\varepsilon$   (against $W=0$)"),
            ("ctrl", r"$1-M$   (against a never-trained gate)"))
    handles = []
    for ci, (arm, arm_title) in enumerate(arms):
        for ri, (met, ylab) in enumerate(mets):
            ax = axes[ri][ci]
            ends = []
            for k, d_f in enumerate(cfg["DENSITIES"]):
                res = cases[d_f]
                Delta = res["Delta"]
                x, y = res["centre"], 1.0 - res[f"{arm}_{met}"]
                ok = np.isfinite(y)
                ln, = ax.semilogx(x[ok], y[ok], color=SERIES[k], linewidth=1.8,
                                  solid_capstyle="round", zorder=3)
                th = mf_curves(d_f, x[ok], Delta)[f"{arm}_{met}"]
                ax.semilogx(x[ok], np.broadcast_to(th, x[ok].shape),
                            color=SERIES[k], linewidth=1.3, linestyle=(0, (1, 2)),
                            zorder=2)
                ends.append((x[ok][-1], y[ok][-1], f"{d_f}"))
                if ri + ci == 0:
                    ln.set_label(f"$d_f=d_b={d_f}$   ($\\Delta={Delta}$)")
                    handles.append(ln)
            ax.axhline(0.0, color=AXIS, linewidth=1.0, zorder=1)
            if ri == 0:
                ax.set_title(arm_title, fontsize=10, color=INK, pad=8)
            if ci == 0:
                ax.set_ylabel(ylab, fontsize=9.5, color=INK2)
            if ri == 1:
                ax.set_xlabel("tasks trained", fontsize=9.5, color=INK2)
            _chrome(ax)
            _end_labels(ax, ends)

    from matplotlib.lines import Line2D
    style = [Line2D([], [], color=MUTED, lw=1.8, label="simulation"),
             Line2D([], [], color=MUTED, lw=1.3, ls=(0, (1, 2)),
                    label="mean field, $\\Gamma\\to d_f$")]
    fig.legend(handles=handles + style, frameon=False, fontsize=8.5, ncol=6,
               loc="upper center", bbox_to_anchor=(0.5, 0.945), labelcolor=INK2)
    fig.suptitle("what a task is worth, against how far into the stream it sits   "
                 f"($N={cfg['N']}$, $N_{{in}}={cfg['D']}$, $K={cfg['K_TASKS']}$, "
                 f"{cfg['SEEDS']} draws;  1 = solved, 0 = baseline)",
                 fontsize=11, color=INK, y=0.995)
    fig.text(0.5, 0.012,
             "each point is the geometric mean of $\\varepsilon$ over its logarithmic bin of "
             "anchors and over the draws.\n"
             "the mean field is a floor, not an estimate: $\\varepsilon$ is quadratic in "
             "$\\Gamma$ and $E[\\Gamma^2]>E[\\Gamma]^2$, so the measured curve sits below it "
             f"here.      stream vs the exact solve (27) at $K={cfg['K_CHECK']}$: "
             f"{solve_gap:.1e}",
             ha="center", va="bottom", fontsize=8, color=MUTED, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.055, 1, 0.925))
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


# ==================================================================== figure 3

def case_vs_density(cfg, d_f, rho):
    """Steady-state metric at one (d_f, rho): the stream's second half, pooled."""
    N, D, K = cfg["N"], cfg["D"], cfg["K_DENSITY"]
    n_f, n_b = counts(N, d_f, rho)
    Delta = retention_lag(N, n_b, cfg["LAG_PAD"])
    lo = K // 2
    pool = {k: [] for k in ("tr_eps", "ret_eps", "tr_ctrl", "ret_ctrl")}
    for s in range(cfg["SEEDS"]):
        v, F, B, Theta = draw(N, D, K, n_f, n_b, seed=200 + s)
        Fc, Tc = draw_controls(N, D, n_f, cfg["N_CTRL"], seed=200 + s)
        e_tr, e_ret, e_ctrl = run_stream(v, F, B, Theta, Delta, Fc, Tc)
        j = np.arange(K)
        m = np.zeros(K, bool)
        m[lo:] = True
        ok = m & np.isfinite(e_ret)
        pool["tr_eps"].append(e_tr[m])
        pool["tr_ctrl"].append(e_tr[m] / e_ctrl[j[m]])
        pool["ret_eps"].append(e_ret[ok])
        pool["ret_ctrl"].append(e_ret[ok] / e_ctrl[j[ok] + 1 + Delta])
    out = {k: 1.0 - geo(np.concatenate(x)) for k, x in pool.items()}
    out["Delta"], out["n_b"], out["t_mid"] = Delta, n_b, 0.75 * K
    return out


def fig_vs_density(cfg, grid, path, solve_gap):
    import matplotlib.pyplot as plt

    d_fs = sorted({d for d, _ in grid})
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.2), sharex=True)
    arms = (("ret", "retention:  read back $\\Delta=\\lceil N/n_b\\rfloor+5$ tasks later"),
            ("tr", "transfer:  read out of the state just before the task trains"))
    mets = (("eps", r"$1-\varepsilon$   (against $W=0$)"),
            ("ctrl", r"$1-M$   (against a never-trained gate)"))
    handles = []
    for ci, (arm, arm_title) in enumerate(arms):
        for ri, (met, ylab) in enumerate(mets):
            ax = axes[ri][ci]
            ends = []
            for k, rho in enumerate(cfg["RHOS"]):
                xs = [d for d in d_fs if (d, rho) in grid]
                ys = [grid[(d, rho)][f"{arm}_{met}"] for d in xs]
                ln, = ax.plot(xs, ys, color=SERIES[k], linewidth=1.8, zorder=3,
                              solid_capstyle="round")
                ends.append((xs[-1], ys[-1], f"{rho:g}"))
                if ri + ci == 0:
                    ln.set_label(f"$d_f/d_b={rho:g}$")
                    handles.append(ln)
            # the mean field cannot see the split: E[Gamma] = d_f whatever d_b is,
            # so one dotted curve serves all three traces
            th = [mf_curves(d, grid[(d, cfg["RHOS"][0])]["t_mid"],
                            grid[(d, cfg["RHOS"][0])]["Delta"])[f"{arm}_{met}"]
                  for d in d_fs]
            ax.plot(d_fs, th, color=MUTED, linewidth=1.4, linestyle=(0, (1, 2)),
                    zorder=2)
            ax.axhline(0.0, color=AXIS, linewidth=1.0, zorder=1)
            if met == "eps":
                # eps runs over sixteen decades once the split destabilises the
                # stream, so the scale is linear through the interesting band and
                # logarithmic outside it
                ax.set_yscale("symlog", linthresh=1.0, linscale=2.2)
                ax.set_ylim(top=1.15)
            if ri == 0:
                ax.set_title(arm_title, fontsize=10, color=INK, pad=8)
            if ci == 0:
                ax.set_ylabel(ylab, fontsize=9.5, color=INK2)
            if ri == 1:
                ax.set_xlabel("read density $d_f$", fontsize=9.5, color=INK2)
            _chrome(ax)
            _end_labels(ax, ends)

    from matplotlib.lines import Line2D
    style = [Line2D([], [], color=MUTED, lw=1.8, label="simulation"),
             Line2D([], [], color=MUTED, lw=1.4, ls=(0, (1, 2)),
                    label="mean field, $\\Gamma\\to d_f$ (one curve for all three)")]
    fig.legend(handles=handles + style, frameon=False, fontsize=8.5, ncol=5,
               loc="upper center", bbox_to_anchor=(0.5, 0.945), labelcolor=INK2)
    fig.suptitle("what a task is worth, against how dense the read gate is   "
                 f"($N={cfg['N']}$, $N_{{in}}={cfg['D']}$, $K={cfg['K_DENSITY']}$, "
                 f"{cfg['SEEDS']} draws, second half of the stream)",
                 fontsize=11, color=INK, y=0.995)
    fig.text(0.5, 0.012,
             "the mean field is one curve for all three split ratios, because "
             "$E[\\Gamma_{kt}]=d_f$ whatever $d_b$ is:\n"
             "everything the split does to the metric lives in second moments and in the "
             "stability of eq. (17).      stream vs the exact solve (27) at "
             f"$K={cfg['K_CHECK']}$: {solve_gap:.1e}",
             ha="center", va="bottom", fontsize=8, color=MUTED, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.055, 1, 0.925))
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


# ==================================================================== figure 4

def case_heat(cfg, d_f, rho):
    """One run to K_HEAT at one cell, read at the three snapshots, pooled over draws."""
    N, D, K = cfg["N"], cfg["D"], cfg["K_HEAT"]
    n_f, n_b = counts(N, d_f, rho)
    Delta = retention_lag(N, n_b, cfg["LAG_PAD"])
    pool = {k: {m: [] for m in ("tr_eps", "ret_eps", "tr_ctrl", "ret_ctrl")}
            for k in cfg["SNAPSHOTS"]}
    for s in range(cfg["SEEDS"]):
        v, F, B, Theta = draw(N, D, K, n_f, n_b, seed=300 + s)
        Fc, Tc = draw_controls(N, D, n_f, cfg["N_CTRL"], seed=300 + s)
        snap = run_stream_snapshots(v, F, B, Theta, Delta, cfg["SNAPSHOTS"],
                                    cfg["HEAT_WINDOW"], Fc, Tc)
        for k in cfg["SNAPSHOTS"]:
            d = snap[k]
            pool[k]["tr_eps"] += list(d["tr"])
            pool[k]["ret_eps"] += list(d["ret"])
            if d["tr"]:
                pool[k]["tr_ctrl"] += list(np.array(d["tr"]) / np.array(d["ctrl_tr"]))
            if d["ret"]:
                pool[k]["ret_ctrl"] += list(np.array(d["ret"]) / np.array(d["ctrl_ret"]))
    return {k: {m: 1.0 - geo(x) for m, x in d.items()} for k, d in pool.items()}, Delta


def fig_heatmaps(cfg, cells, path):
    """Twelve panels of one quantity, read two ways at three times.

    The colour carries $-\\log_{10}$ of the metric rather than the metric itself.
    It is the same quantity with the same baseline -- $-\\log_{10}\\varepsilon$ and
    $1-\\varepsilon$ vanish together at $\\varepsilon=1$, are proportional near it, and
    agree in sign everywhere -- but across this grid $\\varepsilon$ runs from 0.4 to
    $10^{62}$, and no linear scale shows both the mild corner and the blown-up one.
    The two metric blocks get their own scale, since one is bounded and the other is
    not, and within a block the three snapshots share it: they are three readings of
    one run and have to be comparable.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    d_fs = sorted({d for d, _ in cells})
    rhos = list(cfg["HEAT_RHOS"])
    snaps = list(cfg["SNAPSHOTS"])
    blocks = ((("tr_eps", r"transfer,  $-\log_{10}\varepsilon$"),
               ("ret_eps", r"retention,  $-\log_{10}\varepsilon$")),
              (("tr_ctrl", r"transfer,  $-\log_{10}M$"),
               ("ret_ctrl", r"retention,  $-\log_{10}M$")))
    floors = (cfg["HEAT_EPS_FLOOR"], cfg["HEAT_CTRL_FLOOR"])

    planes = {}
    for block in blocks:
        for met, _ in block:
            for k in snaps:
                Z = np.full((len(d_fs), len(rhos)), np.nan)
                for i, d in enumerate(d_fs):
                    for j, rho in enumerate(rhos):
                        if (d, rho) in cells:
                            val = 1.0 - cells[(d, rho)][0][k][met]     # back to eps
                            if np.isfinite(val) and val > 0:
                                Z[i, j] = -np.log10(val)
                planes[(met, k)] = Z

    cmap = diverging_cmap()
    cmap.set_bad((0.0, 0.0, 0.0, 0.0))     # see the hatched axes patch below
    fig, axes = plt.subplots(4, 3, figsize=(11.4, 12.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.115, right=0.855, top=0.905, bottom=0.085,
                        hspace=0.16, wspace=0.09)

    for bi, block in enumerate(blocks):
        vals = np.concatenate([planes[(m, k)][np.isfinite(planes[(m, k)])]
                               for m, _ in block for k in snaps])
        top = float(max(np.nanmax(vals), 0.05)) if vals.size else 1.0
        norm = TwoSlopeNorm(vmin=floors[bi], vcenter=0.0, vmax=top)
        clipped = bool(vals.size and np.nanmin(vals) < floors[bi])
        for si, (met, title) in enumerate(block):
            ri = 2 * bi + si
            for ci, k in enumerate(snaps):
                ax = axes[ri][ci]
                im = ax.imshow(np.clip(planes[(met, k)], floors[bi], top),
                               origin="lower", aspect="auto", cmap=cmap, norm=norm,
                               extent=(rhos[0] - 0.5, rhos[-1] + 0.5,
                                       d_fs[0] - cfg["HEAT_DF_STEP"] / 2,
                                       d_fs[-1] + cfg["HEAT_DF_STEP"] / 2))
                ax.set_xticks(rhos)
                ax.set_yticks(d_fs)
                ax.set_yticklabels([f"{d:.1f}" for d in d_fs])
                if ri == 0:
                    ax.set_title(f"$k={k}$ tasks", fontsize=10, color=INK, pad=6)
                if ri == 3:
                    ax.set_xlabel("splitness  $d_f/d_b$", fontsize=9.5, color=INK2)
                if ci == 0:
                    ax.set_ylabel(f"{title}\n\nread density $d_f$", fontsize=9.5,
                                  color=INK2)
                ax.tick_params(colors=MUTED, labelsize=8, length=0)
                # a cell with no anchor is left blank, and the blank has to be
                # unmistakable: a neutral fill alone reads as "landed on the
                # baseline", which is the one thing it does not mean
                ax.set_facecolor("#f3f2ee")
                ax.patch.set_hatch("///")
                ax.patch.set_edgecolor("#cdccc5")
                ax.patch.set_linewidth(0.0)
                for side in ax.spines:
                    ax.spines[side].set_color(AXIS)
                    ax.spines[side].set_linewidth(0.8)
        box = [axes[2 * bi][2].get_position(), axes[2 * bi + 1][2].get_position()]
        cax = fig.add_axes([0.875, box[1].y0, 0.016, box[0].y1 - box[1].y0])
        cb = fig.colorbar(im, cax=cax, extend="min" if clipped else "neither")
        cb.set_label("worse  $\\leftarrow$   $0$ = no better than the baseline   "
                     "$\\rightarrow$  better", fontsize=8.5, color=INK2)
        cb.ax.tick_params(colors=MUTED, labelsize=8)
        cb.outline.set_edgecolor(AXIS)

    fig.suptitle("density against splitness, three snapshots of one run   "
                 f"($N={cfg['N']}$, $N_{{in}}={cfg['D']}$, $T={cfg['K_HEAT']}$, "
                 f"{cfg['SEEDS']} draws per cell)", fontsize=11, color=INK, y=0.972)
    fig.text(0.5, 0.012,
             "colour is $-\\log_{10}$ of the metric: same baseline as $1-\\varepsilon=0$, "
             "same sign, but $\\varepsilon$ runs to $10^{62}$ at the split corner and no "
             "linear scale shows that\nand the mild corner at once.  the arrow marks cells "
             f"past the floor ($\\varepsilon$ above $10^{{{-int(cfg['HEAT_EPS_FLOOR'])}}}$).  "
             "hatched: the retention lag $\\Delta=\\lceil N/n_b\\rfloor+5$ reaches back past "
             "the start of the stream, so the anchor does not exist.\n"
             "the unclipped numbers, as $1-\\varepsilon$ and $1-M$, are in the .txt.",
             ha="center", va="bottom", fontsize=8, color=MUTED, linespacing=1.5)
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)


# ==================================================================== chrome

def _chrome(ax):
    """Hairline solid grid one shade off the surface, no top or right spine."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(True, color=GRID, linewidth=0.7, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.set_facecolor(SURFACE)


# ================================================================== self-test

def solve_check(cfg, d_f=0.3, rho=2.0, seed=11):
    """The streamed route against the exact triangular solve, eq. (27) and (30).

    Two routes that share no arithmetic: walking the protocol one task at a time,
    and solving (I+L)R = Theta from the coupling alone with nothing trained.  The
    number this returns is the relative gap between them, and it is quoted on the
    figures that rest on the stream.
    """
    N, D, K = cfg["N"], cfg["D"], cfg["K_CHECK"]
    n_f, n_b = counts(N, d_f, rho)
    Delta = retention_lag(N, n_b, cfg["LAG_PAD"])
    v, F, B, Theta = draw(N, D, K, n_f, n_b, seed=seed)
    e_tr, e_ret, _ = run_stream(v, F, B, Theta, Delta)
    G = coupling(v, F, B)
    R = solve_stream(G, Theta)
    norm = (Theta ** 2).sum(1)
    e_tr_d = (R ** 2).sum(1) / norm
    e_ret_d = (lagged(G, R, Delta) ** 2).sum(1) / norm
    ok = np.isfinite(e_ret)
    return float(max(np.abs(e_tr - e_tr_d).max() / np.abs(e_tr_d).max(),
                     np.abs(e_ret[ok] - e_ret_d[ok]).max()
                     / np.abs(e_ret_d[ok]).max()))


def self_test(cfg, verbose=True):
    """Three routes checked against each other, none of which shares code with the
    others: the stream against eq. (27), the integrated flow against eq. (16), and
    the mean-field formulas against the recursion they came from."""
    fails = 0

    def rep(name, got, tol):
        nonlocal fails
        ok = np.all(np.abs(got) <= tol)
        fails += not ok
        if verbose:
            print(f"  [{'ok  ' if ok else 'FAIL'}] {name:<58s} "
                  f"{float(np.max(np.abs(got))):.3e}")

    if verbose:
        print("-- the stream against the exact triangular solve --")
    for d_f, rho in ((0.5, 1.0), (0.3, 2.0), (0.2, 3.0)):
        rep(f"d_f={d_f}, rho={rho}: streamed eps == eq. (27)/(30)",
            solve_check(cfg, d_f, rho), 1e-9)

    if verbose:
        print("-- the integrated flow against eq. (16) --")
    res = flow_case({**cfg, "K_FLOW": 6, "PER_TASK": 60})
    rep("integrated residuals == direct solve", res["box_gap"], 1e-7)
    rep("integrated loss == closed form, per task height", res["flow_gap"], 1e-6)
    rep("stream residuals == direct solve", res["solve_gap"], 1e-12)

    if verbose:
        print("-- the mean field against its own recursion --")
    rng = np.random.default_rng(3)
    for d_f in (0.5, 0.2):
        K, D, M = 200, 32, 3000
        p = 1.0 - d_f
        Th = rng.standard_normal((M, K, D))
        Th /= np.linalg.norm(Th, axis=2, keepdims=True)
        S = np.zeros((M, D))
        Ss = np.empty((M, K, D))
        Rm = np.empty((M, K, D))
        for t in range(K):
            Rm[:, t] = Th[:, t] - d_f * S
            S = p * S + Th[:, t]
            Ss[:, t] = S
        for Delta in (-1, 6, 20):
            for t in (3, 40, 120):
                if Delta >= 0 and t + Delta >= K:
                    continue
                if Delta == -1:
                    got = (Rm[:, t - 1] ** 2).sum(1).mean()
                else:
                    got = (d_f ** 2 * ((Ss[:, t + Delta - 1]
                                        - Ss[:, t - 1]) ** 2).sum(1)).mean()
                want = mf_eps(d_f, t, Delta)
                rep(f"  d_f={d_f}, Delta={Delta:3d}, t={t:3d}: eps = {want:.4f}",
                    got / want - 1.0, 0.03)
        for st in (1, 25, 90):
            got = (1.0 + d_f ** 2 * (Ss[:, st - 1] ** 2).sum(1)).mean()
            rep(f"  d_f={d_f}, control read out of state {st:3d}: "
                f"eps = {mf_ctrl(d_f, st):.4f}",
                got / mf_ctrl(d_f, st) - 1.0, 0.03)

    if verbose:
        print("-- the aggregator --")
    rep("geo() is the exponential of the mean log",
        geo([1.0, 100.0]) - 10.0, 1e-12)
    rep("geo() drops non-finite entries", geo([4.0, np.inf, np.nan]) - 4.0, 1e-12)
    bins = log_bins(1000, 12)
    rep("log_bins tile 1..K exactly once",
        [bins[0][0] - 1, bins[-1][1] - 1001]
        + [b[1] - a[0] for a, b in zip(bins[1:], bins[:-1])], 0)

    if verbose:
        print(f"\n  {'all checks passed' if not fails else str(fails) + ' FAILED'}")
    return fails


# ======================================================================= main

def _grid(lo, hi, step):
    """An inclusive grid, rounded, so that 0.2..1.0 by 0.1 has exactly nine points."""
    n = int(round((hi - lo) / step)) + 1
    return [round(lo + i * step, 9) for i in range(n)]


def main(argv):
    here = Path(__file__).resolve().parent
    want = {a for a in argv if a in {"1", "2", "3", "4"}}
    test_only = "--test" in argv
    cfg = CONFIG

    print("=" * 76)
    print("  neuronal split gating -- experiments and figures")
    print(f"  N={cfg['N']}  N_in={cfg['D']}  seeds={cfg['SEEDS']}")
    print("=" * 76)
    if self_test(cfg) and not test_only:
        print("\n  self-test failed; not drawing anything")
        return 1
    if test_only:
        return 0

    gap = solve_check(cfg)
    lines = [f"stream vs the exact solve (27) at K={cfg['K_CHECK']}: {gap:.3e}",
             "all figures report 1 - (geometric mean of eps over draws and anchors)",
             ""]
    cols = ("ret_eps", "ret_ctrl", "tr_eps", "tr_ctrl")
    head = " ".join(f"{m:>13s}" for m in
                    ("ret 1-eps", "ret 1-M", "tr 1-eps", "tr 1-M"))

    if not want or "1" in want:
        print("\nfigure 1: the flow ...")
        res = flow_case(cfg)
        fig_flow(cfg, res, here / "fig1_flow.png")
        lines += [f"fig1  n_f={res['n_f']} n_b={res['n_b']}  "
                  f"integrated vs closed form {res['flow_gap']:.2e}  "
                  f"integrated residuals vs eq. (27) {res['box_gap']:.2e}  "
                  f"stream vs eq. (27) {res['solve_gap']:.2e}", ""]

    if not want or "2" in want:
        print("figure 2: metric against task index ...")
        cases = {d: case_vs_tasks(cfg, d) for d in cfg["DENSITIES"]}
        fig_vs_tasks(cfg, cases, here / "fig2_vs_tasks.png", gap)
        lines.append(f"fig2  at the last logarithmic bin (K={cfg['K_TASKS']})")
        lines.append(f"  {'d_f':>5s} {'Delta':>6s} " + head)
        for d in cfg["DENSITIES"]:
            r = cases[d]
            lines.append(f"  {d:5.2f} {r['Delta']:6d} "
                         + " ".join(f"{1.0 - r['last'][k]:13.4f}" for k in cols))
        lines.append("")

    if not want or "3" in want:
        print("figure 3: metric against density ...")
        grid = {}
        for rho in cfg["RHOS"]:
            for d in _grid(cfg["D"] / cfg["N"], 1.0, cfg["DF_STEP"]):
                try:
                    grid[(d, rho)] = case_vs_density(cfg, d, rho)
                except ValueError:
                    pass
        fig_vs_density(cfg, grid, here / "fig3_vs_density.png", gap)
        lines.append("fig3  steady state, second half of the stream")
        lines.append(f"  {'d_f':>5s} {'rho':>4s} {'n_b':>4s} {'Delta':>6s} " + head)
        for (d, rho), r in sorted(grid.items()):
            lines.append(f"  {d:5.2f} {rho:4.1f} {r['n_b']:4d} {r['Delta']:6d} "
                         + " ".join(f"{r[k]:13.4g}" for k in cols))
        lines.append("")

    if not want or "4" in want:
        print("figure 4: density against splitness ...")
        cells = {}
        for d in _grid(cfg["D"] / cfg["N"], 1.0, cfg["HEAT_DF_STEP"]):
            for rho in cfg["HEAT_RHOS"]:
                try:
                    cells[(d, rho)] = case_heat(cfg, d, rho)
                except ValueError:
                    pass
        fig_heatmaps(cfg, cells, here / "fig4_heatmaps.png")
        lines.append(f"fig4  {len(cells)} cells, snapshots {cfg['SNAPSHOTS']}, "
                     f"unclipped; the figure shows -log10 of these, floored "
                     f"at {cfg['HEAT_EPS_FLOOR']:g} and {cfg['HEAT_CTRL_FLOOR']:g}")
        lines.append(f"  {'d_f':>5s} {'rho':>4s} {'n_b':>4s} {'Delta':>6s} "
                     f"{'k':>6s} " + head)
        for (d, rho), (snap, Delta) in sorted(cells.items()):
            n_b = counts(cfg["N"], d, rho)[1]
            for k in cfg["SNAPSHOTS"]:
                lines.append(f"  {d:5.2f} {rho:4d} {n_b:4d} {Delta:6d} {k:6d} "
                             + " ".join(f"{snap[k][m]:13.4g}" for m in cols))
        lines.append("")

    (here / "neuronal_curves.txt").write_text("\n".join(lines))
    print(f"\nwritten to {here}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
