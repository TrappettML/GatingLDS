"""Experiments and figures for the neuronal split-gating note, `splitgating_neuronal.tex`.

Standalone: it imports nothing from `hadamard.py`, `lag_curves.py` or
`verify_neuronal.py` and writes no file any of those writes.  The model is the
note's, with gates that open and close whole neurons:

    F_t = diag(f_1..f_N),  B_t = diag(b_1..b_N),  b_i <= f_i,  both N x N
    yhat_t(x) = v^T F_t W x / sqrt(D)
    Wdot      = -B_t v (v^T F_t W - theta_t)                        eq. (6)

One gate pair per task serves the whole network, so the write mass c_t = v^T B_t v
is a scalar, the coupling Gamma_kt is a scalar, and the whole stream is one
triangular system of size K with the D columns of Theta as right-hand sides.

The quantity every figure reports is the residual itself, nothing built on top of it:

    transfer    r_t          = theta_t - thetahat_t^(t-1)           eq. (12)
    retention   r_t^(Delta)  = theta_t - thetahat_t^(t+Delta)       eq. (28)

both as ||r|| / ||theta_t||, which is dimensionless and needs no baseline of its
own: ||r|| = 0 is a solved task and ||r|| = 1 is a state worth no more than W = 0,
since thetahat = 0 there.  Above 1 the state is worse than not having trained.

    python3 neuronal_curves.py            # self-test, then all four figures
    python3 neuronal_curves.py --test     # the self-test alone
    python3 neuronal_curves.py 1 3        # only figures 1 and 3

Figures written next to this file:

    fig1_flow.png       the stream as it actually runs: integrated gradient flow
                        against the two closed forms, eq. (16) and eq. (27)
    fig2_vs_tasks.png   residual against task index, one trace per density
    fig3_vs_density.png residual against read density, one trace per split ratio
    fig4_heatmaps.png   density x splitness, three snapshots of one run to T = 1000
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

    # --- figure 2, residual against task index -----------------------------
    K_TASKS=5000,    # stream length
    DENSITIES=(0.5, 0.3, 0.2, 0.1),   # d_f = d_b, one trace each
    N_BINS_DECADE=12,  # logarithmic bins of anchors per decade.  Every anchor of
                     #   the stream is measured; a point is the geometric mean
                     #   over its bin and over the draws

    # --- figure 3, residual against density --------------------------------
    K_DENSITY=1500,  # stream length; the residual is pooled over the second half
    RHOS=(1.0, 2.0, 3.0),             # d_f / d_b, one trace each
    DF_STEP=0.05,    # read-density grid, from D/N to 1.0

    # --- figure 4, density against splitness -------------------------------
    K_HEAT=1000,     # one run per cell; the three columns are snapshots of it
    SNAPSHOTS=(10, 100, 1000),
    HEAT_DF_STEP=0.1,                 # read density, from D/N to 1.0
    HEAT_RHOS=tuple(range(1, 11)),    # splitness 1..10
    HEAT_WINDOW=0.1, # anchors pooled into a snapshot, as a fraction of k
    HEAT_CEIL=2.0,   # colour ceiling in log10||r||: a cell only has to read as
                     #   "a hundred times the baseline or worse".  the .txt keeps
                     #   the unclipped numbers

    # --- the self-test -----------------------------------------------------
    K_CHECK=600,     # stream length at which the streamed route is checked
                     #   against the exact triangular solve, eq. (27)
)


# ------------------------------------------------------------------ palette
# Categorical slots in the validated order; four is the most any panel uses and
# that set clears the adjacent colour-vision gates.  Aqua and yellow sit below 3:1
# on this surface, so every trace is also labelled at its right-hand end and
# identity is never carried by colour alone.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

# Diverging ramp for the heatmaps.  log10||r|| has a meaningful zero -- the state
# is worth exactly what W = 0 is worth -- with values either side, so one hue per
# sign and a neutral grey between.  Blue is below the baseline, red above it.
RAMP_BLUE = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
RAMP_RED = ("#fbd7d7", "#f3acac", "#e88484", "#e34948", "#c13232", "#992424", "#701919")
NEUTRAL = "#f0efec"


def diverging_cmap():
    """Blue (below the baseline) -> neutral grey -> red (above it).

    Equal step count per arm, and the lightness rises monotonically into the grey
    from either side, which is what makes the midpoint read as "nothing".
    """
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    stops = list(RAMP_BLUE[::-1]) + [NEUTRAL] + list(RAMP_RED)
    lum = [0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
           for c in (to_rgb(h) for h in stops)]
    rise, fall = lum[:len(RAMP_BLUE) + 1], lum[len(RAMP_BLUE):]
    if not (all(np.diff(rise) > 0) and all(np.diff(fall) < 0)):
        raise ValueError("the diverging ramp is not monotone in lightness per arm")
    return LinearSegmentedColormap.from_list("better_worse", stops, N=512)


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
    the family eq. (9) allows.  Teacher rows are unit vectors, so ||r|| is already
    the residual relative to the task's own scale and ||r|| = 1 is exactly the
    null state W = 0.
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(N) / np.sqrt(N)                     # eq. (1)
    rank = np.argsort(np.argsort(rng.random((K, N)), axis=1), axis=1)
    F = (rank < n_f).astype(float)
    B = (rank < n_b).astype(float)
    Theta = rng.standard_normal((K, D))
    Theta /= np.linalg.norm(Theta, axis=1, keepdims=True)
    return v, F, B, Theta


# ============================================================= the stream

def run_stream(v, F, B, Theta, Delta):
    """Walk the stream once, reading the anchor back at two lags as it goes.

    The update is the endpoint of the within-task flow, eq. (16), so no integration
    is needed: the flow has the fixed left factor B_t v, the state moves on a line,
    and the endpoint is the line's end.  State W^s is the state after s tasks have
    trained, W^0 = 0, and task j (0-based) trains into W^{j+1}.

    Two residuals come back, both as ||r|| / ||theta_j||:

        tr[j]    r_j = theta_j - thetahat_j^(j), eq. (12) -- the task read out of
                 the state just before it trains, so 1 means the stream has left
                 nothing useful for it and 0 means it is already solved
        ret[j]   r_j^(Delta), eq. (28) -- the same task read out of W^{j+1+Delta},
                 so 0 means it is still solved and 1 means it has been undone

    Rows of `ret` with no such state are left as NaN.  Only a rolling window of
    Delta+1 gates is needed, so the cost is O(K N D) in time and O(N D) in space
    whatever K is.
    """
    K, N = F.shape
    w = v ** 2
    c = B @ w                                             # write masses, eq. (15)
    if (c <= 0).any():
        raise ValueError("a task has zero write mass; eq. (19) is undefined there")

    W = np.zeros((N, Theta.shape[1]))
    tr = np.empty(K)
    ret = np.full(K, np.nan)
    norm = np.linalg.norm(Theta, axis=1)

    for t in range(K):
        r = Theta[t] - (F[t] * v) @ W                     # eq. (12)
        tr[t] = np.linalg.norm(r) / norm[t]
        W += np.outer(B[t] * v, r) / c[t]                 # eq. (16)
        j = t - Delta
        if j >= 0:
            ret[j] = np.linalg.norm(Theta[j] - (F[j] * v) @ W) / norm[j]
    return tr, ret


def run_stream_snapshots(v, F, B, Theta, Delta, snapshots, window):
    """The same walk, recording only near a few states: figure 4's three columns.

    A snapshot at k is the state W^k.  The transfer reading there is task k-1 read
    out of W^{k-1} and the retention reading is task k-1-Delta read out of W^k,
    both pooled over the `window` anchors ending there so that one heavy-tailed
    draw does not set the cell.  A retention reading needs its anchor to exist, so
    cells with Delta >= k come back empty rather than as a shorter lag.
    """
    K, N = F.shape
    w = v ** 2
    c = B @ w
    W = np.zeros((N, Theta.shape[1]))
    norm = np.linalg.norm(Theta, axis=1)

    snaps = sorted(int(s) for s in snapshots)
    want_tr, want_ret = {}, {}
    for k in snaps:
        nw = max(1, int(round(window * k)))
        for j in range(max(0, k - nw), k):                       # transfer anchors
            want_tr.setdefault(j, []).append(k)
        for j in range(max(0, k - nw - Delta), k - Delta):       # retention anchors
            if j >= 0:
                want_ret.setdefault(j + 1 + Delta, []).append((j, k))
    out = {k: {"tr": [], "ret": []} for k in snaps}

    for t in range(K):
        r = Theta[t] - (F[t] * v) @ W
        if t in want_tr:
            val = np.linalg.norm(r) / norm[t]
            for k in want_tr[t]:
                out[k]["tr"].append(val)
        W += np.outer(B[t] * v, r) / c[t]
        s = t + 1
        if s in want_ret:
            for j, k in want_ret[s]:
                out[k]["ret"].append(
                    np.linalg.norm(Theta[j] - (F[j] * v) @ W) / norm[j])
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

def mf_resid(d_f, t, Delta):
    """The root-mean-square residual with Gamma replaced by its mean, eq. (21).

    For gates drawn independently each task E[Gamma_kt] = d_f whatever the split
    ratio is, so the recursion eq. (24) collapses to a scalar one.  With
    S_t = sum_{s<=t} r_s it reads S_t = (1-d_f) S_{t-1} + theta_t, so with p = 1-d_f

        r_t          = theta_t - d_f S_{t-1}
        r_t^(Delta)  = -d_f (S_{t+Delta} - S_t)
                     = -d_f [ (p^Delta - 1) S_t
                              + sum_{s=t+1}^{t+Delta} p^{t+Delta-s} theta_s ]

    and for teachers that are independent, isotropic and of unit norm the two
    pieces of the second line are uncorrelated, so

        E||r_t||^2         = 1 + d_f (1 - p^{2(t-1)}) / (2 - d_f)
        E||r_t^(Delta)||^2 = d_f [ (1 - p^{2 Delta})
                                   + (1 - p^Delta)^2 (1 - p^{2t}) ] / (2 - d_f)

    using 1 - p^2 = d_f (2 - d_f).  This returns the square root of those, so that
    it is in the same units as the measured curves.  `t` is 1-based, as the note
    counts tasks, and Delta = -1 selects the transfer arm.

    Two things it is not.  It is free of the split ratio, because E[Gamma] is:
    everything the split does to the residual lives in second moments and in the
    stability of eq. (17), neither of which survives replacing Gamma by its mean.

    And it is NOT a bound.  The residuals themselves depend on Gamma, through
    (I + L)^{-1} in eq. (27), so ||r||^2 is rational in Gamma and convexity says
    nothing about the composite.  What happens in fact is that the variance of
    Gamma adds to the diagonal of the read-back and the measured residual runs
    above this curve nearly everywhere -- but "nearly", not "always", and the
    exceptions are at d_f near 1.

    One calibration point is free.  At d_f = 1 the read gate is the identity, so
    Gamma is exactly the all-ones matrix and nothing is being approximated:
    r_t = theta_t - theta_{t-1} in the mean field and in the simulation alike.
    Any gap left there is the estimator and not the physics -- these curves are
    geometric means while this is a root-mean-square -- and it is worth
    exp(-1/4D) to leading order: 1.38 against 1.41 at D = 12.
    """
    p = 1.0 - d_f
    t = np.asarray(t, dtype=float)
    if Delta == -1:
        return np.sqrt(1.0 + d_f * (1.0 - p ** (2 * (t - 1))) / (2.0 - d_f))
    if Delta < 0:
        raise ValueError(f"Delta={Delta}: only -1 and Delta >= 0 are defined here")
    return np.sqrt(d_f * ((1.0 - p ** (2 * Delta))
                          + (1.0 - p ** Delta) ** 2 * (1.0 - p ** (2 * t)))
                   / (2.0 - d_f))


def retention_lag(N, n_b, pad):
    """Delta = round(N / n_b) + pad: one turnover of the write pool, plus a little."""
    return int(round(N / n_b)) + int(pad)


def geo(x):
    """Geometric mean of ||r|| over draws and anchors, the way this file aggregates.

    Once the stream is unstable the residual is heavy-tailed across draws: at
    rho = 3 a single draw can sit seven orders of magnitude above the rest, so an
    arithmetic mean reports that draw and nothing else.  log||r|| is the
    near-Gaussian variable, so the mean is taken there and exponentiated back.
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
    secondary ink rather than the series colour -- the line arriving at the label
    is what carries identity -- and are nudged apart in axis coordinates when they
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


def _baseline(ax):
    """The line at ||r|| = 1: the state is worth exactly what W = 0 is worth."""
    ax.axhline(1.0, color=AXIS, linewidth=1.0, zorder=1)


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
    tau = span[:, None] * np.linspace(0.0, 1.0, per)[None, :]

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
    return dict(
        time=start[:, None] + tau, K=K, n_f=n_f, n_b=n_b,
        loss_int=loss_int, loss_stream=loss_stream, loss_direct=loss_direct,
        box=0.5 * (R_direct ** 2).sum(1),
        flow_gap=float(np.abs(loss_int - loss_stream).max()
                       / loss_stream[:, :1].max()),
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
    ax.set_ylim(res["box"].min() * 1e-15, res["box"].max() * 3.0)
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

def case_vs_tasks(cfg, d_f):
    """One density, every anchor of a stream of K_TASKS, binned logarithmically.

    Every anchor is measured, so a bin near the end of the stream pools hundreds of
    them; that, and not a longer stream, is what makes the tail of the curve smooth.
    """
    N, D, K = cfg["N"], cfg["D"], cfg["K_TASKS"]
    n_f, n_b = counts(N, d_f, 1.0)                 # d_f = d_b on this figure
    Delta = retention_lag(N, n_b, cfg["LAG_PAD"])
    raw = {"tr": [], "ret": []}
    for s in range(cfg["SEEDS"]):
        v, F, B, Theta = draw(N, D, K, n_f, n_b, seed=100 + s)
        tr, ret = run_stream(v, F, B, Theta, Delta)
        raw["tr"].append(tr)
        raw["ret"].append(ret)
    stack = {k: np.vstack(a) for k, a in raw.items()}            # (seeds, K)

    out = {"tr": [], "ret": [], "centre": []}
    for a, b in log_bins(K, cfg["N_BINS_DECADE"]):
        idx = np.arange(a, b)                                    # 1-based anchors
        out["centre"].append(float(np.exp(np.log(idx).mean())))
        for k in ("tr", "ret"):
            out[k].append(geo(stack[k][:, idx - 1]))
    out = {k: np.array(x) for k, x in out.items()}
    out["Delta"], out["n_b"] = Delta, n_b
    out["last"] = {k: float(out[k][np.isfinite(out[k])][-1]) for k in ("tr", "ret")}
    return out


def fig_vs_tasks(cfg, cases, path, solve_gap):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), sharex=True)
    arms = (("ret", r"retention:   $\|r_t^{(\Delta)}\| / \|\theta_t\|$",
             "the task read back $\\Delta$ tasks after it trained"),
            ("tr", r"transfer:   $\|r_t\| / \|\theta_t\|$",
             "the task read out of the state just before it trains"))
    handles = []
    for ci, (arm, ylab, sub) in enumerate(arms):
        ax = axes[ci]
        ends = []
        for k, d_f in enumerate(cfg["DENSITIES"]):
            res = cases[d_f]
            Delta = res["Delta"]
            x, y = res["centre"], res[arm]
            ok = np.isfinite(y)
            ln, = ax.plot(x[ok], y[ok], color=SERIES[k], linewidth=1.8,
                          solid_capstyle="round", zorder=3)
            th = mf_resid(d_f, x[ok], -1 if arm == "tr" else Delta)
            ax.plot(x[ok], np.broadcast_to(th, x[ok].shape), color=SERIES[k],
                    linewidth=1.3, linestyle=(0, (1, 2)), zorder=2)
            ends.append((x[ok][-1], y[ok][-1], f"{d_f}"))
            if ci == 0:
                ln.set_label(f"$d_f=d_b={d_f}$   ($\\Delta={Delta}$)")
                handles.append(ln)
        ax.set_xscale("log")
        _baseline(ax)
        ax.set_title(sub, fontsize=10, color=INK, pad=8)
        ax.set_ylabel(ylab, fontsize=10, color=INK2)
        ax.set_xlabel("tasks trained", fontsize=9.5, color=INK2)
        _chrome(ax)
        ax.annotate("$1$ = no better than $W=0$", xy=(0.0, 1.0),
                    xycoords=("axes fraction", "data"), xytext=(4, 3),
                    textcoords="offset points", ha="left", fontsize=7.5,
                    color=MUTED)
        _end_labels(ax, ends)

    style = [Line2D([], [], color=MUTED, lw=1.8, label="simulation"),
             Line2D([], [], color=MUTED, lw=1.3, ls=(0, (1, 2)),
                    label="mean field, $\\Gamma\\to d_f$ (r.m.s.)")]
    fig.legend(handles=handles + style, frameon=False, fontsize=8.5, ncol=6,
               loc="upper center", bbox_to_anchor=(0.5, 0.935), labelcolor=INK2)
    fig.suptitle("the residual, against how far into the stream the task sits   "
                 f"($N={cfg['N']}$, $N_{{in}}={cfg['D']}$, $K={cfg['K_TASKS']}$, "
                 f"{cfg['SEEDS']} draws;  0 = solved, 1 = no better than $W=0$)",
                 fontsize=11, color=INK, y=0.995)
    fig.text(0.5, 0.012,
             "solid: the geometric mean of $\\|r\\|$ over each logarithmic bin of anchors "
             "and over the draws.   dotted: the root-mean-square the mean field\n"
             "predicts.  it is not a bound -- the residuals depend on $\\Gamma$ themselves, "
             "through $(I+L)^{-1}$ -- but its variance adds to the read-back,\nso the "
             "measured curve runs above it."
             f"      stream vs the exact solve (27) at $K={cfg['K_CHECK']}$: "
             f"{solve_gap:.1e}",
             ha="center", va="bottom", fontsize=8, color=MUTED, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.125, 1, 0.9))
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


# ==================================================================== figure 3

def case_vs_density(cfg, d_f, rho):
    """Steady-state residual at one (d_f, rho): the stream's second half, pooled."""
    N, D, K = cfg["N"], cfg["D"], cfg["K_DENSITY"]
    n_f, n_b = counts(N, d_f, rho)
    Delta = retention_lag(N, n_b, cfg["LAG_PAD"])
    lo = K // 2
    pool = {"tr": [], "ret": []}
    for s in range(cfg["SEEDS"]):
        v, F, B, Theta = draw(N, D, K, n_f, n_b, seed=200 + s)
        tr, ret = run_stream(v, F, B, Theta, Delta)
        pool["tr"].append(tr[lo:])
        pool["ret"].append(ret[lo:][np.isfinite(ret[lo:])])
    out = {k: geo(np.concatenate(x)) for k, x in pool.items()}
    out["Delta"], out["n_b"], out["t_mid"] = Delta, n_b, 0.75 * K
    return out


def fig_vs_density(cfg, grid, path, solve_gap):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    d_fs = sorted({d for d, _ in grid})
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), sharex=True)
    arms = (("ret", r"retention:   $\|r_t^{(\Delta)}\| / \|\theta_t\|$",
             "read back $\\Delta=\\lceil N/n_b\\rfloor+5$ tasks later"),
            ("tr", r"transfer:   $\|r_t\| / \|\theta_t\|$",
             "read out of the state just before the task trains"))
    handles = []
    for ci, (arm, ylab, sub) in enumerate(arms):
        ax = axes[ci]
        ends = []
        for k, rho in enumerate(cfg["RHOS"]):
            xs = [d for d in d_fs if (d, rho) in grid]
            ys = [grid[(d, rho)][arm] for d in xs]
            ln, = ax.plot(xs, ys, color=SERIES[k], linewidth=1.8, zorder=3,
                          solid_capstyle="round")
            ends.append((xs[-1], ys[-1], f"{rho:g}"))
            if ci == 0:
                ln.set_label(f"$d_f/d_b={rho:g}$")
                handles.append(ln)
        # the mean field cannot see the split: E[Gamma] = d_f whatever d_b is,
        # so one dotted curve serves all three traces
        th = [mf_resid(d, grid[(d, cfg["RHOS"][0])]["t_mid"],
                       -1 if arm == "tr" else grid[(d, cfg["RHOS"][0])]["Delta"])
              for d in d_fs]
        ax.plot(d_fs, th, color=MUTED, linewidth=1.4, linestyle=(0, (1, 2)), zorder=2)
        ax.set_yscale("log")
        _baseline(ax)
        ax.set_title(sub, fontsize=10, color=INK, pad=8)
        ax.set_ylabel(ylab, fontsize=10, color=INK2)
        ax.set_xlabel("read density $d_f$", fontsize=9.5, color=INK2)
        _chrome(ax)
        ax.annotate("$1$ = no better than $W=0$", xy=(0.0, 1.0),
                    xycoords=("axes fraction", "data"), xytext=(4, 3),
                    textcoords="offset points", ha="left", fontsize=7.5,
                    color=MUTED)
        _end_labels(ax, ends)

    style = [Line2D([], [], color=MUTED, lw=1.8, label="simulation"),
             Line2D([], [], color=MUTED, lw=1.4, ls=(0, (1, 2)),
                    label="mean field (r.m.s.), one curve for all three")]
    fig.legend(handles=handles + style, frameon=False, fontsize=8.5, ncol=5,
               loc="upper center", bbox_to_anchor=(0.5, 0.935), labelcolor=INK2)
    fig.suptitle("the residual, against how dense the read gate is   "
                 f"($N={cfg['N']}$, $N_{{in}}={cfg['D']}$, $K={cfg['K_DENSITY']}$, "
                 f"{cfg['SEEDS']} draws, second half of the stream)",
                 fontsize=11, color=INK, y=0.995)
    fig.text(0.5, 0.012,
             "one dotted curve serves all three ratios: $E[\\Gamma_{kt}]=d_f$ whatever "
             "$d_b$ is, so everything the split does lives in second moments and in\n"
             "the stability of eq. (17).   at $d_f=1$ the read gate is the identity, "
             "$\\Gamma$ is exactly all-ones and the two must agree; the $1.38$ against\n"
             "$1.41$ left there is the estimator, and the noise floor of the comparison."
             f"      stream vs the exact solve (27) at $K={cfg['K_CHECK']}$: "
             f"{solve_gap:.1e}",
             ha="center", va="bottom", fontsize=8, color=MUTED, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.125, 1, 0.9))
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


# ==================================================================== figure 4

def case_heat(cfg, d_f, rho):
    """One run to K_HEAT at one cell, read at the three snapshots, pooled over draws."""
    N, D, K = cfg["N"], cfg["D"], cfg["K_HEAT"]
    n_f, n_b = counts(N, d_f, rho)
    Delta = retention_lag(N, n_b, cfg["LAG_PAD"])
    pool = {k: {"tr": [], "ret": []} for k in cfg["SNAPSHOTS"]}
    for s in range(cfg["SEEDS"]):
        v, F, B, Theta = draw(N, D, K, n_f, n_b, seed=300 + s)
        snap = run_stream_snapshots(v, F, B, Theta, Delta, cfg["SNAPSHOTS"],
                                    cfg["HEAT_WINDOW"])
        for k in cfg["SNAPSHOTS"]:
            pool[k]["tr"] += list(snap[k]["tr"])
            pool[k]["ret"] += list(snap[k]["ret"])
    return {k: {m: geo(x) for m, x in d.items()} for k, d in pool.items()}, Delta


def fig_heatmaps(cfg, cells, path):
    """Two rows, transfer and retention, at three snapshots of one run.

    The colour carries $\\log_{10}\\|r\\|$ rather than $\\|r\\|$: across this grid the
    residual runs from 0.5 to $10^{31}$, which no linear scale shows, and the log
    has the same zero -- $\\|r\\|=1$, the state worth exactly what $W=0$ is worth --
    with blue below it and red above.  Both rows share one scale; they are two
    readings of one run and have to be comparable.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    d_fs = sorted({d for d, _ in cells})
    rhos = list(cfg["HEAT_RHOS"])
    snaps = list(cfg["SNAPSHOTS"])
    rows = (("tr", r"transfer,  $\log_{10}\|r_t\|$"),
            ("ret", r"retention,  $\log_{10}\|r_t^{(\Delta)}\|$"))

    planes = {}
    for met, _ in rows:
        for k in snaps:
            Z = np.full((len(d_fs), len(rhos)), np.nan)
            for i, d in enumerate(d_fs):
                for j, rho in enumerate(rhos):
                    if (d, rho) in cells:
                        val = cells[(d, rho)][0][k][met]
                        if np.isfinite(val) and val > 0:
                            Z[i, j] = np.log10(val)
            planes[(met, k)] = Z

    cmap = diverging_cmap()
    cmap.set_bad((0.0, 0.0, 0.0, 0.0))     # see the hatched axes patch below
    vals = np.concatenate([planes[(m, k)][np.isfinite(planes[(m, k)])]
                           for m, _ in rows for k in snaps])
    ceil = cfg["HEAT_CEIL"]
    norm = TwoSlopeNorm(vmin=float(min(np.nanmin(vals), -0.05)), vcenter=0.0,
                        vmax=ceil)
    clipped = bool(np.nanmax(vals) > ceil)

    fig, axes = plt.subplots(2, 3, figsize=(11.6, 7.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.105, right=0.865, top=0.865, bottom=0.145,
                        hspace=0.17, wspace=0.09)
    for ri, (met, title) in enumerate(rows):
        for ci, k in enumerate(snaps):
            ax = axes[ri][ci]
            im = ax.imshow(np.clip(planes[(met, k)], None, ceil), origin="lower",
                           aspect="auto", cmap=cmap, norm=norm,
                           extent=(rhos[0] - 0.5, rhos[-1] + 0.5,
                                   d_fs[0] - cfg["HEAT_DF_STEP"] / 2,
                                   d_fs[-1] + cfg["HEAT_DF_STEP"] / 2))
            ax.set_xticks(rhos)
            ax.set_yticks(d_fs)
            ax.set_yticklabels([f"{d:.1f}" for d in d_fs])
            if ri == 0:
                ax.set_title(f"$k={k}$ tasks", fontsize=10, color=INK, pad=6)
            if ri == 1:
                ax.set_xlabel("splitness  $d_f/d_b$", fontsize=9.5, color=INK2)
            if ci == 0:
                ax.set_ylabel(f"{title}\n\nread density $d_f$", fontsize=9.5,
                              color=INK2)
            ax.tick_params(colors=MUTED, labelsize=8, length=0)
            # a cell with no anchor is left blank, and the blank has to be
            # unmistakable: a neutral fill reads as "landed on the baseline",
            # which is the one thing it does not mean
            ax.set_facecolor("#f3f2ee")
            ax.patch.set_hatch("///")
            ax.patch.set_edgecolor("#cdccc5")
            ax.patch.set_linewidth(0.0)
            for side in ax.spines:
                ax.spines[side].set_color(AXIS)
                ax.spines[side].set_linewidth(0.8)

    cax = fig.add_axes([0.885, 0.145, 0.016, 0.72])
    cb = fig.colorbar(im, cax=cax, extend="max" if clipped else "neither")
    cb.set_label("$\\log_{10}\\|r\\|$      $0$ = no better than $W=0$,  "
                 "below = better,  above = worse", fontsize=8.5, color=INK2)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_edgecolor(AXIS)

    fig.suptitle("density against splitness, three snapshots of one run   "
                 f"($N={cfg['N']}$, $N_{{in}}={cfg['D']}$, $T={cfg['K_HEAT']}$, "
                 f"{cfg['SEEDS']} draws per cell)", fontsize=11, color=INK, y=0.965)
    fig.text(0.5, 0.012,
             "each cell is the geometric mean of $\\|r\\|$ over the draws and over the "
             f"anchors in the {cfg['HEAT_WINDOW']:.0%} window ending at $k$.  the scale "
             f"stops at $10^{{{ceil:g}}}$ and the arrow marks cells past it: at the split "
             "corner\n$\\|r\\|$ reaches $10^{31}$, which no scale resolves against a band "
             "of width one.  hatched: the retention lag $\\Delta=\\lceil N/n_b\\rfloor+5$ "
             "reaches back past the start of the stream,\nso the anchor does not exist.  "
             "the unclipped numbers are in the .txt.",
             ha="center", va="bottom", fontsize=8, color=MUTED, linespacing=1.5)
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


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
    tr, ret = run_stream(v, F, B, Theta, Delta)
    G = coupling(v, F, B)
    R = solve_stream(G, Theta)
    norm = np.linalg.norm(Theta, axis=1)
    tr_d = np.linalg.norm(R, axis=1) / norm
    ret_d = np.linalg.norm(lagged(G, R, Delta), axis=1) / norm
    ok = np.isfinite(ret)
    return float(max(np.abs(tr - tr_d).max() / tr_d.max(),
                     np.abs(ret[ok] - ret_d[ok]).max() / ret_d[ok].max()))


def self_test(cfg, verbose=True):
    """Three routes checked against each other, none of which shares code with the
    others: the stream against eq. (27), the integrated flow against eq. (16), and
    the mean-field formula against the recursion it came from."""
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
        rep(f"d_f={d_f}, rho={rho}: streamed ||r|| == eq. (27)/(30)",
            solve_check(cfg, d_f, rho), 1e-9)

    if verbose:
        print("-- the residual at the two reference states --")
    v, F, B, Theta = draw(cfg["N"], cfg["D"], 40, 20, 10, seed=5)
    tr, ret = run_stream(v, F, B, Theta, 3)
    rep("task 1 inherits W = 0, so ||r_1|| / ||theta_1|| = 1", tr[0] - 1.0, 1e-12)
    tr0, ret0 = run_stream(v, F, B, Theta, 0)
    rep("Delta = 0 is a solved task, ||r^(0)|| = 0", ret0[:-1], 1e-12)

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
                want = mf_resid(d_f, t, Delta)
                rep(f"  d_f={d_f}, Delta={Delta:3d}, t={t:3d}: ||r|| = {want:.4f}",
                    np.sqrt(got) / want - 1.0, 0.02)

    if verbose:
        print("-- the mean field is exact at d_f = 1, where Gamma is all-ones --")
    v, F, B, Theta = draw(cfg["N"], cfg["D"], 400, cfg["N"], cfg["N"], seed=9)
    rep("d_f = 1 gives Gamma == 1 everywhere",
        coupling(v, F, B) - 1.0, 1e-12)
    tr, ret = run_stream(v, F, B, Theta, 4)
    direct = np.linalg.norm(Theta[1:] - Theta[:-1], axis=1)
    rep("  and r_t == theta_t - theta_{t-1}", tr[1:] - direct, 1e-12)
    rep("  r.m.s. of it == the mean field, sqrt(2)",
        np.sqrt((tr[1:] ** 2).mean()) / mf_resid(1.0, 200, -1) - 1.0, 0.04)
    rep("  the geometric mean sits below it by exp(-1/4D)",
        geo(tr[1:]) / (np.sqrt(2.0) * np.exp(-1.0 / (4 * cfg["D"]))) - 1.0, 0.02)

    if verbose:
        print("-- the aggregator --")
    rep("geo() is the exponential of the mean log", geo([1.0, 100.0]) - 10.0, 1e-12)
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
             "every number below is the geometric mean of ||r|| / ||theta_t|| over",
             "draws and anchors.  0 = solved, 1 = no better than W = 0.",
             ""]
    head = f"{'retention':>14s} {'transfer':>14s}"

    if not want or "1" in want:
        print("\nfigure 1: the flow ...")
        res = flow_case(cfg)
        fig_flow(cfg, res, here / "fig1_flow.png")
        lines += [f"fig1  n_f={res['n_f']} n_b={res['n_b']}  "
                  f"integrated vs closed form {res['flow_gap']:.2e}  "
                  f"integrated residuals vs eq. (27) {res['box_gap']:.2e}  "
                  f"stream vs eq. (27) {res['solve_gap']:.2e}", ""]

    if not want or "2" in want:
        print("figure 2: residual against task index ...")
        cases = {d: case_vs_tasks(cfg, d) for d in cfg["DENSITIES"]}
        fig_vs_tasks(cfg, cases, here / "fig2_vs_tasks.png", gap)
        lines.append(f"fig2  at the last logarithmic bin (K={cfg['K_TASKS']})")
        lines.append(f"  {'d_f':>5s} {'n_b':>4s} {'Delta':>6s} " + head)
        for d in cfg["DENSITIES"]:
            r = cases[d]
            lines.append(f"  {d:5.2f} {r['n_b']:4d} {r['Delta']:6d} "
                         f"{r['last']['ret']:14.4f} {r['last']['tr']:14.4f}")
        lines.append("")

    if not want or "3" in want:
        print("figure 3: residual against density ...")
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
                         f"{r['ret']:14.4g} {r['tr']:14.4g}")
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
                     f"unclipped (the figure shows log10 of these, capped at "
                     f"{cfg['HEAT_CEIL']:g})")
        lines.append(f"  {'d_f':>5s} {'rho':>4s} {'n_b':>4s} {'Delta':>6s} "
                     f"{'k':>6s} " + head)
        for (d, rho), (snap, Delta) in sorted(cells.items()):
            n_b = counts(cfg["N"], d, rho)[1]
            for k in cfg["SNAPSHOTS"]:
                lines.append(f"  {d:5.2f} {rho:4d} {n_b:4d} {Delta:6d} {k:6d} "
                             f"{snap[k]['ret']:14.4g} {snap[k]['tr']:14.4g}")
        lines.append("")

    (here / "neuronal_curves.txt").write_text("\n".join(lines))
    print(f"\nwritten to {here}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
