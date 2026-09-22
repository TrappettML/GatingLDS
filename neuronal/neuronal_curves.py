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

Two scores, both the raw residual -- no normalising, no squaring, no ratio:

    transfer(K)    ||r_K||,  r_K = theta_K - thetahat_K^(K-1)       eq. (12)

        the task that arrives at step K, read out of the state just before it
        trains: what the stream has left for it before it does any work of its own

    retention(K)   (1/K) sum_{i=1}^{K} ||r_i^(K)||,
                   r_i^(K) = theta_i - thetahat_i^(K)               eq. (28)

        every task trained so far, read out of the current state, averaged.  There
        is no lag to choose: the average runs over all the lags the stream has had
        time to produce, K-1 down to 0.

Teacher rows are drawn on the unit sphere, so both scores sit on a scale where 0 is
a solved task and 1 is a state worth no more than W = 0, at which thetahat = 0.

    python3 neuronal_curves.py            # self-test, then all five figures
    python3 neuronal_curves.py --test     # the self-test alone
    python3 neuronal_curves.py 1 4        # only figures 1 and 4

Figures written next to this file:

    fig1_flow.png         the stream as it actually runs: integrated gradient flow
                          against the two closed forms, eq. (16) and eq. (27)
    fig2_vs_tasks.png     against task index, one trace per density, d_f = d_b
    fig3_vs_tasks_split.png   against task index, one row per density, one trace
                          per split ratio
    fig4_vs_density.png   against read density, one trace per split ratio
    fig5_heatmaps.png     density x splitness, three snapshots of one run
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

    # --- figure 1, the flow ------------------------------------------------
    K_FLOW=16,       # tasks drawn out in flow time
    DF_FLOW=0.5,     # read density
    RHO_FLOW=2.0,    # split ratio d_f / d_b for the flow figure
    FLOW_SEED=0,
    PER_TASK=240,    # points recorded inside each task's window
    DECAY=20.0,      # window width, in time constants 1/c_t
    SUBSTEPS=4,      # integrator steps between two recorded points, at least
    MAX_RATE_STEP=0.1,   # ceiling on c_t dtau

    # --- figures 2 and 3, against task index -------------------------------
    K_TASKS=5000,    # stream length
    DENSITIES=(0.5, 0.3, 0.2, 0.1),   # figure 2: d_f = d_b, one trace each
    SPLIT_DENSITIES=(0.5, 0.3, 0.2),  # figure 3: one row each
    SPLIT_RHOS=(1.0, 2.0, 5.0),       # figure 3: one trace each
    FIG3_FLOOR=1e-2,  # figure 3's y-limits.  the split ratio 5 runs to 1e86 by the
    FIG3_CEIL=1e6,    #   end of the stream, and letting the axis follow it would
                      #   flatten the other two traces into one line, so the axis
                      #   stops here and a clipped trace is labelled with where it
                      #   actually ended
    N_POINTS_DECADE=12,   # states sampled per decade of the logarithmic axis.
                     #      retention is evaluated exactly at those states;
                     #      transfer is the geometric mean over the interval
                     #      ending at each of them

    # --- figure 4, against density -----------------------------------------
    K_DENSITY=1500,  # stream length; both scores are read at the end of it
    RHOS=(1.0, 2.0, 3.0),             # d_f / d_b, one trace each
    DF_STEP=0.05,    # read-density grid, from D/N to 1.0
    DENSITY_TAIL=0.25,  # fraction of the stream transfer is pooled over

    # --- figure 5, density against splitness -------------------------------
    K_HEAT=1000,     # one run per cell; the three columns are snapshots of it
    SNAPSHOTS=(10, 100, 1000),
    HEAT_DF_STEP=0.1,                 # read density, from D/N to 1.0
    HEAT_RHOS=tuple(range(1, 11)),    # splitness 1..10
    HEAT_WINDOW=0.1, # anchors transfer is pooled over, as a fraction of k
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
    the family eq. (9) allows.

    Teacher rows are drawn on the unit sphere.  That is a property of the ensemble,
    not a normalisation applied to the scores: the residuals below are reported raw.
    It is what puts them on a scale where ||r|| = 1 is the null state W = 0.
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(N) / np.sqrt(N)                     # eq. (1)
    rank = np.argsort(np.argsort(rng.random((K, N)), axis=1), axis=1)
    F = (rank < n_f).astype(float)
    B = (rank < n_b).astype(float)
    Theta = rng.standard_normal((K, D))
    Theta /= np.linalg.norm(Theta, axis=1, keepdims=True)
    return v, F, B, Theta


def rownorm(X):
    """Euclidean norm of each row, scaled so a large state cannot overflow the square.

    Past the split threshold the state grows like exp(lambda K) and reaches 1e87 at
    the corner of figure 3.  Squaring that is still finite, but only just, so the
    rows are scaled by their own largest entry first.
    """
    X = np.atleast_2d(X)
    m = np.abs(X).max(axis=-1)
    m = np.where(m == 0.0, 1.0, m)
    return m * np.sqrt(((X / m[:, None]) ** 2).sum(-1))


# ============================================================= the stream

def run_stream(v, F, B, Theta, eval_states=()):
    """Walk the stream once, scoring it as it goes.

    The update is the endpoint of the within-task flow, eq. (16), so no integration
    is needed: the flow has the fixed left factor B_t v, the state moves on a line,
    and the endpoint is the line's end.  State W^s is the state after s tasks have
    trained, W^0 = 0, and task j (0-based) trains into W^{j+1}.

    Returns

        tr    (K,)  ||r_t|| for every task, eq. (12): the task read out of the
                    state just before it trains
        ret   dict  state s -> (1/s) sum_{i<=s} ||theta_i - v^T F_i W^s||, eq. (28)
                    averaged over every task the stream has trained so far

    Retention needs the whole history read out of one state, which is a single
    (s, N) x (N, D) product, so a state costs O(s N D) and the walk itself costs
    O(K N D).  Evaluating it at logarithmically spaced states keeps the total
    within a small multiple of the walk.
    """
    K, N = F.shape
    w = v ** 2
    c = B @ w                                             # write masses, eq. (15)
    if (c <= 0).any():
        raise ValueError("a task has zero write mass; eq. (19) is undefined there")

    Fv = F * v[None, :]
    Bv = B * v[None, :]
    W = np.zeros((N, Theta.shape[1]))
    tr = np.empty(K)
    ret = {}
    want = {int(s) for s in eval_states}

    for t in range(K):
        r = Theta[t] - Fv[t] @ W                          # eq. (12)
        tr[t] = float(rownorm(r[None, :])[0])
        W += np.outer(Bv[t], r) / c[t]                    # eq. (16)
        s = t + 1
        if s in want:
            ret[s] = float(rownorm(Theta[:s] - Fv[:s] @ W).mean())
    return tr, ret


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


def read_back(Gamma, R, s):
    """Every task i <= s read out of state s: -sum_{u=i+1}^{s} Gamma_iu r_u, eq. (30).

    The same band operator as eq. (30), taken all the way out to the current state
    rather than to a fixed lag: row i uses the superdiagonal entries between i and
    s, so row s is empty and the task just trained comes back solved.
    """
    U = np.triu(Gamma[:s, :s], 1)
    return -U @ R[:s]


def exact_scores(v, F, B, Theta, eval_states):
    """The same two scores from the closed form, with nothing trained.

    Route two.  Build the coupling eq. (19) from the gates and the readout, solve
    the whole stream in one triangular system eq. (27), and read every task back
    out of the wanted states with the band operator eq. (30).  The walk in
    `run_stream` never happens: no state is ever formed, no task is ever trained.

    The note says the two routes give the same numbers, and the figures draw both
    so that the claim is visible rather than asserted.  They are not independent
    -- both use the one draw of (v, F, B, Theta) -- but they share no arithmetic,
    so a disagreement would be real.  Cost is O(K^2 N) to build Gamma and
    O(K^2 D) to solve, against O(K N D) for the walk, and Gamma is 200 MB at
    K = 5000; that is why this is a second pass and not the main one.
    """
    G = coupling(v, F, B)
    R = solve_stream(G, Theta)
    U = np.triu(G, 1)
    ret = {}
    for s_ in eval_states:
        s_ = int(s_)
        ret[s_] = float(rownorm(-(U[:s_, :s_] @ R[:s_])).mean())
    return rownorm(R), ret


# ============================================== the mean field, Gamma -> d_f

def mf_resid(d_f, t, Delta):
    """The root-mean-square residual with Gamma replaced by its mean, eq. (21).

    For gates drawn independently each task E[Gamma_kt] = d_f whatever the split
    ratio is, so the recursion eq. (24) collapses to a scalar one.  With
    S_t = sum_{s<=t} r_s it reads S_t = (1-d_f) S_{t-1} + theta_t, so with p = 1-d_f

        r_t          = theta_t - d_f S_{t-1}
        r_t^(Delta)  = -d_f (S_{t+Delta} - S_t)

    and for teachers that are independent, isotropic and of unit norm the pieces of
    the second line are uncorrelated, so

        E||r_t||^2         = 1 + d_f (1 - p^{2(t-1)}) / (2 - d_f)
        E||r_t^(Delta)||^2 = d_f [ (1 - p^{2 Delta})
                                   + (1 - p^Delta)^2 (1 - p^{2t}) ] / (2 - d_f)

    using 1 - p^2 = d_f (2 - d_f).  This returns the square roots, so that it is in
    the same units as the measured curves.  `t` is 1-based, as the note counts
    tasks; Delta = -1 selects the transfer arm; both may be arrays.

    Two things it is not.  It is free of the split ratio, because E[Gamma] is:
    everything the split does to the residual lives in second moments and in the
    stability of eq. (17), neither of which survives replacing Gamma by its mean.

    And it is NOT a bound.  The residuals themselves depend on Gamma, through
    (I + L)^{-1} in eq. (27), so ||r||^2 is rational in Gamma and convexity says
    nothing about the composite.  What happens in fact is that the variance of
    Gamma adds to the diagonal of the read-back and the measured residual runs
    above this curve nearly everywhere -- "nearly", not "always", and the
    exceptions are at d_f near 1, where the calibration point below explains them.

    That calibration point is free.  At d_f = 1 the read gate is the identity, so
    Gamma is exactly the all-ones matrix and nothing is being approximated:
    r_t = theta_t - theta_{t-1} in the mean field and in the simulation alike.  Any
    gap left there is the estimator -- these curves are geometric means over draws
    while this is a root-mean-square -- and it is worth exp(-1/4D) to leading
    order: 1.38 against 1.41 at D = 12.
    """
    p = 1.0 - d_f
    t = np.asarray(t, dtype=float)
    Delta = np.asarray(Delta, dtype=float)
    lag = np.maximum(Delta, 0.0)
    second = np.where(
        Delta < 0,
        1.0 + d_f * (1.0 - p ** (2.0 * (t - 1.0))) / (2.0 - d_f),
        d_f * ((1.0 - p ** (2.0 * lag))
               + (1.0 - p ** lag) ** 2 * (1.0 - p ** (2.0 * t))) / (2.0 - d_f))
    return np.sqrt(np.maximum(second, 0.0))


def mf_retention(d_f, K):
    """The mean field's retention score at state K: the same average, term by term.

    (1/K) sum_{i=1}^{K} of the root-mean-square residual of task i read out of
    state K, which is `mf_resid` at t = i and Delta = K - i.  The i = K term is
    zero -- the task just trained is solved -- so the average starts at 0 and
    climbs as the stream fills with older tasks.
    """
    K = int(K)
    i = np.arange(1, K + 1, dtype=float)
    return float(mf_resid(d_f, i, K - i).mean())


def geo(x):
    """Geometric mean over draws, the way this file combines realisations.

    Once the stream is unstable the residual is heavy-tailed across draws: at
    rho = 5 a single draw can sit sixty orders of magnitude above the rest, so an
    arithmetic mean reports that draw and nothing else.  log||r|| is the
    near-Gaussian variable, so the mean is taken there and exponentiated back.
    The average *within* a draw -- retention's sum over tasks -- is the plain
    arithmetic mean the definition asks for.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        return np.nan
    return float(np.exp(np.log(x).mean()))


def log_points(K, per_decade):
    """States to score at, logarithmically spaced over 1..K, no repeats."""
    n = max(2, int(round(np.log10(K) * per_decade)) + 1)
    pts = np.unique(np.round(np.logspace(0.0, np.log10(K), n)).astype(int))
    return pts[(pts >= 1) & (pts <= K)]


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


def _baseline(ax, note=True):
    """The line at ||r|| = 1: the state is worth exactly what W = 0 is worth."""
    ax.axhline(1.0, color=AXIS, linewidth=1.0, zorder=1)
    if note:
        ax.annotate("$1$ = no better than $W=0$", xy=(0.0, 1.0),
                    xycoords=("axes fraction", "data"), xytext=(4, 3),
                    textcoords="offset points", ha="left", fontsize=7.5,
                    color=MUTED)


ARMS = (("ret", r"retention:   $\frac{1}{K}\sum_{i\leq K}\|r_i^{(K)}\|$",
         "every task trained so far, read out of the current state"),
        ("tr", r"transfer:   $\|r_K\|$",
         "the arriving task, read out of the state just before it trains"))


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
    c = B @ v ** 2
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


# ============================================================ figures 2 and 3

def case_vs_tasks(cfg, d_f, rho=1.0, seed0=100):
    """One (d_f, rho), scored at logarithmically spaced states, by both routes.

    Retention is evaluated exactly at those states -- it is already an average over
    every task the stream has trained, so nothing more is pooled within a draw.
    Transfer is one number per task, so it is pooled over the interval of arrivals
    ending at each state.  Draws are combined geometrically in both cases, and the
    walk and the closed form see the same draws.
    """
    N, D, K = cfg["N"], cfg["D"], cfg["K_TASKS"]
    n_f, n_b = counts(N, d_f, rho)
    pts = log_points(K, cfg["N_POINTS_DECADE"])
    edges = np.concatenate([[0], pts])
    shape = (cfg["SEEDS"], pts.size)
    raw = {k: np.full(shape, np.nan) for k in
           ("ret", "tr", "ret_x", "tr_x")}
    gap = 0.0
    for s in range(cfg["SEEDS"]):
        v, F, B, Theta = draw(N, D, K, n_f, n_b, seed=seed0 + s)
        tr, ret = run_stream(v, F, B, Theta, pts)
        tr_x, ret_x = exact_scores(v, F, B, Theta, pts)
        gap = max(gap, _route_gap(tr, tr_x), _route_gap(
            np.array([ret[int(k)] for k in pts]),
            np.array([ret_x[int(k)] for k in pts])))
        raw["ret"][s] = [ret[int(k)] for k in pts]
        raw["ret_x"][s] = [ret_x[int(k)] for k in pts]
        for q in range(pts.size):
            raw["tr"][s, q] = geo(tr[edges[q]:edges[q + 1]])
            raw["tr_x"][s, q] = geo(tr_x[edges[q]:edges[q + 1]])
    out = {k: np.array([geo(a[:, q]) for q in range(pts.size)])
           for k, a in raw.items()}
    out["x"], out["n_b"], out["gap"] = pts.astype(float), n_b, gap
    return out


def _route_gap(a, b):
    """Largest relative disagreement between the two routes, ignoring exact zeros."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 0)
    if not m.any():
        return 0.0
    return float(np.max(np.abs(a[m] - b[m]) / np.abs(b[m])))


def _three(ax, x, sim, exact, mf, colour, label=None):
    """One series, three ways: simulation, the closed form, and the mean field.

    The walk is drawn broad and translucent and the closed form thin on top of it,
    because the two are the same numbers: what the reader should see is a dashed
    line riding inside a solid band, not two curves to compare.  The mean field is
    dotted and is a different quantity, so it is drawn at full weight underneath.
    """
    ok = np.isfinite(sim) & (sim > 0)
    ln, = ax.plot(x[ok], sim[ok], color=colour, linewidth=2.8, alpha=0.40,
                  solid_capstyle="round", zorder=3, label=label)
    if exact is not None:
        okx = np.isfinite(exact) & (exact > 0)
        ax.plot(x[okx], exact[okx], color=colour, linewidth=1.2,
                linestyle=(0, (5, 2)), zorder=4)
    if mf is not None:
        m = np.asarray(mf, dtype=float)
        okm = np.isfinite(m) & (m > 0)
        ax.plot(x[okm], m[okm], color=colour, linewidth=1.3,
                linestyle=(0, (1, 2)), zorder=2)
    return ln, ((float(x[ok][-1]), float(sim[ok][-1])) if ok.any() else None)


def _route_legend(colour=None):
    """The three line styles, named once per figure."""
    from matplotlib.lines import Line2D
    c = colour or MUTED
    return [Line2D([], [], color=c, lw=2.8, alpha=0.40, label="simulation"),
            Line2D([], [], color=c, lw=1.2, ls=(0, (5, 2)),
                   label="whole stream, eq. (27) & (30)"),
            Line2D([], [], color=c, lw=1.3, ls=(0, (1, 2)),
                   label="mean field, $\\Gamma\\to d_f$ (r.m.s.)")]


def _mf_curve(d_f, x, arm):
    """The mean field on the same abscissa as the measured traces."""
    if arm == "ret":
        return np.array([mf_retention(d_f, int(k)) for k in x])
    return np.asarray(mf_resid(d_f, x, -1), dtype=float)


def fig_vs_tasks(cfg, cases, path, solve_gap):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), sharex=True)
    handles = []
    worst = max(c["gap"] for c in cases.values())
    for ci, (arm, ylab, sub) in enumerate(ARMS):
        ax = axes[ci]
        ends = []
        for k, d_f in enumerate(cfg["DENSITIES"]):
            res = cases[d_f]
            ln, end = _three(ax, res["x"], res[arm], res[arm + "_x"],
                             _mf_curve(d_f, res["x"], arm), SERIES[k],
                             f"$d_f=d_b={d_f}$" if ci == 0 else None)
            if ci == 0:
                handles.append(ln)
            if end:
                ends.append((end[0], end[1], f"{d_f}"))
        ax.set_xscale("log")
        _baseline(ax)
        ax.set_title(sub, fontsize=10, color=INK, pad=8)
        ax.set_ylabel(ylab, fontsize=10, color=INK2)
        ax.set_xlabel("tasks trained, $K$", fontsize=9.5, color=INK2)
        _chrome(ax)
        _end_labels(ax, ends)

    fig.legend(handles=handles + _route_legend(), frameon=False, fontsize=8.5,
               ncol=7, loc="upper center", bbox_to_anchor=(0.5, 0.935),
               labelcolor=INK2)
    fig.suptitle("the raw residual, against how far into the stream the task sits   "
                 f"($N={cfg['N']}$, $N_{{in}}={cfg['D']}$, $K={cfg['K_TASKS']}$, "
                 f"{cfg['SEEDS']} draws;  0 = solved, 1 = no better than $W=0$)",
                 fontsize=11, color=INK, y=0.995)
    fig.text(0.5, 0.012,
             "the dashed line is the closed form on the same draws -- the coupling "
             "eq. (19), one triangular solve eq. (27) and the band eq. (30), with no "
             "state ever formed.\nit rides inside the solid band because the two are "
             "the same numbers; here they agree to "
             f"{worst:.0e} relative, worst case over every point drawn.\n"
             "the mean field is a different quantity and is not a bound: the residuals "
             "depend on $\\Gamma$ themselves, through $(I+L)^{-1}$, but its variance "
             "adds to the read-back, so the measured curve runs above it.",
             ha="center", va="bottom", fontsize=8, color=MUTED, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.125, 1, 0.9))
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


def fig_vs_tasks_split(cfg, cases, path, solve_gap):
    """Figure 2's axes again, one row per density, one trace per split ratio.

    The vertical range is capped.  Past the threshold of eq. (40) the state grows
    like exp(lambda K), so the d_f/d_b = 5 trace ends between 1e61 and 1e86
    depending on the density; an axis that followed it would compress the other two
    into a single line.  A trace that leaves the panel is labelled with the decade
    it actually reached.
    """
    import matplotlib.pyplot as plt

    dens = cfg["SPLIT_DENSITIES"]
    rhos = cfg["SPLIT_RHOS"]
    lo, hi = cfg["FIG3_FLOOR"], cfg["FIG3_CEIL"]
    worst = max(c["gap"] for c in cases.values())
    fig, axes = plt.subplots(len(dens), 2, figsize=(12.6, 10.4), sharex=True)
    handles = []
    for ri, d_f in enumerate(dens):
        for ci, (arm, ylab, sub) in enumerate(ARMS):
            ax = axes[ri][ci]
            ends, res = [], None
            for k, rho in enumerate(rhos):
                if (d_f, rho) not in cases:
                    continue
                res = cases[(d_f, rho)]
                ln, end = _three(ax, res["x"], res[arm], res[arm + "_x"], None,
                                 SERIES[k],
                                 f"$d_f/d_b={rho:g}$" if ri + ci == 0 else None)
                if ri + ci == 0:
                    handles.append(ln)
                if end:
                    last = end[1]
                    text = (f"{rho:g}" if last <= hi else
                            f"{rho:g} $\\to10^{{{int(np.log10(last))}}}$")
                    ends.append((end[0], min(last, hi), text))
            # one dotted curve for the row: the mean field cannot see the split
            th = _mf_curve(d_f, res["x"], arm)
            good = th > 1e-9
            ax.plot(res["x"][good], th[good], color=MUTED, linewidth=1.3,
                    linestyle=(0, (1, 2)), zorder=2)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(lo, hi)
            _baseline(ax, note=(ri == 0))
            if ri == 0:
                ax.set_title(sub, fontsize=10, color=INK, pad=8)
            if ri == len(dens) - 1:
                ax.set_xlabel("tasks trained, $K$", fontsize=9.5, color=INK2)
            ax.set_ylabel(f"$d_f={d_f}$\n\n{ylab}" if ci == 0 else ylab,
                          fontsize=9.5, color=INK2)
            _chrome(ax)
            _end_labels(ax, ends, gap=0.075)

    fig.legend(handles=handles + _route_legend(), frameon=False, fontsize=8.5,
               ncol=6, loc="upper center", bbox_to_anchor=(0.5, 0.96),
               labelcolor=INK2)
    fig.suptitle("the raw residual against task index, at three read densities   "
                 f"($N={cfg['N']}$, $N_{{in}}={cfg['D']}$, $K={cfg['K_TASKS']}$, "
                 f"{cfg['SEEDS']} draws;  1 = no better than $W=0$)",
                 fontsize=11, color=INK, y=0.995)
    fig.text(0.5, 0.008,
             "the vertical axis stops at $10^{6}$: past the threshold of eq. (40) the "
             "state grows like $\\exp(\\lambda K)$ and $d_f/d_b=5$ ends between "
             "$10^{61}$ and $10^{86}$, so a trace that leaves\nthe panel is labelled "
             "with the decade it reached.  the dashed closed form rides inside the "
             f"solid band, agreeing to {worst:.0e} relative at worst -- eighty decades "
             "of growth\ncost the triangular solve nothing.  the dotted mean field is "
             "the same in every trace of a row, because $E[\\Gamma_{kt}]=d_f$ whatever "
             "$d_b$ is.",
             ha="center", va="bottom", fontsize=8, color=MUTED, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.055, 1, 0.945))
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


# ==================================================================== figure 4

def case_vs_density(cfg, d_f, rho):
    """Both scores at the end of a stream of K_DENSITY, by both routes.

    Retention is the average over every task the stream trained, read out of the
    final state.  Transfer is pooled over the arrivals in the last DENSITY_TAIL of
    the stream, which is where it has settled.
    """
    N, D, K = cfg["N"], cfg["D"], cfg["K_DENSITY"]
    n_f, n_b = counts(N, d_f, rho)
    lo = int(round((1.0 - cfg["DENSITY_TAIL"]) * K))
    acc = {k: [] for k in ("ret", "tr", "ret_x", "tr_x")}
    gap = 0.0
    for s in range(cfg["SEEDS"]):
        v, F, B, Theta = draw(N, D, K, n_f, n_b, seed=200 + s)
        tr, ret = run_stream(v, F, B, Theta, (K,))
        tr_x, ret_x = exact_scores(v, F, B, Theta, (K,))
        gap = max(gap, _route_gap(tr, tr_x),
                  _route_gap([ret[K]], [ret_x[K]]))
        acc["ret"].append(ret[K])
        acc["ret_x"].append(ret_x[K])
        acc["tr"].append(geo(tr[lo:]))
        acc["tr_x"].append(geo(tr_x[lo:]))
    out = {k: geo(x) for k, x in acc.items()}
    out["n_b"], out["K"], out["gap"] = n_b, K, gap
    return out


def fig_vs_density(cfg, grid, path, solve_gap):
    import matplotlib.pyplot as plt

    d_fs = sorted({d for d, _ in grid})
    K = cfg["K_DENSITY"]
    worst = max(c["gap"] for c in grid.values())
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), sharex=True)
    handles = []
    for ci, (arm, ylab, sub) in enumerate(ARMS):
        ax = axes[ci]
        ends = []
        for k, rho in enumerate(cfg["RHOS"]):
            xs = np.array([d for d in d_fs if (d, rho) in grid])
            sim = np.array([grid[(d, rho)][arm] for d in xs])
            exa = np.array([grid[(d, rho)][arm + "_x"] for d in xs])
            ln, end = _three(ax, xs, sim, exa, None, SERIES[k],
                             f"$d_f/d_b={rho:g}$" if ci == 0 else None)
            if ci == 0:
                handles.append(ln)
            if end:
                ends.append((end[0], end[1], f"{rho:g}"))
        th = np.array([mf_retention(d, K) if arm == "ret"
                       else float(mf_resid(d, K, -1)) for d in d_fs])
        ax.plot(d_fs, th, color=MUTED, linewidth=1.4, linestyle=(0, (1, 2)),
                zorder=2)
        ax.set_yscale("log")
        _baseline(ax)
        ax.set_title(sub, fontsize=10, color=INK, pad=8)
        ax.set_ylabel(ylab, fontsize=10, color=INK2)
        ax.set_xlabel("read density $d_f$", fontsize=9.5, color=INK2)
        _chrome(ax)
        _end_labels(ax, ends)

    fig.legend(handles=handles + _route_legend(), frameon=False, fontsize=8.5,
               ncol=6, loc="upper center", bbox_to_anchor=(0.5, 0.935),
               labelcolor=INK2)
    fig.suptitle("the raw residual at the end of the stream, against read density   "
                 f"($N={cfg['N']}$, $N_{{in}}={cfg['D']}$, $K={K}$, "
                 f"{cfg['SEEDS']} draws)", fontsize=11, color=INK, y=0.995)
    fig.text(0.5, 0.012,
             "the dashed closed form rides inside the solid band, agreeing to "
             f"{worst:.0e} relative at worst.  one dotted curve serves all three "
             "ratios: $E[\\Gamma_{kt}]=d_f$ whatever\n$d_b$ is, so everything the "
             "split does lives in second moments and in the stability of eq. (17).   "
             "at $d_f=1$ the read gate is the identity, $\\Gamma$ is exactly\nall-ones "
             "and the mean field must agree; the gap left there is the estimator, "
             "$1.38$ against $1.41$, and the noise floor of the comparison.",
             ha="center", va="bottom", fontsize=8, color=MUTED, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.125, 1, 0.9))
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


# ==================================================================== figure 5

def case_heat(cfg, d_f, rho):
    """One run to K_HEAT at one cell, scored at the three snapshots.

    Retention at a snapshot is the average over every task trained by then, so
    every cell of the grid has one -- there is no lag that can reach back past the
    start of the stream and no cell to leave blank.
    """
    N, D, K = cfg["N"], cfg["D"], cfg["K_HEAT"]
    n_f, n_b = counts(N, d_f, rho)
    snaps = tuple(int(s) for s in cfg["SNAPSHOTS"])
    out = {k: {"ret": [], "tr": []} for k in snaps}
    for s in range(cfg["SEEDS"]):
        v, F, B, Theta = draw(N, D, K, n_f, n_b, seed=300 + s)
        tr, ret = run_stream(v, F, B, Theta, snaps)
        for k in snaps:
            nw = max(1, int(round(cfg["HEAT_WINDOW"] * k)))
            out[k]["ret"].append(ret[k])
            out[k]["tr"].append(geo(tr[k - nw:k]))
    return {k: {m: geo(x) for m, x in d.items()} for k, d in out.items()}, n_b


def fig_heatmaps(cfg, cells, path):
    """Two rows, retention and transfer, at three snapshots of one run.

    The colour carries log10||r|| rather than ||r||: across this grid the residual
    runs from 0.2 to 1e31, which no linear scale shows, and the log has the same
    zero -- ||r|| = 1, the state worth exactly what W = 0 is worth -- with blue
    below it and red above.  Both rows share one scale; they are two readings of
    one run and have to be comparable.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    d_fs = sorted({d for d, _ in cells})
    rhos = list(cfg["HEAT_RHOS"])
    snaps = list(cfg["SNAPSHOTS"])
    rows = (("ret", r"retention,  $\log_{10}$"),
            ("tr", r"transfer,  $\log_{10}$"))

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
    cmap.set_bad((0.0, 0.0, 0.0, 0.0))
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
            ax.set_facecolor("#f3f2ee")
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
             "retention at $k$ averages every task trained by then, so no cell is "
             f"missing.  transfer is pooled over the arrivals in the "
             f"{cfg['HEAT_WINDOW']:.0%} window ending at $k$.\nthe scale stops at "
             f"$10^{{{ceil:g}}}$ and the arrow marks cells past it: at the split corner "
             "$\\|r\\|$ reaches $10^{31}$, which no scale resolves against a band of "
             "width one.\nthe unclipped numbers are in the .txt.",
             ha="center", va="bottom", fontsize=8, color=MUTED, linespacing=1.5)
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


# ================================================================== self-test

def solve_check(cfg, d_f=0.3, rho=2.0, seed=11):
    """The walk against the closed form, on exactly the path the figures draw."""
    N, D, K = cfg["N"], cfg["D"], cfg["K_CHECK"]
    n_f, n_b = counts(N, d_f, rho)
    states = (K // 4, K // 2, K)
    v, F, B, Theta = draw(N, D, K, n_f, n_b, seed=seed)
    tr, ret = run_stream(v, F, B, Theta, states)
    tr_x, ret_x = exact_scores(v, F, B, Theta, states)
    return max(_route_gap(tr, tr_x),
               _route_gap([ret[s] for s in states], [ret_x[s] for s in states]))


def self_test(cfg, verbose=True):
    """Routes that share no code, checked against each other."""
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
        rep(f"d_f={d_f}, rho={rho}: streamed scores == eq. (27)/(30)",
            solve_check(cfg, d_f, rho), 1e-9)

    if verbose:
        print("-- the two scores at their reference states --")
    v, F, B, Theta = draw(cfg["N"], cfg["D"], 40, 20, 10, seed=5)
    tr, ret = run_stream(v, F, B, Theta, (1, 2, 40))
    rep("task 1 inherits W = 0, so ||r_1|| = ||theta_1|| = 1", tr[0] - 1.0, 1e-12)
    rep("retention(1) = 0: the only task trained is solved", ret[1], 1e-12)
    # two steps replayed by hand: task 2 is solved, so retention(2) is half of
    # what is left of task 1
    W1 = np.outer(B[0] * v, Theta[0]) / float(B[0] @ v ** 2)
    W2 = W1 + np.outer(B[1] * v, Theta[1] - (F[1] * v) @ W1) / float(B[1] @ v ** 2)
    rep("retention(2) == ||r_1^(2)|| / 2, hand-replayed",
        ret[2] - 0.5 * float(rownorm((Theta[0] - (F[0] * v) @ W2)[None, :])[0]),
        1e-12)
    rep("the residual is raw: ||theta_t|| == 1, nothing is divided out",
        np.linalg.norm(Theta, axis=1) - 1.0, 1e-12)

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
                want = float(mf_resid(d_f, t, Delta))
                rep(f"  d_f={d_f}, Delta={Delta:3d}, t={t:3d}: ||r|| = {want:.4f}",
                    np.sqrt(got) / want - 1.0, 0.02)
        # the retention score is the same formula averaged term by term
        for Ks in (20, 150):
            per = np.sqrt((d_f ** 2 * ((Ss[:, Ks - 1][:, None, :]
                                        - Ss[:, :Ks]) ** 2).sum(2)).mean(0))
            rep(f"  d_f={d_f}, retention({Ks}) = {mf_retention(d_f, Ks):.4f}",
                per.mean() / mf_retention(d_f, Ks) - 1.0, 0.02)

    if verbose:
        print("-- the mean field is exact at d_f = 1, where Gamma is all-ones --")
    v, F, B, Theta = draw(cfg["N"], cfg["D"], 400, cfg["N"], cfg["N"], seed=9)
    rep("d_f = 1 gives Gamma == 1 everywhere", coupling(v, F, B) - 1.0, 1e-12)
    tr, ret = run_stream(v, F, B, Theta, (400,))
    rep("  and r_t == theta_t - theta_{t-1}",
        tr[1:] - np.linalg.norm(Theta[1:] - Theta[:-1], axis=1), 1e-12)
    rep("  r.m.s. of it == the mean field, sqrt(2)",
        np.sqrt((tr[1:] ** 2).mean()) / float(mf_resid(1.0, 200, -1)) - 1.0, 0.04)
    rep("  the geometric mean sits below it by exp(-1/4D)",
        geo(tr[1:]) / (np.sqrt(2.0) * np.exp(-1.0 / (4 * cfg["D"]))) - 1.0, 0.02)

    if verbose:
        print("-- the aggregator --")
    rep("geo() is the exponential of the mean log", geo([1.0, 100.0]) - 10.0, 1e-12)
    rep("geo() drops non-finite entries", geo([4.0, np.inf, np.nan]) - 4.0, 1e-12)
    pts = log_points(1000, 12)
    rep("log_points are increasing and inside 1..K",
        [pts[0] - 1, pts[-1] - 1000, int((np.diff(pts) <= 0).sum())], 0)
    big = np.array([[1e200, 1e200]])
    rep("rownorm survives an entry whose square would overflow",
        rownorm(big)[0] / (1e200 * np.sqrt(2.0)) - 1.0, 1e-12)

    if verbose:
        print(f"\n  {'all checks passed' if not fails else str(fails) + ' FAILED'}")
    return fails


# ======================================================================= main

def _grid(lo, hi, step):
    """An inclusive grid, rounded, so that 0.2..1.0 by 0.1 has exactly nine points."""
    n = int(round((hi - lo) / step)) + 1
    return [round(lo + i * step, 9) for i in range(n)]


def _write_log(path, lines, want):
    """Write the run log, keeping the sections this run did not regenerate.

    Asking for a subset of the figures used to leave a log describing only that
    subset, which is worse than no log at all.  Sections are keyed by their "figN"
    heading, the ones just produced replace their old text, and the rest are
    carried over in order.
    """
    def split(text):
        head, out, key = [], {}, None
        for ln in text.split("\n"):
            if ln[:3] == "fig" and ln[3:4].isdigit():
                key = ln[:4]
                out[key] = []
            (out[key] if key else head).append(ln)
        return head, out

    head, fresh = split("\n".join(lines))
    kept = {}
    if path.exists():
        _, kept = split(path.read_text())
    kept.update(fresh)
    body = []
    for k in sorted(kept):
        if want and k[3:] not in want and k in fresh:
            continue
        body += kept[k]
    path.write_text("\n".join(head + body).rstrip() + "\n")


def main(argv):
    here = Path(__file__).resolve().parent
    want = {a for a in argv if a in {"1", "2", "3", "4", "5"}}
    cfg = CONFIG

    print("=" * 76)
    print("  neuronal split gating -- experiments and figures")
    print(f"  N={cfg['N']}  N_in={cfg['D']}  seeds={cfg['SEEDS']}")
    print("=" * 76)
    if self_test(cfg) and "--test" not in argv:
        print("\n  self-test failed; not drawing anything")
        return 1
    if "--test" in argv:
        return 0

    gap = solve_check(cfg)
    lines = [f"stream vs the exact solve (27) at K={cfg['K_CHECK']}: {gap:.3e}",
             "scores are the RAW residual, ||r||, with no normalising and no squaring.",
             "  transfer(K)  = ||theta_K - thetahat_K^(K-1)||",
             "  retention(K) = (1/K) sum_{i<=K} ||theta_i - thetahat_i^(K)||",
             "0 = solved, 1 = no better than W = 0.  draws combined geometrically.",
             "the eq.27/30 columns are the same scores from the closed form, with",
             "nothing trained; 'route gap' is the largest relative disagreement.",
             ""]
    head = (f"{'retention':>14s} {'transfer':>14s}"
            f"{'ret, eq.27/30':>15s} {'tr, eq.27/30':>15s} {'route gap':>11s}")

    if not want or "1" in want:
        print("\nfigure 1: the flow ...")
        res = flow_case(cfg)
        fig_flow(cfg, res, here / "fig1_flow.png")
        lines += [f"fig1  n_f={res['n_f']} n_b={res['n_b']}  "
                  f"integrated vs closed form {res['flow_gap']:.2e}  "
                  f"integrated residuals vs eq. (27) {res['box_gap']:.2e}  "
                  f"stream vs eq. (27) {res['solve_gap']:.2e}", ""]

    if not want or "2" in want:
        print("figure 2: against task index, one trace per density ...")
        cases = {d: case_vs_tasks(cfg, d) for d in cfg["DENSITIES"]}
        fig_vs_tasks(cfg, cases, here / "fig2_vs_tasks.png", gap)
        lines.append(f"fig2  at the end of the stream (K={cfg['K_TASKS']})")
        lines.append(f"  {'d_f':>5s} {'n_b':>4s} " + head)
        for d in cfg["DENSITIES"]:
            r = cases[d]
            lines.append(f"  {d:5.2f} {r['n_b']:4d} "
                         f"{r['ret'][-1]:14.4g} {r['tr'][-1]:14.4g}"
                         f"{r['ret_x'][-1]:15.4g} {r['tr_x'][-1]:15.4g}"
                         f"{r['gap']:11.1e}")
        lines.append("")

    if not want or "3" in want:
        print("figure 3: against task index, one trace per split ratio ...")
        cases = {}
        for d in cfg["SPLIT_DENSITIES"]:
            for rho in cfg["SPLIT_RHOS"]:
                try:
                    cases[(d, rho)] = case_vs_tasks(cfg, d, rho, seed0=400)
                except ValueError:
                    pass
        fig_vs_tasks_split(cfg, cases, here / "fig3_vs_tasks_split.png", gap)
        lines.append(f"fig3  at the end of the stream (K={cfg['K_TASKS']})")
        lines.append(f"  {'d_f':>5s} {'rho':>4s} {'n_b':>4s} " + head)
        for (d, rho), r in sorted(cases.items()):
            lines.append(f"  {d:5.2f} {rho:4.1f} {r['n_b']:4d} "
                         f"{r['ret'][-1]:14.4g} {r['tr'][-1]:14.4g}"
                         f"{r['ret_x'][-1]:15.4g} {r['tr_x'][-1]:15.4g}"
                         f"{r['gap']:11.1e}")
        lines.append("")

    if not want or "4" in want:
        print("figure 4: against density ...")
        grid = {}
        for rho in cfg["RHOS"]:
            for d in _grid(cfg["D"] / cfg["N"], 1.0, cfg["DF_STEP"]):
                try:
                    grid[(d, rho)] = case_vs_density(cfg, d, rho)
                except ValueError:
                    pass
        fig_vs_density(cfg, grid, here / "fig4_vs_density.png", gap)
        lines.append(f"fig4  at the end of the stream (K={cfg['K_DENSITY']})")
        lines.append(f"  {'d_f':>5s} {'rho':>4s} {'n_b':>4s} " + head)
        for (d, rho), r in sorted(grid.items()):
            lines.append(f"  {d:5.2f} {rho:4.1f} {r['n_b']:4d} "
                         f"{r['ret']:14.4g} {r['tr']:14.4g}"
                         f"{r['ret_x']:15.4g} {r['tr_x']:15.4g}"
                         f"{r['gap']:11.1e}")
        lines.append("")

    if not want or "5" in want:
        print("figure 5: density against splitness ...")
        cells = {}
        for d in _grid(cfg["D"] / cfg["N"], 1.0, cfg["HEAT_DF_STEP"]):
            for rho in cfg["HEAT_RHOS"]:
                try:
                    cells[(d, rho)] = case_heat(cfg, d, rho)
                except ValueError:
                    pass
        fig_heatmaps(cfg, cells, here / "fig5_heatmaps.png")
        lines.append(f"fig5  {len(cells)} cells, snapshots {cfg['SNAPSHOTS']}, "
                     f"unclipped (the figure shows log10, capped at "
                     f"{cfg['HEAT_CEIL']:g})")
        lines.append(f"  {'d_f':>5s} {'rho':>4s} {'n_b':>4s} {'k':>6s} " + head)
        for (d, rho), (snap, n_b) in sorted(cells.items()):
            for k in cfg["SNAPSHOTS"]:
                lines.append(f"  {d:5.2f} {rho:4d} {n_b:4d} {k:6d} "
                             f"{snap[k]['ret']:14.4g} {snap[k]['tr']:14.4g}")
        lines.append("")

    _write_log(here / "neuronal_curves.txt", lines, want)
    print(f"\nwritten to {here}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
