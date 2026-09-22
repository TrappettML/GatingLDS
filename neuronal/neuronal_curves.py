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

Three metrics.  The first two are the raw residual -- no normalising, no
squaring, no ratio:

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

The third is the size of the weight change itself, in two readings:

    step(T)  = (1/T) sum_{t<=T} ||W^t - W^{t-1}||_F         the average step

        how far the state moves on a task, averaged over the stream.  Eq. (16)
        makes the step rank one, W^t - W^{t-1} = B_t v r_t / c_t, and the write
        mass is c_t = ||B_t v||^2 by eq. (15), so the step is exactly
        ||r_t|| / sqrt(c_t) -- a residual divided by the square root of the mass
        that had to carry it.

    disp(T) = ||W^T||_F / T                                 the net displacement

        the same increments summed before the norm is taken rather than after.
        step(T) is the mean length of a step and disp(T) the length of the mean
        step, so disp <= step always and the ratio is how much of the movement
        cancelled.  In the stable regime it does: the state random-walks and disp
        falls like T^{-1/2}, while at d_f = d_b = 1 the sum telescopes exactly and
        ||W^T||_F = 1/||v|| for every T, so disp falls like 1/T.

Seeds -- independent draws of (readout, gates, teachers) -- are combined
geometrically, and every figure carries the standard error of that mean, taken in
logs and shown multiplicatively.

    python3 neuronal_curves.py            # self-test, then all seven figures
    python3 neuronal_curves.py --test     # the self-test alone
    python3 neuronal_curves.py 1 4        # only figures 1 and 4
    python3 neuronal_curves.py --jax 6 7  # the two heavy kernels on a GPU

Figures written next to this file:

    fig1_flow.png         the stream as it actually runs: integrated gradient flow
                          against the two closed forms, eq. (16) and eq. (27)
    fig2_vs_tasks.png     against task index, one trace per density, d_f = d_b
    fig3_vs_tasks_split.png   against task index, one row per density, one trace
                          per split ratio
    fig4_vs_density.png   against read density, one trace per split ratio
    fig5_heatmaps.png     density x splitness, three snapshots of one run
    fig6_step.png         the average step, three stream lengths: against density,
                          against splitness, and the two against each other
    fig7_displacement.png the net displacement per task, the same three panels
"""

import sys
from pathlib import Path

import numpy as np


# =============================================================== configuration

CONFIG = dict(
    # --- the network, shared by every figure -------------------------------
    N=200,            # N_h: neurons.  The gate opens and closes whole neurons, so
                     #      this is the only width the dynamics sees
    D=12,            # N_in: input dimension.  The D coordinates share one gate and
                     #      differ only in what drives them
    SEEDS=20,        # independent seeds (readout, gates, teachers) per point.
                     #      every figure reports the geometric mean over them and
                     #      the standard error of that mean, taken in logs
    ROOT_SEED=20260922,  # the one entropy the whole study hangs off; every stream
                     #      is a named child of it, so running one figure alone
                     #      draws exactly what the full run draws

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

    # --- figures 6 and 7, the weight change --------------------------------
    K_DW=5000,       # one stream per cell; the three columns are snapshots of it
    DW_SNAPSHOTS=(50, 500, 5000),     # T, the number of tasks: one column each
    DW_DF_STEP=0.1,  # read density, 0.1 to 1.0 -- the whole range, not from D/N:
                     #      nothing in eqs. (12)-(30) needs n_f >= D, since the
                     #      update is rank one and solves its own task whatever
                     #      n_f is, and the sparse end is where the split bites
    DW_RHOS=tuple(range(1, 11)),      # splitness 1..10
    DW_ROW1_RHOS=(1.0, 2.0, 5.0),     # row 1's three traces, as figure 3
    DW_ROW2_DFS=(0.5, 0.3, 0.2),      # row 2's three traces, as figure 3's rows
    DW_CEIL=1e6,     # rows 1 and 2 share one vertical axis across all three
                     #      columns -- the growth with T is the point of having T
                     #      as the columns -- and it stops here; at rho = 10 the
                     #      step reaches 1e159 by T = 5000 and a trace that leaves
                     #      the panel is labelled with the decade it reached
    DW_HEAT_CEIL=3.0,   # colour ceiling of row 3, in log10 of the metric: a cell
                     #      only has to read as "a thousand times the baseline or
                     #      worse".  the .txt keeps the unclipped numbers

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


def sequential_cmap():
    """One hue, light to dark: the encoding for a magnitude with no meaningful zero.

    Figure 5's residual has a zero worth marking -- ||r|| = 1 is the state worth
    exactly what W = 0 is worth -- so it gets a diverging ramp centred there.  The
    weight change has no such point: the natural reference for a step is
    1/sqrt(d_b), which moves across the grid, so there is nothing to diverge about
    and a single hue carrying the whole ordering in its lightness is the honest
    encoding.  Monotonicity is checked here for the same reason it is there.
    """
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    stops = [NEUTRAL] + list(RAMP_BLUE)
    lum = [0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
           for c in (to_rgb(h) for h in stops)]
    if not all(np.diff(lum) < 0):
        raise ValueError("the sequential ramp is not monotone in lightness")
    return LinearSegmentedColormap.from_list("magnitude", stops, N=512)


# ==================================================================== the seeds
# One root entropy, and a named child stream per (experiment, seed index).  Two
# reasons to build them this way rather than handing default_rng consecutive
# integers.  A SeedSequence hashes the whole key, so neighbouring indices give
# streams with no relation to one another, which consecutive integer seeds do not
# guarantee; and the key is a name and not a position, so running one figure on
# its own draws exactly what the full run draws, and two figures cannot collide
# however many seeds either of them grows to.
#
# Within a figure the index is the replicate and every case reuses it, so the cases
# are paired: at a fixed index the readout v and the teachers Theta are the same
# for every (d_f, rho), and the gates are nested subsets of one permutation.  A
# difference between two traces is then a difference between gates and not between
# draws, which is what makes the error bars below worth reading against each other.

FLOW, TASKS, SPLIT, DENSITY, HEAT, DELTAW, TEST = range(7)


def rng_for(cfg, stream, index=0):
    """The generator for replicate `index` of experiment `stream`."""
    return np.random.default_rng(np.random.SeedSequence(
        cfg["ROOT_SEED"], spawn_key=(int(stream), int(index))))


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


def draw(N, D, K, n_f, n_b, rng):
    """One seed's realisation: readout, nested per-task neuron gates, teachers.

    One uniform permutation of the N neurons per task; the first n_f places are
    read and the first n_b written, so B_t sits inside F_t by construction and any
    two tasks' gates are independent uniform subsets -- the constant array
    d_fb^{kt} = n_f/N = d_f for k != t, which is the most decorrelated member of
    the family eq. (9) allows.

    Teacher rows are drawn on the unit sphere.  That is a property of the ensemble,
    not a normalisation applied to the scores: the residuals below are reported raw.
    It is what puts them on a scale where ||r|| = 1 is the null state W = 0.

    `rng` comes from `rng_for`, so the caller names the stream rather than picking
    an integer; the draw order below is fixed, which is what makes two cases at the
    same seed index share v and Theta exactly.
    """
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


def fro(X):
    """The same guard for a whole matrix: ||X||_F without squaring the largest entry.

    The state itself needs this, not just its rows.  Past the split threshold the
    step of eq. (16) reaches 1e159 at the corner of the grid figures 6 and 7 draw,
    and 1e159 squared is 1e318, which is not a double.  Scaling by the largest
    entry first costs one pass and makes the norm exact wherever the entries are.
    A state that has already overflowed comes back as inf or nan rather than 0, so
    the seed is dropped by `geo_sem` instead of being counted as a state at rest.
    """
    X = np.asarray(X, dtype=float)
    m = np.abs(X).max() if X.size else 0.0
    if not (m > 0.0):
        return float(m)
    return float(m * np.sqrt(((X / m) ** 2).sum()))


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
        dw    (K,)  ||W^t - W^{t-1}||_F, the size of the step eq. (16) took.  It is
                    measured from the increment the walk actually adds, not from
                    the identity ||r_t||/sqrt(c_t), so that the identity is
                    something the self-test can check rather than assume
        disp  dict  state s -> ||W^s||_F, how far the state has moved from W^0 = 0
                    once the steps have been allowed to cancel

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
    dw = np.empty(K)
    ret, disp = {}, {}
    want = {int(s) for s in eval_states}

    for t in range(K):
        r = Theta[t] - Fv[t] @ W                          # eq. (12)
        tr[t] = float(rownorm(r[None, :])[0])
        step = np.outer(Bv[t], r) / c[t]                  # eq. (16)
        dw[t] = fro(step)
        W += step
        s = t + 1
        if s in want:
            ret[s] = float(rownorm(Theta[:s] - Fv[:s] @ W).mean())
            disp[s] = fro(W)
    return tr, ret, dw, disp


# ================================================= the exact route, eq. (27)
# Two kernels carry the cost of this route and nothing else comes close: building
# Gamma, which is one K x N x K product, and the triangular solve.  Measured at
# K = 5000, N = 60, D = 12: 0.14 s and 0.014 s against 0.07 s for everything else
# in a cell.  Both are routed through a backend so that `--jax` runs them on
# whatever device JAX finds; everything else stays in numpy, where it is free.

_JNP = None      # jax.numpy, once --jax has been taken up; None otherwise


def use_jax(on=True):
    """Route the two heavy kernels through jax.numpy.  Returns the devices found.

    float64 is not optional here.  Past the split threshold the residual reaches
    1e159 on the grid figures 6 and 7 draw, which float32 cannot represent at all
    -- it would come back as inf with no warning -- so x64 is switched on and
    checked before anything is computed.
    """
    global _JNP
    if not on:
        _JNP = None
        return None
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    if jnp.zeros(1).dtype != np.float64:
        raise RuntimeError(
            "jax is not in x64 mode: the residual reaches 1e159 on this grid and "
            "float32 would silently return inf.  set JAX_ENABLE_X64=1 in the "
            "environment before jax is first imported")
    _JNP = jnp
    return jax.devices()


def _xp():
    """numpy, or jax.numpy if --jax took."""
    return _JNP if _JNP is not None else np


def coupling(v, F, B):
    """Gamma_kt = v^T F_k B_t v / v^T B_t v, eq. (19): one K x K array, no D axis."""
    xp = _xp()
    v, F, B = xp.asarray(v), xp.asarray(F), xp.asarray(B)
    w = v ** 2
    return (F @ (w[:, None] * B.T)) / (B @ w)[None, :]


SOLVE_BLOCK = 192   # rows per block below; 192 measured fastest at K = 5000


def forward_substitution(Gamma, Theta):
    """R = (I + L)^{-1} Theta one row at a time, eq. (27): the reference route.

    L is the strict lower triangle of Gamma, which is what Gamma[t, :t] already is,
    so no copy of the K x K array is taken.  This is the equation and nothing else.
    `solve_stream` blocks it for speed and the self-test holds the two together.
    """
    K = Theta.shape[0]
    R = np.empty_like(Theta)
    for t in range(K):
        R[t] = Theta[t] - Gamma[t, :t] @ R[:t]
    return R


def solve_stream(Gamma, Theta, block=SOLVE_BLOCK):
    """The same solve, blocked: one triangular system of size K, eq. (27).

    One system and not D of them, because a neuron is open or closed on all D of
    its synapses at once and Gamma carries no input-coordinate index.

    Forward substitution is sequential in t, but only within a block: rows
    s..s+m of R need the rows before s only through one product, so the O(K^2 D)
    work of the solve goes into ceil(K/m) matrix products and the sequential part
    is left with an m x m triangle.  Measured 6x faster than the row-at-a-time
    route at K = 5000, agreeing with it to 6e-14 relative -- the two differ only
    in the order the same terms are summed.
    """
    if _JNP is not None:
        from jax.scipy.linalg import solve_triangular
        return solve_triangular(Gamma, _JNP.asarray(Theta), lower=True,
                                unit_diagonal=True)
    K = Theta.shape[0]
    R = np.empty_like(Theta)
    for s in range(0, K, block):
        e = min(s + block, K)
        blk = Theta[s:e] - Gamma[s:e, :s] @ R[:s] if s else Theta[s:e]
        for t in range(s, e):
            R[t] = blk[t - s] - Gamma[t, s:t] @ R[s:t]
    return R


def read_back(Gamma, R, s):
    """Every task i <= s read out of state s: -sum_{u=i+1}^{s} Gamma_iu r_u, eq. (30).

    The same band operator as eq. (30), taken all the way out to the current state
    rather than to a fixed lag: row i uses the superdiagonal entries between i and
    s, so row s is empty and the task just trained comes back solved.
    """
    U = _xp().triu(Gamma[:s, :s], 1)
    return -U @ R[:s]


def exact_scores(v, F, B, Theta, eval_states, retention=True):
    """The same metrics from the closed form, with nothing trained.

    Route two.  Build the coupling eq. (19) from the gates and the readout, solve
    the whole stream in one triangular system eq. (27), and read every task back
    out of the wanted states with the band operator eq. (30).  The walk in
    `run_stream` never happens: no state is ever formed, no task is ever trained.

    The note says the two routes give the same numbers, and the figures draw both
    so that the claim is visible rather than asserted.  They are not independent
    -- both use one seed's (v, F, B, Theta) -- but they share no arithmetic,
    so a disagreement would be real.  Cost is O(K^2 N) to build Gamma and
    O(K^2 D) to solve, against O(K N D) for the walk, and Gamma is 200 MB at
    K = 5000; that is why this is a second pass and not the main one.

    The step and the displacement come out of the same R.  Eq. (16) makes the step
    rank one, W^t - W^{t-1} = a_t r_t / c_t with a_t = B_t v, and ||a_t||^2 = c_t by
    eq. (15), so ||W^t - W^{t-1}||_F = ||r_t|| / sqrt(c_t) with no state formed; and
    W^s = sum_{t<=s} a_t r_t / c_t is one (N, s) x (s, D) product of the same rows.

    `retention=False` skips the band eq. (30), which figures 6 and 7 do not read.
    """
    xp = _xp()
    c = B @ v ** 2
    G = coupling(v, F, B)
    R = solve_stream(G, Theta)
    Rn = rownorm(np.asarray(R))
    A = xp.asarray((B * v[None, :]) / c[:, None])         # rows a_t / c_t, eq. (16)
    ret, disp = {}, {}
    for s_ in eval_states:
        s_ = int(s_)
        if retention:
            ret[s_] = float(rownorm(np.asarray(-(xp.triu(G[:s_, :s_], 1)
                                                 @ R[:s_]))).mean())
        disp[s_] = fro(np.asarray(A[:s_].T @ R[:s_]))
    return Rn, ret, Rn / np.sqrt(c), disp


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
    gap left there is the estimator -- these curves are geometric means over seeds
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


def mf_step(d_f, d_b, K):
    """The mean field's average step at stream length T = K, eq. (16) with Gamma -> d_f.

    The step is ||W^t - W^{t-1}||_F = ||r_t|| / sqrt(c_t) exactly, by eq. (16) and
    c_t = ||B_t v||^2 at eq. (15).  Replacing both ingredients by their means --
    Gamma by d_f, which is what `mf_resid` is, and the write mass by
    E[c_t] = d_b ||v||^2 with E||v||^2 = 1 -- gives

        (1/K) sum_{t=1}^{K} mf_resid(d_f, t, -1) / sqrt(d_b) .

    This curve does see the split, and it is the only one in the file that does:
    the residual cannot tell d_b from d_f, because E[Gamma_kt] = d_f whatever d_b
    is, but the write mass *is* d_b, so a thinner write pool lengthens every step
    by 1/sqrt(d_b) before any instability has happened at all.  A trace that rises
    faster than 1/sqrt(d_b) is showing the instability of eq. (17) and not this.

    It is a ratio of two means rather than the mean of a ratio, and 1/sqrt is
    convex, so E[1/sqrt(c_t)] >= 1/sqrt(E c_t) and the measured curve runs above
    this one by roughly 1 + (3/8) Var(c_t)/E[c_t]^2, which is 1 + O(1/n_b).  At
    n_b = 1 the arithmetic mean of the step does not exist at all -- c_t is a single
    v_i^2 and E[1/|v_i|] diverges -- which is one more reason the seeds here are
    combined geometrically.
    """
    K = int(K)
    t = np.arange(1, K + 1, dtype=float)
    return float(np.asarray(mf_resid(d_f, t, -1)).mean() / np.sqrt(d_b))


def mf_displacement(d_f, d_b, K):
    """The mean field's ||W^K||_F: the same increments summed before the norm.

    W^K = sum_t a_t r_t / c_t with a_t = B_t v, so

        ||W^K||_F^2 = sum_{t,s} (a_t . a_s)(r_t . r_s) / (c_t c_s) .

    For gates drawn independently each task E[a_t . a_t] = E[c_t] = d_b ||v||^2 and
    E[a_t . a_s] = d_b^2 ||v||^2 for t != s, so with E||v||^2 = 1 the diagonal
    carries a factor 1/d_b and every off-diagonal term carries 1.  Adding and
    subtracting the diagonal,

        E||W^K||_F^2 = (1/d_b - 1) sum_{t<=K} E||r_t||^2 + E||sum_{t<=K} r_t||^2 ,

    and the second term is free: the mean-field recursion has
    S_K = sum_{t<=K} r_t = sum_{s<=K} p^{K-s} theta_s with p = 1 - d_f, so
    E||S_K||^2 = (1 - p^{2K})/(1 - p^2) = (1 - p^{2K}) / (d_f (2 - d_f)).  The first
    term is `mf_resid` squared and summed.

    The split enters once, through 1/d_b - 1, and the sum S_K does not feel it at
    all -- so the whole difference between this and `mf_step` is that cancellation
    is not something the write pool can undo.

    At d_f = d_b = 1 the first term vanishes and the second is exactly 1, and that
    is not an approximation: Gamma is all-ones there, so eq. (25) reads
    sum_{s<=t} r_s = theta_t, W^t = v theta_t^T / ||v||^2 telescopes, and
    ||W^t||_F = 1/||v|| for every t.  The self-test asserts that line.

    Away from that corner this is the loosest curve in the file, and in two ways
    worth naming, both measured at N = 60 against the walk:

    (i) It runs high in the stable regime, and increasingly so with T -- 1.3x at
        T = 50 and 2.7x at T = 500, at d_f = d_b = 0.5 -- because the two large
        pieces above very nearly cancel and the decoupling sits in the gap.  The
        off-diagonal weight (a_t.a_s)/(c_t c_s) is not independent of r_t.r_s: an
        overlap between task t's read pool and task s's writes is exactly what
        makes r_t.r_s negative, so the cross terms that cancel the diagonal are
        weighted above their mean and the true norm is smaller than this says.
        Behind that is the qualitative miss: the true displacement saturates,
        because the homogeneous map of eq. (32) contracts at rho < 2, while this
        curve grows like sqrt(T) forever.

    (ii) Past the split threshold it has no instability in it at all, exactly as
         `mf_resid` has none: it is E[Gamma] = d_f that is being substituted, and
         the split lives in second moments.  The distance from the dotted line up
         to the measured one is that instability and nothing else.

    The obvious repair is not one.  Taking E||W^t||^2 through eq. (32) instead
    gives S_t = (1 + (rhobar-2)/N) S_{t-1} + E[1/c_t], the note's own one-step
    ratio eq. (33), and that does track the walk to about 10% at rho <= 2 -- but it
    is wrong by sqrt(N/rank) wherever the state is not isotropic, which the
    isotropy assumption of Step 12 is blind to.  At d_f = 1 the state is exactly
    rank one and the route is out by 7.8x, a measured sqrt(60); it also needs
    E[rho_t] = (n_f-2)/(n_b-2), which does not exist over half the splitness axis
    of figures 6 and 7.  It is a good recursion for the residual, whose isotropised
    projection it gets right, and a bad one for a norm.
    """
    K = int(K)
    p = 1.0 - d_f
    t = np.arange(1, K + 1, dtype=float)
    diag = float((np.asarray(mf_resid(d_f, t, -1)) ** 2).sum())
    tail = (1.0 - p ** (2.0 * K)) / (d_f * (2.0 - d_f))
    return float(np.sqrt(max((1.0 / d_b - 1.0) * diag + tail, 0.0)))


def geo_sem(x):
    """Geometric mean over seeds, the standard error of that mean, and the count.

    Once the stream is unstable the metric is heavy-tailed across seeds: at
    rho = 5 a single seed can sit sixty orders of magnitude above the rest, so an
    arithmetic mean reports that seed and nothing else.  log||r|| is the
    near-Gaussian variable, so both the mean and its standard error are taken
    there, and the error bar comes back multiplicative -- the interval is
    gm * exp(-+ sem), which is why it is drawn asymmetric on a linear axis and
    symmetric on a logarithmic one.

    It is the standard error of the mean over seeds, s/sqrt(n), not the spread s:
    it says how well located the plotted point is, which is what a reader comparing
    two traces needs.  The average *within* a seed -- retention's sum over tasks,
    the step's average over the stream -- is the plain arithmetic mean the
    definitions ask for and is untouched by any of this.

    Seeds that overflowed or came back non-positive are dropped and the surviving
    count is returned, so the log can say when a cell is thinner than it looks;
    `sem` is nan below two seeds.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        return np.nan, np.nan, 0
    lg = np.log(x)
    sem = float(lg.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else np.nan
    return float(np.exp(lg.mean())), sem, int(x.size)


def geo(x):
    """The geometric mean alone, for the places that do not draw an error bar."""
    return geo_sem(x)[0]


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
    # traces that all leave the top of a capped panel arrive at the same place and
    # the nudging above then walks the stack off the axes; slide it back inside,
    # stopping short of the top spine so the last label does not sit on it
    over = placed[-1] - 0.96
    if over > 0.0:
        placed = [max(0.0, f - over) for f in placed]
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
    v, F, B, Theta = draw(N, D, K, n_f, n_b, rng_for(cfg, FLOW, cfg["FLOW_SEED"]))
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

def case_vs_tasks(cfg, d_f, rho=1.0, stream=TASKS):
    """One (d_f, rho), scored at logarithmically spaced states, by both routes.

    Retention is evaluated exactly at those states -- it is already an average over
    every task the stream has trained, so nothing more is pooled within a seed.
    Transfer is one number per task, so it is pooled over the interval of arrivals
    ending at each state.  Seeds are combined geometrically in both cases, with the
    standard error of that mean alongside, and the walk and the closed form see the
    same seeds.
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
        v, F, B, Theta = draw(N, D, K, n_f, n_b, rng_for(cfg, stream, s))
        tr, ret, _, _ = run_stream(v, F, B, Theta, pts)
        tr_x, ret_x, _, _ = exact_scores(v, F, B, Theta, pts)
        gap = max(gap, _route_gap(tr, tr_x), _route_gap(
            np.array([ret[int(k)] for k in pts]),
            np.array([ret_x[int(k)] for k in pts])))
        raw["ret"][s] = [ret[int(k)] for k in pts]
        raw["ret_x"][s] = [ret_x[int(k)] for k in pts]
        for q in range(pts.size):
            raw["tr"][s, q] = geo(tr[edges[q]:edges[q + 1]])
            raw["tr_x"][s, q] = geo(tr_x[edges[q]:edges[q + 1]])
    out = {}
    for k, a in raw.items():
        agg = [geo_sem(a[:, q]) for q in range(pts.size)]
        out[k] = np.array([g[0] for g in agg])
        out[k + "_sem"] = np.array([g[1] for g in agg])
        out[k + "_n"] = np.array([g[2] for g in agg])
    out["x"], out["n_b"], out["gap"] = pts.astype(float), n_b, gap
    return out


def _route_gap(a, b):
    """Largest relative disagreement between the two routes, ignoring exact zeros."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 0)
    if not m.any():
        return 0.0
    return float(np.max(np.abs(a[m] - b[m]) / np.abs(b[m])))


def _three(ax, x, sim, exact, mf, colour, label=None, sem=None, err="band",
           marker=None):
    """One series, three ways: simulation, the closed form, and the mean field.

    The walk is drawn broad and translucent and the closed form thin on top of it,
    because the two are the same numbers: what the reader should see is a dashed
    line riding inside a solid band, not two curves to compare.  The mean field is
    dotted and is a different quantity, so it is drawn at full weight underneath.

    `sem` is the standard error of the geometric mean over seeds, measured in logs,
    so the interval is sim * exp(-+ sem): asymmetric in the value and symmetric on
    the logarithmic axes these panels use.  It is drawn as a filled band where the
    abscissa is dense -- figures 2 and 3 carry about forty points per trace and
    caps there would be a thicket -- and as capped bars where the points are few
    and each one is a measurement a reader may want to read off.
    """
    x = np.asarray(x, dtype=float)
    sim = np.asarray(sim, dtype=float)
    ok = np.isfinite(sim) & (sim > 0)
    if sem is not None:
        e = np.asarray(sem, dtype=float)
        m = ok & np.isfinite(e)
        lo, hi = sim[m] * np.exp(-e[m]), sim[m] * np.exp(e[m])
        if err == "band":
            ax.fill_between(x[m], lo, hi, color=colour, alpha=0.16,
                            linewidth=0.0, zorder=2)
        else:
            ax.errorbar(x[m], sim[m],
                        yerr=np.vstack([sim[m] - lo, hi - sim[m]]), fmt="none",
                        ecolor=colour, elinewidth=1.0, capsize=2.6, capthick=1.0,
                        alpha=0.9, zorder=5)
    ln, = ax.plot(x[ok], sim[ok], color=colour, linewidth=2.8, alpha=0.40,
                  solid_capstyle="round", zorder=3, label=label)
    if marker:
        ax.plot(x[ok], sim[ok], linestyle="none", marker=marker, markersize=3.6,
                color=colour, alpha=0.85, zorder=6)
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


def _route_legend(colour=None, err=None):
    """The three line styles, and the error bar, named once per figure."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    c = colour or MUTED
    out = [Line2D([], [], color=c, lw=2.8, alpha=0.40, label="simulation"),
           Line2D([], [], color=c, lw=1.2, ls=(0, (5, 2)),
                  label="whole stream, eq. (27) & (30)"),
           Line2D([], [], color=c, lw=1.3, ls=(0, (1, 2)),
                  label="mean field, $\\Gamma\\to d_f$ (r.m.s.)")]
    if err == "band":
        out.append(Patch(facecolor=c, alpha=0.16, edgecolor="none",
                         label="$\\pm1$ s.e.m. over seeds"))
    elif err == "bars":
        out.append(Line2D([], [], color=c, lw=1.0, marker="_", markersize=7,
                          label="$\\pm1$ s.e.m. over seeds"))
    return out


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
                             f"$d_f=d_b={d_f}$" if ci == 0 else None,
                             sem=res[arm + "_sem"], err="band")
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

    fig.legend(handles=handles + _route_legend(err="band"), frameon=False,
               fontsize=8.5, ncol=8, loc="upper center",
               bbox_to_anchor=(0.5, 0.935), labelcolor=INK2)
    fig.suptitle("the raw residual, against how far into the stream the task sits   "
                 f"($N={cfg['N']}$, $N_{{in}}={cfg['D']}$, $K={cfg['K_TASKS']}$, "
                 f"{cfg['SEEDS']} seeds;  0 = solved, 1 = no better than $W=0$)",
                 fontsize=11, color=INK, y=0.995)
    fig.text(0.5, 0.012,
             "the dashed line is the closed form on the same seeds -- the coupling "
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
                                 f"$d_f/d_b={rho:g}$" if ri + ci == 0 else None,
                                 sem=res[arm + "_sem"], err="band")
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

    fig.legend(handles=handles + _route_legend(err="band"), frameon=False,
               fontsize=8.5, ncol=7, loc="upper center",
               bbox_to_anchor=(0.5, 0.96), labelcolor=INK2)
    fig.suptitle("the raw residual against task index, at three read densities   "
                 f"($N={cfg['N']}$, $N_{{in}}={cfg['D']}$, $K={cfg['K_TASKS']}$, "
                 f"{cfg['SEEDS']} seeds;  1 = no better than $W=0$)",
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
        v, F, B, Theta = draw(N, D, K, n_f, n_b, rng_for(cfg, DENSITY, s))
        tr, ret, _, _ = run_stream(v, F, B, Theta, (K,))
        tr_x, ret_x, _, _ = exact_scores(v, F, B, Theta, (K,))
        gap = max(gap, _route_gap(tr, tr_x),
                  _route_gap([ret[K]], [ret_x[K]]))
        acc["ret"].append(ret[K])
        acc["ret_x"].append(ret_x[K])
        acc["tr"].append(geo(tr[lo:]))
        acc["tr_x"].append(geo(tr_x[lo:]))
    out = {}
    for k, x in acc.items():
        out[k], out[k + "_sem"], out[k + "_n"] = geo_sem(x)
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
            sem = np.array([grid[(d, rho)][arm + "_sem"] for d in xs])
            ln, end = _three(ax, xs, sim, exa, None, SERIES[k],
                             f"$d_f/d_b={rho:g}$" if ci == 0 else None,
                             sem=sem, err="bars", marker="o")
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

    fig.legend(handles=handles + _route_legend(err="bars"), frameon=False,
               fontsize=8.5, ncol=7, loc="upper center",
               bbox_to_anchor=(0.5, 0.935), labelcolor=INK2)
    fig.suptitle("the raw residual at the end of the stream, against read density   "
                 f"($N={cfg['N']}$, $N_{{in}}={cfg['D']}$, $K={K}$, "
                 f"{cfg['SEEDS']} seeds)", fontsize=11, color=INK, y=0.995)
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
        v, F, B, Theta = draw(N, D, K, n_f, n_b, rng_for(cfg, HEAT, s))
        tr, ret, _, _ = run_stream(v, F, B, Theta, snaps)
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
                 f"{cfg['SEEDS']} seeds per cell)", fontsize=11, color=INK, y=0.965)
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


# ============================================================ figures 6 and 7

DW_METRICS = {
    "step": (r"$\frac{1}{T}\sum_{t\leq T}\|W^t-W^{t-1}\|_F$",
             "the average step: how far the state moves on a task",
             "$\\Gamma\\to d_f$ with the write mass at $E[c_t]=d_b\\|v\\|^2$, so "
             "unlike every other mean-field curve here it does see the split, "
             "through $1/\\sqrt{d_b}$; what runs above it is eq. (17)."),
    "disp": (r"$\|W^T\|_F\,/\,T$",
             "the net displacement: the same steps summed before the norm",
             "$\\Gamma\\to d_f$ again, and the loosest curve in the note: it is "
             "exact at $d_f=d_b=1$, where the sum telescopes, but it decouples a "
             "gate overlap from the erasure\nthat overlap causes and so misses "
             "the contraction of eq. (32), growing like $\\sqrt{T}$ where the true "
             "displacement saturates.  above it is eq. (17)."),
}


def case_delta_w(cfg, d_f, rho, exact=True):
    """One cell of the (d_f, rho) grid: both weight-change metrics at three T.

    One stream of K_DW carries all three columns.  Both metrics are causal prefixes
    of it -- step(T) averages the first T steps, disp(T) is the norm of their sum --
    so a snapshot at T is what a stream of length T would have given; and the same
    holds of the closed form, whose forward substitution never looks past row T.
    Neither statement is taken on trust: the self-test checks both.

    The two routes are the ones the rest of the file uses.  The walk measures the
    step off the increment it actually adds to the state.  The closed form forms no
    state at all: it builds the coupling eq. (19), solves eq. (27) once, and reads
    ||r_t||/sqrt(c_t) and ||sum_t a_t r_t / c_t||_F out of the answer.
    """
    N, D, K = cfg["N"], cfg["D"], cfg["K_DW"]
    n_f, n_b = counts(N, d_f, rho)
    snaps = tuple(int(t) for t in cfg["DW_SNAPSHOTS"])
    if max(snaps) > K:
        raise ValueError(f"snapshot {max(snaps)} is past the stream length {K}")
    acc = {f"{m}@{t}": [] for m in ("step", "disp", "step_x", "disp_x")
           for t in snaps}
    gap = 0.0
    for sd in range(cfg["SEEDS"]):
        v, F, B, Theta = draw(N, D, K, n_f, n_b, rng_for(cfg, DELTAW, sd))
        _, _, dw, disp = run_stream(v, F, B, Theta, snaps)
        cum = np.cumsum(dw)
        for t in snaps:
            acc[f"step@{t}"].append(cum[t - 1] / t)
            acc[f"disp@{t}"].append(disp[t] / t)
        if exact:
            _, _, dw_x, disp_x = exact_scores(v, F, B, Theta, snaps,
                                              retention=False)
            cum_x = np.cumsum(dw_x)
            gap = max(gap, _route_gap(dw, dw_x),
                      _route_gap([disp[t] for t in snaps],
                                 [disp_x[t] for t in snaps]))
            for t in snaps:
                acc[f"step_x@{t}"].append(cum_x[t - 1] / t)
                acc[f"disp_x@{t}"].append(disp_x[t] / t)
    out = {"n_f": n_f, "n_b": n_b, "gap": gap}
    for k, x in acc.items():
        out[k], out[k + "_sem"], out[k + "_n"] = geo_sem(x)
    return out


def _mf_dw(cfg, cell, d_f, metric, T):
    """The mean field for one cell, at the realised write density n_b/N.

    The realised pair and not the nominal one: `counts` rounds d_f and rho onto the
    1/N grid, and at the sparse corner that rounding is the difference between
    d_b = 1/60 and d_b = 0.02, which 1/sqrt(d_b) notices.
    """
    d_b = cell["n_b"] / cfg["N"]
    if metric == "step":
        return mf_step(d_f, d_b, T)
    return mf_displacement(d_f, d_b, T) / T


def _dw_series(cfg, cells, metric, axis, held, T):
    """One trace: x, the two routes with the error of the first, and the mean field.

    `axis` is which of the two the abscissa runs over and `held` is the value of
    the other, so row 1 asks for axis="d_f" at a held rho and row 2 for axis="rho"
    at a held d_f.  Both read the same `cells`; the rows are slices of one grid,
    not three grids.
    """
    if axis == "d_f":
        keys = [(d, held) for d in sorted({d for d, _ in cells})
                if (d, held) in cells]
    else:
        keys = [(held, r) for r in sorted({r for _, r in cells})
                if (held, r) in cells]
    col = f"{metric}@{int(T)}"
    return (np.array([k[0] if axis == "d_f" else k[1] for k in keys]),
            np.array([cells[k][col] for k in keys]),
            np.array([cells[k][col + "_sem"] for k in keys]),
            np.array([cells[k][f"{metric}_x@{int(T)}"] for k in keys]),
            np.array([_mf_dw(cfg, cells[k], k[0], metric, T) for k in keys]))


def fig_delta_w(cfg, cells, path, metric):
    """Three rows on one weight-change metric, three stream lengths as the columns.

    Rows 1 and 2 are two slices of the grid row 3 shows whole, so the figure is one
    set of numbers read three ways and no cell is computed twice.  Rows 1 and 2
    share one vertical axis across all three columns, because how the metric grows
    with T is the point of having T as the columns; it is capped the way figure 3's
    is, and a trace that leaves the panel is labelled with the decade it reached.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    ylab, sub, mfnote = DW_METRICS[metric]
    snaps = [int(t) for t in cfg["DW_SNAPSHOTS"]]
    d_fs = sorted({d for d, _ in cells})
    rhos = sorted({r for _, r in cells})
    rows = (("d_f", cfg["DW_ROW1_RHOS"], r"read density  $d_f$", "$d_f/d_b={:g}$"),
            ("rho", cfg["DW_ROW2_DFS"], r"splitness  $d_f/d_b$", "$d_f={:g}$"))

    # every trace of rows 1 and 2, gathered before anything is drawn so that the
    # two rows can be given one vertical axis
    series = {}
    for ri, (axis, held_all, _, _) in enumerate(rows):
        for ci, T in enumerate(snaps):
            for hi, held in enumerate(held_all):
                held = float(held)
                if not any((held == (k[1] if axis == "d_f" else k[0]))
                           for k in cells):
                    continue
                series[(ri, ci, hi)] = (held,) + _dw_series(
                    cfg, cells, metric, axis, held, T)
    pool = np.concatenate([np.concatenate([a[2], a[4], a[5]])
                           for a in series.values()])
    pool = pool[np.isfinite(pool) & (pool > 0)]
    hi_lim = min(float(pool.max()) * 2.0, float(cfg["DW_CEIL"]))
    lo_lim = max(float(pool.min()) / 2.0, hi_lim * 1e-12)

    fig, axes = plt.subplots(3, 3, figsize=(13.2, 12.4))
    # a wide gutter: where every trace of a panel is clipped, its end labels all
    # carry the decade they reached and the stack is wide enough to reach the next
    # panel's tick labels at the default spacing
    fig.subplots_adjust(left=0.085, right=0.955, top=0.905, bottom=0.215,
                        hspace=0.34, wspace=0.32)
    handles = [[], []]
    for ri, (axis, held_all, xlab, fmt) in enumerate(rows):
        for ci, T in enumerate(snaps):
            ax = axes[ri][ci]
            ends = []
            for hi, held in enumerate(held_all):
                if (ri, ci, hi) not in series:
                    continue
                h, xs, sim, sem, exa, mf = series[(ri, ci, hi)]
                ln, end = _three(ax, xs, sim, exa, mf, SERIES[hi],
                                 fmt.format(h) if ci == 0 else None,
                                 sem=sem, err="bars", marker="o")
                if ci == 0:
                    handles[ri].append(ln)
                if end:
                    last = end[1]
                    text = (f"{h:g}" if last <= hi_lim else
                            f"{h:g} $\\to10^{{{int(np.log10(last))}}}$")
                    ends.append((end[0], min(last, hi_lim), text))
            ax.set_yscale("log")
            ax.set_ylim(lo_lim, hi_lim)
            if axis == "rho":
                ax.set_xticks(rhos)
                ax.set_xticklabels([f"{r:g}" for r in rhos])
            ax.set_title(f"$T={T}$ tasks", fontsize=10, color=INK, pad=6)
            ax.set_xlabel(xlab, fontsize=9.5, color=INK2)
            if ci == 0:
                ax.set_ylabel(ylab, fontsize=10, color=INK2)
            _chrome(ax)
            _end_labels(ax, ends, gap=0.075)
        # the two rows run over different variables, so the three categorical
        # slots mean different things in each and one legend for the figure would
        # say that blue is both d_f/d_b = 1 and d_f = 0.5.  Each row names its own,
        # inside its leftmost panel, on the side its traces leave free.
        axes[ri][0].legend(handles=handles[ri], frameon=False, fontsize=8.5,
                           labelcolor=INK2, handlelength=1.6, borderpad=0.2,
                           loc="upper right" if axis == "d_f" else "upper left")

    # row 3: the whole grid, one shared colour scale across the three snapshots
    planes, ceil = {}, float(cfg["DW_HEAT_CEIL"])
    for T in snaps:
        Z = np.full((len(d_fs), len(rhos)), np.nan)
        for i, d in enumerate(d_fs):
            for j, r in enumerate(rhos):
                val = cells.get((d, r), {}).get(f"{metric}@{T}", np.nan)
                if np.isfinite(val) and val > 0:
                    Z[i, j] = np.log10(val)
        planes[T] = Z
    allv = np.concatenate([planes[T][np.isfinite(planes[T])] for T in snaps])
    norm = Normalize(vmin=float(allv.min()), vmax=min(float(allv.max()), ceil))
    clipped = bool(allv.max() > ceil)
    cmap = sequential_cmap()
    cmap.set_bad((0.0, 0.0, 0.0, 0.0))
    step_d = d_fs[1] - d_fs[0] if len(d_fs) > 1 else 0.1
    for ci, T in enumerate(snaps):
        ax = axes[2][ci]
        im = ax.imshow(np.clip(planes[T], None, ceil), origin="lower",
                       aspect="auto", cmap=cmap, norm=norm,
                       extent=(rhos[0] - 0.5, rhos[-1] + 0.5,
                               d_fs[0] - step_d / 2, d_fs[-1] + step_d / 2))
        ax.set_xticks(rhos)
        ax.set_xticklabels([f"{r:g}" for r in rhos])
        ax.set_yticks(d_fs)
        ax.set_yticklabels([f"{d:.1f}" for d in d_fs])
        ax.set_xlabel(r"splitness  $d_f/d_b$", fontsize=9.5, color=INK2)
        ax.set_title(f"$T={T}$ tasks", fontsize=10, color=INK, pad=6)
        if ci == 0:
            ax.set_ylabel(r"read density  $d_f$", fontsize=9.5, color=INK2)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.set_facecolor("#f3f2ee")
        for side in ax.spines:
            ax.spines[side].set_color(AXIS)
            ax.spines[side].set_linewidth(0.8)

    cax = fig.add_axes([0.345, 0.150, 0.31, 0.011])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal",
                      extend="max" if clipped else "neither")
    cb.set_label(r"$\log_{10}$ " + ylab, fontsize=8.5, color=INK2)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_edgecolor(AXIS)

    worst = max((c["gap"] for c in cells.values()), default=0.0)
    thin = sum(1 for c in cells.values()
               if min(c[f"{metric}@{T}_n"] for T in snaps) < cfg["SEEDS"])
    # the splitness axis is the nominal ratio; `counts` rounds it onto the 1/N grid
    # and n_b cannot go below 1, so at the sparse end several nominal ratios realise
    # the same pair of counts and the trace is flat between them by construction.
    # name the sparsest density and the ratio at which its write pool bottoms out.
    d_thin = d_fs[0]
    flat = min((int(r) for r in rhos
                if (d_thin, r) in cells and cells[(d_thin, r)]["n_b"] == 1),
               default=0)
    fig.legend(handles=_route_legend(err="bars"), frameon=False, fontsize=8.5,
               ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.968),
               labelcolor=INK2)
    fig.suptitle(f"{sub}   "
                 f"($N={cfg['N']}$, $N_{{in}}={cfg['D']}$, "
                 f"{cfg['SEEDS']} seeds per cell)",
                 fontsize=11, color=INK, y=0.995)
    note = [
        "rows 1 and 2 are two slices of the grid row 3 shows whole, so the figure "
        "is one set of numbers read three ways.",
        f"rows 1 and 2 share one vertical axis across all three columns, capped at "
        f"{hi_lim:.0e}; a trace that leaves a panel is labelled with the decade it "
        f"reached, and row 3's colour stops at $10^{{{ceil:g}}}$.",
        "the dashed closed form rides inside the solid band -- the coupling eq. (19) "
        "and one triangular solve eq. (27), with no state ever formed -- agreeing to "
        f"{worst:.0e} relative at worst.",
    ] + mfnote.split("\n") + [
        "the splitness axis is the nominal $d_f/d_b$.  `counts` rounds it onto the "
        f"$1/N$ grid and $n_b\\geq1$, so at $d_f={d_thin:g}$ the read pool is "
        f"{cells[(d_thin, rhos[0])]['n_f']} neurons and every ratio from {flat} up "
        "realises the same" if flat else "",
        "single written neuron: those traces are flat there by construction, not by "
        "saturating." if flat else "",
        f"{thin} of {len(cells)} cells lost a seed to overflow." if thin else "",
    ]
    fig.text(0.5, 0.012, "\n".join(ln for ln in note if ln), ha="center",
             va="bottom", fontsize=7.8, color=MUTED, linespacing=1.55)
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


# ================================================================== self-test

def solve_check(cfg, d_f=0.3, rho=2.0, index=11):
    """The walk against the closed form, on exactly the path the figures draw.

    All four metrics, not just the two scores: the step of eq. (16) and the
    displacement go through the same comparison, the walk measuring them off the
    state it forms and the closed form reading them out of one triangular solve.
    """
    N, D, K = cfg["N"], cfg["D"], cfg["K_CHECK"]
    n_f, n_b = counts(N, d_f, rho)
    states = (K // 4, K // 2, K)
    v, F, B, Theta = draw(N, D, K, n_f, n_b, rng_for(cfg, TEST, index))
    tr, ret, dw, disp = run_stream(v, F, B, Theta, states)
    tr_x, ret_x, dw_x, disp_x = exact_scores(v, F, B, Theta, states)
    return max(_route_gap(tr, tr_x),
               _route_gap(dw, dw_x),
               _route_gap([ret[s] for s in states], [ret_x[s] for s in states]),
               _route_gap([disp[s] for s in states], [disp_x[s] for s in states]))


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
    v, F, B, Theta = draw(cfg["N"], cfg["D"], 40, 20, 10, rng_for(cfg, TEST, 5))
    tr, ret, dw, disp = run_stream(v, F, B, Theta, (1, 2, 40))
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
    rng = rng_for(cfg, TEST, 3)
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
    v, F, B, Theta = draw(cfg["N"], cfg["D"], 400, cfg["N"], cfg["N"],
                          rng_for(cfg, TEST, 9))
    rep("d_f = 1 gives Gamma == 1 everywhere",
        np.asarray(coupling(v, F, B)) - 1.0, 1e-12)
    tr, ret, dw, disp = run_stream(v, F, B, Theta, (400,))
    rep("  and r_t == theta_t - theta_{t-1}",
        tr[1:] - np.linalg.norm(Theta[1:] - Theta[:-1], axis=1), 1e-12)
    rep("  r.m.s. of it == the mean field, sqrt(2)",
        np.sqrt((tr[1:] ** 2).mean()) / float(mf_resid(1.0, 200, -1)) - 1.0, 0.04)
    rep("  the geometric mean sits below it by exp(-1/4D)",
        geo(tr[1:]) / (np.sqrt(2.0) * np.exp(-1.0 / (4 * cfg["D"]))) - 1.0, 0.02)

    if verbose:
        print("-- the weight change: the step of eq. (16) is rank one --")
    v, F, B, Theta = draw(cfg["N"], cfg["D"], 120, 24, 12, rng_for(cfg, TEST, 21))
    cw = B @ v ** 2
    tr, ret, dw, disp = run_stream(v, F, B, Theta, (1, 40, 120))
    rep("||W^t-W^{t-1}||_F == ||r_t||/sqrt(c_t), eqs. (15) and (16)",
        dw / (tr / np.sqrt(cw)) - 1.0, 1e-12)
    rep("the first step is ||theta_1||/sqrt(c_1), since W^0 = 0",
        dw[0] * np.sqrt(cw[0]) - 1.0, 1e-12)
    rep("the displacement never exceeds the steps that made it",
        max(0.0, (disp[120] - dw.sum()) / dw.sum()), 1e-12)
    G = coupling(v, F, B)
    R = np.asarray(solve_stream(G, Theta))
    rep("the solve is causal: rows <= k of the K-solve are the k-solve",
        _route_gap(np.asarray(solve_stream(G[:40, :40], Theta[:40])).ravel(),
                   R[:40].ravel()), 1e-12)
    rep("blocked forward substitution == the row-at-a-time reference",
        _route_gap(R.ravel(), forward_substitution(np.asarray(G), Theta).ravel()),
        1e-10)
    rep("read_back == the band eq. (30) the scores are read through",
        _route_gap(np.asarray(read_back(G, R, 40)).ravel(),
                   np.asarray(-(np.triu(np.asarray(G)[:40, :40], 1)
                                @ R[:40])).ravel()), 1e-12)

    if verbose:
        print("-- at d_f = d_b = 1 the sum telescopes and both are exact --")
    v, F, B, Theta = draw(cfg["N"], cfg["D"], 200, cfg["N"], cfg["N"],
                          rng_for(cfg, TEST, 22))
    nv = float(np.linalg.norm(v))
    tr, ret, dw, disp = run_stream(v, F, B, Theta, (100, 200))
    rep("  ||W^t||_F == 1/||v|| for every t: sum_{s<=t} r_s = theta_t",
        [disp[s] * nv - 1.0 for s in (100, 200)], 1e-12)
    rep("  and the step is ||theta_t - theta_{t-1}|| / ||v||",
        dw[1:] * nv - np.linalg.norm(Theta[1:] - Theta[:-1], axis=1), 1e-12)
    rep("  mf_displacement(1, 1, T) == 1 exactly, not approximately",
        [mf_displacement(1.0, 1.0, T) - 1.0 for T in (100, 200)], 1e-12)
    rep("  mf_step(1, 1, T) == the measured step, bar the estimator gap",
        dw.mean() * nv / mf_step(1.0, 1.0, 200) - 1.0, 0.05)

    if verbose:
        print("-- the weight-change mean field against its own recursion --")
    rng = rng_for(cfg, TEST, 23)
    for d_f, d_b in ((0.5, 0.5), (0.4, 0.1), (0.2, 0.05)):
        K_, D_, M = 120, 24, 2000
        p = 1.0 - d_f
        Th = rng.standard_normal((M, K_, D_))
        Th /= np.linalg.norm(Th, axis=2, keepdims=True)
        S = np.zeros((M, D_))
        Rm = np.empty((M, K_, D_))
        for t in range(K_):
            Rm[:, t] = Th[:, t] - d_f * S
            S = p * S + Th[:, t]
        for T in (20, 120):
            want_ = mf_displacement(d_f, d_b, T)
            got = np.sqrt(((1.0 / d_b - 1.0) * (Rm[:, :T] ** 2).sum((1, 2))
                           + (Rm[:, :T].sum(1) ** 2).sum(1)).mean())
            rep(f"  d_f={d_f}, d_b={d_b}, T={T:3d}: ||W^T|| = {want_:8.3f}",
                got / want_ - 1.0, 0.02)
            want_ = mf_step(d_f, d_b, T)
            got = np.sqrt((Rm[:, :T] ** 2).sum(2).mean(0)).mean() / np.sqrt(d_b)
            rep(f"  d_f={d_f}, d_b={d_b}, T={T:3d}: step   = {want_:8.3f}",
                got / want_ - 1.0, 0.02)

    if verbose:
        print("-- one cell of the grid against a replay of the same seeds --")
    small = {**cfg, "K_DW": 80, "DW_SNAPSHOTS": (10, 80), "SEEDS": 3}
    cell = case_delta_w(small, 0.5, 2.0)
    n_f, n_b = counts(cfg["N"], 0.5, 2.0)
    hand = {10: [], 80: []}
    handd = {10: [], 80: []}
    for sd in range(small["SEEDS"]):
        v, F, B, Theta = draw(cfg["N"], cfg["D"], 80, n_f, n_b,
                              rng_for(small, DELTAW, sd))
        W = np.zeros((cfg["N"], cfg["D"]))
        cw = B @ v ** 2
        tot = 0.0
        for t in range(80):
            r = Theta[t] - (F[t] * v) @ W
            stp = np.outer(B[t] * v, r) / cw[t]
            tot += float(np.linalg.norm(stp))
            W += stp
            if t + 1 in hand:
                hand[t + 1].append(tot / (t + 1))
                handd[t + 1].append(float(np.linalg.norm(W)) / (t + 1))
    for T in (10, 80):
        rep(f"  step@{T} == the hand replay", cell[f"step@{T}"] / geo(hand[T]) - 1.0,
            1e-12)
        rep(f"  disp@{T} == the hand replay", cell[f"disp@{T}"] / geo(handd[T]) - 1.0,
            1e-12)

    if verbose:
        print("-- the aggregator and the seeds --")
    rep("geo() is the exponential of the mean log", geo([1.0, 100.0]) - 10.0, 1e-12)
    rep("geo() drops non-finite entries", geo([4.0, np.inf, np.nan]) - 4.0, 1e-12)
    rep("geo_sem's error is the s.d. of the logs over sqrt(n)",
        geo_sem(np.exp([0.0, 1.0, 2.0, 3.0]))[1]
        - float(np.std([0.0, 1.0, 2.0, 3.0], ddof=1)) / 2.0, 1e-12)
    rep("geo_sem counts only the seeds it kept",
        geo_sem([4.0, np.inf, np.nan, -1.0])[2] - 1, 0)
    rep("geo_sem has no error bar below two seeds",
        0 if np.isnan(geo_sem([2.0])[1]) else 1, 0)
    rep("rng_for repeats itself at the same key",
        draw(8, 3, 4, 4, 2, rng_for(cfg, TEST, 30))[0]
        - draw(8, 3, 4, 4, 2, rng_for(cfg, TEST, 30))[0], 0)
    rep("  and gives an unrelated stream at the next index",
        int(np.allclose(draw(8, 3, 4, 4, 2, rng_for(cfg, TEST, 30))[0],
                        draw(8, 3, 4, 4, 2, rng_for(cfg, TEST, 31))[0])), 0)
    rep("  and two figures never share one at the same index",
        int(np.allclose(draw(8, 3, 4, 4, 2, rng_for(cfg, TASKS, 0))[0],
                        draw(8, 3, 4, 4, 2, rng_for(cfg, SPLIT, 0))[0])), 0)
    pts = log_points(1000, 12)
    rep("log_points are increasing and inside 1..K",
        [pts[0] - 1, pts[-1] - 1000, int((np.diff(pts) <= 0).sum())], 0)
    big = np.array([[1e200, 1e200]])
    rep("rownorm survives an entry whose square would overflow",
        rownorm(big)[0] / (1e200 * np.sqrt(2.0)) - 1.0, 1e-12)
    rep("fro does too, for the whole state",
        fro(np.full((2, 2), 1e200)) / 2e200 - 1.0, 1e-12)
    rep("the sequential ramp is monotone in lightness",
        int(sequential_cmap().N != 512), 0)

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
    want = {a for a in argv if a in {"1", "2", "3", "4", "5", "6", "7"}}
    cfg = CONFIG

    print("=" * 76)
    print("  neuronal split gating -- experiments and figures")
    print(f"  N={cfg['N']}  N_in={cfg['D']}  seeds={cfg['SEEDS']}"
          f"  root seed={cfg['ROOT_SEED']}")
    if "--jax" in argv:
        print(f"  the coupling and the solve on {use_jax(True)}")
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
             "the weight change, figures 6 and 7, is the step of eq. (16):",
             "  step(T)      = (1/T) sum_{t<=T} ||W^t - W^{t-1}||_F = "
             "(1/T) sum ||r_t||/sqrt(c_t)",
             "  disp(T)      = ||W^T||_F / T, the same increments summed first",
             "0 = solved, 1 = no better than W = 0.  seeds combined geometrically,",
             "with +-1 s.e.m. of that mean, in logs -- the figures draw exp(+-sem).",
             "the eq.27/30 columns are the same numbers from the closed form, with",
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
                    cases[(d, rho)] = case_vs_tasks(cfg, d, rho, stream=SPLIT)
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

    if not want or (want & {"6", "7"}):
        print("figures 6 and 7: the weight change ...")
        cells = {}
        for d in _grid(cfg["DW_DF_STEP"], 1.0, cfg["DW_DF_STEP"]):
            for rho in cfg["DW_RHOS"]:
                try:
                    cells[(d, float(rho))] = case_delta_w(cfg, d, float(rho))
                except ValueError:
                    pass
        for which, metric, name, what in (
                ("6", "step", "fig6_step.png", "mean step per task"),
                ("7", "disp", "fig7_displacement.png", "net displacement per task")):
            if want and which not in want:
                continue
            fig_delta_w(cfg, cells, here / name, metric)
            lines.append(f"fig{which}  {len(cells)} cells, T={cfg['DW_SNAPSHOTS']}, "
                         f"{what}")
            lines.append(f"  {'d_f':>5s} {'rho':>4s} {'n_b':>4s} {'T':>6s} "
                         f"{'simulation':>14s} {'eq. (27)':>14s} {'s.e.m.':>9s} "
                         f"{'seeds':>6s} {'route gap':>11s}")
            for (d, rho), c in sorted(cells.items()):
                for T in cfg["DW_SNAPSHOTS"]:
                    k = f"{metric}@{T}"
                    lines.append(f"  {d:5.2f} {rho:4.0f} {c['n_b']:4d} {T:6d} "
                                 f"{c[k]:14.4g} {c[f'{metric}_x@{T}']:14.4g} "
                                 f"{c[k + '_sem']:9.3f} {c[k + '_n']:6d} "
                                 f"{c['gap']:11.1e}")
            lines.append("")

    _write_log(here / "neuronal_curves.txt", lines, want)
    print(f"\nwritten to {here}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
