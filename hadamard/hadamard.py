"""Split gating with entrywise masks, the read mask and the write mask kept apart.

A synapse of task t is read if F_t selects it and written if B_t does, and the
nesting B . F = B says the written set sits inside the read one: a synapse may be
written only if it is read.  Past that the two densities are free,

    n_f = d_f N   ones per column of F        (eq. 10)
    n_b = d_b N   ones per column of B,       n_b <= n_f
    rho = d_f / d_b >= 1                      the split ratio,

and rho = 1 is the unsplit case B_t = F_t this file used to be restricted to.

One question, the same one as before.  A stream of K tasks is trained in order,
each to convergence, each reading through its own read mask and writing only into
the synapses its own write mask selects.  There are two ways to find the residual
r_t that each task inherits from the ones before it:

  the stream      run it.  Task by task,
                      r_t = theta_t - v^T (F_t . W^{t-1})
                      W^t = W^{t-1} + B_t . ( v (r_t / c_t) ),  c_t = v^T B_t v
                  which is the converged flow, eq. (18), not an approximation to
                  it -- the flow moves on a line and its endpoint is closed form.
                  Read through F, write through B; the write mass c is B's.

  the direct      don't run it.  Build the coupling between every pair of tasks,
  solve               Gamma[k,t,a] = sum_i F[k,i,a] B[t,i,a] v_i^2
                                     -----------------------------
                                        sum_i B[t,i,a] v_i^2
                  -- reader's F against writer's B -- keep its strict lower
                  triangle L (only s < t: a later task reading through an earlier
                  task's writes), and solve
                      (I + L[a]) R[:,a] = Theta[:,a],   a = 1..D
                  which is eq. (29), one triangular system per input coordinate.

Do they agree?  Exactly, and for the same reason as at rho = 1: the derivation of
(29) from (18) uses the masks only through the nesting, never through F = B, and
takes no expectation and makes no approximation.  So any gap is rounding, on the
whole (d_f, d_b) half-plane and not only on its diagonal.  That is what this file
measures.

What the split does change is the geometry and the spread, and both are measured
here too:

    obliquity   the rank-one map (19) is symmetric iff B = F.  Its obliquity is
                the squared sine between B^[a] v and F^[a] v, which the nesting
                collapses to 1 - c^[a]_t / (v^T F^[a]_t v), about 1 - 1/rho.

    spread      Gamma[k,t,a] is an average of d_f-coins over the write pool of
                task t, so its mean is d_f whatever d_b is, while its spread is
                set by the pool size n_b.  Thinning B at fixed F therefore leaves
                the coupling where it was and makes it noisier.

The other free choice is how COARSE the gating is, which the note leaves open in a
line -- a row of ones is a gated neuron, a single entry a gated synapse, and a block
of entries a gated dendritic branch, and the algebra does not distinguish them.  A
neuron here is a row of W and its N_in = D entries are the synapses on it, so the
granularity is a grouping of the D columns:

    N_d   dendrites per neuron, a whole factor of D, carrying N_s = D / N_d
          synapses each.  Columns a and a' share a mask iff a // N_s == a' // N_s,
          and different dendrites are drawn independently.

    N_d = D       one synapse per dendrite, every column independent: SYNAPTIC
                  gating, what this file did before N_d existed, and the default
    N_d = 1       one dendrite carrying the row, so the masks are row-constant and
                  a neuron is on or off as a whole: NEURONAL gating
    1 < N_d < D   blocks of N_s columns gated together: DENDRITIC gating

Nothing in the algebra notices.  Gamma[k,t,a], c[t,a] and L^[a] simply repeat across
the N_s columns of a dendrite -- at N_d = 1 the D triangular solves collapse into
one, which is the row-constant case the note writes out beside eq. (29) -- while
Theta does not repeat, so R still differs column by column.  The saving is left on
the table: D solves of a system that is N_s-fold degenerate cost nothing here worth
a special case.

What the coarsening does move is the ensemble.  Everything this file reduces over
the column axis -- the spread of Gamma, the loss, the residual norm -- is then an
average over N_d independent draws rather than D, so it gets noisier as the gating
coarsens even though its mean stays put.  The per-column predictions are untouched.

    N     width, synapses per input coordinate
    D     input dimension (the note's d); the D coordinates never mix
    K     stream length, number of tasks
    d_f   read density;  n_f = round(d_f N) ones in every column of every F_t
    d_b   write density; n_b = round(d_b N) ones in every column of every B_t,
          nested inside that task's read set.  Defaults to d_f, the unsplit case.
    N_d   dendrites per neuron, a whole factor of D.  Defaults to D: one synapse
          per dendrite, the independent columns this file started with.

Shapes:  v (N,)   F, B (K,N,D)   Theta (K,D)   Gamma (K,K,D)   L (D,K,K)   R (K,D)

There is a third way, used only to check the first two: integrate the flow with a
stepper that is told nothing, and see whether it traces out the curve the closed
form claims.  `integrate_stream` does that.

Run the file to do the comparison.  It prints the tables and writes four files
next to itself, unless a directory is given on the command line:

    split_gating_check.txt        the same tables, and the (d_f, d_b) grid
    split_gating_check.png        four panels per run: the gap seed by seed, the
                                  residuals, the gap against K, and the coupling
                                  the stream actually saw
    split_gating_check_flow.png   the learning curve of one stream per run, read
                                  three ways, one full-width row each
    split_gating_check_split.png  the (d_f, d_b) grid: the gap, the coupling and
                                  the cost of the stream, against rho

API, with d_b defaulting to d_f everywhere it appears:

    counts(N, d_f, d_b)             -> n_f, n_b, with the nesting checked
    dendrites(D, N_d)               -> N_d, N_s, with the grouping checked
    draw(N, D, K, d_f, d_b, seed, N_d)
                                    -> v, F, B, Theta, (n_f, n_b)
    coupling(v, F, B)               -> Gamma (K,K,D), write masses c (K,D)
    obliquity(v, F, B, c)           -> 1 - c/(v'Fv) per (t,a), and one Pi checked
    coupling_spread(Gamma, v, B, c, n_f)
                                    -> mean, s.d., the s.d. the pool accounts for
    solve_direct(Gamma, Theta)      -> R (K,D), the whole-stream triangular solve
    run_stream(v, F, B, Theta)      -> R (K,D), final W, fit error, rates C
    integrate_stream(v, F, B, Theta)-> time, loss, inherited residuals
    compare(N, D, K, d_f, d_b, seeds, flow, N_d) -> dict of per-seed arrays
    summarise(res)                  -> printable mean / s.d. / worst table
    sweep(param, values, ..., N_d)  -> one of N, D, K, d_f, d_b, rho varied, the
                                       gating granularity held fixed
    split_grid(d_f_values, d_b_values, ..., N_d) -> the whole rectangle
    format_grid(grid)               -> the rectangle as tables
    plot_comparison(res), plot_flow(res), plot_split_grid(grid)
"""

import sys
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------ the draw

def counts(N, d_f, d_b):
    """The two per-column counts, exact, and the nesting checked before anything runs.

    Eq. (10) asks for n_f = d_f N and n_b = d_b N as counts, not as expectations,
    so the densities are rounded to the 1/N grid once here and everything
    downstream uses the realised pair.  The only real constraint is the nesting:
    a column cannot write more synapses than it reads.
    """
    n_f = int(round(d_f * N))
    n_b = int(round(d_b * N))
    if not 0 < n_f <= N:
        raise ValueError(f"d_f={d_f} gives n_f={n_f} with N={N}; need 1 <= n_f <= N")
    if n_b < 1:
        raise ValueError(
            f"d_b={d_b} gives n_b={n_b} with N={N}: an empty write column has no "
            "write mass and the coupling is undefined there.  Raise N or d_b")
    if n_b > n_f:
        raise ValueError(
            f"d_b={d_b} > d_f={d_f} (n_b={n_b} > n_f={n_f}): the nesting (9) says a "
            "synapse may be written only if it is read, so rho = d_f/d_b >= 1")
    return n_f, n_b


def dendrites(D, N_d=None):
    """N_d dendrites per neuron and the N_s = D/N_d synapses on each, checked once here.

    A neuron is a row of W and its N_in = D entries are the synapses on it, so how
    coarse the gate is, is a grouping of the D columns: columns a and a' carry one
    mask iff a // N_s == a' // N_s, and different dendrites are independent.

        N_d = D       N_s = 1    every column independent -- synaptic gating
        N_d = 1       N_s = D    one mask for the whole row -- neuronal gating
        1 < N_d < D              blocks of N_s columns -- dendritic gating

    N_d has to divide D exactly.  A ragged last block would leave one dendrite
    carrying a different number of synapses from the rest, and d_f would no longer
    mean one thing across the row, so it is a hard error and not a rounding, in the
    way that n_f and n_b are roundings.  None means D, the independent columns,
    which is the default everywhere and is what this code did before N_d existed.
    """
    N_d = D if N_d is None else int(N_d)
    if not 1 <= N_d <= D:
        raise ValueError(
            f"N_d={N_d} with D={D}: need 1 <= N_d <= D, from one dendrite carrying "
            "the whole row (neuronal) to one synapse per dendrite (synaptic)")
    if D % N_d:
        raise ValueError(
            f"N_d={N_d} does not divide D={D}: the dendrites would not all carry the "
            f"same number of synapses.  Use a factor of D: "
            f"{[k for k in range(1, D + 1) if D % k == 0]}")
    return N_d, D // N_d


def draw(N, D, K, d_f, d_b=None, seed=0, N_d=None):
    """One realisation: readout, nested masks, teachers.

    Each DENDRITE of each task gets its own uniform permutation of 0..N-1; the first
    n_f places in it are read and the first n_b written, and the N_s columns on that
    dendrite are handed the same pair.  So B sits inside F by construction, every
    column carries exactly its count, and any two columns belonging to different
    tasks -- or to different dendrites of one task -- are independent uniform
    subsets, which makes the pairwise overlap d_fb^{kt} = n_f/N exactly for k != t,
    independent of n_b and of N_d.  This is the independent-mask case, the one the
    free array (11) is free not to be.

    At N_d = D the repeat is the identity and the draw is the fully independent one,
    variate for variate.  Below that the draw consumes K N_d N variates rather than
    K D N, so a seed does not name the same teachers at two granularities; seeds are
    the sample axis, not a paired control.

    The teachers are arbitrary; nothing anywhere uses anything about them.  Unit
    rows just put the loss on a scale where 1 means "output nothing".
    """
    d_b = d_f if d_b is None else d_b
    n_f, n_b = counts(N, d_f, d_b)
    N_d, N_s = dendrites(D, N_d)
    rng = np.random.default_rng(seed)

    v = rng.standard_normal(N)/np.sqrt(N)
    v = v / np.linalg.norm(v)

    # rank[k,g,i] is where synapse i landed in DENDRITE g of task k's permutation;
    # the repeat hands each dendrite's mask to the N_s columns sitting on it, so
    # column a takes dendrite a // N_s.  N_s = 1 makes the repeat the identity.
    rank = np.argsort(np.argsort(rng.random((K, N_d, N)), axis=-1), axis=-1)
    F = np.repeat(np.moveaxis((rank < n_f).astype(float), 1, -1),  # (K,N_d,N)
                  N_s, axis=-1)                                    # -> (K, N, D)
    B = np.repeat(np.moveaxis((rank < n_b).astype(float), 1, -1), N_s, axis=-1)

    Theta = rng.standard_normal((K, D))
    Theta = Theta / np.linalg.norm(Theta, axis=1, keepdims=True)

    return v, F, B, Theta, (n_f, n_b)


# --------------------------------------------------------------- the coupling

def coupling(v, F, B):
    """Gamma[k,t,a] and the write masses c[t,a].

    Gamma[k,t,a] is the fraction of the write mass task t laid down in input
    coordinate a that task k reads back: the reader's F against the writer's B,
    eq. (21).  Reader index first, writer second, and it is not symmetric -- a
    later task reading through an earlier task's writes is a different event from
    the reverse, and with F != B it is not even symmetric within one task pair for
    a symmetric reason.

    Two values fall out with no computation: Gamma[t,t,a] = 1 by the nesting, and
    every entry lies in [0,1], being a fraction of a mass.
    """
    w = v ** 2
    c = np.einsum("i,tia->ta", w, B)                 # write mass, eq. (17)
    if np.any(c <= 0):
        raise ValueError("a write column is empty; the coupling is undefined there")
    num = np.einsum("i,kia,tia->kta", w, F, B, optimize=True)
    return np.clip(num / c[None, :, :], 0.0, 1.0), c


def obliquity(v, F, B, c):
    """How far the rank-one map (19) is from symmetric, and one explicit check of it.

    Pi^[a]_t = B^[a]_t v v^T F^[a]_t / c^[a]_t is a projection for any nested pair,
    and symmetric iff B^[a]_t = F^[a]_t.  The two directions it pairs are B^[a] v
    and F^[a] v, whose squared sine the nesting collapses to

        1 - (u.z)^2 / (|u|^2 |z|^2) = 1 - c^[a]_t / (v^T F^[a]_t v),

    since u.z = |u|^2 = c.  It is zero exactly at rho = 1 and about 1 - 1/rho
    otherwise, with no matrix built.  One Pi is built anyway, for column 0 of task
    0, so that idempotency (20) is checked against something and not just asserted.
    """
    f_mass = np.einsum("i,tia->ta", v ** 2, F)
    obl = 1.0 - c / f_mass

    u, z = B[0, :, 0] * v, F[0, :, 0] * v
    Pi = np.outer(u, z) / c[0, 0]
    idem = np.abs(Pi @ Pi - Pi).max()
    trace = abs(np.trace(Pi) - 1.0)
    return obl, idem, trace


def coupling_spread(Gamma, v, B, c, n_f):
    """What the off-diagonal coupling looks like, measured and predicted.

    Measured: the mean and s.d. of Gamma[k,t,a] over every ordered pair k != t and
    every input coordinate.

    Predicted: conditional on the write pool of task t and on v, Gamma[k,t,a] is
    sum_i p_i f^{(k)}_ia with weights p_i = b^{(t)}_ia v_i^2 / c^[a]_t summing to
    one, and f^{(k)}_.a a uniform n_f-subset of N.  So its mean is d_f = n_f/N
    exactly -- whatever d_b is -- and its variance is

        d_f (1 - d_f) [ (sum_i p_i^2) N/(N-1) - 1/(N-1) ],

    the bracket being the sampling-without-replacement correction.  With near-flat
    weights sum p_i^2 is about 1/n_b, so the spread falls like 1/sqrt(n_b): the
    split does not move the coupling, it only makes it noisier.

    Both statements are per (t,a) and neither notices N_d.  What N_d moves is the
    MEASUREMENT: at N_d < D each distinct value of Gamma[k,t,.] appears N_s times in
    the sample below, so the mean and the s.d. are still unbiased but carry N_d
    independent columns rather than D.
    """
    K = Gamma.shape[0]
    N = v.size
    off = ~np.eye(K, dtype=bool)
    g = Gamma[off]                                   # (K(K-1), D)

    d_f = n_f / N
    p2 = np.einsum("i,tia->ta", v ** 4, B) / c ** 2   # sum_i p_i^2, per (t,a)
    var = d_f * (1.0 - d_f) * (p2 * N / (N - 1.0) - 1.0 / (N - 1.0))
    return g.mean(), g.std(), np.sqrt(np.clip(var, 0.0, None).mean()), g.ravel()


# ------------------------------------------------------ the two ways to get R

def solve_direct(Gamma, Theta):
    """R from the whole stream at once: (I + L[a]) R[:,a] = Theta[:,a], eq. (29).

    L[a] is the strict lower triangle of Gamma[:,:,a], so only the couplings with
    s < t survive and the system is triangular.  I + L[a] is unit lower
    triangular, so its determinant is 1 for every realisation -- there is no
    invertibility condition and nothing to regularise, and nothing about that
    depends on the split.
    """
    K, _, D = Gamma.shape
    L = np.tril(np.moveaxis(Gamma, -1, 0), -1)       # (D, K, K)
    R = np.linalg.solve(np.eye(K)[None] + L, Theta.T[..., None])[..., 0]
    return R.T                                       # (K, D)


def run_stream(v, F, B, Theta):
    """R from running the stream, one task at a time, eq. (18).

    Each task is trained to convergence before the next arrives.  No integration
    is done because none is needed: the flow has a fixed left factor B^[a]_t v, so
    the state moves on a line and the endpoint is the line's end.  The read is
    through F_t, the write through B_t, and applying the read to the update
    returns theta_t exactly by the nesting, so the task just trained sits at zero
    loss; `fit` reports how exactly.

    The write masses C[t,a] come back too.  They are the rates: a coordinate
    relaxes at c_{t,a}, so the whole curve the flow traces out is fixed by R and C
    together, which is what `flow_curves` uses.  Thinning B slows the stream down
    as well as changing what it writes.
    """
    K, N, D = F.shape
    w = v ** 2
    W = np.zeros((N, D))
    R = np.empty((K, D))
    C = np.empty((K, D))
    fit = 0.0

    for t in range(K):
        c = w @ B[t]
        r = Theta[t] - np.einsum("i,ia,ia->a", v, F[t], W)
        R[t] = r
        C[t] = c
        W = W + B[t] * np.outer(v, r / c)
        fit = max(fit, np.abs(np.einsum("i,ia,ia->a", v, F[t], W) - Theta[t]).max())

    return R, W, fit, C


# ------------------------------------------- the flow, written out in time

PER_TASK = 240      # points recorded inside each task's window
DECAY = 20.0        # window width, in slowest-coordinate time constants
SUBSTEPS = 4        # integrator steps between two recorded points, at least
MAX_RATE_STEP = 0.1 # ceiling on c dtau for the fastest coordinate in a task


def flow_curves(R, C, per_task=PER_TASK, decay=DECAY):
    """The loss along the flow, one window of flow time per task, laid end to end.

    This is the closed form, and it assumes what `integrate_stream` checks.
    Within task t the read error obeys de_a/dtau = -c_{t,a} e_a -- the flow has a
    fixed left factor, so the D coordinates never mix and each relaxes at its own
    rate, the rate being the write mass of B and not of F -- so

        e_a(tau) = r_{t,a} exp(-c_{t,a} tau),   loss_t(tau) = 1/2 sum_a e_a^2.

    At tau = 0 that is 1/2 ||r_t||^2, the value the task starts from, and it is
    the whole content of the residual.  Feed it R and C from the stream and it
    draws eq. (18); feed it R from the direct solve and the same C -- which is a
    property of the masks, not of any training -- and it draws eq. (29).

    Each task gets its own window, `decay` of its own slowest time constant wide,
    rather than one window sized by the slowest coordinate anywhere in the stream.
    A thin write mask makes the write masses small and ragged, so one window for
    all of them would draw most tasks as vertical lines.

    Returns time (K, per_task) and loss (K, per_task), one row per task, so the
    rows can be drawn separately and the jump between tasks is a gap, not a line.
    """
    K, D = R.shape
    span = decay / C.min(axis=1)                       # (K,)
    start = np.concatenate([[0.0], np.cumsum(span)[:-1]])
    tau = span[:, None] * np.linspace(0.0, 1.0, per_task)[None, :]
    loss = 0.5 * np.einsum("ta,tap->tp", R ** 2,
                           np.exp(-2.0 * C[:, :, None] * tau[:, None, :]))
    return start[:, None] + tau, loss


def integrate_stream(v, F, B, Theta, per_task=PER_TASK, decay=DECAY,
                     substeps=SUBSTEPS, max_rate_step=MAX_RATE_STEP):
    """The stream a third time, by integrating the gradient flow and nothing else.

    Nothing here is told the answer.  The state is stepped with a fourth-order
    explicit scheme (classical Runge-Kutta) from

        dW/dtau = B_t . ( v (theta_t - v^T (F_t . W)) ),

    the masked flow (6) with both masks in their own places, the loss is read off
    the state as it goes, and the residual each task inherits is whatever the
    previous task's integration happened to leave behind.  So this tests both
    things the closed form asserts: that the relaxation inside a task is
    exponential at rate c_{t,a}, and that the residual handed down the stream is
    the one eq. (29) predicts.

    "Trained to convergence" becomes a finite window here, `decay` slowest time
    constants wide, which leaves a relative error of order exp(-decay) for the
    next task to inherit.  That truncation, and not the integrator, is what sets
    the agreement of the residuals; raising `decay` drives it down, and changing
    the step does not move it at all.  The agreement of the curves is the other
    way round -- it is the step, spent at the top of each window where the fast
    coordinates are still alive -- so the two numbers this file reports have two
    different causes and should not be read as one.

    Keeping it that way takes some care once the masks can be split.  The window
    is set by the slowest coordinate of the task and the step by the fastest, and
    an explicit stepper is accurate on a decaying mode only while c dtau is small
    -- past about 2.8 it is not even stable.  A thin write mask makes the write
    masses both small and ragged, so the two rates can differ by a hundred, and a
    step tied to the recording grid alone would integrate the fast coordinates
    wrongly or not at all.  So `substeps` is a floor, and the count actually used
    holds c dtau below `max_rate_step` for the fastest coordinate in each task.

    Returns time (K, per_task), loss (K, per_task) and the inherited residuals
    R (K, D) measured at each window's start.
    """
    K, N, D = F.shape
    c = np.einsum("i,tia->ta", v ** 2, B)
    span = decay / c.min(axis=1)                       # one window per task
    start = np.concatenate([[0.0], np.cumsum(span)[:-1]])
    frac = np.linspace(0.0, 1.0, per_task)

    W = np.zeros((N, D))
    time = np.empty((K, per_task))
    loss = np.empty((K, per_task))
    R = np.empty((K, D))

    for t in range(K):
        Ft, Bt, theta = F[t], B[t], Theta[t]
        tau = span[t] * frac
        nsub = max(substeps,
                   int(np.ceil(c[t].max() * (tau[1] - tau[0]) / max_rate_step)))
        dt = (tau[1] - tau[0]) / nsub

        def rhs(state):
            e = theta - np.einsum("i,ia,ia->a", v, Ft, state)
            return Bt * np.outer(v, e)

        for p in range(per_task):
            e = theta - np.einsum("i,ia,ia->a", v, Ft, W)
            loss[t, p] = 0.5 * (e ** 2).sum()
            if p == 0:
                R[t] = e
            if p + 1 < per_task:
                for _ in range(nsub):
                    k1 = rhs(W)
                    k2 = rhs(W + 0.5 * dt * k1)
                    k3 = rhs(W + 0.5 * dt * k2)
                    k4 = rhs(W + dt * k3)
                    W = W + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        time[t] = start[t] + tau

    return time, loss, R


# ------------------------------------------------------ the comparison itself

def compare(N, D, K, d_f, d_b=None, seeds=range(20), flow=True, N_d=None):
    """The two routes, on `len(seeds)` independent realisations, at one (d_f, d_b).

    Returns per-seed arrays so the statistics can be done downstream however you
    like.  The one to look at is `gap`, the largest disagreement between the
    stream's residuals and the direct solve's, and `gap_rel`, the same divided by
    the size of the residuals themselves.

    `amp` is the largest row sum of (I + L[a])^-1.  It is not a property of the
    code but of the stream: the factor by which rounding in Gamma is entitled to
    show up in R.  Rounding enters multiplied by that and by the size of R itself,
    so the dimensionless quantity to read the gap against is

        gap_amp = gap / (eps * amp * |R|),

    which stays below one everywhere, while the raw gap and gap/(eps*amp) both
    climb once the write pool is thin enough to make R large.  The worst case
    allowed by Gamma in [0,1] is 2^(K-1) for the amplification; it does not
    happen, because the couplings are all positive and of one size and the partial
    sums behave like a stable first-order recursion -- though thinning B widens
    them, and a wide coupling is what makes the amplification grow.

    The split shows up in `obliq`, `gam_sd` and `gam_sd_pred`: how far from
    symmetric the update is, and how much wider the coupling gets as the write
    pool is thinned.  `gam_mean` should sit at d_f however narrow that pool is.

    `flow=False` skips the integration of seed 0, which is all that is expensive
    here; sweeps and grids pass it.
    """
    d_b = d_f if d_b is None else d_b
    n_f, n_b = counts(N, d_f, d_b)
    N_d, N_s = dendrites(D, N_d)
    seeds = list(seeds)
    keys = ("gap", "gap_rel", "gap_amp", "fit", "amp", "diag", "resid", "loss",
            "gam_mean", "gam_sd", "gam_sd_pred", "obliq", "idem")
    out = {k: np.empty(len(seeds)) for k in keys}
    out["seeds"] = np.array(seeds)
    resid_curves = np.empty((len(seeds), K))
    eps = np.finfo(float).eps

    for j, seed in enumerate(seeds):
        v, F, B, Theta, _ = draw(N, D, K, d_f, d_b, seed, N_d=N_d)
        Gamma, c = coupling(v, F, B)

        R_direct = solve_direct(Gamma, Theta)
        R_stream, W, fit, C = run_stream(v, F, B, Theta)

        gam_mean, gam_sd, gam_sd_pred, gam_off = coupling_spread(Gamma, v, B, c, n_f)
        obl, idem, _ = obliquity(v, F, B, c)

        if j == 0:
            # one seed's worth of the flow itself, for the learning-curve figure:
            # three readings of one curve.  The integrator is told nothing, the
            # closed form is eq. (18) with the stream's residuals, and the third
            # is the same closed form with eq. (29)'s residuals, which cost no
            # training at all.  `flow_gap` is the worst disagreement between the
            # integrated curve and the closed form, relative to the height each
            # task starts from, and `box_gap` the same question asked of the
            # residuals alone.
            out["flow_seed"] = seed
            out["gamma_off"] = gam_off
            if flow:
                time_i, loss_i, R_int = integrate_stream(v, F, B, Theta)
                _, loss_stream = flow_curves(R_stream, C)
                _, loss_direct = flow_curves(R_direct, c)
                out["flow_time"] = time_i
                out["flow_loss_int"] = loss_i
                out["flow_loss_stream"] = loss_stream
                out["flow_loss_direct"] = loss_direct
                out["box_time"] = time_i[:, 0]
                out["box_loss"] = 0.5 * (R_direct ** 2).sum(axis=1)
                out["flow_gap"] = (np.abs(loss_i - loss_stream).max(axis=1)
                                   / loss_stream[:, 0]).max()
                out["box_gap"] = np.abs(R_int - R_direct).max()

        gap = np.abs(R_stream - R_direct).max()
        scale = np.abs(R_stream).max()

        L = np.tril(np.moveaxis(Gamma, -1, 0), -1)
        amp = np.abs(np.linalg.inv(np.eye(K)[None] + L)).sum(axis=-1).max()

        # what the stream cost: every task's loss measured against the end state
        Theta_hat_end = np.einsum("i,kia,ia->ka", v, F, W)
        loss = 0.5 * ((Theta_hat_end - Theta) ** 2).sum(axis=1).mean() / D

        out["gap"][j] = gap
        out["gap_rel"][j] = gap / scale
        out["gap_amp"][j] = gap / (eps * amp * scale)
        out["fit"][j] = fit
        out["amp"][j] = amp
        out["diag"][j] = np.abs(np.einsum("tta->ta", Gamma) - 1.0).max()
        out["resid"][j] = np.linalg.norm(R_stream, axis=1).mean()
        out["loss"][j] = loss
        out["gam_mean"][j] = gam_mean
        out["gam_sd"][j] = gam_sd
        out["gam_sd_pred"][j] = gam_sd_pred
        out["obliq"][j] = obl.mean()
        out["idem"][j] = idem
        resid_curves[j] = np.linalg.norm(R_stream, axis=1)

    out["resid_curves"] = resid_curves
    out["N"], out["D"], out["K"] = N, D, K
    out["d_f"], out["d_b"] = d_f, d_b
    out["n_f"], out["n_b"] = n_f, n_b
    out["N_d"], out["N_s"] = N_d, N_s
    out["d_f_realised"], out["d_b_realised"] = n_f / N, n_b / N
    out["rho"] = n_f / n_b
    return out


def gating_name(N_d, D):
    """What to call a grouping of D columns into N_d dendrites."""
    return "synaptic" if N_d == D else "neuronal" if N_d == 1 else "dendritic"


def gating_lines(N_d, N_s, D, indent="  "):
    """The granularity in words, and what it does to every average taken over D."""
    out = [f"{indent}the gating is {N_d} dendrite{'' if N_d == 1 else 's'} of "
           f"{N_s} synapse{'' if N_s == 1 else 's'} per neuron "
           f"({gating_name(N_d, D)}):"]
    if N_d == D:
        out.append(f"{indent}  every column carries its own mask, so the column "
                   f"averages above have {D} independent terms")
    else:
        out.append(f"{indent}  Gamma, c and L repeat across each block of {N_s} "
                   f"columns, so the column")
        out.append(f"{indent}  averages above have {N_d} independent term"
                   f"{'' if N_d == 1 else 's'} and not {D}")
    return out


def header(res):
    """The one line that says which run this is: the settings, the gating, the seeds."""
    return (f"N={res['N']}  D={res['D']}  K={res['K']}  "
            f"d_f={res['d_f_realised']:.3f} (n_f={res['n_f']})  "
            f"d_b={res['d_b_realised']:.3f} (n_b={res['n_b']})  "
            f"rho={res['rho']:.2f}  "
            f"N_d={res['N_d']} (N_s={res['N_s']}, "
            f"{gating_name(res['N_d'], res['D'])})  "
            f"{len(res['seeds'])} seeds")


def summarise(res):
    """Mean, spread and worst case of each per-seed quantity, as a printable table."""
    n = len(res["seeds"])
    eps = np.finfo(float).eps
    rows = [
        ("gap  (stream vs direct solve)", res["gap"]),
        ("gap / |R|", res["gap_rel"]),
        ("gap / (eps * amp * |R|)", res["gap_amp"]),
        ("fit  (zero loss on current task)", res["fit"]),
        ("max |Gamma_tt - 1|  (the nesting)", res["diag"]),
        ("max |Pi^2 - Pi|  (one column)", res["idem"]),
        ("||(I+L)^-1||_inf", res["amp"]),
        ("mean off-diagonal Gamma", res["gam_mean"]),
        ("s.d. of off-diagonal Gamma", res["gam_sd"]),
        ("  the write pool accounts for", res["gam_sd_pred"]),
        ("obliquity 1 - c/(v'Fv)", res["obliq"]),
        ("mean ||r_t||", res["resid"]),
        ("mean loss at end of stream", res["loss"]),
    ]
    lines = [header(res), "-" * 78,
             f"  {'':<34}{'mean':>11}{'s.d.':>11}{'worst':>11}"]
    for name, a in rows:
        sd = a.std(ddof=1) if n > 1 else 0.0
        lines.append(f"  {name:<34}{a.mean():>11.3e}{sd:>11.3e}{a.max():>11.3e}")
    lines.append("-" * 78)
    lines.append(f"  machine epsilon is {eps:.2e};  "
                 f"worst gap is {res['gap'].max() / eps:.1f} of it")
    lines.append(f"  the coupling should sit at d_f = {res['d_f_realised']:.3f} "
                 f"whatever d_b is, and the obliquity at 1 - 1/rho = "
                 f"{1.0 - 1.0 / res['rho']:.3f}")
    lines.extend(gating_lines(res["N_d"], res["N_s"], res["D"]))
    if "flow_gap" in res:
        lines.append(
            f"  seed {res['flow_seed']} integrated: curve within {res['flow_gap']:.2e} "
            f"of the closed form, residuals within {res['box_gap']:.2e} of eq. (29);")
        lines.append(
            f"  the residuals are the finite training window, exp(-{DECAY:.0f}) = "
            f"{np.exp(-DECAY):.1e}; the curve is the step")
    return "\n".join(lines)


# ---------------------------------------------------------------- the sweeps

def sweep(param, values, N=200, D=8, K=24, d_f=0.4, d_b=None, seeds=range(10),
          flow=False, N_d=None):
    """Repeat the comparison while one of N, D, K, d_f, d_b, rho varies.

    `rho` is the split ratio: the value given is d_f/d_b, so d_b = d_f/rho and the
    read mask is held fixed while the write pool is thinned.  Sweeping `d_b` does
    the same thing in the other parametrisation.  Rounding to the 1/N grid happens
    in `counts`, so ask for rho values the width can actually represent.

    `N_d` is held fixed across the sweep, not varied: it is the gating granularity,
    and the point of a sweep is to move one thing.  Sweeping `D` with an explicit
    N_d therefore fails on the first value D is not a multiple of, which is the
    honest outcome -- N_d = 3 does not mean the same gate at D = 6 and at D = 8.

    Returns the values and, for each, the per-seed mean and worst of every
    quantity `compare` produces, plus the realised densities.
    """
    if param not in ("N", "D", "K", "d_f", "d_b", "rho"):
        raise ValueError("param must be one of N, D, K, d_f, d_b, rho")
    keys = ("gap", "gap_rel", "gap_amp", "fit", "amp", "resid", "loss",
            "gam_mean", "gam_sd", "gam_sd_pred", "obliq")
    out = {"param": param, "values": np.asarray(values, dtype=float)}
    for k in keys:
        out[k] = np.empty(len(values))
        out[k + "_max"] = np.empty(len(values))
    out["rho_realised"] = np.empty(len(values))

    base = dict(N=N, D=D, K=K, d_f=d_f, d_b=d_f if d_b is None else d_b, N_d=N_d)
    for i, val in enumerate(values):
        kw = dict(base)
        if param == "rho":
            kw["d_b"] = kw["d_f"] / float(val)
        elif param in ("d_f", "d_b"):
            kw[param] = float(val)
            if param == "d_f" and base["d_b"] > float(val):
                raise ValueError(
                    f"d_f={val} with d_b={base['d_b']} breaks the nesting; pass a "
                    "smaller d_b, or sweep d_b or rho instead")
        else:
            kw[param] = int(val)
        res = compare(seeds=seeds, flow=flow, **kw)
        for k in keys:
            out[k][i] = res[k].mean()
            out[k + "_max"][i] = res[k].max()
        out["rho_realised"][i] = res["rho"]
    return out


def split_grid(d_f_values, d_b_values, N=40, D=4, K=20, seeds=range(8), N_d=None):
    """The whole (d_f, d_b) rectangle, skipping the cells the nesting forbids.

    Every combination with d_b <= d_f and n_b >= 1 is run; the rest come back NaN.
    The question the grid answers is the one the file exists for -- whether the
    stream and the triangular solve still agree away from the diagonal d_f = d_b
    -- and the columns beside the gap say what the split did to the stream while
    they were agreeing.

    Returns a dict of (len(d_f_values), len(d_b_values)) arrays.
    """
    keys = ("gap", "gap_max", "gap_amp_max", "amp", "amp_max", "resid", "loss",
            "gam_mean", "gam_sd", "gam_sd_pred", "obliq", "rho", "n_b")
    shape = (len(d_f_values), len(d_b_values))
    out = {k: np.full(shape, np.nan) for k in keys}
    out["d_f_values"] = np.asarray(d_f_values, dtype=float)
    out["d_b_values"] = np.asarray(d_b_values, dtype=float)
    out["N"], out["D"], out["K"], out["seeds"] = N, D, K, list(seeds)
    out["N_d"], out["N_s"] = dendrites(D, N_d)

    for i, d_f in enumerate(d_f_values):
        for j, d_b in enumerate(d_b_values):
            try:
                counts(N, d_f, d_b)
            except ValueError:
                continue
            res = compare(N, D, K, d_f, d_b, seeds=seeds, flow=False, N_d=N_d)
            out["gap"][i, j] = res["gap"].mean()
            out["gap_max"][i, j] = res["gap"].max()
            out["gap_amp_max"][i, j] = res["gap_amp"].max()
            out["amp_max"][i, j] = res["amp"].max()
            for k in ("amp", "resid", "loss", "gam_mean", "gam_sd", "gam_sd_pred",
                      "obliq"):
                out[k][i, j] = res[k].mean()
            out["rho"][i, j] = res["rho"]
            out["n_b"][i, j] = res["n_b"]
    return out


def format_grid(grid):
    """The grid as a stack of small tables, read across d_b and down d_f.

    The first two are the claim: the raw gap in units of machine epsilon, which
    the split does inflate, and the same gap divided by the amplification and the
    size of the residuals, which is what the inflation is made of and which stays
    below one everywhere.  The rest is what the split did while they agreed.
    """
    eps = np.finfo(float).eps
    d_f_values, d_b_values = grid["d_f_values"], grid["d_b_values"]
    panels = (("worst gap, in units of machine epsilon", grid["gap_max"] / eps),
              ("the same, over eps * amplification * |R|", grid["gap_amp_max"]),
              ("worst amplification ||(I+L)^-1||_inf", grid["amp_max"]),
              ("mean off-diagonal Gamma  (should be d_f, down the row)",
               grid["gam_mean"]),
              ("s.d. of off-diagonal Gamma, measured", grid["gam_sd"]),
              ("s.d. of off-diagonal Gamma, from the write pool", grid["gam_sd_pred"]),
              ("mean loss at end of stream", grid["loss"]))

    lines = [f"the (d_f, d_b) grid   N={grid['N']}  D={grid['D']}  K={grid['K']}  "
             f"N_d={grid['N_d']} ({gating_name(grid['N_d'], grid['D'])})  "
             f"{len(grid['seeds'])} seeds;  blank where the nesting forbids the cell",
             "-" * 78]
    for title, table in panels:
        lines.append(f"  {title}")
        lines.append("  " + "d_f \\ d_b".ljust(12)
                     + "".join(f"{b:>10.3f}" for b in d_b_values))
        for i, d_f in enumerate(d_f_values):
            cells = "".join("         ." if np.isnan(x) else f"{x:>10.3g}"
                            for x in table[i])
            lines.append("  " + f"{d_f:<12.3f}" + cells)
        lines.append("")
    lines.append("-" * 78)
    return "\n".join(lines)


# --------------------------------------------------------------------- plots

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#898781"
GREEN, PURPLE = "#2f8f5b", "#8452c4"


def plot_flow(res, ax=None):
    """The learning curve of one stream, read three ways, on its own wide axis.

    Three traces of one quantity.  The integrated one is the only one that ran
    anything; the two closed forms differ only in where their residuals came from,
    the stream or the triangular solve.  If the closed form is right they are
    indistinguishable, and the boxes -- the height eq. (29) says each task begins
    at, obtained without training -- sit on the left end of every tooth.

    The scale is logarithmic because the relaxation is exponential: a task is a
    straight line, and the jump to the next task is the gap between teeth.  The
    slope of a tooth is the write mass, so thinning B tilts every tooth toward the
    horizontal without changing where it starts.
    """
    import matplotlib.pyplot as plt

    if "flow_time" not in res:
        raise ValueError("this result was made with flow=False; nothing to draw")
    if ax is None:
        fig, ax = plt.subplots(figsize=(13.0, 4.6))

    # Each task gets an equal slot, because the windows do not have equal widths:
    # they are 20 time constants of their own slowest coordinate, and a thin write
    # mask makes those vary by an order of magnitude from task to task.  Drawn
    # against raw flow time, one slow task would take half the axis and crush the
    # rest into vertical strokes.  Within a slot the horizontal coordinate is the
    # fraction of that task's window, so every tooth descends at the same slope
    # and the picture compares heights, which is what the residuals are.
    raw = res["flow_time"]
    K = raw.shape[0]
    time = (np.arange(K)[:, None]
            + (raw - raw[:, :1]) / (raw[:, -1:] - raw[:, :1]))
    for t in range(K):
        first = t == 0
        ax.semilogy(time[t], res["flow_loss_int"][t], color=GREY, linewidth=3.0,
                    alpha=0.55, solid_capstyle="round",
                    label="gradient flow, integrated" if first else None)
        ax.semilogy(time[t], res["flow_loss_stream"][t], color=BLUE,
                    linewidth=1.2,
                    label="closed form, stream residuals (18)" if first else None)
        ax.semilogy(time[t], res["flow_loss_direct"][t], color=ORANGE,
                    linewidth=1.0, linestyle="--",
                    label="closed form, direct residuals (29)" if first else None)
    ax.semilogy(np.arange(K), res["box_loss"], linestyle="none", marker="s",
                markersize=11, markerfacecolor="none", markeredgecolor=ORANGE,
                markeredgewidth=1.6,
                label=r"direct solve, $\frac{1}{2}\|r_t\|^2$")

    ax.set_xlabel("task, and the fraction of its window "
                  r"($20/c_{\min}$ of flow time each)")
    ax.set_ylabel(r"$\frac{1}{2}\|\hat\theta_t - \theta_t\|^2$")
    ax.set_title(f"the stream as it actually runs  "
                 f"({header(res).rsplit('  ', 1)[0]}, seed {res['flow_seed']})",
                 pad=30)
    # each window runs down to exp(-2 decay) of its start, seventeen decades, which
    # would draw every task as a vertical line; the axis is cut five decades below
    # the shallowest tooth, where the descent still reads as the straight line an
    # exponential should be
    top, bottom = res["box_loss"].max(), res["box_loss"].min()
    ax.set_ylim(bottom * 1e-15, top * 3.0)
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="lower left",
              bbox_to_anchor=(0.0, 1.005))
    ax.annotate(
        f"integrated vs closed form: {res['flow_gap']:.1e} of each task's height "
        f"(the step, spent at the top of each tooth)\n"
        f"integrated residuals vs eq. (29): {res['box_gap']:.1e}  "
        f"(the finite window, $e^{{-{DECAY:.0f}}}$ = {np.exp(-DECAY):.0e})",
        xy=(0.008, 0.03), xycoords="axes fraction", fontsize=8, color=GREY,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2.0))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    return ax


def plot_comparison(res, axes=None):
    """Four panels: the gap per seed, the residuals, the gap against K, the coupling.

    The gap panel is logarithmic with machine epsilon drawn, because that is the
    only scale on which "they agree" means anything.  The residual panel shows the
    two routes on top of each other; they are two readings of one quantity, so
    they share one axis.  The fourth is the coupling this run actually saw: it
    should be centred on d_f wherever d_b sits, and as wide as the write pool
    allows.
    """
    import matplotlib.pyplot as plt

    if axes is None:
        fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.4))
    eps = np.finfo(float).eps
    floor = 1e-20

    # 1. every seed's gap
    ax = axes[0]
    y = np.maximum(res["gap"], floor)
    ax.semilogy(res["seeds"], y, marker="o", linestyle="none", color=BLUE)
    ax.axhline(eps, color=GREY, linewidth=1.0, linestyle=":")
    ax.annotate("machine $\\epsilon$", xy=(res["seeds"][0], eps), xytext=(2, 4),
                textcoords="offset points", fontsize=8, color=GREY)
    ax.set_xlabel("seed")
    ax.set_ylabel("max |R$_{stream}$ - R$_{direct}$|")
    ax.set_title("the gap, seed by seed")
    ax.set_ylim(min(y.min(), eps) / 30, max(y.max(), eps) * 30)

    # 2. the residuals the stream inherits
    ax = axes[1]
    curves = res["resid_curves"]
    t = np.arange(curves.shape[1])
    lo, mid, hi = np.percentile(curves, [10, 50, 90], axis=0)
    ax.fill_between(t, lo, hi, color=BLUE, alpha=0.18, linewidth=0)
    ax.plot(t, mid, color=BLUE)
    ax.set_xlabel("task index t")
    ax.set_ylabel(r"$\|r_t\|$")
    ax.set_title("residual each task inherits\n(median, 10-90% over seeds)")

    # 3. the gap as the stream gets longer.  It does drift upward, and the
    # amplification is why: rounding in Gamma reaches R multiplied by
    # ||(I+L)^-1||_inf, so the two curves belong on one axis, both dimensionless
    # and both measured at their worst over the same seeds.  Their ratio, not the
    # gap alone, is the flat thing.  A thin write pool moves both up together --
    # by orders of magnitude at small n_b and long K, which is why the axis is
    # logarithmic here and was not when the masks could not be split.
    ax = axes[2]
    Ks = [k for k in (4, 8, 16, 32, 64, 128) if k <= max(32, 4 * res["K"])]
    sw = sweep("K", Ks, N=res["N"], D=res["D"], d_f=res["d_f"], d_b=res["d_b"],
               seeds=res["seeds"][:5], N_d=res["N_d"])
    ax.plot(sw["values"], sw["gap_max"] / eps, marker="o", color=ORANGE,
            label=r"worst gap / $\epsilon$")
    ax.plot(sw["values"], sw["amp_max"], marker="s", markerfacecolor="none",
            color=BLUE, label=r"worst $\|(I+L)^{-1}\|_\infty$")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("K")
    ax.set_ylabel("dimensionless")
    ax.set_title("the gap tracks the amplification,\nnot the stream length")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    # 4. the coupling one seed saw: mean at d_f, width set by the write pool
    ax = axes[3]
    g = res["gamma_off"]
    ax.hist(g, bins=30, color=BLUE, alpha=0.55, edgecolor="none")
    ax.axvline(res["d_f_realised"], color=ORANGE, linewidth=1.4)
    ax.annotate(f"$d_f$ = {res['d_f_realised']:.3f}",
                xy=(res["d_f_realised"], 1.0), xycoords=("data", "axes fraction"),
                xytext=(4, -12), textcoords="offset points", fontsize=8,
                color=ORANGE)
    ax.set_xlabel(r"$\Gamma_{kt}^{[a]}$,  $k \neq t$")
    ax.set_ylabel("count")
    ax.set_title(f"the coupling, seed {res['flow_seed']}\n"
                 f"s.d. {res['gam_sd'][0]:.3f} vs {res['gam_sd_pred'][0]:.3f} "
                 f"predicted")

    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(True, color="#e1e0d9", linewidth=0.8)
        ax.set_axisbelow(True)
    return axes


def plot_split_grid(grid, axes=None):
    """The grid against the split ratio: agreement, coupling, and what it cost.

    One line per d_f, walking rightward as the write pool is thinned.

    The first panel is the claim.  The raw gap (filled) does climb with the split,
    and the amplification (open) climbs with it: thinning B widens the coupling,
    a wide coupling makes (I+L)^-1 big, and rounding in Gamma reaches R multiplied
    by it.  The two rise together, which is the statement that nothing but
    rounding is happening.

    The second is what the split did to the coupling: the mean sits on its own d_f
    (dotted) wherever d_b goes, while the bar -- one s.d. either side -- fans out
    like 1/sqrt(n_b), open squares marking the width the write pool accounts for.

    The third is what it cost the stream.
    """
    import matplotlib.pyplot as plt

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.6))
    eps = np.finfo(float).eps
    colors = [BLUE, ORANGE, GREEN, PURPLE, GREY]

    for i, d_f in enumerate(grid["d_f_values"]):
        col = colors[i % len(colors)]
        rho = grid["rho"][i]
        ok = ~np.isnan(rho)
        if not ok.any():
            continue
        lab = f"$d_f$ = {d_f:.2f}"
        x, mean, sd = rho[ok], grid["gam_mean"][i][ok], grid["gam_sd"][i][ok]

        axes[0].plot(x, grid["gap_max"][i][ok] / eps, marker="o", color=col,
                     label=lab)
        axes[0].plot(x, grid["amp_max"][i][ok], marker="s", linestyle="--",
                     markerfacecolor="none", color=col)

        axes[1].errorbar(x, mean, yerr=sd, marker="o", color=col, capsize=3,
                         linewidth=1.0, label=lab)
        axes[1].plot(x, mean + grid["gam_sd_pred"][i][ok], marker="s",
                     markerfacecolor="none", linestyle="none", markersize=5,
                     color=col)
        axes[1].plot(x, mean - grid["gam_sd_pred"][i][ok], marker="s",
                     markerfacecolor="none", linestyle="none", markersize=5,
                     color=col)
        axes[1].axhline(d_f, color=col, linestyle=":", linewidth=1.0)

        axes[2].plot(x, grid["loss"][i][ok], marker="o", color=col, label=lab)

    axes[0].set_yscale("log")
    axes[0].set_ylabel("dimensionless")
    axes[0].set_title("the split raises the gap\nby raising the amplification")
    axes[0].annotate("filled: worst gap / $\\epsilon$\nopen: "
                     r"$\|(I+L)^{-1}\|_\infty$",
                     xy=(0.03, 0.80), xycoords="axes fraction", fontsize=8,
                     color=GREY)
    axes[1].set_ylim(-0.05, 1.35)
    axes[1].set_ylabel(r"$\Gamma_{kt}$, mean $\pm$ s.d.")
    axes[1].set_title("the split widens the coupling,\nit does not move it")
    axes[1].annotate("dotted: $d_f$;  open: the width the pool predicts",
                     xy=(0.03, 0.97), xycoords="axes fraction", fontsize=8,
                     color=GREY, va="top")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("mean loss at end of stream")
    axes[2].set_title("what the split cost the stream")

    for ax in axes:
        ax.set_xlabel(r"split ratio $\rho = d_f / d_b$")
        ax.set_xscale("log", base=2)
        ax.legend(frameon=False, fontsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(True, color="#e1e0d9", linewidth=0.8)
        ax.set_axisbelow(True)
    return axes


# ------------------------------------------------------------------ the files

def save_figure(results, path, dpi=200):
    """One png, one run per row of four panels, each row labelled by its settings.

    Drawing goes through the Agg backend, which needs no display, so the file is
    written the same way whether or not anything is watching.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = list(results)
    fig = plt.figure(figsize=(15.5, 3.9 * len(results)), layout="constrained")
    rows = fig.subfigures(len(results), 1) if len(results) > 1 else [fig]

    for row, res in zip(rows, results):
        axes = row.subplots(1, 4)
        row.suptitle(header(res), fontsize=10, color=GREY, x=0.01, ha="left")
        plot_comparison(res, axes=axes)

    path = Path(path)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def save_flow_figure(results, path, dpi=200):
    """The learning curves, one full-width row per run, in a file of their own.

    They get their own file because they want the width: the boxes have to be big
    enough to read against three traces, and three panels of other things beside
    them would not leave the room.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = [r for r in results if "flow_time" in r]
    if not results:
        return None
    fig, axes = plt.subplots(len(results), 1, squeeze=False,
                             figsize=(13.0, 4.6 * len(results)),
                             layout="constrained")
    for ax, res in zip(axes[:, 0], results):
        plot_flow(res, ax=ax)

    path = Path(path)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def save_split_figure(grid, path, dpi=200):
    """The (d_f, d_b) grid, three panels, in a file of its own."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), layout="constrained")
    plot_split_grid(grid, axes=axes)
    fig.suptitle(f"N={grid['N']}  D={grid['D']}  K={grid['K']}  "
                 f"{len(grid['seeds'])} seeds", fontsize=10, color=GREY,
                 x=0.01, ha="left")
    path = Path(path)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def save_summary(results, path, grid=None):
    """The same tables that go to the terminal, one after another, as plain text."""
    path = Path(path)
    text = "\n\n".join(summarise(res) for res in results)
    if grid is not None:
        text += "\n\n" + format_grid(grid)
    path.write_text(text + "\n")
    return path


def run_all(configs, seeds=range(20), outdir=None, stem="split_gating_check",
            grid=None):
    """Every configuration compared, printed, and written to the files.

    `grid`, if given, is the keyword dict for `split_grid`; it adds the table and
    the fourth figure.  Returns the results, the grid, and a dict of the paths
    written -- the files are a side effect, not the only place the numbers end up.
    (The unsplit version of this file returned `results, txt, png, flow_png`; the
    paths moved into the dict when the fourth file arrived.)
    """
    outdir = Path(outdir) if outdir is not None else Path(__file__).resolve().parent
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    for kw in configs:
        res = compare(seeds=seeds, **kw)
        print(summarise(res))
        print()
        results.append(res)

    grid_out = None
    if grid is not None:
        grid_out = split_grid(**grid)
        print(format_grid(grid_out))
        print()

    files = {"txt": save_summary(results, outdir / f"{stem}.txt", grid=grid_out),
             "png": save_figure(results, outdir / f"{stem}.png"),
             "flow": save_flow_figure(results, outdir / f"{stem}_flow.png")}
    if grid_out is not None:
        files["split"] = save_split_figure(grid_out, outdir / f"{stem}_split.png")
    for path in files.values():
        if path is not None:
            print(f"wrote {path}")
    return results, grid_out, files


# ---------------------------------------------------------------------- main

# one read density, three write densities under it: the unsplit case and two
# thinnings of it.  Everything else is held fixed, so the rows differ only in rho.
CONFIGS = (dict(N=20, D=4, K=40, d_f=0.50, d_b=0.50),      # rho = 1
           dict(N=20, D=4, K=40, d_f=0.50, d_b=0.25),      # rho = 2
           dict(N=20, D=4, K=40, d_f=0.50, d_b=0.10))      # rho = 5

GRID = dict(d_f_values=(0.2, 0.4, 0.8),
            d_b_values=(0.025, 0.05, 0.1, 0.2, 0.4, 0.8),
            N=40, D=4, K=200, seeds=range(8))

if __name__ == "__main__":
    run_all(CONFIGS, seeds=range(20), grid=GRID,
            outdir=sys.argv[1] if len(sys.argv) > 1 else None)
