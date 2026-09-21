"""Transfer and retention as functions of the lag, the whole family in one picture.

Companion to `hadamard.py`, which asks whether the stream and the triangular solve
agree on the residual each task *inherits*.  They do, exactly.  This file asks the
same question of the whole lag family, on both sides of zero, and draws the answer.

One task is singled out -- the anchor, task t* = K/2, chosen so that half the stream
lies on either side of it -- and read out of every state the stream visits:

    r_t*^(Delta) = theta_t* - thetahat_t*^(t*+Delta),      -t* <= Delta <= K - t*

    Delta < 0   the state is |Delta| tasks BEFORE the anchor was trained: it has seen
                everything up to task t* - |Delta| and has NOT yet seen the |Delta| - 1
                tasks immediately preceding the anchor, nor the anchor itself.  This is
                transfer -- what the network already does on a task it has not been
                given.  At Delta = -t* the state is W^0 = 0 and the residual is
                theta_t* itself.

    Delta = 0   the anchor, just trained.  Zero, by eq. (18).

    Delta > 0   the state is Delta tasks AFTER.  This is retention: how much of the
                anchor survives Delta further tasks of writing.

Both sides come out of eq. (29), R = (I + L)^-1 Theta, with no second solve and no
rerun of anything.  Reading eq. (24) at the two states and subtracting:

    Delta >= 0      R^(Delta)[:,a] = -U^[a](Delta) R[:,a]                  eq. (32)
                    the first Delta SUPERdiagonals of Gamma, eq. (31)

    Delta = -m <= -1
                    R^(-m)[:,a] = (I + L^[a](m-1)) R[:,a]
                    the first m-1 SUBdiagonals of Gamma, and the identity

the second being eq. (26) read only part way down the same triangle eq. (29) inverts:
r_t^(-m) is r_t plus the m-1 couplings between the anchor and its nearest predecessors,
which the state at t*-m has not yet laid down.  The two meet at the seam the note fixes,
U(-1) = -I, where both give R itself, eq. (14); and the two ends are exact, not fitted:

    Delta = -t*     r = theta_t*,  the null state                  eps = 1
    Delta = 0       r = 0,         the task just trained           eps = 0

Two metrics are drawn, in two figures of the same shape.

    eps     the note's normalised loss, eq. (34),

                eps_t^(Delta) = L_t(W^(t+Delta)) / L_t(0)
                              = ||r_t^(Delta)||^2 / ||theta_t||^2 ,

            plotted as 1 - eps so that up is better: 1 is the task solved, 0 is the
            null state -- no better than outputting nothing -- and negative is worse
            than the null state.

    M       the same thing divided by a never-trained control.  A read mask F_ctrl and
            a teacher theta_ctrl are drawn and never written to; the control is read
            out of the same states as the anchor, giving eps_ctrl^(Delta), and

                M^(Delta) = eps^(Delta) / eps_ctrl^(Delta) ,

            plotted as 1 - M.  Here 1 is the task solved and 0 means the state is no
            better on the anchor than on a task it was never given -- every trace of
            the anchor's training is gone.  It costs one extra row of Gamma and no
            extra solve, since it reuses the same R.

            The second metric exists because the first one is not only about memory.
            Once the stream itself is unstable -- see `lyapunov` below -- eps carries
            the growth of the whole state, which is the same for every task and has
            nothing to do with the anchor.  M divides that out.  Read together they
            separate "the anchor was forgotten" from "everything blew up".

Three curves per panel, the split ratio rho = d_f / d_b in {1, 3, 5}: the read mask is
held at d_f and the write pool is thinned under it, so the three curves are three
write-gate densities, d_b = d_f, d_f/3, d_f/5, reading through the same gate.

How COARSE the gating is, is the other free choice, and here it is a hyperparameter and
not a curve.  A neuron is a row of W and its N_in = D entries are the synapses on it, so
the granularity is a grouping of the D columns into N_d dendrites of N_s = D / N_d
synapses each, N_d a whole factor of D:

    N_d = D       every column independent -- SYNAPTIC gating, the default, and what
                  this file did before N_d existed
    N_d = 1       row-constant masks, a neuron on or off as a whole -- NEURONAL gating
    1 < N_d < D   blocks of N_s columns sharing one mask -- DENDRITIC gating

Columns on one dendrite carry one mask, so Gamma, c and L^[a] repeat across the block
while Theta does not, and at N_d = 1 the D triangular systems collapse into the single
one the note writes out beside eq. (29).  All three rows below stay exact at every N_d:
the derivation uses the masks only through the nesting, never through their shape.

What the coarsening moves is the ensemble, not the algebra.  eps sums over the D columns
and every column average here has N_d independent terms rather than D, so the curves get
noisier and more heavy-tailed as the gating coarsens -- at N_d = 1 the whole row shares
one realised growth rate instead of averaging D of them, which is the same reason the
band widens, not a different one.  Nothing about the threshold moves: rho_eff is a
per-column quantity and eq. (19) acts on each column separately whatever N_d is.

Three rows:

    1  the direct solve.  Eq. (29) banded for transfer, eq. (32) for retention.  No
       training is run at all: the masks and the teachers are drawn, Gamma is built,
       one triangular system per input coordinate is solved, and the whole curve is
       read off its bands.

    2  the stream.  Eq. (19), W^t = (I - Pi_t) W^{t-1} + B_t v theta_t / c_t, stepped
       forward K times with the anchor's read applied to every state along the way.
       Nothing here knows about Gamma, L, U or the solve.

    3  the RELATIVE error, |row 1 - row 2| / |row 1|, against machine epsilon.  The two
       rows are two readings of one quantity, so this is rounding.  It has to be
       relative: at rho = 5 the metric itself reaches 10^9, so an absolute gap of 1
       would be agreement to ten digits and would still draw sixteen decades above
       machine epsilon.

Run the file to draw it.  Four files are written next to it, unless a directory is
given on the command line:

    lag_curves.png              the 3 x 2 figure, metric eps
    lag_curves_control.png      the same, metric M
    lag_curves_lyapunov.png     the stability threshold, two panels
    lag_curves.txt              the numbers as tables, and the self-test report

`python3 lag_curves.py --test` runs the self-test alone.  The self-test is the third
perspective on the same algebra: naive triply-nested loops for Gamma and for both band
operators, a direct simulation of the protocol that rebuilds every state from scratch,
and a fourth-order explicit integration of the true gradient flow, eq. (6), which is
told nothing and checked against all of it.

Shapes:  v (N,)  F, B (K,N,D)  Theta (K,D)  Gamma (K,K,D)  R (K,D)  lags (n_lag,)
The D axis of F, B, Gamma and c carries only N_d distinct values, repeated in blocks of
N_s; Theta and R carry D.
Index convention: tasks are 0-based here, 0..K-1; the note's task t is index t-1.  The
state index s counts tasks trained, so W^s is the state after task s-1 finished, W^0 = 0
and a stream of K tasks visits K+1 states.  The anchor sits at index `ts`, its own state
is s = ts + 1, and Delta = s - (ts + 1).
"""

import sys
from pathlib import Path

import numpy as np


# =============================================================== configuration

# Kept in one dict rather than as bare N, D, K so that nothing at module scope can be
# picked up by accident inside a function that meant to unpack its own shapes.
CONFIG = dict(N=60,        # width, synapses per input coordinate
              D=12,        # input dimension (the note's d); the D columns never mix.
                           #   12 rather than a prime so that N_d has somewhere to go:
                           #   its factors are 1, 2, 3, 4, 6, 12
              K=300,       # stream length, so that K/2 = 150 sits above N
              d_f=0.50,    # read density, held fixed across the three curves
              N_d=None)    # dendrites per neuron, a whole factor of D.  None = D, one
                           #   synapse per dendrite: synaptic gating, every column
                           #   independent.  1 is neuronal gating, the whole row at
                           #   once; anything between is dendritic

RHOS = (1, 3, 5)            # split ratios d_f / d_b; d_b = d_f / rho
SEEDS = tuple(range(20))    # independent realisations
N_ANCHORS = 1               # anchors averaged per seed; see `anchor_window`.  1 keeps
                            # the lag range at its longest, +-K/2, at one sample a seed


def anchor_index(K):
    """0-based index of the anchor task, so that both lag arms are K//2 long.

    The note counts tasks from 1, so its task K/2 is index K//2 - 1 here.  That choice
    is what makes Delta reach -K/2 on the transfer side (the state W^0, before anything
    is trained) and +K/2 on the retention side (the end of the stream).
    """
    if K < 2:
        raise ValueError(f"K={K}: a lag family needs at least two tasks")
    return K // 2 - 1


def lag_range(K, ts):
    """Every lag the anchor admits: one per state, K+1 of them.

    Delta = s - (ts+1) with s = 0..K, so the range is -(ts+1) .. K-1-ts and its length
    is K+1 whatever ts is.  With ts = anchor_index(K) the two arms are K//2 each.
    """
    if not 0 <= ts < K:
        raise ValueError(f"anchor index {ts} outside 0..{K - 1}")
    return np.arange(-(ts + 1), K - ts)


def anchor_window(K, n_anchors=1, ts=None):
    """`n_anchors` consecutive anchors centred on `ts`, and the lags all of them admit.

    One anchor per seed is what the lag range +-K/2 costs: the reach is min(t*, K-t*),
    so only the middle task has both arms at their longest.  That leaves one sample of
    the metric per seed, and once the stream is unstable the sample is heavy-tailed, so
    the curve is noisy where the three densities are closest together.

    Widening the window trades reach for samples, symmetrically: with n anchors the lags
    shorten by (n-1)//2 on each side and every seed contributes n curves.  The anchors
    within one seed share that seed's realised growth rate, so they are far from
    independent -- averaging over a window of 41 cuts the variance of the seed mean by
    about 36 at rho = 1, where the spread is mask noise, and by nothing at all at
    rho = 5, where it is the exponent.  `run_case` therefore averages over the window
    inside each seed and keeps seeds as the sample axis, so the error band stays honest.

    Returns (anchors, lags).
    """
    if n_anchors < 1:
        raise ValueError(f"n_anchors={n_anchors} must be at least 1")
    ts = anchor_index(K) if ts is None else ts
    if not 0 <= ts < K:
        raise ValueError(f"anchor index {ts} outside 0..{K - 1}")
    half = (n_anchors - 1) // 2
    anchors = ts + np.arange(-half, n_anchors - half)
    if anchors.min() < 0 or anchors.max() >= K:
        raise ValueError(
            f"{n_anchors} anchors centred on {ts} run outside 0..{K - 1}; "
            f"use at most {2 * min(ts, K - 1 - ts) + 1}")
    reach = min(int(anchors.min()) + 1, K - 1 - int(anchors.max()))
    return anchors, np.arange(-reach, reach + 1)


def _as_lags(lags):
    """Lags are task counts: integers, in a 1-D array.  Anything else is a mistake.

    Left to `np.asarray(..., dtype=int)` a float lag would be silently truncated toward
    zero -- so -2.9 would become -2, not -3 -- and a scalar would be accepted by one
    route and rejected by the other.  Both are caught here instead.
    """
    lags = np.atleast_1d(np.asarray(lags))
    if not np.issubdtype(lags.dtype, np.integer):
        raise TypeError(f"lags must be integers, got dtype {lags.dtype}")
    if lags.ndim != 1:
        raise ValueError(f"lags must be 1-D, got shape {lags.shape}")
    return lags


# =================================================================== the draw
# counts() and draw() are hadamard.py's, unchanged apart from the optional teacher
# correlation, so that the two files look at the same ensemble and their numbers can be
# compared across files.  At phi = 0 the draw is bit-identical to the parent's.

def counts(N, d_f, d_b):
    """The two per-column counts, exact, and the nesting checked before anything runs.

    Eq. (10) asks for n_f = d_f N and n_b = d_b N as counts, not as expectations, so the
    densities are rounded to the 1/N grid once here and everything downstream uses the
    realised pair.  The only real constraint is the nesting (9): a column cannot write
    more synapses than it reads.
    """
    n_f = int(round(d_f * N))
    n_b = int(round(d_b * N))
    if not 0 < n_f <= N:
        raise ValueError(f"d_f={d_f} gives n_f={n_f} with N={N}; need 1 <= n_f <= N")
    if n_b < 1:
        raise ValueError(
            f"d_b={d_b} gives n_b={n_b} with N={N}: an empty write column has no write "
            "mass and the coupling is undefined there.  Raise N or d_b")
    if n_b > n_f:
        raise ValueError(
            f"d_b={d_b} > d_f={d_f} (n_b={n_b} > n_f={n_f}): the nesting (9) says a "
            "synapse may be written only if it is read, so rho = d_f/d_b >= 1")
    return n_f, n_b


def dendrites(D, N_d=None):
    """N_d dendrites per neuron and the N_s = D/N_d synapses on each, checked once here.

    A neuron is a row of W and its N_in = D entries are the synapses on it, so how coarse
    the gate is, is a grouping of the D columns: columns a and a' carry one mask iff
    a // N_s == a' // N_s, and different dendrites are independent.

        N_d = D       N_s = 1    every column independent -- synaptic gating
        N_d = 1       N_s = D    one mask for the whole row -- neuronal gating
        1 < N_d < D              blocks of N_s columns -- dendritic gating

    N_d has to divide D exactly.  A ragged last block would leave one dendrite carrying a
    different number of synapses from the rest, and d_f would no longer mean one thing
    across the row, so it is a hard error and not a rounding, in the way that n_f and n_b
    are roundings.  None means D, the independent columns, which is the default
    everywhere and is what this file did before N_d existed.
    """
    N_d = D if N_d is None else int(N_d)
    if not 1 <= N_d <= D:
        raise ValueError(
            f"N_d={N_d} with D={D}: need 1 <= N_d <= D, from one dendrite carrying the "
            "whole row (neuronal) to one synapse per dendrite (synaptic)")
    if D % N_d:
        raise ValueError(
            f"N_d={N_d} does not divide D={D}: the dendrites would not all carry the "
            f"same number of synapses.  Use a factor of D: "
            f"{[k for k in range(1, D + 1) if D % k == 0]}")
    return N_d, D // N_d


def gating_name(N_d, D):
    """What to call a grouping of D columns into N_d dendrites."""
    return "synaptic" if N_d == D else "neuronal" if N_d == 1 else "dendritic"


def gating_lines(N_d, N_s, D):
    """The granularity as two rows of `summarise`'s table: what it is, what it costs.

    eps sums over the D columns, so what the granularity buys or spends is the number
    of independent terms in that sum, which is N_d and not D.
    """
    pad = " " * 33
    out = [f"  gating                         {N_d} dendrite"
           f"{'' if N_d == 1 else 's'} of {N_s} synapse{'' if N_s == 1 else 's'} "
           f"per neuron ({gating_name(N_d, D)})"]
    if N_d == D:
        out.append(pad + "every column carries its own mask, so eps")
        out.append(pad + f"averages all {D} independent columns")
    else:
        out.append(pad + f"Gamma, c and L repeat across each block of {N_s},")
        out.append(pad + f"so eps averages {N_d} independent column"
                         f"{'' if N_d == 1 else 's'} and not {D}")
    return out


def draw(N, D, K, d_f, d_b=None, seed=0, phi=0.0, N_d=None):
    """One realisation: readout, nested masks, teachers.

    Each DENDRITE of each task gets its own uniform permutation of 0..N-1; the first n_f
    places in it are read and the first n_b written, and the N_s columns on that dendrite
    are handed the same pair.  So B sits inside F by construction, every column carries
    exactly its count, and any two columns belonging to different tasks -- or to
    different dendrites of one task -- are independent uniform subsets, which makes the
    pairwise overlap d_fb^{kt} = n_f/N exactly for k != t, independent of n_b and of N_d.
    This is the independent-mask case, the one the free array (11) is free not to be, and
    it is the most decorrelated member of that family; §`lyapunov` says why that matters.
    N_d does not touch that: it correlates columns WITHIN a task, never masks ACROSS
    tasks, which is the correlation the exponent is sensitive to.

    At N_d = D the repeat is the identity and the draw is the fully independent one,
    variate for variate.  Below that the draw consumes K N_d N variates rather than K D N,
    so a seed does not name the same teachers at two granularities; seeds are the sample
    axis, not a paired control.

    `phi` is the only addition to hadamard.py's draw: a first-order autoregressive
    coefficient on the teacher sequence,

        theta_t = phi theta_{t-1} + sqrt(1 - phi^2) xi_t,   rows renormalised,

    so that phi = 0 is exactly hadamard.py's independent draw and phi > 0 makes
    consecutive teachers similar.  It is off by default and nothing in the algebra uses
    it -- eq. (26) and eq. (30) hold for an arbitrary fixed teacher sequence -- but the
    transfer panel is where teacher similarity would show, so the knob is here to be
    turned rather than argued about.  It is drawn after the masks and consumes no extra
    variates before them, so v, F and B do not depend on it.
    """
    d_b = d_f if d_b is None else d_b
    n_f, n_b = counts(N, d_f, d_b)
    N_d, N_s = dendrites(D, N_d)
    if not 0.0 <= phi < 1.0:
        raise ValueError(f"phi={phi} must lie in [0, 1)")
    rng = np.random.default_rng(seed)

    v = rng.standard_normal(N)
    v = v / np.linalg.norm(v)

    # rank[k,g,i] is where synapse i landed in DENDRITE g of task k's permutation; the
    # repeat hands each dendrite's mask to the N_s columns sitting on it, so column a
    # takes dendrite a // N_s.  N_s = 1 makes the repeat the identity.
    rank = np.argsort(np.argsort(rng.random((K, N_d, N)), axis=-1), axis=-1)
    F = np.repeat(np.moveaxis((rank < n_f).astype(float), 1, -1),   # (K, N_d, N)
                  N_s, axis=-1)                                     # -> (K, N, D)
    B = np.repeat(np.moveaxis((rank < n_b).astype(float), 1, -1), N_s, axis=-1)

    xi = rng.standard_normal((K, D))
    if phi == 0.0:
        Theta = xi
    else:
        Theta = np.empty((K, D))
        Theta[0] = xi[0]
        for t in range(1, K):
            Theta[t] = phi * Theta[t - 1] + np.sqrt(1.0 - phi ** 2) * xi[t]
    Theta = Theta / np.linalg.norm(Theta, axis=1, keepdims=True)

    return v, F, B, Theta, (n_f, n_b)


def draw_control(N, D, n_f, seed=0, N_d=None):
    """A read mask and a teacher for a task that is never trained.

    The control is the baseline the metric M divides by: a task drawn from the same
    ensemble as the stream's, with the same read density AND the same gating
    granularity, read out of every state the stream visits but never written into.  It
    has no write mask, because it never writes, and its teacher never enters eq. (26).

    Its own generator is keyed on (seed, a fixed word), so that adding the control to a
    run does not disturb v, F, B or Theta: the stream is bit-identical with and without
    it, and the same seed gives the same control.
    """
    N_d, N_s = dendrites(D, N_d)
    rng = np.random.default_rng((seed, 0xC0FFEE))
    rank = np.argsort(np.argsort(rng.random((N_d, N)), axis=-1), axis=-1)
    F_ctrl = np.repeat(np.moveaxis((rank < n_f).astype(float), 0, -1),  # (N_d, N)
                       N_s, axis=-1)                                    # -> (N, D)
    theta = rng.standard_normal(D)
    return F_ctrl, theta / np.linalg.norm(theta)


# =============================================================== the coupling

def strict_lower(Gamma):
    """L^[a], the strict lower triangle of Gamma[:,:,a], as a (D, K, K) stack.

    Only the couplings with writer s < reader t survive -- a later task reading through
    an earlier task's writes -- which is the half eq. (26) uses.  Split out of
    `solve_direct` so that the amplification diagnostic can reuse it rather than rebuild
    a second copy of a (D, K, K) array.
    """
    return np.tril(np.moveaxis(Gamma, -1, 0), -1)


def coupling(v, F, B):
    """Gamma[k,t,a] and the write masses c[t,a], eq. (21) and eq. (17).

    Gamma[k,t,a] is the fraction of the write mass task t laid down in input coordinate
    a that task k reads back: the reader's F against the writer's B.  Reader index
    first, writer second, and it is not symmetric.  Gamma[t,t,a] = 1 by the nesting, and
    every entry lies in [0,1], being a fraction of a mass.
    """
    w = v ** 2
    c = np.einsum("i,tia->ta", w, B)                 # write mass, eq. (17)
    if np.any(c <= 0):
        raise ValueError("a write column is empty; the coupling is undefined there")
    num = np.einsum("i,kia,tia->kta", w, F, B, optimize=True)
    return num / c[None, :, :], c


def coupling_row(v, F_read, B, c):
    """One reader's row of Gamma against every writer in the stream: (K, D).

    Eq. (21) with the reader's mask supplied directly rather than taken from the stream,
    which is what the never-trained control needs.  `Gamma[k]` from `coupling` is the
    same object for a reader that is in the stream.
    """
    return np.einsum("i,ia,tia->ta", v ** 2, F_read, B) / c


# ================================================= route 1, the direct solve

def solve_direct(Gamma, Theta):
    """R from the whole stream at once: (I + L[a]) R[:,a] = Theta[:,a], eq. (29).

    I + L[a] is unit lower triangular, so its determinant is 1 for every realisation:
    there is no invertibility condition and nothing to regularise, and nothing about
    that depends on the split.
    """
    K, _, _ = Gamma.shape
    L = strict_lower(Gamma)                          # (D, K, K)
    R = np.linalg.solve(np.eye(K)[None] + L, Theta.T[..., None])[..., 0]
    return R.T                                       # (K, D)


def lag_residuals_direct(Gamma_row, R, ts, lags):
    """The whole lag family of one reader, from R alone: one band multiply per lag.

    Only the reader's own row of the coupling is needed, so neither band operator is
    built.  With g = Gamma_row (g[j,a] = Gamma^[a]_{reader, j}), in 0-based indices:

        Delta >= 0   r = - sum_{j = ts+1}^{ts+Delta} g[j] * R[j]          eq. (32)
        Delta = -m   r =   R[ts] + sum_{j = ts-m+1}^{ts-1} g[j] * R[j]    eq. (29)

    The first is minus the lag-Delta band of eq. (31) applied to R.  The second is R
    itself plus the first m-1 subdiagonals, which is eq. (26) read part way:

        theta_t  = r_t + sum_{s<t}       Gamma_{ts} r_s   (all of the lower triangle)
        r_t^(-m) = r_t + sum_{t-m<s<t}   Gamma_{ts} r_s   (its last m-1 terms)

    so the omitted head of that sum, sum_{s <= t-m}, is exactly what the state at t*-m
    had already written and the anchor therefore already reads.  At m = ts+1 the whole
    lower triangle is in and r = theta_t*; at Delta = 0 both sums are empty, giving 0.

    Both are cumulative sums walking outward from ts, so the whole curve costs one pass
    in each direction, O(K D) for the pair, and no second solve.  Returns (n_lag, D).

    For the anchor, `Gamma_row` is `Gamma[ts]` and R[ts] is the anchor's own inherited
    residual.  For the never-trained control there is no such row of R, and the caller
    uses `lag_readout_direct` below instead.
    """
    K, D = R.shape
    if Gamma_row.shape != (K, D):
        raise ValueError(f"Gamma_row has shape {Gamma_row.shape}, expected {(K, D)}")
    if not 0 <= ts < K:
        raise ValueError(f"anchor index {ts} outside 0..{K - 1}")
    lags = _as_lags(lags)
    if lags.min() < -(ts + 1) or lags.max() > K - 1 - ts:
        raise ValueError(
            f"lags must lie in [{-(ts + 1)}, {K - 1 - ts}] for anchor index {ts} of a "
            f"stream of K={K}")
    g = Gamma_row

    # forward arm, Delta >= 0: partial sums of g[j] R[j] over j = ts+1 .. ts+Delta,
    # prepended with a zero row so that fwd[q] is the sum at Delta = q and fwd[0] = 0.
    tail = g[ts + 1:] * R[ts + 1:]                   # (K-1-ts, D)
    fwd = np.concatenate([np.zeros((1, D)), np.cumsum(tail, axis=0)], axis=0)

    # backward arm, Delta = -m: R[ts] plus the sum over j = ts-m+1 .. ts-1, walking
    # leftward from ts-1.  bwd is 0-based in m-1, so bwd[0] = R[ts] (an empty sum) and
    # bwd[ts] = Theta[ts] (the whole lower triangle, by eq. (26)).
    head = (g[:ts] * R[:ts])[::-1]                   # j = ts-1, ts-2, ..., 0
    bwd = R[ts][None, :] + np.concatenate(
        [np.zeros((1, D)), np.cumsum(head, axis=0)], axis=0)

    out = np.empty((lags.size, D))
    neg = lags < 0
    out[neg] = bwd[-lags[neg] - 1]
    out[~neg] = -fwd[lags[~neg]]
    return out


def lag_readout_direct(Gamma_row, R, theta_ctrl, ts, lags):
    """The same family for a reader that is NOT in the stream: the never-trained control.

    Eq. (24) has no special case for the reader, so for any read mask at all

        thetahat^{(s)}_ctrl = sum_{u <= s} Gamma_ctrl,u . r_u ,

    and the residual is theta_ctrl minus that.  The control was never trained, so there
    is no r_ctrl to start from and no seam at Delta = 0: the sum simply runs over every
    task the state has seen, s = ts + 1 + Delta of them.  One cumulative sum, no solve.
    """
    K, D = R.shape
    if Gamma_row.shape != (K, D):
        raise ValueError(f"Gamma_row has shape {Gamma_row.shape}, expected {(K, D)}")
    if not 0 <= ts < K:
        raise ValueError(f"anchor index {ts} outside 0..{K - 1}")
    lags = _as_lags(lags)
    s = ts + 1 + lags
    if s.min() < 0 or s.max() > K:
        raise ValueError(f"lags reach states outside 0..{K}")
    acc = np.concatenate([np.zeros((1, D)), np.cumsum(Gamma_row * R, axis=0)], axis=0)
    return theta_ctrl[None, :] - acc[s]


# ======================================================= route 2, the stream

def run_lds(v, F, B, Theta, reads):
    """The stream by stepping eq. (19) K times, reading out of every state as it goes.

    Column by column, eq. (19) is

        W^t[:,a] = (I_N - Pi^[a]_t) W^{t-1}[:,a] + B^[a]_t v theta_{t,a} / c^[a]_t,
        Pi^[a]_t = B^[a]_t v v^T F^[a]_t / c^[a]_t,

    and it is written out here in exactly those two pieces -- the projection removed,
    the teacher's own direction added -- rather than in the equivalent residual form of
    eq. (18), so that the two routes share no line of algebra beyond the masks
    themselves.  Neither Pi nor any N x N object is built: the map is rank one, so
    v^T F^[a]_t W[:,a] is a contraction and the outer product is formed in place.

    Nothing here knows about Gamma, L, U or the triangular solve.  `reads` is a stack of
    read masks, (n_read, N, D) -- the anchors of the window and the never-trained control
    -- and what comes back is

        traces[j,s,a] = thetahat^{(s)}_{j, a} = v^T (F_j . W^s)[:,a],   s = 0..K,

    each of them read out of every state the stream visits, including W^0 = 0 before
    anything is trained; and R_stream, the residual every task inherited, which is the
    Delta = -1 column of the same object and is kept for the comparison with eq. (29).
    A read mask need not belong to the stream: eq. (24) has no special case for the
    reader, which is what makes the control possible.

    `fit` is the worst |thetahat^{(t)}_t - theta_t| over the stream: eq. (18) says the
    task just trained sits at zero loss, and this is how exactly.
    """
    K, N, D = F.shape
    reads = np.asarray(reads, dtype=float)
    if reads.ndim != 3 or reads.shape[1:] != (N, D):
        raise ValueError(f"reads has shape {reads.shape}, expected (n_read, {N}, {D})")
    w = v ** 2
    W = np.zeros((N, D))
    traces = np.empty((reads.shape[0], K + 1, D))
    R = np.empty((K, D))
    fit = 0.0

    traces[:, 0] = np.einsum("i,jia,ia->ja", v, reads, W)    # W^0 = 0, so 0
    for t in range(K):
        Ft, Bt = F[t], B[t]
        c = w @ Bt                                           # c^[a]_t, eq. (17)
        read = np.einsum("i,ia,ia->a", v, Ft, W)             # v^T F^[a]_t W[:,a]
        R[t] = Theta[t] - read                               # eq. (14), for the record
        # eq. (19), the two pieces kept apart
        W = W - Bt * np.outer(v, read / c) + Bt * np.outer(v, Theta[t] / c)
        fit = max(fit, np.abs(np.einsum("i,ia,ia->a", v, Ft, W) - Theta[t]).max())
        traces[:, t + 1] = np.einsum("i,jia,ia->ja", v, reads, W)

    return traces, R, fit


def lag_residuals_stream(trace, theta, ts, lags):
    """theta minus the reader's own read at each state, straight off the trace.

    Delta = s - (ts + 1) by the index convention, so state s = ts + 1 + Delta.  This is
    the definition of eq. (30) and nothing else; it is what row 2 of the figure plots
    and what row 3 measures the direct solve against.  It serves the anchor and the
    control alike -- only the trace and the teacher differ.
    """
    lags = _as_lags(lags)
    s = ts + 1 + lags
    if s.min() < 0 or s.max() >= trace.shape[0]:
        raise ValueError(f"lags reach states outside 0..{trace.shape[0] - 1}")
    return np.asarray(theta)[None, :] - trace[s]


# ==================================================================== metrics

def eps_metric(r_lag, theta):
    """eq. (34): the loss at the state over the loss of the null state.

    L_t(W) = ||thetahat - theta||^2 / 2d by eq. (4), and L_t(0) = ||theta||^2 / 2d since
    thetahat = 0 at W = 0, so the 1/2d cancels and the ratio is free of d and of the
    scale of the teacher.  Zero when the task is solved, one when the state is worth no
    more than outputting nothing, and greater than one when it is worth less.
    """
    return (np.asarray(r_lag) ** 2).sum(axis=-1) / (np.asarray(theta) ** 2).sum()


def score(eps):
    """1 - eps, the way up that makes more mean better.

        1   solved exactly
        0   no better than the baseline -- W = 0 for eps, a never-trained task for M
        <0  worse than the baseline
    """
    return 1.0 - np.asarray(eps)


# =========================================================== stability of (19)

def second_moment_rate(v, F, B):
    """Where the stream turns unstable, in closed form, with nothing run.

    Strip the teacher off eq. (19) and an error already in the weights obeys
    x <- (I - Pi_t) x.  With Pi = a b^T / (b^T a), a = B^[a]_t v and b = F^[a]_t v, the
    nesting (9) gives b^T a = v^T F B v = c and ||a||^2 = v^T B v = c, so for an
    isotropic x in R^N

        E||(I - Pi)x||^2 / E||x||^2 = 1 - 2 (b^T a)/c + ||a||^2 ||b||^2 / c^2 ... = 1 +
        (rho_eff - 2) / N ,        rho_eff := v^T F^[a]_t v / v^T B^[a]_t v ,

    -- one step grows the second moment iff rho_eff > 2, and the margin is (rho_eff-2)/N.
    So the threshold is not rho > 1.  At rho slightly above 1 the map is oblique,
    ||I - Pi||_2 = sqrt(rho_eff) > 1, and yet the product still contracts, because what
    decides is the average step and not the worst one.

    rho_eff is the realised ratio of read mass to write mass, which is the nominal
    rho = d_f/d_b only in median: c fluctuates over a pool of just n_b synapses, and
    averaging 1/c inflates the mean.  At N = 60, d_f = 0.5 the nominal rho = 5 has
    rho_eff with median 5.5 and mean 7.3, spread over [2.9, 11.6] between the tenth and
    ninetieth percentiles.

    Returns (mean rate per step, rho_eff as a (K, D) array).
    """
    w = v ** 2
    N = v.size
    rho_eff = np.einsum("i,tia->ta", w, F) / np.einsum("i,tia->ta", w, B)
    return float((rho_eff - 2.0).mean() / N), rho_eff


LYAP_BURN = 200     # steps discarded before the exponent is accumulated


def lyapunov(v, F, B, col=0, burn=LYAP_BURN, x0_seed=1):
    """The growth rate per task of the homogeneous part of eq. (19), measured.

    A unit vector is walked through the product of the complements I - Pi_t -- each a
    projection of rank N-1, the complement of the rank-one Pi of eq. (20) -- renormalised
    at every step, and the logs are accumulated.  That is the top Lyapunov exponent of
    the product.  No teacher, no residual, no Gamma, no solve: the LDS by itself.

    `burn` discards the first steps.  The initial vector's alignment costs an O(1) log
    that a short walk would divide by K and report as a rate; at rho = 1 that artefact
    alone reads as lambda = -0.004 at K = 300 and -0.0005 at K = 4800, a "rate" that
    shrinks with the length of the walk.  Discarding the head removes most of it.

    What the sign means, and it is the whole story of the split:

        rho_eff < 2   the product contracts.  At rho = 1 the nesting forces B = F, so Pi
                      is symmetric, eq. (20) makes it an ORTHOGONAL projection, and
                      I - Pi is nonexpansive step by step: the exponent cannot be
                      positive however long the stream runs.  Between 1 and the
                      threshold it is oblique and still contracting, and the decay is
                      slower than exponential, so a fitted exponent drifts toward zero
                      as K grows and only its sign should be read.

        rho_eff > 2   the exponent is positive and, near the threshold, close to the
                      closed form above: 2 N lambda = rho_eff - 2, checked from
                      rho_eff = 2.07 (0.029 against 0.074) to 3.24 (1.27 against 1.24),
                      drifting apart further out where the average of logs and the log
                      of the average part company.  Since lambda ~ 1/N, the residuals
                      grow like exp(lambda K) = exp(c(rho) K / 2N): the blow-up is a
                      function of K/N alone, and the regime eq. (13) asks for, K >> N,
                      is the one in which a split mask diverges.

    Two things this rests on, neither of them in the note.  The first is that each task
    is trained to its exact fixed point: stopping the inner flow early replaces Pi by
    alpha Pi, and the threshold becomes alpha < 2/rho_eff, so any finite training budget
    can restore stability.  The second is the mask draw: `draw` re-permutes independently
    every task, which is the most decorrelated member of the free array (11).  Masks that
    drift, cycle, or prefer the heaviest v_i^2 all reduce the exponent by one to three
    orders of magnitude.  The divergence is a property of split gating run to
    convergence with independent masks, not of split gating.

    None of it touches eq. (29) or eq. (32), which stay exact throughout; that is row 3.
    """
    K, N, _ = F.shape
    if burn >= K:
        burn = 0
    w = v ** 2
    x = np.random.default_rng(x0_seed).standard_normal(N)
    x = x / np.linalg.norm(x)
    total, used = 0.0, 0
    for t in range(K):
        u, z = B[t, :, col] * v, F[t, :, col] * v
        x = x - u * ((z @ x) / (w @ B[t, :, col]))
        n = np.linalg.norm(x)
        if n == 0.0:                      # the error was entirely inside the read set
            return -np.inf
        if t >= burn:
            total += np.log(n)
            used += 1
        x = x / n
    return total / used


def growth_rate(R, skip=0.5):
    """Growth rate per task of ||r_t||, by a straight-line fit to its log over the tail.

    The same number `lyapunov` measures, arrived at from the other end: the inhomogeneous
    stream driven by an arbitrary teacher inherits the homogeneous map's exponent
    wherever that exponent is positive, so a fit to the residuals the triangular solve
    returns reproduces the exponent of a product the solve never forms.  Where the
    exponent is negative the two need not agree and this one is the meaningful number:
    a contracting homogeneous map leaves the driven residuals bounded, not decaying,
    because every task brings a fresh teacher.  `skip` drops the leading transient.
    """
    n = np.linalg.norm(R, axis=1)[int(skip * R.shape[0]):]
    if n.size < 4:
        return np.nan
    return float(np.polyfit(np.arange(n.size), np.log(np.maximum(n, 1e-300)), 1)[0])


# ================================================================ the sweep

METRICS = ("eps", "control")


def geometric_mean(x, axis=0):
    """exp of the mean log, with an exact zero carried through as zero.

    eps vanishes exactly at Delta = 0 and nowhere else, so the special case is that one
    column and it is not an approximation: every member of the average is 0 there.
    """
    x = np.asarray(x, dtype=float)
    if (x < 0).any():
        raise ValueError("geometric_mean needs non-negative input")
    with np.errstate(divide="ignore"):
        return np.exp(np.log(x).mean(axis=axis))   # a single -inf carries the mean


def run_case(N, D, K, d_f, d_b, seeds, ts=None, phi=0.0, n_anchors=1, N_d=None):
    """Both routes, both metrics, every seed, at one (d_f, d_b).

    The two routes are run from the same draw, so any difference between them is
    arithmetic and nothing else.  eps and M come back as (n_seed, n_lag) arrays for each
    route, and the diagnostics hadamard.py reports at a single lag are carried along so
    that the figure can be read against the same numbers.

    With n_anchors > 1 each seed's curve is the geometric mean over a window of anchors
    (see `anchor_window`, which also shortens the lag range to what every anchor in the
    window admits).  Seeds remain the sample axis, so the error band is still an interval
    over independent realisations and not over anchors that share one.

    `N_d` sets the gating granularity, the stream and its control alike, and is held
    fixed for the case: every seed of one case is gated the same way.  Nothing below
    branches on it -- both routes read the masks they are given -- which is the point.
    """
    n_f, n_b = counts(N, d_f, d_b)
    N_d, N_s = dendrites(D, N_d)
    seeds = list(seeds)
    anchors, lags = anchor_window(K, n_anchors, ts)
    n_s, n_l, n_a = len(seeds), lags.size, anchors.size

    out = {k: np.empty((n_s, n_l)) for k in
           ("eps_direct", "eps_stream", "ctl_direct", "ctl_stream")}
    diag = {k: np.empty(n_s) for k in
            ("resid", "amp", "gam_mean", "gam_sd", "fit", "gap", "lam", "lam_resid",
             "smr", "rho_eff")}

    for j, seed in enumerate(seeds):
        v, F, B, Theta, _ = draw(N, D, K, d_f, d_b, seed, phi=phi, N_d=N_d)
        F_ctrl, th_ctrl = draw_control(N, D, n_f, seed, N_d=N_d)
        Gamma, c = coupling(v, F, B)
        g_ctrl = coupling_row(v, F_ctrl, B, c)

        R_direct = solve_direct(Gamma, Theta)
        reads = np.concatenate([F[anchors], F_ctrl[None]], axis=0)
        traces, R_stream, fit = run_lds(v, F, B, Theta, reads)

        per = {k: np.empty((n_a, n_l)) for k in out}
        for q, a in enumerate(anchors):
            per["eps_direct"][q] = eps_metric(
                lag_residuals_direct(Gamma[a], R_direct, a, lags), Theta[a])
            per["eps_stream"][q] = eps_metric(
                lag_residuals_stream(traces[q], Theta[a], a, lags), Theta[a])
            per["ctl_direct"][q] = eps_metric(
                lag_readout_direct(g_ctrl, R_direct, th_ctrl, a, lags), th_ctrl)
            per["ctl_stream"][q] = eps_metric(
                lag_residuals_stream(traces[-1], th_ctrl, a, lags), th_ctrl)
        for k in out:
            out[k][j] = per[k][0] if n_a == 1 else geometric_mean(per[k], axis=0)

        L = strict_lower(Gamma)
        smr, rho_eff = second_moment_rate(v, F, B)
        off = ~np.eye(K, dtype=bool)
        g = Gamma[off]
        diag["resid"][j] = np.linalg.norm(R_stream, axis=1).mean()
        diag["amp"][j] = np.abs(np.linalg.inv(np.eye(K)[None] + L)).sum(axis=-1).max()
        diag["gam_mean"][j], diag["gam_sd"][j] = g.mean(), g.std()
        diag["fit"][j] = fit
        diag["gap"][j] = np.abs(R_stream - R_direct).max()
        diag["lam"][j] = lyapunov(v, F, B)
        diag["lam_resid"][j] = growth_rate(R_direct)
        diag["smr"][j] = smr
        diag["rho_eff"][j] = np.median(rho_eff)

    res = dict(lags=lags, ts=int(anchors[anchors.size // 2]), anchors=anchors,
               seeds=np.array(seeds), phi=phi,
               N=N, D=D, K=K, d_f=d_f, d_b=d_b, n_f=n_f, n_b=n_b,
               N_d=N_d, N_s=N_s,
               d_f_realised=n_f / N, d_b_realised=n_b / N, rho=n_f / n_b,
               **out, **diag)
    # M = eps / eps_ctrl, both routes.  eps_ctrl is never zero: the control is never
    # trained, so no state solves it exactly.
    res["M_direct"] = out["eps_direct"] / out["ctl_direct"]
    res["M_stream"] = out["eps_stream"] / out["ctl_stream"]
    return res


def metric_arrays(res, metric):
    """(direct, stream) for whichever metric is being drawn, with its axis label."""
    if metric == "eps":
        return res["eps_direct"], res["eps_stream"]
    if metric == "control":
        return res["M_direct"], res["M_stream"]
    raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")


def sweep_rho(rhos=RHOS, seeds=SEEDS, phi=0.0, n_anchors=N_ANCHORS, **cfg):
    """One case per split ratio, the read mask held fixed and the write pool thinned.

    Keyword overrides go to `CONFIG`, so `sweep_rho(K=120)` is the whole experiment on a
    shorter stream.

    The gating granularity comes from `CONFIG["N_d"]` like everything else and is the
    same for all three curves: rho is what varies, and N_d is held.

    rho is asked for exactly: n_f must be divisible by every rho so that d_b = d_f/rho
    lands on the 1/N grid and the realised ratio is the requested one, not a rounding of
    it.  The check is here rather than in `counts` because it is a property of the
    experiment -- three comparable curves, differing in d_b and in nothing else -- and
    not of the algebra, which is happy with any nested pair.
    """
    cfg = {**CONFIG, **cfg}
    N_, D_, K_, d_f = cfg["N"], cfg["D"], cfg["K"], cfg["d_f"]
    N_d = cfg.get("N_d")
    n_f = int(round(d_f * N_))
    bad = [r for r in rhos if n_f % r]
    if bad:
        raise ValueError(
            f"n_f = d_f N = {n_f} is not divisible by {bad}; rho would be rounded rather "
            f"than realised.  Pick d_f with n_f a multiple of "
            f"{int(np.lcm.reduce(np.asarray(rhos, dtype=int)))}")
    return [run_case(N_, D_, K_, d_f, d_f / r, seeds, phi=phi, n_anchors=n_anchors,
                     N_d=N_d)
            for r in rhos]


def lyapunov_scan(Ns=(60, 120, 240), rhos=(1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 5.0),
                  n_f_frac=0.5, K=2000, seeds=(0, 1, 2)):
    """The exponent against the realised ratio, at several widths: the threshold, drawn.

    One column is enough (D = 1) because the columns never mix, and the teachers are not
    used at all, so this is cheap: a walk of K steps on a vector of length N.  D = 1 also
    makes N_d moot -- one column is one dendrite -- which is the honest statement of what
    the granularity does to the threshold, namely nothing: eq. (19) acts on each column
    separately, so rho_eff and the exponent are per-column quantities and a coarser gate
    only decides how many columns share one realisation of them.  Returns
    rho_eff and 2 N lambda, both averaged over seeds, so that the closed form of
    `second_moment_rate` -- 2 N lambda = rho_eff - 2 -- is a straight line through the
    origin at rho_eff = 2 whatever N is.
    """
    out = {}
    for N in Ns:
        n_f = int(round(n_f_frac * N))
        re_, sc_ = [], []
        for rho in rhos:
            n_b = int(round(n_f / rho))
            if not 1 <= n_b <= n_f:
                continue
            ls, rs = [], []
            for s in seeds:
                v, F, B, _, _ = draw(N, 1, K, n_f / N, n_b / N, s)
                ls.append(lyapunov(v, F, B))
                _, rho_eff = second_moment_rate(v, F, B)
                rs.append(np.median(rho_eff))
            re_.append(np.mean(rs))
            sc_.append(2 * N * np.mean(ls))
        out[N] = (np.array(re_), np.array(sc_))
    return out


# ================================================================ the figure

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#898781"
GREEN, PURPLE = "#2f8f5b", "#8452c4"
COLORS = (BLUE, ORANGE, GREEN, PURPLE, GREY)
GAP_FLOOR = 1e-18           # a gap of exactly zero, drawn on a logarithmic axis

METRIC_LABEL = {
    "eps": (r"$1-\varepsilon_{t^*}^{(\Delta)}$   (up is better)",
            "$1$ = solved exactly\n$0$ = no better than $W=0$\n$<0$ = worse than $W=0$",
            "symlog"),
    "control": (r"$1-M_{t^*}^{(\Delta)}$   (up is better)",
                "$1$ = solved exactly\n$0$ = no better than a task\n"
                "        never trained on\n$<0$ = worse than that",
                "linear"),
}


def _agg(eps_curves):
    """Geometric mean over seeds, with two standard errors of the log either side.

    The arithmetic mean of eps is the wrong summary here and the plain percentile band is
    the wrong companion.  Once rho_eff > 2 the ensemble is heavy-tailed -- each seed
    carries its own realised exponent and the metric is quadratic in the residual -- so
    the mean is set by one or two seeds in a hundred, while log eps is close to Gaussian
    across seeds.  Averaging the logs and exponentiating back gives a summary whose
    twenty-seed standard error is a fifth of a decade instead of larger than the quantity
    itself, and the band is then an interval on the line drawn rather than a spread of
    seeds that a reader will mistake for one.

    eps = 0 exactly, which happens only at Delta = 0, is carried through as 0.
    """
    eps = np.asarray(eps_curves, dtype=float)
    n = eps.shape[0]
    with np.errstate(divide="ignore"):
        lg = np.log(eps)
    ok = np.isfinite(lg).all(axis=0)
    mid = np.zeros(eps.shape[1])
    lo, hi = np.zeros_like(mid), np.zeros_like(mid)
    m = lg[:, ok].mean(axis=0)
    se = lg[:, ok].std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(m)
    mid[ok], lo[ok], hi[ok] = np.exp(m), np.exp(m - 2 * se), np.exp(m + 2 * se)
    return mid, lo, hi


def _band(ax, x, eps_curves, color, label):
    """One curve: the geometric mean, turned up-is-better, with its 2 s.e. interval."""
    mid, lo, hi = _agg(eps_curves)
    ax.fill_between(x, score(hi), score(lo), color=color, alpha=0.18, linewidth=0)
    ax.plot(x, score(mid), color=color, linewidth=1.4, label=label)
    return mid


def plot_lag_curves(cases, axes=None, metric="eps"):
    """Three rows by two columns: the solve, the stream, and the gap between them.

    Columns are the two arms of the same lag family, split at the anchor: transfer on the
    left, Delta running from -K/2 up to -1, and retention on the right, Delta from 0 to
    +K/2.  Rows 1 and 2 plot the metric turned so that up is better; row 3 plots their
    relative difference on a logarithmic axis with machine epsilon drawn, because that is
    the only scale on which "they agree" says anything.

    The horizontal axis is symmetric-logarithmic with a linear core one task wide.  The
    couplings are positive and of order d_f, so the partial sums behave like a stable
    first-order recursion with a correlation time of a few tasks: both arms do most of
    their moving inside the first ten or twenty lags and then follow the envelope of the
    stream itself.  Drawn linearly out to +-K/2 that whole near-anchor transition is one
    pixel wide.

    The vertical axis depends on the metric.  For M it is linear, since M is a ratio of
    two things that grow together and stays of order one.  For eps it is
    symmetric-logarithmic with a linear core one unit wide, so that [-1, 1] -- solved,
    null, and as-far-wrong-as-the-teacher-is-big -- is linear and everything beyond is
    logarithmic; that interval is where the whole rho = 1 curve lives, while at
    rho_eff > 2 the stream diverges and 1 - eps runs to -10^9 by the end of a stream of
    300.  Those two facts cannot share a linear panel, which is the reason the M figure
    exists.
    """
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(3, 2, figsize=(12.0, 10.5), layout="constrained")
    eps_mach = np.finfo(float).eps
    ylabel, key, yscale = METRIC_LABEL[metric]

    for i, res in enumerate(cases):
        col = COLORS[i % len(COLORS)]
        lags = res["lags"]
        neg, pos = lags < 0, lags >= 0
        direct, stream = metric_arrays(res, metric)
        lab = (f"$\\rho$ = {res['rho']:.0f},  $d_b$ = {res['d_b_realised']:.3f} "
               f"($n_b$ = {res['n_b']}),  "
               f"$\\rho_{{\\rm eff}}$ = {res['rho_eff'].mean():.1f},  "
               f"$\\lambda$ = {res['lam'].mean():+.4f}/task")

        for r, curves in enumerate((direct, stream)):
            _band(axes[r][0], lags[neg], curves[:, neg], col, lab)
            _band(axes[r][1], lags[pos], curves[:, pos], col, lab)

        # The gap is relative.  Delta = 0 is dropped: both routes return exactly 0 for
        # eps there, by eq. (18) and by the empty band, so the relative gap is 0/0.
        rel = np.abs(direct - stream) / np.maximum(np.abs(direct), 1e-300)
        for c, m in enumerate((neg, pos & (lags != 0))):
            if not m.any():
                continue
            e = rel[:, m]
            axes[2][c].plot(lags[m], np.maximum(e.mean(axis=0), GAP_FLOOR), color=col,
                            linewidth=1.2, label=lab)
            axes[2][c].plot(lags[m], np.maximum(e.max(axis=0), GAP_FLOOR), color=col,
                            linewidth=0.8, linestyle=":", alpha=0.8)

    ref = cases[0]
    K_, ts = ref["K"], ref["ts"]
    titles = (("transfer  --  eq. (29), banded", "retention  --  eq. (32)"),
              ("transfer  --  the stream, eq. (19)",
               "retention  --  the stream, eq. (19)"),
              ("the relative gap, transfer", "the relative gap, retention"))
    for r in range(3):
        for c in range(2):
            ax = axes[r][c]
            ax.set_xscale("symlog", linthresh=1.0)
            ax.set_title(titles[r][c], fontsize=10)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.grid(True, color="#e1e0d9", linewidth=0.8)
            ax.set_axisbelow(True)

    for r in (0, 1):
        for c in range(2):
            ax = axes[r][c]
            if yscale == "symlog":
                ax.set_yscale("symlog", linthresh=1.0, linscale=1.2)
            ax.axhline(0.0, color=GREY, linewidth=1.0, linestyle=":")
            ax.axhline(1.0, color=GREY, linewidth=0.8, linestyle="--", alpha=0.7)
        axes[r][0].set_ylabel(ylabel)
        lo = min(axes[r][c].get_ylim()[0] for c in range(2))
        hi = max(axes[r][c].get_ylim()[1] for c in range(2))
        for c in range(2):
            axes[r][c].set_ylim(lo, hi if yscale == "linear" else 3.0)

    axes[0][0].annotate(key + ("\n$[-1,1]$ linear, outside it logarithmic"
                               if yscale == "symlog" else ""),
                        xy=(0.03, 0.03), xycoords="axes fraction", fontsize=8,
                        color=GREY)
    axes[0][0].legend(fontsize=7.5, loc="upper right", framealpha=0.85,
                      edgecolor="none")
    axes[0][1].annotate(
        "line: geometric mean over seeds\nband: $\\pm2$ s.e. of the mean log\n"  # noqa
        + (f"anchor $t^*$ = task {ts + 1} of {K_}\n" if ref["anchors"].size == 1 else
           f"{ref['anchors'].size} anchors centred on task {ts + 1} of {K_}\n")
        + f"lags $\\Delta \\in [{ref['lags'].min()}, +{ref['lags'].max()}]$"
        + ("\n    $\\Delta=-{}$ is $W^0=0$".format(ts + 1)
           if ref["lags"].min() == -(ts + 1) else ""),
        xy=(0.03, 0.03), xycoords="axes fraction", fontsize=8, color=GREY)

    floors = [l.get_ydata().min() for c in range(2)
              for l in axes[2][c].get_lines() if l.get_ydata().size]
    floor = min(floors) if floors else eps_mach
    for c in range(2):
        ax = axes[2][c]
        ax.set_yscale("log")
        ax.set_ylim(min(floor, eps_mach) / 20.0, None)
        ax.axhline(eps_mach, color=GREY, linewidth=1.0, linestyle=":")
        ax.annotate("machine $\\epsilon$", xy=(0.98, eps_mach),
                    xycoords=("axes fraction", "data"), xytext=(0, 4),
                    textcoords="offset points", fontsize=8, color=GREY, ha="right",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.0))
        ax.set_xlabel(r"lag $\Delta$   (tasks; negative = before the anchor)")
    axes[2][0].set_ylabel(r"$|$row 1 $-$ row 2$| \, / \, |$row 1$|$")
    lo = min(axes[2][c].get_ylim()[0] for c in range(2))
    hi = max(axes[2][c].get_ylim()[1] for c in range(2))
    for c in range(2):
        axes[2][c].set_ylim(lo, hi)
    axes[2][0].annotate("solid: mean over seeds\ndotted: worst seed\n"
                        r"($\Delta=0$ dropped: both routes give exactly $0$)",
                        xy=(0.03, 0.02), xycoords="axes fraction", fontsize=8,
                        color=GREY, va="bottom",
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8,
                                  pad=2.0))
    return axes


def plot_lyapunov(scan, cases=None, axes=None):
    """Where the stream turns unstable, and where the three curves of the other figures sit.

    Left: 2 N lambda against the realised ratio rho_eff, one line per width.  The closed
    form of `second_moment_rate` says an isotropic error grows iff rho_eff > 2, with
    margin (rho_eff - 2)/N, so on these axes every width should fall on the same straight
    line 2 N lambda = rho_eff - 2 and cross zero at 2.  It does, near the threshold; the
    measured exponent falls below the line further out, which is the average of logs
    sitting under the log of the average.

    Right: the distribution of rho_eff over tasks and columns at the settings the other
    figures use.  The nominal rho is the median of this, not its mean: c is a sum over a
    pool of only n_b synapses, so thinning B widens the ratio and pulls its mean up.  It
    is the realised ratio, not the nominal one, that has to clear 2.
    """
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(11.0, 3.8), layout="constrained")

    ax = axes[0]
    for i, (N, (re_, sc_)) in enumerate(sorted(scan.items())):
        ax.plot(re_, sc_, marker="o", color=COLORS[i % len(COLORS)], label=f"$N$ = {N}")
    x = np.linspace(1.0, max(r.max() for r, _ in scan.values()), 50)
    ax.plot(x, x - 2.0, color=GREY, linestyle="--", linewidth=1.2,
            label=r"$2N\lambda = \rho_{\rm eff}-2$")
    ax.axhline(0.0, color=GREY, linewidth=1.0, linestyle=":")
    ax.axvline(2.0, color=GREY, linewidth=1.0, linestyle=":")
    ax.annotate("stable", xy=(1.05, 0.04), xycoords=("data", "axes fraction"),
                fontsize=8, color=GREY)
    ax.annotate("divergent", xy=(2.15, 0.04), xycoords=("data", "axes fraction"),
                fontsize=8, color=GREY)
    ax.set_xlabel(r"$\rho_{\rm eff} = v^\top\!F v \, / \, v^\top\!B v$   (median)")
    ax.set_ylabel(r"$2N\lambda$")
    ax.set_title("the threshold is $\\rho_{\\rm eff}=2$, not $\\rho>1$", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[1]
    if cases:
        drawn = []
        for res in cases:
            v, F, B, _, _ = draw(res["N"], res["D"], res["K"], res["d_f"], res["d_b"],
                                 int(res["seeds"][0]), N_d=res["N_d"])
            drawn.append(second_moment_rate(v, F, B)[1].ravel())
        # a thin write pool puts a long right tail on 1/c; the axis follows the bulk and
        # the tail is counted in the last bin rather than given nine tenths of the panel
        hi = float(max(np.percentile(d, 99) for d in drawn))
        bins = np.linspace(0.0, hi, 45)
        for i, (res, d) in enumerate(zip(cases, drawn)):
            ax.hist(np.clip(d, None, bins[-1]), bins=bins, color=COLORS[i % len(COLORS)],
                    alpha=0.5, edgecolor="none",
                    label=f"$\\rho$ = {res['rho']:.0f}  (median {np.median(d):.2f})")
        ax.set_xlim(0.0, hi)
        ax.axvline(2.0, color=GREY, linewidth=1.4, linestyle="--")
        ax.annotate("threshold", xy=(2.0, 0.97), xycoords=("data", "axes fraction"),
                    xytext=(4, -10), textcoords="offset points", fontsize=8, color=GREY)
        ax.set_xlabel(r"$\rho_{\rm eff}$, per task and column")
        ax.set_ylabel("count")
        ax.set_title("the realised ratio at the settings above, one seed", fontsize=10)
        ax.legend(frameon=False, fontsize=8)

    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(True, color="#e1e0d9", linewidth=0.8)
        ax.set_axisbelow(True)
    return axes


# ================================================================== reporting

def header(res):
    """The one line that says which run this is."""
    return (f"N={res['N']}  D={res['D']}  K={res['K']}  "
            f"d_f={res['d_f_realised']:.3f} (n_f={res['n_f']})  "
            f"d_b={res['d_b_realised']:.3f} (n_b={res['n_b']})  "
            f"rho={res['rho']:.2f}  "
            f"N_d={res['N_d']} (N_s={res['N_s']}, "
            f"{gating_name(res['N_d'], res['D'])})  "
            + (f"anchor t*={res['ts'] + 1}  " if res["anchors"].size == 1 else
               f"{res['anchors'].size} anchors around t*={res['ts'] + 1}  ")
            + f"{len(res['seeds'])} seeds"
            + (f"  phi={res['phi']:.2f}" if res["phi"] else ""))


def summarise(cases):
    """The curves at a handful of lags, and the diagnostics, as a printable table."""
    eps_mach = np.finfo(float).eps
    ref = cases[0]
    lags = ref["lags"]
    want = (int(lags.min()), -100, -10, -3, -1, 0, 1, 3, 10, 100, int(lags.max()))
    picks = list(dict.fromkeys(l for l in want if l in set(lags.tolist())))
    idx = [int(np.where(lags == l)[0][0]) for l in picks]

    lines = []
    for res in cases:
        lines.append(header(res))
        lines.append("-" * 78)
        lines.append("  " + "Delta".ljust(10) + "".join(f"{l:>10d}" for l in picks))
        for name, key in (("1 - eps, eq. (29)/(32)", "eps_direct"),
                          ("1 - eps, the stream", "eps_stream"),
                          ("1 - M,   eq. (29)/(32)", "M_direct"),
                          ("1 - M,   the stream", "M_stream")):
            m = score(_agg(res[key])[0])[idx]
            lines.append(f"  {name}   (geometric mean over seeds)")
            lines.append("    " + "".join(f"{x:>10.4g}" for x in m))
        # Delta = 0 has no relative gap to report: the direct route returns exactly 0
        # there -- an empty band -- so the ratio has nothing to divide by.
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = (np.abs(res["eps_direct"] - res["eps_stream"])
                   / res["eps_direct"]).max(axis=0)
        lines.append("  worst RELATIVE gap on eps, in units of machine eps")
        lines.append("    " + "".join("         ." if not np.isfinite(x)
                                      else f"{x / eps_mach:>10.3g}"
                                      for x in rel[idx]))
        lines.append("")
        lam, lam_r = res["lam"].mean(), res["lam_resid"].mean()
        lines.append(f"  mean ||r_t||                   {res['resid'].mean():.4g}")
        lines.append(f"  rho_eff (median over t, a)     {res['rho_eff'].mean():.3f}   "
                     f"(nominal rho = {res['rho']:.2f}; the threshold is 2)")
        lines.append(f"  (rho_eff - 2)/N, closed form   {res['smr'].mean():+.5f}/step")
        lines.append(f"  Lyapunov exponent of (19)      {lam:+.5f}/task   "
                     f"(measured, burn-in {LYAP_BURN})")
        lines.append(f"  the same from ||r_t||          {lam_r:+.5f}/task   "
                     f"(fit to the residuals eq. (29) returns)")
        lines.append(f"    over K = {res['K']} tasks        exp(lambda K) = "
                     f"{np.exp(lam * res['K']):.3e}")
        lines.append(f"  worst ||(I+L)^-1||_inf         {res['amp'].max():.4e}")
        lines.append(f"  off-diagonal Gamma             {res['gam_mean'].mean():.4f} "
                     f"+- {res['gam_sd'].mean():.4f}   "
                     f"(mean should sit at d_f = {res['d_f_realised']:.3f})")
        lines.extend(gating_lines(res["N_d"], res["N_s"], res["D"]))
        lines.append(f"  worst fit, eq. (18)            {res['fit'].max():.3e}")
        lines.append(f"  worst |R_stream - R_(29)|      {res['gap'].max():.3e}  "
                     f"(relative: {res['gap'].max() / res['resid'].mean():.3e})")
        lines.append("")
    lines.append("-" * 78)
    lines.append(f"  machine epsilon is {eps_mach:.2e}")
    lines.append("  the values the curves must hit exactly, for every rho and seed:")
    if ref["lags"].min() == -(ref["ts"] + 1):
        lines.append(f"    Delta = {ref['lags'].min()}: the state is W^0 = 0, so "
                     "eps = 1, eps_ctrl = 1, M = 1, and both scores are 0")
    lines.append("    Delta = 0: the anchor was just trained, so eps = 0, M = 0, and "
                 "both scores are 1")
    lines.append("")
    for line in (
        "  What sets the scale of the three eps curves is the growth rate above, not",
        "  the lag.  One step of eq. (19) grows the second moment of an error already",
        "  in the weights iff rho_eff = v'Fv / v'Bv exceeds 2, with margin",
        "  (rho_eff - 2)/N, so the threshold is NOT rho > 1: at rho = 1 the nesting",
        "  makes Pi symmetric, eq. (20), and I - Pi is an orthogonal projection, while",
        "  between 1 and the threshold it is oblique and the product still contracts.",
        "  Past the threshold the exponent is positive and of order 1/N, so the",
        "  residuals grow like exp(lambda K) = exp(c(rho) K / 2N) and the blow-up is a",
        "  function of K/N alone.  The regime eq. (13) asks for, K >> N, is therefore",
        "  the one in which a split mask diverges: bounded eps curves at rho = 5 need K",
        "  well below N, which is the opposite regime, so the two cannot be had at",
        "  once.  That is what the M figure is for -- dividing by a task the stream was",
        "  never given removes the growth that every task shares and leaves the part",
        "  that is about the anchor.  Nothing about eq. (29) or eq. (32) fails while",
        "  any of this happens; that is row 3, which stays within a few parts in 10^10",
        "  of zero throughout.",
        "",
        "  Two assumptions carry the divergence, and neither is in the note.  Each task",
        "  is trained to its exact fixed point -- stopping the inner flow early replaces",
        "  Pi by alpha Pi and moves the threshold to alpha < 2/rho_eff -- and `draw`",
        "  re-permutes the masks independently every task, the most decorrelated member",
        "  of the free array (11).  Masks that drift or cycle cut the exponent by one to",
        "  three orders of magnitude.",
        "",
        "  The gating granularity N_d is orthogonal to all of that.  Eq. (19) acts on",
        "  each column separately, so rho_eff and the exponent are per-column",
        "  quantities and a coarser gate does not move either: it decides how many of",
        "  the D columns share one realisation of them.  At N_d = D the D columns eps",
        "  sums over carry D independent growth rates and eps averages them; at",
        "  N_d = 1 they carry one, so the same mean arrives with a wider spread across",
        "  seeds and a heavier tail.  Read a change in the band between granularities",
        "  as that, and not as a change in the dynamics.",
    ):
        lines.append(line)
    return "\n".join(lines)


# =============================================================== the self-test
# Three more readings of the same algebra, each as unlike the fast path as it can be
# made: naive loops, a simulation that rebuilds every state from scratch, and an
# integration of the flow itself that is told nothing.

def _naive_coupling(v, F, B):
    """Gamma by triply-nested loops, eq. (21) copied off the page."""
    K, N, Dd = F.shape
    Gamma = np.zeros((K, K, Dd))
    for a in range(Dd):
        for t in range(K):
            c = sum(B[t, i, a] * v[i] ** 2 for i in range(N))
            for k in range(K):
                num = sum(F[k, i, a] * B[t, i, a] * v[i] ** 2 for i in range(N))
                Gamma[k, t, a] = num / c
    return Gamma


def _naive_lag_direct(Gamma_row, R, ts, lags):
    """The two band formulas as written sums: no cumulative sums, no slicing tricks."""
    Dd = R.shape[1]
    out = np.zeros((len(lags), Dd))
    for q, dl in enumerate(lags):
        for a in range(Dd):
            if dl >= 0:
                s = 0.0
                for j in range(ts + 1, ts + dl + 1):
                    s -= Gamma_row[j, a] * R[j, a]
            else:
                s = R[ts, a]
                for j in range(ts + dl + 1, ts):
                    s += Gamma_row[j, a] * R[j, a]
            out[q, a] = s
    return out


def _naive_protocol(v, F, B, Theta, ts, lags, F_read=None, theta_read=None):
    """The protocol itself: train the stream, rebuild each state, read it.

    Each state is rebuilt from W = 0 by replaying the first s updates, so nothing is
    carried between lags and an error in the stepping of one state cannot hide in the
    next.  The update is eq. (18) -- residual form -- which is the one form of the fixed
    point neither `run_lds` nor the direct solve uses.  `F_read` / `theta_read` default
    to the anchor's own; pass the control's to check that route.
    """
    K, Nn, Dd = F.shape
    w = v ** 2
    F_read = F[ts] if F_read is None else F_read
    theta_read = Theta[ts] if theta_read is None else theta_read

    updates = []
    W = np.zeros((Nn, Dd))
    for t in range(K):
        c = w @ B[t]
        r = Theta[t] - np.einsum("i,ia,ia->a", v, F[t], W)
        upd = B[t] * np.outer(v, r / c)
        updates.append(upd)
        W = W + upd

    out = np.zeros((len(lags), Dd))
    for q, dl in enumerate(lags):
        Ws = np.zeros((Nn, Dd))
        for t in range(ts + 1 + dl):
            Ws = Ws + updates[t]
        out[q] = theta_read - np.einsum("i,ia,ia->a", v, F_read, Ws)
    return out


def _integrate_stream(v, F, B, Theta, ts, lags, decay=26.0, steps=900):
    """The flow, eq. (6), integrated, and the anchor read out of the states it leaves.

        dW/dtau = B_t . ( v (theta_t - v^T (F_t . W)) ),

    stepped with a fourth-order explicit scheme (classical Runge-Kutta).  Nothing here is
    told the fixed point; "trained to convergence" becomes a window `decay` of the task's
    own slowest time constant wide, which leaves a relative error of order exp(-decay)
    for the next task to inherit.  That truncation, not the stepper, is what sets the
    agreement, and it is why this check runs at a tolerance and the other two at machine
    precision.  The step is held below a tenth of the fastest coordinate's time constant,
    because a thin write mask makes the rates ragged and an explicit stepper is not even
    stable past about 2.8.
    """
    K, Nn, Dd = F.shape
    c = np.einsum("i,tia->ta", v ** 2, B)
    W = np.zeros((Nn, Dd))
    states = {0: W.copy()}
    want = set(int(ts + 1 + dl) for dl in lags)

    for t in range(K):
        Ft, Bt, th = F[t], B[t], Theta[t]
        span = decay / c[t].min()
        nstep = max(steps, int(np.ceil(c[t].max() * span / 0.1)))
        dt = span / nstep

        def rhs(state):
            e = th - np.einsum("i,ia,ia->a", v, Ft, state)
            return Bt * np.outer(v, e)

        for _ in range(nstep):
            k1 = rhs(W)
            k2 = rhs(W + 0.5 * dt * k1)
            k3 = rhs(W + 0.5 * dt * k2)
            k4 = rhs(W + dt * k3)
            W = W + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if t + 1 in want:
            states[t + 1] = W.copy()

    out = np.zeros((len(lags), Dd))
    for q, dl in enumerate(lags):
        out[q] = Theta[ts] - np.einsum("i,ia,ia->a", v, F[ts], states[int(ts + 1 + dl)])
    return out


def self_test(verbose=True):
    """Every claim this file rests on, checked against something that does not share its
    code path.  Returns (ok, report)."""
    eps_mach = np.finfo(float).eps
    lines, ok = [], True

    def check(name, value, tol):
        nonlocal ok
        good = bool(np.isfinite(value)) and value <= tol
        ok = ok and good
        lines.append(f"  [{'ok ' if good else 'FAIL'}]  {name:<52}"
                     f"{value:>10.2e}  (tol {tol:.0e})")

    def note(good, name):
        nonlocal ok
        ok = ok and good
        lines.append(f"  [{'ok ' if good else 'FAIL'}]  {name}")

    # -- small, exhaustively checkable ------------------------------------------
    Ns, Ds, Ks = 20, 3, 12
    for d_f, d_b in ((0.5, 0.5), (0.5, 0.25), (0.75, 0.15)):
        for seed in (0, 1):
            v, F, B, Theta, (n_f, n_b) = draw(Ns, Ds, Ks, d_f, d_b, seed)
            F_ctrl, th_ctrl = draw_control(Ns, Ds, n_f, seed)
            ts = anchor_index(Ks)
            lags = lag_range(Ks, ts)
            Gamma, c = coupling(v, F, B)
            tag = f"d_f={d_f} d_b={d_b} seed={seed}"

            # the masks
            check(f"nesting B.F = B                      [{tag}]",
                  np.abs(B * F - B).max(), 0.0)
            check(f"counts n_f={n_f} n_b={n_b} per column       [{tag}]",
                  max(np.abs(F.sum(axis=1) - n_f).max(),
                      np.abs(B.sum(axis=1) - n_b).max(),
                      np.abs(F_ctrl.sum(axis=0) - n_f).max()), 0.0)

            # the coupling
            check(f"Gamma vs naive loops                 [{tag}]",
                  np.abs(Gamma - _naive_coupling(v, F, B)).max(), 1e-13)
            check(f"Gamma_tt = 1, eq. (22)               [{tag}]",
                  np.abs(np.einsum("tta->ta", Gamma) - 1.0).max(), 1e-13)
            check(f"Gamma in [0,1], eq. (21)             [{tag}]",
                  max(-Gamma.min(), Gamma.max() - 1.0, 0.0), 1e-13)
            check(f"coupling_row vs Gamma[k]             [{tag}]",
                  np.abs(coupling_row(v, F[3], B, c) - Gamma[3]).max(), 1e-14)

            # the rank-one map of eq. (19), built explicitly for one column
            Pi = np.outer(B[0, :, 0] * v, F[0, :, 0] * v) / c[0, 0]
            check(f"Pi^2 = Pi, eq. (20)                  [{tag}]",
                  np.abs(Pi @ Pi - Pi).max(), 1e-12)
            check(f"tr Pi = 1, eq. (20)                  [{tag}]",
                  abs(np.trace(Pi) - 1.0), 1e-12)
            check(f"rank(I - Pi) = N - 1                 [{tag}]",
                  abs(np.linalg.matrix_rank(np.eye(Ns) - Pi) - (Ns - 1)), 0.0)
            # symmetric exactly at rho = 1 and oblique otherwise, with
            # ||I - Pi||_2 = ||Pi||_2 = sqrt(v'Fv / c) = sqrt(rho_eff)
            f_mass = np.einsum("i,tia->ta", v ** 2, F)
            if n_f == n_b:
                check(f"Pi symmetric at rho = 1              [{tag}]",
                      np.abs(Pi - Pi.T).max(), 1e-12)
            else:
                check(f"||I-Pi|| = ||Pi|| = sqrt(rho_eff)    [{tag}]",
                      max(abs(np.linalg.norm(np.eye(Ns) - Pi, 2)
                              - np.linalg.norm(Pi, 2)),
                          abs(np.linalg.norm(Pi, 2)
                              - np.sqrt(f_mass[0, 0] / c[0, 0]))), 1e-10)

            # the two routes, anchor and control
            R_direct = solve_direct(Gamma, Theta)
            traces, R_stream, fit = run_lds(
                v, F, B, Theta, np.stack([F[ts], F_ctrl]))
            trace, trace_c = traces[0], traces[1]
            check(f"zero loss on the current task, (18)  [{tag}]", fit, 1e-12)
            check(f"R: stream vs eq. (29)                [{tag}]",
                  np.abs(R_stream - R_direct).max(), 1e-10)

            rd = lag_residuals_direct(Gamma[ts], R_direct, ts, lags)
            rs = lag_residuals_stream(trace, Theta[ts], ts, lags)
            check(f"lag family: eq. (29)/(32) vs stream  [{tag}]",
                  np.abs(rd - rs).max(), 1e-10)
            check(f"band formulas vs naive loops         [{tag}]",
                  np.abs(rd - _naive_lag_direct(Gamma[ts], R_direct, ts, lags)).max(),
                  1e-12)
            check(f"lag family vs replayed protocol      [{tag}]",
                  np.abs(rd - _naive_protocol(v, F, B, Theta, ts, lags)).max(), 1e-10)

            g_ctrl = coupling_row(v, F_ctrl, B, c)
            cd = lag_readout_direct(g_ctrl, R_direct, th_ctrl, ts, lags)
            cs = lag_residuals_stream(trace_c, th_ctrl, ts, lags)
            check(f"control: eq. (24) vs stream          [{tag}]",
                  np.abs(cd - cs).max(), 1e-10)
            check(f"control vs replayed protocol         [{tag}]",
                  np.abs(cd - _naive_protocol(v, F, B, Theta, ts, lags, F_ctrl,
                                              th_ctrl)).max(), 1e-10)
            check(f"control eps never 0 (never trained)  [{tag}]",
                  max(0.0, 1e-9 - eps_metric(cd, th_ctrl).min()), 1e-12)

            # the seams and the exact ends
            i0 = int(np.where(lags == 0)[0][0])
            im1 = int(np.where(lags == -1)[0][0])
            check(f"Delta = 0 gives r = 0, eq. (18)      [{tag}]",
                  np.abs(rd[i0]).max(), 1e-11)
            check(f"Delta = -1 gives r_t, eq. (14)       [{tag}]",
                  np.abs(rd[im1] - R_direct[ts]).max(), 1e-13)
            check(f"Delta = -t* gives theta_t*, eq. (26) [{tag}]",
                  np.abs(rd[0] - Theta[ts]).max(), 1e-11)
            check(f"control at Delta = -t* is theta_ctrl [{tag}]",
                  np.abs(cd[0] - th_ctrl).max(), 1e-13)
            e = eps_metric(rd, Theta[ts])
            M = e / eps_metric(cd, th_ctrl)
            check(f"eps: score 1 at 0, 0 at -t*          [{tag}]",
                  max(abs(score(e)[i0] - 1.0), abs(score(e)[0])), 1e-10)
            check(f"M:   score 1 at 0, 0 at -t*          [{tag}]",
                  max(abs(score(M)[i0] - 1.0), abs(score(M)[0])), 1e-10)

    # -- the gating granularity: N_d dendrites of N_s = D/N_d synapses each -------
    # N_d moves how COARSE the gate is, from one synapse per dendrite (every column
    # independent, which is what this file did before N_d existed) to one dendrite
    # per neuron (a whole row on or off).  The claim is that the algebra does not
    # notice, so that is what is checked: the masks acquire the block structure and
    # nothing else, Gamma, c and L inherit it, and both routes still agree at every
    # granularity -- against each other, against the naive loops, against the
    # replayed protocol, and once against the integrated flow.
    Ng, Dg, Kg = 16, 12, 9
    note(dendrites(Dg, None) == (Dg, 1) and dendrites(Dg, 1) == (1, Dg)
         and dendrites(Dg, 3) == (3, 4),
         f"dendrites(D={Dg}):  None -> {dendrites(Dg, None)},  1 -> {dendrites(Dg, 1)},"
         f"  3 -> {dendrites(Dg, 3)}")
    note(all(gating_name(k, Dg) == n for k, n in
             ((Dg, "synaptic"), (1, "neuronal"), (3, "dendritic"))),
         "the three granularities are named synaptic / neuronal / dendritic")
    check("N_d = D is the pre-N_d draw, bit for bit",
          max(float(np.abs(a - b).max()) for a, b in
              zip(draw(Ng, Dg, Kg, 0.5, 0.25, 0, N_d=Dg)[:4],
                  draw(Ng, Dg, Kg, 0.5, 0.25, 0)[:4])), 0.0)
    check("N_d = D leaves the control's draw alone too",
          float(np.abs(draw_control(Ng, Dg, 8, 0, N_d=Dg)[0]
                       - draw_control(Ng, Dg, 8, 0)[0]).max()), 0.0)

    for N_d_ in (1, 2, 3, 4, 6, Dg):
        N_s_ = Dg // N_d_
        v, F, B, Theta, (n_f, n_b) = draw(Ng, Dg, Kg, 0.5, 0.25, 7, N_d=N_d_)
        F_ctrl, th_ctrl = draw_control(Ng, Dg, n_f, 7, N_d=N_d_)
        ts = anchor_index(Kg)
        lags = lag_range(Kg, ts)
        Gamma, c = coupling(v, F, B)
        L = strict_lower(Gamma)
        tag = f"N_d={N_d_:<2} N_s={N_s_:<2}"

        # the block structure, and that it is the ONLY structure: the N_s columns on a
        # dendrite carry one mask, and neighbouring dendrites do not.
        def blocked(x, _N_d=N_d_, _N_s=N_s_):
            """How far x strays from its own dendrite's first column: 0 iff blocked."""
            y = np.asarray(x)
            y = y.reshape(*y.shape[:-1], _N_d, _N_s)
            return float(np.abs(y - y[..., :1]).max())
        check(f"one mask per dendrite, F, B and the control [{tag}]",
              max(blocked(F), blocked(B), blocked(F_ctrl)), 0.0)
        note(N_d_ == 1 or not np.array_equal(F[:, :, 0], F[:, :, N_s_]),
             f"neighbouring dendrites carry different masks  [{tag}]")
        check(f"the nesting and the counts survive it       [{tag}]",
              max(float(np.abs(B * F - B).max()),
                  float(np.abs(F.sum(axis=1) - n_f).max()),
                  float(np.abs(B.sum(axis=1) - n_b).max()),
                  float(np.abs(F_ctrl.sum(axis=0) - n_f).max())), 0.0)

        # Gamma, c and L inherit it; L is (D,K,K), so its dendrite axis is the first
        check(f"Gamma, c and L repeat across a dendrite     [{tag}]",
              max(blocked(Gamma), blocked(c),
                  float(np.abs(L.reshape(N_d_, N_s_, Kg, Kg)
                               - L.reshape(N_d_, N_s_, Kg, Kg)[:, :1]).max())), 0.0)

        # and the whole pipeline is still exact at this granularity
        R_direct = solve_direct(Gamma, Theta)
        traces, R_stream, fit = run_lds(v, F, B, Theta, np.stack([F[ts], F_ctrl]))
        rd = lag_residuals_direct(Gamma[ts], R_direct, ts, lags)
        cd = lag_readout_direct(coupling_row(v, F_ctrl, B, c), R_direct, th_ctrl,
                                ts, lags)
        check(f"zero loss on the current task, eq. (18)     [{tag}]", fit, 1e-12)
        check(f"R: stream vs eq. (29)                       [{tag}]",
              float(np.abs(R_stream - R_direct).max()), 1e-11)
        check(f"lag family: eq. (29)/(32) vs the stream     [{tag}]",
              float(np.abs(rd - lag_residuals_stream(
                  traces[0], Theta[ts], ts, lags)).max()), 1e-11)
        check(f"control: eq. (24) vs the stream             [{tag}]",
              float(np.abs(cd - lag_residuals_stream(
                  traces[-1], th_ctrl, ts, lags)).max()), 1e-11)
        check(f"lag family vs the replayed protocol         [{tag}]",
              float(np.abs(rd - _naive_protocol(v, F, B, Theta, ts, lags)).max()), 1e-11)
        e = eps_metric(rd, Theta[ts])
        check(f"eps: score 1 at Delta = 0, 0 at -t*         [{tag}]",
              max(abs(score(e)[int(np.where(lags == 0)[0][0])] - 1.0),
                  abs(score(e)[0])), 1e-10)
        if N_d_ in (1, 3, Dg):          # the triply-nested loops, at three of the six
            check(f"Gamma vs naive loops                        [{tag}]",
                  float(np.abs(Gamma - _naive_coupling(v, F, B)).max()), 1e-13)
            check(f"band formulas vs naive loops                [{tag}]",
                  float(np.abs(rd - _naive_lag_direct(
                      Gamma[ts], R_direct, ts, lags)).max()), 1e-12)

    # neuronal gating is the row-constant case the note writes out beside eq. (29):
    # one mask for all D columns, so the D triangular systems are one system with D
    # right-hand sides.  Checked against a solve that never sees the other D-1.
    v, F, B, Theta, _ = draw(Ng, Dg, Kg, 0.5, 0.25, 2, N_d=1)
    Gm1 = coupling(v, F, B)[0]
    L1 = strict_lower(Gm1)
    check("N_d = 1: L^[a] is one matrix for every a",
          float(np.abs(L1 - L1[:1]).max()), 0.0)
    check("N_d = 1: the D solves collapse into one",
          float(np.abs(solve_direct(Gm1, Theta)
                       - np.linalg.solve(np.eye(Kg) + L1[0], Theta)).max()), 1e-13)
    note(not np.allclose(solve_direct(Gm1, Theta)[:, 0], solve_direct(Gm1, Theta)[:, 1]),
         "N_d = 1: R still differs column by column, because Theta does not repeat")

    # the flow itself, once, at a coarse gate: the one route told nothing at all
    v, F, B, Theta, _ = draw(12, 6, 7, 0.5, 0.25, 3, N_d=2)
    ts6 = anchor_index(7)
    lg6 = lag_range(7, ts6)
    Gm6 = coupling(v, F, B)[0]
    check("N_d = 2: lag family vs the integrated flow, eq. (6)",
          float(np.abs(lag_residuals_direct(Gm6[ts6], solve_direct(Gm6, Theta), ts6, lg6)
                       - _integrate_stream(v, F, B, Theta, ts6, lg6)).max()), 1e-8)

    # end to end, through run_case, at both extremes
    for N_d_ in (1, Dg):
        rc = run_case(Ng, Dg, Kg, 0.5, 0.25, (0, 1), N_d=N_d_)
        i0 = int(np.where(rc["lags"] == 0)[0][0])
        check(f"run_case at N_d = {N_d_:<2}: the two routes agree on eps",
              float(np.abs(rc["eps_direct"] - rc["eps_stream"]).max()), 1e-10)
        check(f"run_case at N_d = {N_d_:<2}: the two routes agree on M",
              float(np.abs(rc["M_direct"] - rc["M_stream"]).max()), 1e-10)
        check(f"run_case at N_d = {N_d_:<2}: 1 at Delta = 0, 0 at -t*",
              max(float(np.abs(score(rc["M_direct"])[:, i0] - 1.0).max()),
                  float(np.abs(score(rc["eps_direct"])[:, 0]).max())), 1e-10)
        note(rc["N_d"] == N_d_ and rc["N_s"] == Dg // N_d_
             and f"N_d={N_d_}" in header(rc),
             f"run_case records N_d = {rc['N_d']}, N_s = {rc['N_s']}, and header says so")
    note(sweep_rho(rhos=(1,), seeds=(0,), N=20, D=6, K=8, d_f=0.5,
                   N_d=2)[0]["N_d"] == 2,
         "sweep_rho carries N_d out of CONFIG into every case")

    # -- the metric's normalisation, which the unit-norm draw hides ---------------
    # draw() returns unit teacher rows, so ||theta||^2 = 1 in every path above and an
    # error in the denominator would not show.  Scale the teacher by hand instead.
    r_, th_ = np.array([[0.3, -0.4, 0.5]]), np.array([1.0, 2.0, 2.0])
    check("eps normalises by ||theta||^2",
          abs(eps_metric(r_, th_)[0] - 0.5 / 9.0), 1e-14)
    check("eps invariant to r and theta scaled together",
          abs(eps_metric(7.0 * r_, 7.0 * th_)[0] - eps_metric(r_, th_)[0]), 1e-14)
    check("eps reduces over the input axis, not the lag",
          np.abs(eps_metric(np.vstack([r_, 2 * r_]), th_)
                 - np.array([0.5, 2.0]) / 9.0).max(), 1e-14)

    # -- the growth rate, three ways ---------------------------------------------
    # At rho = 1 the nesting makes Pi symmetric, so I - Pi is an orthogonal projection
    # and every step is nonexpansive: a machine-precision statement, checked step by
    # step rather than through an exponent.  Past the threshold the exponent must be
    # positive, must agree with a fit to the residuals the triangular solve returns, and
    # must sit near the closed form (rho_eff - 2)/2N -- three computations sharing
    # nothing but the masks.
    v, F, B, Theta, _ = draw(60, 2, 400, 0.5, 0.5, 5)
    w, worst = v ** 2, 0.0
    x = np.random.default_rng(0).standard_normal(60)
    for t in range(400):
        u, z = B[t, :, 0] * v, F[t, :, 0] * v
        y = x - u * ((z @ x) / (w @ B[t, :, 0]))
        worst = max(worst, np.linalg.norm(y) / np.linalg.norm(x) - 1.0)
        x = y / max(np.linalg.norm(y), 1e-300)
    check("rho = 1: every step of eq. (19) nonexpansive", max(worst, 0.0), 1e-12)
    check("rho = 1: Lyapunov exponent <= 0", max(0.0, lyapunov(v, F, B)), 1e-12)

    # rho = 1.5 is oblique -- ||I - Pi|| > 1 -- and must STILL be stable, because
    # rho_eff < 2.  This is the case the "rho > 1 diverges" reading gets wrong.
    v, F, B, _, _ = draw(120, 1, 1200, 0.5, 0.5 / 1.5, 3)
    smr, rho_eff = second_moment_rate(v, F, B)
    check("rho = 1.5: rho_eff still below 2", max(0.0, np.median(rho_eff) - 2.0), 1e-12)
    check("rho = 1.5: oblique but stable, lambda <= 0",
          max(0.0, lyapunov(v, F, B)), 1e-12)

    # Both rates are estimated off one finite stream, so they are compared as an
    # ensemble: four seeds of K = 800 brings the two means within a few per cent.
    for rho_ in (3, 5):
        lams, rates, smrs = [], [], []
        for s in range(4):
            v, F, B, Theta, _ = draw(60, 2, 800, 0.5, 0.5 / rho_, s)
            lams.append(lyapunov(v, F, B))
            rates.append(growth_rate(solve_direct(coupling(v, F, B)[0], Theta)))
            smrs.append(second_moment_rate(v, F, B)[0])
        lam, lam_r, smr = float(np.mean(lams)), float(np.mean(rates)), float(np.mean(smrs))
        check(f"rho = {rho_}: Lyapunov exponent > 0", max(0.0, 1e-3 - lam), 1e-12)
        check(f"rho = {rho_}: exponent vs ||r_t|| growth (rel.)",
              abs(lam - lam_r) / abs(lam), 0.20)
        # the closed form is a second moment, the exponent an average of logs, so the
        # exponent sits below it; they agree in sign and within a factor of three here
        note(0.0 < lam <= smr / 2.0 * 3.0,
             f"rho = {rho_}: lambda ({lam:+.5f}) positive and below "
             f"(rho_eff-2)/2N ({smr / 2:+.5f}) by < 3x")

    # -- the flow itself, once; the tolerance is the finite training window ------
    v, F, B, Theta, _ = draw(12, 2, 7, 0.5, 0.25, 3)
    ts = anchor_index(7)
    lags = lag_range(7, ts)
    Gamma, _ = coupling(v, F, B)
    rd = lag_residuals_direct(Gamma[ts], solve_direct(Gamma, Theta), ts, lags)
    check("lag family vs the integrated flow, eq. (6)",
          np.abs(rd - _integrate_stream(v, F, B, Theta, ts, lags)).max(), 1e-8)

    # -- the pipeline above the algebra, which the checks so far never touch -----
    cases = sweep_rho(rhos=(1, 2), seeds=(0, 1, 2), N=20, D=3, K=12, d_f=0.5)
    res = cases[0]
    i0 = int(np.where(res["lags"] == 0)[0][0])
    for key in ("eps_direct", "eps_stream", "M_direct", "M_stream"):
        check(f"run_case: {key:<11} 1 at Delta=0, 0 at -t*",
              max(np.abs(score(res[key])[:, i0] - 1.0).max(),
                  np.abs(score(res[key])[:, 0]).max()), 1e-10)
    check("run_case: the two routes agree on eps",
          np.abs(res["eps_direct"] - res["eps_stream"]).max(), 1e-10)
    check("run_case: the two routes agree on M",
          np.abs(res["M_direct"] - res["M_stream"]).max(), 1e-10)
    v, F, B, Theta, _ = draw(20, 3, 12, 0.5, 0.5, 0)
    amp_ref = np.abs(np.linalg.inv(np.eye(12)[None] + strict_lower(coupling(v, F, B)[0]))
                     ).sum(axis=-1).max()
    check("run_case: amp against an independent inverse",
          abs(res["amp"][0] - amp_ref) / amp_ref, 1e-12)

    # the anchor window: a window of one must reproduce the single-anchor path exactly,
    # a wider one must shorten the lags symmetrically and still hit both exact ends, and
    # its curve must be the geometric mean of the curves of its own anchors
    w1 = run_case(20, 3, 12, 0.5, 0.25, (0, 1), n_anchors=1)
    w3 = run_case(20, 3, 12, 0.5, 0.25, (0, 1), n_anchors=3)
    check("window of 1 reproduces the single anchor",
          np.abs(w1["eps_direct"] - run_case(20, 3, 12, 0.5, 0.25, (0, 1),
                                             ts=anchor_index(12))["eps_direct"]).max(),
          0.0)
    note(w3["lags"].min() == w1["lags"].min() + 1
         and w3["lags"].max() == w1["lags"].max() - 1
         and list(w3["anchors"]) == [w1["ts"] - 1, w1["ts"], w1["ts"] + 1],
         f"window of 3 shortens the lags by one each side: {list(w3['lags'][[0, -1]])}")
    i0 = int(np.where(w3["lags"] == 0)[0][0])
    check("window of 3: still exactly 1 at Delta = 0",
          np.abs(score(w3["M_direct"])[:, i0] - 1.0).max(), 1e-10)
    check("window of 3: routes still agree",
          np.abs(w3["eps_direct"] - w3["eps_stream"]).max(), 1e-10)
    # rebuild seed 0's window curve by hand from its three anchors
    v, F, B, Theta, _ = draw(20, 3, 12, 0.5, 0.25, 0)
    Fc, thc = draw_control(20, 3, counts(20, 0.5, 0.25)[0], 0)
    Gm3, c3 = coupling(v, F, B)
    Rd3 = solve_direct(Gm3, Theta)
    by_hand = geometric_mean(np.stack([
        eps_metric(lag_residuals_direct(Gm3[a], Rd3, a, w3["lags"]), Theta[a])
        for a in w3["anchors"]]), axis=0)
    check("window of 3 is the geometric mean of its anchors",
          np.abs(by_hand - w3["eps_direct"][0]).max(), 1e-12)
    note(res["lags"].size == 13 and res["eps_direct"].shape == (3, 13),
         f"run_case: shapes {res['eps_direct'].shape} for 3 seeds and K+1 = 13 lags")
    note(np.isfinite(np.asarray(_agg(res["eps_direct"]))).all(),
         "the geometric-mean aggregate is finite everywhere, eps = 0 included")
    note(len(summarise(cases).splitlines()) > 20, "summarise() produces a table")

    # phi must move the teachers and nothing else
    a = draw(20, 3, 12, 0.5, 0.25, 0, phi=0.0)
    b = draw(20, 3, 12, 0.5, 0.25, 0, phi=0.7)
    note(np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])
         and np.array_equal(a[0], b[0]), "phi leaves v, F and B bit-identical")
    note(not np.allclose(a[3], b[3]), "phi does move the teachers")
    note(np.array_equal(draw_control(20, 3, 10, 0)[0], draw_control(20, 3, 10, 0)[0])
         and not np.array_equal(draw_control(20, 3, 10, 0)[0],
                                draw_control(20, 3, 10, 1)[0]),
         "draw_control is reproducible per seed and differs across seeds")
    note(np.array_equal(draw(20, 3, 12, 0.5, 0.25, 0)[1],
                        draw(20, 3, 12, 0.5, 0.25, 0)[1]),
         "draw is reproducible")

    # -- shapes, ranges, and the guards -----------------------------------------
    K_ = 300
    ts_ = anchor_index(K_)
    lg = lag_range(K_, ts_)
    check("anchor puts K/2 tasks on each side",
          abs(min(ts_ + 1, K_ - 1 - ts_) - K_ // 2), 0.0)
    check("lag range is one per state, K+1 of them",
          abs(lg.size - (K_ + 1)) + abs(lg.min() + ts_ + 1)
          + abs(lg.max() - (K_ - 1 - ts_)), 0.0)

    Gm, c_ = coupling(*draw(20, 3, 12, 0.5, 0.25, 0)[:3])
    R_ = np.zeros((12, 3))
    guards = [
        ("counts() rejects d_b > d_f", lambda: counts(40, 0.2, 0.4), ValueError),
        ("dendrites() rejects an N_d that does not divide D",
         lambda: dendrites(12, 5), ValueError),
        ("dendrites() rejects N_d below 1", lambda: dendrites(12, 0), ValueError),
        ("dendrites() rejects N_d above D", lambda: dendrites(12, 13), ValueError),
        ("draw() rejects a ragged N_d",
         lambda: draw(20, 3, 12, 0.5, 0.25, 0, N_d=2), ValueError),
        ("draw_control() rejects a ragged N_d",
         lambda: draw_control(20, 3, 10, 0, N_d=2), ValueError),
        ("run_case() rejects a ragged N_d",
         lambda: run_case(20, 3, 12, 0.5, 0.25, (0,), N_d=2), ValueError),
        ("counts() rejects an empty write column", lambda: counts(40, 0.5, 0.0),
         ValueError),
        ("anchor_index rejects K < 2", lambda: anchor_index(1), ValueError),
        ("lag_range rejects an anchor outside the stream",
         lambda: lag_range(12, 12), ValueError),
        ("lag_residuals_direct rejects a negative anchor",
         lambda: lag_residuals_direct(Gm[0], R_, -1, [0]), ValueError),
        ("run_lds rejects a mis-shaped read stack",
         lambda: run_lds(*draw(20, 3, 12, 0.5, 0.25, 0)[:4], np.zeros((20, 3))),
         ValueError),
        ("anchor_window rejects n_anchors = 0",
         lambda: anchor_window(12, 0), ValueError),
        ("anchor_window rejects a window that runs off the stream",
         lambda: anchor_window(12, 99), ValueError),
        ("geometric_mean rejects a negative input",
         lambda: geometric_mean(np.array([[-1.0]])), ValueError),
        ("lag_residuals_stream rejects a lag off the trace",
         lambda: lag_residuals_stream(np.zeros((13, 3)), np.zeros(3), 5, [99]),
         ValueError),
        ("lag_residuals_direct rejects a float lag",
         lambda: lag_residuals_direct(Gm[5], R_, 5, [1.5]), TypeError),
        ("lag_residuals_stream rejects a float lag",
         lambda: lag_residuals_stream(np.zeros((13, 3)), np.zeros(3), 5, [1.5]),
         TypeError),
        ("metric_arrays rejects an unknown metric",
         lambda: metric_arrays({}, "nope"), ValueError),
        ("sweep_rho rejects a rho that n_f cannot realise",
         lambda: sweep_rho(rhos=(1, 7), seeds=(0,), N=60, D=2, K=6, d_f=0.5),
         ValueError),
    ]
    for name, fn, exc in guards:
        try:
            fn()
            note(False, f"{name}  -- it did not raise")
        except exc:
            note(True, name)
        except Exception as err:                                   # noqa: BLE001
            note(False, f"{name}  -- raised {type(err).__name__} instead: {err}")
    # a scalar lag is accepted by both routes, identically
    note(lag_residuals_direct(Gm[5], R_, 5, 2).shape == (1, 3)
         and lag_residuals_stream(np.zeros((13, 3)), np.zeros(3), 5, 2).shape == (1, 3),
         "both routes accept a scalar lag and return (1, D)")
    # the degenerate anchors, where one arm is empty
    for ts_edge in (0, 11):
        v, F, B, Theta, _ = draw(20, 3, 12, 0.5, 0.25, 0)
        Gm2, _ = coupling(v, F, B)
        Rd = solve_direct(Gm2, Theta)
        lg2 = lag_range(12, ts_edge)
        a = lag_residuals_direct(Gm2[ts_edge], Rd, ts_edge, lg2)
        b = lag_residuals_stream(run_lds(v, F, B, Theta, F[ts_edge][None])[0][0],
                                 Theta[ts_edge], ts_edge, lg2)
        check(f"anchor at the edge, ts = {ts_edge:<2}: routes agree",
              np.abs(a - b).max(), 1e-11)

    report = "\n".join(
        ["self-test", "=" * 78] + lines + ["=" * 78,
         f"  {'all checks passed' if ok else 'SOMETHING FAILED -- see above'};  "
         f"machine epsilon is {eps_mach:.2e}"])
    if verbose:
        print(report)
    return ok, report


# ==================================================================== the files

def save_figure(cases, path, metric="eps", dpi=180):
    """One 3 x 2 figure.  Agg, so the file is written whether or not anything watches."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(12.0, 10.5), layout="constrained")
    plot_lag_curves(cases, axes=axes, metric=metric)
    ref = cases[0]
    what = ("the note's normalised loss $\\varepsilon$, eq. (34)" if metric == "eps"
            else "$M=\\varepsilon/\\varepsilon_{\\rm ctrl}$, against a never-trained task")
    fig.suptitle(
        f"transfer and retention against the lag $\\Delta$   --   {what}\n"
        f"$N$={ref['N']}   $D$={ref['D']}   $K$={ref['K']}   "
        f"$d_f$={ref['d_f_realised']:.3f} ($n_f$={ref['n_f']})   "
        f"$N_d$={ref['N_d']} ($N_s$={ref['N_s']}, "
        f"{gating_name(ref['N_d'], ref['D'])} gating)   "
        f"{len(ref['seeds'])} seeds",
        fontsize=11, color=GREY, x=0.01, ha="left")
    path = Path(path)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def save_lyapunov_figure(scan, cases, path, dpi=180):
    """The stability threshold, two panels, in a file of its own."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9), layout="constrained")
    plot_lyapunov(scan, cases=cases, axes=axes)
    fig.suptitle("when the stream of eq. (19) diverges", fontsize=11, color=GREY,
                 x=0.01, ha="left")
    path = Path(path)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def main(outdir=None, stem="lag_curves", test=True, lyap=True):
    outdir = Path(outdir) if outdir is not None else Path(__file__).resolve().parent
    outdir.mkdir(parents=True, exist_ok=True)

    report = ""
    if test:
        ok, report = self_test()
        print()
        if not ok:
            raise SystemExit("self-test failed; not drawing anything")

    cases = sweep_rho()
    text = summarise(cases)
    print(text)

    paths = [save_figure(cases, outdir / f"{stem}.png", metric="eps"),
             save_figure(cases, outdir / f"{stem}_control.png", metric="control")]
    if lyap:
        paths.append(save_lyapunov_figure(lyapunov_scan(), cases,
                                          outdir / f"{stem}_lyapunov.png"))
    txt = outdir / f"{stem}.txt"
    txt.write_text(text + "\n\n" + report + "\n")
    paths.append(txt)
    print()
    for p in paths:
        print(f"wrote {p}")
    return cases


if __name__ == "__main__":
    args = list(sys.argv[1:])
    if "--test" in args:
        raise SystemExit(0 if self_test()[0] else 1)
    main(outdir=args[0] if args else None, lyap="--no-lyap" not in args)
