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
squaring, no ratio -- and both are named for what a large value means, since
higher is worse:

    interference(K)  ||r_K||,  r_K = theta_K - thetahat_K^(K-1)     eq. (12)

        the task that arrives at step K, read out of the state just before it
        trains: what the stream has left for it before it does any work of its own.
        One task, one number: nothing is pooled over the tasks around it, and the
        only average is over seeds

    forgetting(K)    (1/K) sum_{i=1}^{K} ||r_i^(K)||,
                     r_i^(K) = theta_i - thetahat_i^(K)             eq. (28)

        every task trained so far, read out of the current state, averaged.  There
        is no lag to choose: the average runs over all the lags the stream has had
        time to produce, K-1 down to 0.

(The code keys them "tr" and "ret", the names they had before they were renamed.)

Teacher rows are drawn on the unit sphere, so both scores sit on a scale where 0 is
a solved task and 1 is a state worth no more than W = 0, at which thetahat = 0.

Task similarity.  The teachers can share a direction:

    theta_t = sqrt(q) theta_0 + sqrt(1 - q) thetatilde_t

with theta_0 drawn once per seed, before any task, and thetatilde_t afresh for every
task, each a unit vector drawn exactly as a teacher row is.  So E||theta_t||^2 = 1,
the scale above is untouched, and E[theta_t . theta_u] = q for t != u.  theta_0
comes from its own stream of the replicate (`rng_for`, THETA0_PART), so q = 0 is the
stream every other figure runs, number for number, and at a fixed seed index every
q sees the same thetatilde, theta_0, readout, gates and W^0.  Only the teachers
change; the walk and the closed form take whatever Theta they are handed, and the
mean-field functions see q as a drive split into a part every task shares, of power
q, and a part of power 1 - q that is new on every task (`_drive_split`).

No figure draws the mean field (Gamma -> d_f).  It is right over the first few tasks
of a stream and wrong almost everywhere after, so every figure shows the two routes
that agree -- the walk and the closed form -- and nothing else.  The mean-field
functions stay in the file, checked against their own recursion by the self-test.

The stream starts from W^0, drawn once per seed with iid N(0, s^2/N_in) entries: the
scale a linear layer of fan-in N_in is initialised at, times s = W0_SCALE.  s = 0 is
the note's protocol, W^0 = 0, and reproduces the runs made from it exactly.  W^0 is
a baseline every task reads before anything has trained,
thetahat_t^(0) = v^T F_t W^0, so the first task inherits
||theta_1 - thetahat_1^(0)||, of mean square 1 + s^2 v^T F_1 v rather than 1.  It
enters the closed form in one place: the teachers are shifted by it,
Theta -> Theta - thetahat^(0) on the right of eq. (27), and the band eq. (30) is
unchanged.  The stream forgets it over about N tasks, as the writes overwrite W^0
neuron by neuron.

The third is the size of the weight change itself, in two readings:

    step(T)  = (1/T) sum_{t<=T} ||W^t - W^{t-1}||_F         the average step

        how far the state moves on a task, averaged over the stream.  Eq. (16)
        makes the step rank one, W^t - W^{t-1} = B_t v r_t / c_t, and the write
        mass is c_t = ||B_t v||^2 by eq. (15), so the step is exactly
        ||r_t|| / sqrt(c_t) -- a residual divided by the square root of the mass
        that had to carry it.

    disp(T) = ||W^T - W^0||_F                               the net displacement

        how far the state has moved from where it started: the same increments
        summed before the norm is taken rather than after.  It is not divided by
        T -- it measures the weights, not a rate -- so a state that has stopped
        growing reads flat.  T step(T) is the length of the path and disp(T) the
        distance between its ends, so disp(T) <= T step(T) always, and the ratio
        of the two is the fraction of the movement that did not cancel.  W^0 is
        subtracted because ||W^0||_F is about s sqrt(N), 14 at N = 200 and s = 1,
        which would bury the movement under the starting point.

        Measured at N = 200 from W^0 = 0: in the stable regime the state
        saturates, 7.5, 14.9 and 18.6 at T = 50, 500 and 5000 for
        d_f = d_b = 0.5 -- the contraction of eq. (32).  At d_f = 1 and rho > 1
        the readout pins v^T W to the latest teacher and the directions it cannot
        see random-walk, so disp grows like sqrt(T): 9.8, 31 and 98 at rho = 2.
        At d_f = d_b = 1 the sum telescopes exactly, sum_{t<=T} r_t =
        theta_T - v^T W^0, so disp(T) = ||theta_T - v^T W^0|| / ||v||: 1/||v||
        for every T from W^0 = 0.

        From a random W^0 the displacement also carries the erasure of W^0.  The
        part of the stream W^0 drives is linear and independent of the teachers,
        so it adds to disp^2: in the stable regime it climbs toward ||W^0||_F as
        the writes overwrite the initial weights -- about 4, 9 and 12 of 14 at
        T = 50, 500 and 5000, d_f = d_b = 0.5, s = 1 -- and figure 7 reads 8.57,
        17.5 and 22.5 there, where the zero start gave 7.57, 15.2 and 18.8.  At
        d_f = 1 one step removes the one direction every task reads and nothing
        more is ever touched: that part is ||v^T W^0|| / sqrt(c_1), 1.4, for good,
        and disp at rho = 2 reads 10.2, 31.9 and 101 against 10.1, 31.9 and 101.

Figure 8 reads the same weights against the task index rather than averaged over
the stream: the left panel is the step of eq. (16) that the arriving task itself
took, ||W^K - W^{K-1}||_F (the update magnitude), and the middle one is
disp(K) = ||W^K - W^0||_F once it has trained (the weight displacement) --
figure 7's metric, in the same units as the step beside it, so the middle panel
over the left says how many steps long the state is.  Like interference, the step
is task K's own and is not pooled over the tasks around it.  The right panel is the
size of the weights themselves, ||W^K||_F (the weight magnitude), the one metric
with a value before any task has trained: its traces start at K = 0 from
||W^0||_F, on a task axis that is linear below one task to make room for it.

Seeds -- independent draws of (readout, gates, teachers, initial state) -- are
combined geometrically, and every figure carries the standard error of that mean,
taken in logs and shown multiplicatively.

    python3 neuronal_curves.py            # self-test, then all eight figures
    python3 neuronal_curves.py --test     # the self-test alone
    python3 neuronal_curves.py 1 4        # only figures 1 and 4
    python3 neuronal_curves.py --jax 6 7  # the two heavy kernels on a GPU
    python3 neuronal_curves.py --sim      # self-test, then the task-similarity set
    python3 neuronal_curves.py --sim 3 5  # only its figures 3a/3b and 5a/5b

Figures written to CONFIG["OUTDIR"], a folder next to this file (this file's own
folder when OUTDIR is empty):

    fig1_flow.png         the stream as it actually runs: integrated gradient flow
                          against the two closed forms, eq. (16) and eq. (27), over
                          a stream of K_FLOW tasks -- the first and the last
                          FLOW_SHOW of them, with a break in the task axis between
    fig2a_vs_tasks.png    the two scores against task index, one trace per
                          density, d_f = d_b
    fig2b_vs_split.png    the same at d_f = FIG2B_DF, one trace per split ratio
    fig3_vs_tasks_split.png   against task index, one row per density, one trace
                          per split ratio
    fig4_vs_density.png   against read density, one trace per split ratio
    fig5_heatmaps.png     density x splitness, three snapshots of one run
    fig6_step.png         the average step, three stream lengths: against density,
                          against splitness, and the two against each other
    fig7_displacement.png the net displacement ||W^T - W^0||_F, the same three
                          panels
    fig8a_weight_vs_tasks.png   figure 2a's axes, carrying the weights instead of
                          the residual: the step of the arriving task, the net
                          displacement it leaves behind, and the weights
                          themselves, from W^0 at K = 0
    fig8b_weight_vs_split.png   the same at d_f = FIG2B_DF, per split ratio

Figures 2 and 8 come in an `a` and a `b`: `a` varies the density at d_f = d_b, `b`
holds d_f and varies the split ratio over the same three values figure 3 uses.  The
four share two sweeps between them, so no case is walked or solved twice.

`--sim` draws the task-similarity set instead, to CONFIG["SIM_OUTDIR"], numbered
after the figure each one reworks.  Every `a` holds d_f = d_b and varies the
density; every `b` holds d_f = SIM_SPLIT_DF and varies the split ratio:

    fig1_flow.png         figure 1 once per q, one row each, over a stream of
                          SIM_K_FLOW tasks: the first and the last SIM_FLOW_SHOW of
                          them, with a break in the task axis between the two
    fig2_vs_tasks.png     forgetting and interference against task index, one trace
                          per q, at d_f = d_b = SIM_DF
    fig3a_vs_tasks_density.png   one row per density, one trace per q, and five
                          columns: forgetting, interference, and figure 8's three
                          weight metrics -- update magnitude, weight displacement
                          and weight magnitude
    fig3b_vs_tasks_split.png     one row per split ratio, one trace per q
    fig4a_vs_q_density.png       both scores at the end of the stream against q,
                          one trace per density
    fig4b_vs_q_split.png         the same, one trace per split ratio
    fig5a_heatmaps_density.png   q against density, three snapshots of one run
    fig5b_heatmaps_split.png     q against splitness
    fig6a_metrics_density.png    five metrics off the penultimate state W^{K-1} of a
                          K_DW stream, one column each -- forgetting, interference
                          (the last task, read out of W^{K-1}), ||W^{K-1} - W^0||_F,
                          ||W^{K-1} - W^{K-2}||_F and ||W^{K-1}||_F -- and three
                          rows: against density (one trace per q), against q (one
                          per density), and the grid
    fig6b_step_split.png         the average step, 3x3: against splitness (one trace
                          per q), against q (one per split ratio), and the grid,
                          at three stream lengths
    fig7a_displacement_density.png, fig7b_displacement_split.png
                          the net displacement ||W^T - W^0||_F, 6b's panels with
                          density or splitness
    fig8_weight_vs_tasks.png     figure 8's three weight panels, one trace per q

Wherever a density or a split ratio is the trace it is one of SIM_DENSITIES or
SIM_RHOS, and those two are also the rows of figures 3a and 3b.  q is the trace
wherever it is not an axis, over SIM_QS.
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
    W0_SCALE=3,    # s: the initial state W^0 has iid N(0, s^2/N_in) entries, the
                     #      scale a linear layer of fan-in N_in is initialised at,
                     #      times s.  Every task then reads v^T F_t W^0 before it
                     #      has trained, of mean square s^2 v^T F_t v ~ s^2 d_f
                     #      against the teacher's 1.  0 is the zero start, and
                     #      reproduces the runs made before W^0 existed exactly
    SEEDS=20,        # independent seeds (readout, gates, teachers) per point.
                     #      every figure reports the geometric mean over them and
                     #      the standard error of that mean, taken in logs
    ROOT_SEED=20260922,  # the one entropy the whole study hangs off; every stream
                     #      is a named child of it, so running one figure alone
                     #      draws exactly what the full run draws
    OUTDIR="independentTasks",  # figures and the log go here, relative to this
                     #      file; "" writes next to it, where the W^0 = 0 set lives

    # --- task similarity: the --sim figures --------------------------------
    # theta_t = sqrt(q) theta_0 + sqrt(1-q) thetatilde_t, both unit vectors drawn
    # as teachers always are, theta_0 once per seed and thetatilde_t every task.
    # These are the traces and the held values every --sim figure reads; the
    # figure-by-figure settings further down (stream lengths, snapshots, grids,
    # caps) are shared with the figure each one reworks.
    SIM_QS=(0.0, 0.3, 0.5, 0.7, 0.9),   # q wherever similarity is the trace: one
                     #      trace each, light to dark on one hue (q is ordered),
                     #      at most five -- the ramp has five steps far enough
                     #      apart to tell neighbours by lightness
    SIM_DENSITIES=(0.9, 0.5, 0.3, 0.1),      # d_f = d_b wherever density is the trace
                     #      (figures 4a, 6a, 7a), and the rows of figure 3a, in
                     #      this order.  at most four: one categorical slot each
    SIM_RHOS=(1.0, 2.0, 5.0),           # d_f/d_b wherever splitness is the trace
                     #      (figures 4b, 6b, 7b), and the rows of figure 3b
    SIM_DF=0.3,      # figures 2 and 8: the density d_f = d_b they hold.  one of
                     #      SIM_DENSITIES shares figure 3a's sweep; any other value
                     #      is walked on its own
    SIM_SPLIT_DF=0.5,   # every b figure: the read density held while the split
                     #      varies.  not 1: there Gamma is all-ones, the split
                     #      cannot touch the residual, and 3b/4b/5b go flat
    SIM_Q_MAX=0.9,   # the q axes run from 0 to here
    SIM_Q_STEP=0.05,      # figure 4's q axis
    SIM_HEAT_Q_STEP=0.1,  # the q axis of figures 5, 6 and 7
    SIM_K_FLOW=1000,      # figure 1: the stream the flow is integrated over ...
    SIM_FLOW_SHOW=10,     #   ... and how many tasks at each end are drawn; the
                          #   original figure 1's K_FLOW and FLOW_SHOW, per q
    SIM_OUTDIR="TaskSimilarity_w0_3",   # where --sim writes, relative to this file

    # --- figure 1, the flow ------------------------------------------------
    K_FLOW=1000,     # the stream the flow is integrated over, every task of it ...
    FLOW_SHOW=10,    #   ... and how many tasks at each end are drawn out in flow
                     #   time, with a break in the task axis between, as --sim
                     #   draws it.  2 FLOW_SHOW >= K_FLOW draws the whole stream
    DF_FLOW=0.5,     # read density
    RHO_FLOW=2.0,    # split ratio d_f / d_b for the flow figure
    FLOW_SEED=0,
    PER_TASK=240,    # points recorded inside each task's window
    DECAY=20.0,      # window width, in time constants 1/c_t
    SUBSTEPS=4,      # integrator steps between two recorded points, at least
    MAX_RATE_STEP=0.1,   # ceiling on c_t dtau

    # --- figures 2 and 3, against task index -------------------------------
    K_TASKS=10000,    # stream length
    DENSITIES=(0.7,0.5, 0.3, 0.1),   # figure 2: d_f = d_b, one trace each
    SPLIT_DENSITIES=(1.0, 0.3, 0.2),  # figure 3: one row each
    SPLIT_RHOS=(1.0, 2.0, 5.0),       # figure 3: one trace each
    FIG3_FLOOR=1e-2,  # figure 3's y-limits.  the split ratio 5 runs to 1e86 by the
    FIG3_CEIL=1e6,    #   end of the stream, and letting the axis follow it would
                      #   flatten the other two traces into one line, so the axis
                      #   stops here and a clipped trace is labelled with where it
                      #   actually ended
    FIG2B_DF=1.0,    # figures 2b and 8b: the density held fixed while the split
                     #      varies over SPLIT_RHOS, the same three ratios as fig. 3
    FIG8_FLOOR=1e-1, # figure 8b's y-limits, as FIG3_* are figure 2b's.  neither
    FIG8_CEIL=1e6,   #      arm goes much below 1: the first step alone is
                     #      ||r_1||/sqrt(c_1), with c_1 <= ||v||^2 and
                     #      E||r_1||^2 = 1 + s^2 v^T F_1 v
    N_POINTS_DECADE=12,   # states sampled per decade of the logarithmic axis.
                     #      every arm is read exactly there and nowhere else:
                     #      forgetting and the displacement at state K, interference
                     #      and the step at task K, the arrival that made it

    # --- figure 4, against density -----------------------------------------
    K_DENSITY=1500,  # stream length; both scores are read at the end of it
    RHOS=(1.0, 2.0, 3.0),             # d_f / d_b, one trace each
    DF_STEP=0.05,    # read-density grid, from D/N to 1.0

    # --- figure 5, density against splitness -------------------------------
    K_HEAT=1000,     # one run per cell; the three columns are snapshots of it
    SNAPSHOTS=(10, 100, 1000),
    HEAT_DF_STEP=0.1,                 # read density, from D/N to 1.0
    HEAT_RHOS=tuple(range(1, 11)),    # splitness 1..10
    HEAT_CEIL=2.0,   # colour ceiling in log10||r||: a cell only has to read as
                     #   "a hundred times the baseline or worse".  the .txt keeps
                     #   the unclipped numbers

    # --- figures 6 and 7, the weight change --------------------------------
    K_DW=5000,       # one stream per cell; the three columns are snapshots of it.
                     #      --sim figure 6a reads its five metrics off the same
                     #      stream's penultimate state, K_DW - 1
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

# The ordinal ramp, for task similarity.  q is ordered -- where a trace sits in the
# sequence is what it says -- so it takes one hue in monotone lightness steps
# rather than categorical slots: steps 300 to 700 of the blue ramp, a hundred
# apart.  Fifty apart the ramp's neighbours differ by 0.047 in OKLCH lightness, under
# the 0.06 two ordinal steps need, so a hundred is the closest they may sit, and
# five is as many as fit above the lightest step that still clears 2:1 on the
# surface.  Validated as an ordinal ramp: monotone, adjacent dL >= 0.06, one hue,
# light end 2.44:1.  That is below 3:1, so these traces are labelled at their
# right-hand ends like every other.
Q_RAMP = ("#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")


def q_colours(qs):
    """One colour per value of q, light to dark as q rises, off the ordinal ramp.

    Fewer than five values spread over the whole ramp, so neighbours stay at least
    a hundred apart; a sixth would put two closer than the ramp validates, so it is
    refused rather than squeezed in.
    """
    qs = [float(q) for q in qs]
    if len(qs) > len(Q_RAMP):
        raise ValueError(
            f"{len(qs)} values of q and {len(Q_RAMP)} ordinal steps: a sixth step "
            f"would sit closer to its neighbour than lightness can tell apart.  "
            f"trim SIM_QS to five")
    if len(qs) == 1:
        return {qs[0]: Q_RAMP[len(Q_RAMP) // 2]}
    step = np.round(np.linspace(0, len(Q_RAMP) - 1, len(qs))).astype(int)
    rank = np.argsort(qs, kind="stable")
    return {qs[i]: Q_RAMP[step[r]] for r, i in enumerate(rank)}


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
# are paired: at a fixed index the readout v, the teachers Theta and the initial
# state W^0 are the same for every (d_f, rho), and the gates are nested subsets of
# one permutation.  A difference between two traces is then a difference between
# gates and not between draws, which is what makes the error bars below worth
# reading against each other.  The same holds across q: thetatilde and theta_0 are
# fixed by the index, and q only mixes them.

FLOW, TASKS, SPLIT, DENSITY, HEAT, DELTAW, TEST = range(7)
W0_PART = 1      # the second stream of a replicate: the initial state comes from it
THETA0_PART = 2  # the third: the shared teacher theta_0 of task similarity


def rng_for(cfg, stream, index=0, part=None):
    """The generator for replicate `index` of experiment `stream`.

    `part` names a further stream belonging to the same replicate: the initial
    state uses one (W0_PART) and the shared teacher another (THETA0_PART).
    Drawing each apart from everything `draw` draws means adding it moved none of
    those numbers -- a run at any W0_SCALE or q is paired seed for seed with the
    zero-started, dissimilar one, on the same v, gates and thetatilde -- and
    neither depends on the stream length.  Keys without a part are the ones every
    earlier run used.
    """
    key = (int(stream), int(index)) + (() if part is None else (int(part),))
    return np.random.default_rng(np.random.SeedSequence(cfg["ROOT_SEED"],
                                                        spawn_key=key))


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


def initial_state(N, D, scale, rng):
    """W^0, the state before the first task: iid N(0, scale^2 / N_in) entries.

    1/N_in is the variance a linear layer of fan-in N_in is initialised at, as v
    is at fan-in N by eq. (1).  Row i of W^0 has mean square norm scale^2, and task
    t reads out

        thetahat_t^(0) = v^T F_t W^0,     E||thetahat_t^(0)||^2 = scale^2 v^T F_t v

    before it has trained: about scale^2 d_f against the teacher's 1.  Two tasks
    read overlapping pools, so their baselines are correlated,
    E[thetahat_t^(0) . thetahat_u^(0)] = scale^2 v^T F_t F_u v, which is
    scale^2 d_f^2 ||v||^2 on average for t != u.  `mf_resid` carries both.

    scale = 0 returns exact zeros and draws nothing.
    """
    if scale == 0:
        return np.zeros((N, D))
    return rng.standard_normal((N, D)) * (float(scale) / np.sqrt(D))


def shared_teacher(D, rng):
    """theta_0, the direction similar tasks share: one unit vector per seed.

    Drawn exactly as a single teacher row is, so it and every thetatilde_t come
    from the same law and E||theta_t||^2 = q + (1 - q) = 1 whatever q is.
    """
    th = rng.standard_normal(D)
    return th / np.linalg.norm(th)


def similar_teachers(Theta, theta0, q):
    """theta_t = sqrt(q) theta_0 + sqrt(1 - q) thetatilde_t, one row per task.

    The rows of `Theta` are the thetatilde_t.  theta_0 and thetatilde_t are
    independent unit vectors with isotropic laws, so

        E||theta_t||^2 = 1,     E[theta_t . theta_u] = q   (t != u) :

    the similarity is the mean overlap of two teachers, and the scale every score
    is read on -- 1 is a state worth no more than W = 0 -- does not move.  A single
    row's norm is not exactly 1 once 0 < q < 1: ||theta_t||^2 = 1 +
    2 sqrt(q(1-q)) theta_0 . thetatilde_t, whose spread across tasks is
    2 sqrt(q(1-q)/N_in), 0.29 at q = 1/2.  q = 0 returns `Theta` itself, untouched.
    """
    q = float(q)
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"task similarity q={q} is outside [0, 1]")
    if q == 0.0:
        return Theta
    return np.sqrt(q) * theta0[None, :] + np.sqrt(1.0 - q) * Theta


def draw_seed(cfg, K, n_f, n_b, stream, index, q=0.0):
    """Everything one seed fixes: `draw`'s readout, gates and teachers, and W^0.

    Each comes from its own stream of the same replicate (`rng_for`), so v and the
    gates are exactly what `draw` alone gives at this index, W^0 is the same for
    every (d_f, rho) and every stream length of a figure, as v is, and so is the
    shared teacher theta_0.  At q = 0 the teachers are `draw`'s too and theta_0 is
    never drawn; at q > 0 they are `similar_teachers` of `draw`'s rows, which play
    thetatilde, so the same index gives every q the same thetatilde and theta_0.
    """
    N, D = cfg["N"], cfg["D"]
    v, F, B, Theta = draw(N, D, K, n_f, n_b, rng_for(cfg, stream, index))
    W0 = initial_state(N, D, cfg["W0_SCALE"], rng_for(cfg, stream, index, W0_PART))
    if q:
        theta0 = shared_teacher(D, rng_for(cfg, stream, index, THETA0_PART))
        Theta = similar_teachers(Theta, theta0, q)
    return v, F, B, Theta, W0


def initial_readout(v, F, W0):
    """thetahat_t^(0) = v^T F_t W^0 for every task: what W^0 reads out, one row each.

    This is the whole of W^0's effect on every metric in the file.  Summing eq.
    (16) from W^0 rather than 0 adds this row to every readout, so eq. (12) becomes

        r_t = theta_t - thetahat_t^(0) - sum_{s<t} Gamma_ts r_s ,

    and the solve eq. (27) takes Theta - thetahat^(0) as its right-hand side.  The
    band eq. (30) is unchanged, because theta_i - thetahat_i^(0) is exactly the row
    it cancels, and so is the step, and the displacement is sum_t a_t r_t / c_t
    as before -- W^0 moves them all only through R.
    """
    return (F * v[None, :]) @ W0


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

def run_stream(v, F, B, Theta, eval_states=(), W0=None, mag=None):
    """Walk the stream once, scoring it as it goes.

    The update is the endpoint of the within-task flow, eq. (16), so no integration
    is needed: the flow has the fixed left factor B_t v, the state moves on a line,
    and the endpoint is the line's end.  State W^s is the state after s tasks have
    trained, W^0 the initial state (zero when none is given), and task j (0-based)
    trains into W^{j+1}.  The walk forms the state itself, from W^0, and reads
    every task out of it with v^T F_t W; it never shifts the teachers, which is how
    the closed form takes W^0 in, so the two routes still share no arithmetic.

    Returns

        tr    (K,)  ||r_t|| for every task, eq. (12): the task read out of the
                    state just before it trains
        ret   dict  state s -> (1/s) sum_{i<=s} ||theta_i - v^T F_i W^s||, eq. (28)
                    averaged over every task the stream has trained so far
        dw    (K,)  ||W^t - W^{t-1}||_F, the size of the step eq. (16) took.  It is
                    measured from the increment the walk actually adds, not from
                    the identity ||r_t||/sqrt(c_t), so that the identity is
                    something the self-test can check rather than assume
        disp  dict  state s -> ||W^s - W^0||_F, how far the state has moved from
                    where it started once the steps have been allowed to cancel

    and, when a dict is handed in as `mag`, fills it with state s -> ||W^s||_F,
    the size of the weights themselves, at the same states.  It is an argument
    rather than a fifth return so that every caller before it is untouched.

    Forgetting needs the whole history read out of one state, which is a single
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
    W0 = (np.zeros((N, Theta.shape[1])) if W0 is None
          else np.asarray(W0, dtype=float))
    W = W0.copy()
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
            disp[s] = fro(W - W0)
            if mag is not None:
                mag[s] = fro(W)
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


def exact_scores(v, F, B, Theta, eval_states, retention=True, W0=None, mag=None):
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
    W^s - W^0 = sum_{t<=s} a_t r_t / c_t is one (N, s) x (s, D) product of the same
    rows.

    `W0` enters in one place, the right-hand side of eq. (27): the solve is run on
    Theta - thetahat^(0), the teachers less what W^0 already reads out
    (`initial_readout`), and everything after it is unchanged.

    `retention=False` skips the band eq. (30), which figures 6 and 7 do not read,
    and a collection of states reads it at those of `eval_states` alone: figure 6a
    of --sim wants forgetting at one state of a stream scored at four.

    A dict handed in as `mag` is filled with state s -> ||W^s||_F, the size of the
    weights: W^0 added back to the same sum, W^s = W^0 + sum_{t<=s} a_t r_t / c_t,
    so the one place W^0 enters the solve is joined by the one place it enters
    the state.
    """
    xp = _xp()
    c = B @ v ** 2
    G = coupling(v, F, B)
    R = solve_stream(G, Theta if W0 is None
                     else Theta - initial_readout(v, F, W0))
    Rn = rownorm(np.asarray(R))
    A = xp.asarray((B * v[None, :]) / c[:, None])         # rows a_t / c_t, eq. (16)
    if (retention is None or isinstance(retention, (bool, np.bool_))
            or np.isscalar(retention)):
        band = {int(s_) for s_ in eval_states} if retention else set()
    else:
        band = {int(s_) for s_ in retention}
    ret, disp = {}, {}
    for s_ in eval_states:
        s_ = int(s_)
        if s_ in band:
            ret[s_] = float(rownorm(np.asarray(-(xp.triu(G[:s_, :s_], 1)
                                                 @ R[:s_]))).mean())
        Ws = np.asarray(A[:s_].T @ R[:s_])             # W^s - W^0, (16) summed
        disp[s_] = fro(Ws)
        if mag is not None:
            mag[s_] = fro(Ws if W0 is None else Ws + W0)
    return Rn, ret, Rn / np.sqrt(c), disp


# ============================================== the mean field, Gamma -> d_f
# No figure draws these (since 2026-09-28): they are right over the first few tasks
# and wrong almost everywhere after.  They are kept, with the self-test's checks
# against their own recursion, for the note's derivations and for reference; the
# docstrings' "the measured curve" and "this curve" are the comparisons made when
# the figures still drew them.

def mf_resid(d_f, t, Delta, w0=0.0, q=0.0):
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
    tasks; Delta = -1 selects the interference arm; both may be arrays.

    `w0` is the scale of the initial state, W0_SCALE.  A non-zero W^0 adds its
    readout to every teacher: task t is driven by theta_t - thetahat_t^(0) (see
    `initial_readout`), and replacing Gamma by its mean leaves that drive as it is,
    the way it leaves theta_t.  What the drive brings is two second moments.  With
    E||v||^2 = 1,

        E[thetahat_t^(0) . thetahat_u^(0)] = w0^2 d_f  (t = u),   w0^2 d_f^2  (t != u)

    -- two tasks' read pools share d_f^2 N neurons on average -- so the drive is an
    independent part of power a = 1 + w0^2 d_f (1 - d_f) and a part every task
    shares, of power b = w0^2 d_f^2.  The independent part scales the lines above
    by a.  The shared part is summed by the recursion like a constant, S_t carries
    it as (1 - p^t)/d_f, and what reaches a residual decays like p^t:

        E||r_t||^2         = a [1 + d_f (1 - p^{2(t-1)}) / (2 - d_f)] + b p^{2(t-1)}
        E||r_t^(Delta)||^2 = a d_f [ (1 - p^{2 Delta})
                                     + (1 - p^Delta)^2 (1 - p^{2t}) ] / (2 - d_f)
                             + b p^{2t} (1 - p^Delta)^2

    Exact at t = 1 for every density, where it is E||theta_1 - thetahat_1^(0)||^2
    = 1 + w0^2 d_f, and at d_f = 1 for every t, where a = 1: every task reads the
    same v^T W^0, which the first one clears.  Where it fails is the independent
    part, and it fails for good.  The stream cancels task t's own readout of W^0
    through the overlap between its read pool and the pools written before it --
    the fluctuation of Gamma about d_f, which this discards -- so it forgets W^0
    over about N tasks as the writes overwrite it, and the mean field never does.
    Measured at N = 200, d_f = 0.5, rho = 1, w0 = 1, the part of E||r_t||^2 that
    W^0 is responsible for runs 0.51, 0.35, 0.15, 0.067 and 0.015 at t = 1, 10,
    100, 300 and 1000; here it is 0.50, then 0.33 for ever.  Past K ~ N the curve
    keeps a factor sqrt(a) the stream has shed: 1.12 at d_f = 0.5.

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
    r_t = theta_t - theta_{t-1} in the mean field and in the simulation alike, for
    every t > 1 and from any W^0 (r_1 = theta_1 - v^T W^0).  Any
    gap left there is the estimator -- these curves are geometric means over seeds
    while this is a root-mean-square -- and it is worth exp(-1/4D) to leading
    order: 1.38 against 1.41 at D = 12.

    `q` is the task similarity, and it enters exactly as W^0's shared part does.
    The teachers theta_t = sqrt(q) theta_0 + sqrt(1-q) thetatilde_t have
    E[theta_t . theta_u] = 1 (t = u) and q (t != u), so the drive gains a shared
    part of power q and its independent part loses the same:

        a = 1 - q + w0^2 d_f (1 - d_f),      b = q + w0^2 d_f^2 ,

    and the two lines above stand as written.  In the mean field the shared
    teacher is a constant drive, and a constant drive is cleared at the rate p^t:
    after 1/d_f tasks only the new part 1 - q is left.  The stream gets most of the
    way as fast and the rest far slower.  To read sqrt(q) theta_0 out of every
    random read pool the state needs v_i W_i equal on every neuron, and a write
    moves neuron i by the fraction v_i^2 / c_t of what it carries, so the neurons
    the readout barely weights are the last to get there.  What is left after the
    first 1/d_f tasks is that spread across the pool, of about
    sqrt(q) sqrt(2(1 - d_f)/n_f) -- the relative spread of v_i^2 is sqrt(2), and a
    pool of n_f samples it -- and the stream then clears it over thousands of
    tasks.  Measured at N = 200, d_f = d_b = 0.3, q = 1, from W^0 = 0: the r.m.s.
    ||r_t|| is 1, 0.48, 0.13 at t = 1, 3, 10, then 0.14, 0.07, 0.03 at t = 200,
    1000, 10^4, where this curve is 1, 0.49, 0.04 and then 1e-31 by t = 200.  So
    the curve runs below the stream by that remnant on top of the gap the
    variance of Gamma opens at q = 0, and at late times the stream's scores come
    back to sqrt(1 - q) times their q = 0 values: 0.98 and 0.44 against 1.38 at
    q = 0.5 and 0.9, K = 3000.  At d_f = 1 every pool is the whole network and
    there is no spread: r_t = sqrt(1-q) (thetatilde_t - thetatilde_{t-1}) exactly,
    of mean square 2(1 - q), and the mean field is exact there again.
    """
    p = 1.0 - d_f
    a, b = _drive_split(d_f, w0, q)
    t = np.asarray(t, dtype=float)
    Delta = np.asarray(Delta, dtype=float)
    lag = np.maximum(Delta, 0.0)
    second = np.where(
        Delta < 0,
        a * (1.0 + d_f * (1.0 - p ** (2.0 * (t - 1.0))) / (2.0 - d_f))
        + b * p ** (2.0 * (t - 1.0)),
        a * d_f * ((1.0 - p ** (2.0 * lag))
                   + (1.0 - p ** lag) ** 2 * (1.0 - p ** (2.0 * t))) / (2.0 - d_f)
        + b * p ** (2.0 * t) * (1.0 - p ** lag) ** 2)
    return np.sqrt(np.maximum(second, 0.0))


def _drive_split(d_f, w0, q=0.0):
    """The drive theta_t - thetahat_t^(0) as (independent power, shared power).

    Unit teachers of similarity q plus the readout of a W^0 at scale w0, with
    E||v||^2 = 1; the derivation is in `mf_resid`.  The teachers and W^0 are
    independent, so their powers add: theta_0 is shared by every task, power q,
    and W^0's readout is shared through the overlap of two read pools.  At
    w0 = 0, q = 0 this is (1, 0) and every mean-field curve is the zero-start,
    dissimilar one, to the last bit; at q = 0 it is the W^0 split alone, to the
    last bit.
    """
    s2 = float(w0) ** 2
    q = float(q)
    return 1.0 - q + s2 * d_f * (1.0 - d_f), q + s2 * d_f ** 2


def mf_retention(d_f, K, w0=0.0, q=0.0):
    """The mean field's forgetting score at state K: the same average, term by term.

    (1/K) sum_{i=1}^{K} of the root-mean-square residual of task i read out of
    state K, which is `mf_resid` at t = i and Delta = K - i.  The i = K term is
    zero -- the task just trained is solved -- so the average starts at 0 and
    climbs as the stream fills with older tasks.
    """
    K = int(K)
    i = np.arange(1, K + 1, dtype=float)
    return float(mf_resid(d_f, i, K - i, w0, q).mean())


def mf_step(d_f, d_b, K, w0=0.0, q=0.0):
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
    return float(np.asarray(mf_resid(d_f, t, -1, w0, q)).mean() / np.sqrt(d_b))


def mf_displacement(d_f, d_b, K, w0=0.0, q=0.0):
    """The mean field's ||W^K - W^0||_F: the same increments summed before the norm.

    W^K - W^0 = sum_t a_t r_t / c_t with a_t = B_t v, so

        ||W^K - W^0||_F^2 = sum_{t,s} (a_t . a_s)(r_t . r_s) / (c_t c_s) .

    For gates drawn independently each task E[a_t . a_t] = E[c_t] = d_b ||v||^2 and
    E[a_t . a_s] = d_b^2 ||v||^2 for t != s, so with E||v||^2 = 1 the diagonal
    carries a factor 1/d_b and every off-diagonal term carries 1.  Adding and
    subtracting the diagonal,

        E||W^K - W^0||_F^2 = (1/d_b - 1) sum_{t<=K} E||r_t||^2
                             + E||sum_{t<=K} r_t||^2 ,

    and the second term is free: the mean-field recursion has
    S_K = sum_{t<=K} r_t = sum_{s<=K} p^{K-s} theta_s with p = 1 - d_f, so
    E||S_K||^2 = (1 - p^{2K})/(1 - p^2) = (1 - p^{2K}) / (d_f (2 - d_f)).  From a
    non-zero W^0 the drive is theta_s - thetahat_s^(0), with the independent part a
    and the shared part of `mf_resid`; the sum weights the shared one by
    (1 - p^K)/d_f, so E||S_K||^2 = a (1 - p^{2K}) / (d_f (2 - d_f))
    + w0^2 (1 - p^K)^2.  The first term is `mf_resid` squared and summed.
    Task similarity adds its shared power q to b, and so q/d_f^2 (1 - p^K)^2 to
    E||S_K||^2: the state the mean field needs to read sqrt(q) theta_0 out of a
    pool of d_f of the network is sqrt(q) theta_0 / d_f, reached within 1/d_f
    tasks.  The stream reaches that scale as fast and then keeps growing, slowly:
    it goes on writing theta_0 into the neurons the readout barely weights, where
    a small readout needs a large weight.  At d_f = d_b = 0.3, q = 1, from W^0 = 0,
    ||W^T - W^0||_F is 4.0 at T = 10 and 7.7 at T = 10^4, where this is 3.9 and
    then 4.0 for good.

    The split enters once, through 1/d_b - 1, and the sum S_K does not feel it at
    all -- so the whole difference between this and `mf_step` is that cancellation
    is not something the write pool can undo.

    At d_f = d_b = 1 the first term vanishes and the second is exactly 1 + w0^2,
    and that is not an approximation: Gamma is all-ones there, so eq. (25) reads
    sum_{s<=t} r_s = theta_t - v^T W^0, W^t - W^0 = v (theta_t - v^T W^0)^T/||v||^2
    telescopes, and ||W^t - W^0||_F = ||theta_t - v^T W^0|| / ||v||, of mean square
    (1 + w0^2 ||v||^2) / ||v||^2 -- and 1/||v|| exactly, for every t, from W^0 = 0.
    The self-test asserts that line.

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
         the split lives in second moments.  The distance from this curve up
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

    From a non-zero W^0 there is a third miss.  The stream erases W^0: the part of
    it W^0 drives is linear and independent of the teachers, so it adds to the
    displacement in quadrature, and in the stable regime it climbs toward
    ||W^0||_F -- about w0 sqrt(N) -- as the writes overwrite the initial weights.
    A recursion on residual sums sees W^0 only through what it reads out, carries
    the independent part of that readout for ever (see `mf_resid`), and so grows
    this curve like sqrt(K) where the true erasure saturates.
    """
    K = int(K)
    p = 1.0 - d_f
    a, _ = _drive_split(d_f, w0, q)
    t = np.arange(1, K + 1, dtype=float)
    diag = float((np.asarray(mf_resid(d_f, t, -1, w0, q)) ** 2).sum())
    # the shared power b is w0^2 d_f^2 + q, and the sum weights it by
    # ((1 - p^K)/d_f)^2; written as w0^2 + q/d_f^2 so that q = 0 is the old line
    shared = float(w0) ** 2 + (float(q) / d_f ** 2 if q else 0.0)
    tail = (a * (1.0 - p ** (2.0 * K)) / (d_f * (2.0 - d_f))
            + shared * (1.0 - p ** K) ** 2)
    return float(np.sqrt(max((1.0 / d_b - 1.0) * diag + tail, 0.0)))


def mf_step_at(d_f, d_b, t, w0=0.0, q=0.0):
    """The mean field's step on task t alone: `mf_step` before the average over t.

    `mf_step` is the step averaged over the stream, as figures 6 and 7 plot it;
    this is the step the current task actually takes, as figure 8 plots it.
    """
    return np.asarray(mf_resid(d_f, t, -1, w0, q), dtype=float) / np.sqrt(d_b)


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
    two traces needs.  The average *within* a seed -- forgetting's sum over tasks,
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


def _w0_tag(cfg):
    """The initial state, as the figure titles name it."""
    s = float(cfg["W0_SCALE"])
    if s == 0.0:
        return "$W^0=0$"
    return f"$W^0\\sim\\mathcal{{N}}(0,{s * s:g}/N_{{in}})$"


def _baseline(ax, note=True):
    """The line at ||r|| = 1: the state is worth exactly what W = 0 is worth."""
    ax.axhline(1.0, color=AXIS, linewidth=1.0, zorder=1)
    if note:
        ax.annotate("$1$ = no better than $W=0$", xy=(0.0, 1.0),
                    xycoords=("axes fraction", "data"), xytext=(4, 3),
                    textcoords="offset points", ha="left", fontsize=7.5,
                    color=MUTED)


ARMS = (("ret", r"forgetting:   $\frac{1}{K}\sum_{i\leq K}\|r_i^{(K)}\|$",
         "every task trained so far, read out of the current state"),
        ("tr", r"interference:   $\|r_K\|$",
         "the arriving task, read out of the state just before it trains"))

# figure 8's three.  The first two have the shapes of ARMS -- one quantity
# belonging to the arriving task, one to the state it leaves behind -- read off the
# weights instead of the residual.  Both are the same norm, one of an increment and
# one of the sum of them, so the middle panel over the left is how many steps long
# the state is.  The third is the size of the weights themselves, which has a value
# before any task has trained, ||W^0||_F, and so is drawn from K = 0
# (`_task_axis`).  ||W^K||^2 = ||W^0||^2 + ||W^K - W^0||^2 + 2 <W^0, W^K - W^0>,
# and the last term is how much of W^0 the writes cancel.
ARMS_W = (("dw", r"$\|W^K-W^{K-1}\|_F$",
           "update magnitude: the step the arriving task took, eq. (16)"),
          ("disp", r"$\|W^K-W^0\|_F$",
           "weight displacement: how far the state has moved from $W^0$"),
          ("mag", r"$\|W^K\|_F$",
           "weight magnitude: the size of the weights themselves"))

# the five together, in the order --sim figure 3a draws them as columns, with the
# short name each column goes by and the metric alone, without the name ARMS puts
# in front of it
ARMS_ALL = ARMS + ARMS_W
ARM_NAME = {"ret": "forgetting", "tr": "interference", "dw": "update magnitude",
            "disp": "weight displacement", "mag": "weight magnitude"}
ARM_FORMULA = {"ret": r"$\frac{1}{K}\sum_{i\leq K}\|r_i^{(K)}\|$",
               "tr": r"$\|r_K\|$", "dw": r"$\|W^K-W^{K-1}\|_F$",
               "disp": r"$\|W^K-W^0\|_F$", "mag": r"$\|W^K\|_F$"}


def _task_axis(ax, K):
    """A task axis that has room for K = 0, the state before any task trained.

    The magnitude of the weights is the one metric with a value at K = 0, ||W^0||,
    and a logarithmic axis has no place for it.  A symmetric-log axis linear below
    one task keeps the decades where they were and puts 0 a short, linear step to
    the left of 1, so the trace can start where the stream did.  Every panel of a
    figure that draws the magnitude gets this axis, so the tasks line up.
    """
    ax.set_xscale("symlog", linthresh=1.0, linscale=0.45)
    ax.set_xlim(-0.12, K * 1.6)


def _mag_series(res, arm):
    """The arm's abscissa and its arrays, with the magnitude's K = 0 point in front.

    Returns (x, sim, sem, exact) for fig_vs_tasks and fig_sim_rows.  For every
    other arm it is the case's own arrays.  For the magnitude, K = 0 is W^0, the
    same state for both routes, and a seed's W^0 is the same for every case at its
    index; at W0_SCALE = 0 it is zero and has no place on a log axis, so the trace
    then starts at K = 1 like the rest.
    """
    x = res["x"]
    if arm != "mag" or not (np.isfinite(res["mag0"]) and res["mag0"] > 0):
        return x, res[arm], res[arm + "_sem"], res[arm + "_x"]
    return (np.r_[0.0, x], np.r_[res["mag0"], res["mag"]],
            np.r_[res["mag0_sem"], res["mag_sem"]],
            np.r_[res["mag0"], res["mag_x"]])


# ==================================================================== figure 1

def flow_case(cfg, q=0.0, K=None, show=None):
    """One stream, three ways: integrated, closed form, closed form from (27).

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

    All three start from the seed's W^0.  The integration and the protocol form
    the state from it; the solve sees it only as the shifted teachers
    Theta - thetahat^(0), so the first box is ||theta_1 - thetahat_1^(0)||^2 / 2.

    `q` is the task similarity of the teachers, `K` the stream length (K_FLOW when
    not given) and `show` how many tasks at each end have their windows recorded:
    FLOW_SHOW for figure 1, SIM_FLOW_SHOW for its --sim rows, and None records
    every task, as the self-test's short streams do.  A task that is not recorded is
    still integrated, through the same window of DECAY/c_t, but at the coarser
    step MAX_RATE_STEP/c_t.  RK4 misses e^-z by z^5/120 a step, so over the
    DECAY/z steps of a window its error is DECAY z^4/120 of what is left at the
    end, and what is left is e^-DECAY of where it started: at the defaults a
    relative 2e-5 of 2e-9, so the state such a task hands on moves by about 3e-14
    of its residual -- far below the e^-DECAY every window leaves behind and the
    recorded tasks compare against.  The self-test measures it.  The returned
    arrays hold the recorded tasks only, in order, with their indices in `shown`;
    the residuals R_int, R_stream and R_direct keep every task.
    """
    N, D = cfg["N"], cfg["D"]
    K = cfg["K_FLOW"] if K is None else int(K)
    n_f, n_b = counts(N, cfg["DF_FLOW"], cfg["RHO_FLOW"])
    v, F, B, Theta, W0 = draw_seed(cfg, K, n_f, n_b, FLOW, cfg["FLOW_SEED"], q=q)
    c = B @ v ** 2
    per, decay = cfg["PER_TASK"], cfg["DECAY"]
    if show is None:
        shown = np.arange(K)
    else:
        m = min(int(show), K)
        shown = np.unique(np.r_[np.arange(m), np.arange(K - m, K)])
    recorded = np.zeros(K, dtype=bool)
    recorded[shown] = True
    span = decay / c
    start = np.concatenate([[0.0], np.cumsum(span)[:-1]])
    tau = span[:, None] * np.linspace(0.0, 1.0, per)[None, :]

    # route 1: the protocol, eq. (16), from W^0
    W = W0.copy()
    R_stream = np.empty((K, D))
    for t in range(K):
        r = Theta[t] - (F[t] * v) @ W
        R_stream[t] = r
        W += np.outer(B[t] * v, r) / c[t]

    # route 2: the triangular solve, eq. (27) -- no training at all, and W^0 only
    # through what it reads out
    R_direct = solve_stream(coupling(v, F, B), Theta - initial_readout(v, F, W0))

    def closed(R):
        return (0.5 * (R[shown] ** 2).sum(1)[:, None]
                * np.exp(-2.0 * c[shown, None] * tau[shown]))

    # route 3: integrate the flow from W^0 and be told nothing
    W = W0.copy()
    loss_int = np.empty((shown.size, per))
    R_int = np.empty((K, D))
    dt_rec = tau[:, 1] - tau[:, 0]
    row = 0
    for t in range(K):
        Ft, Bt, th = F[t] * v, B[t] * v, Theta[t]

        def rhs(state):
            return np.outer(Bt, th - Ft @ state)

        if not recorded[t]:
            # the same window at the coarse step: nothing is read inside it
            R_int[t] = th - Ft @ W
            n = int(np.ceil(decay / cfg["MAX_RATE_STEP"]))
            h = span[t] / n
            for _ in range(n):
                k1 = rhs(W)
                k2 = rhs(W + 0.5 * h * k1)
                k3 = rhs(W + 0.5 * h * k2)
                k4 = rhs(W + h * k3)
                W += (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            continue
        nsub = max(cfg["SUBSTEPS"],
                   int(np.ceil(c[t] * dt_rec[t] / cfg["MAX_RATE_STEP"])))
        h = dt_rec[t] / nsub

        for p in range(per):
            e = th - Ft @ W
            loss_int[row, p] = 0.5 * (e ** 2).sum()
            if p == 0:
                R_int[t] = e
            if p + 1 < per:
                for _ in range(nsub):
                    k1 = rhs(W)
                    k2 = rhs(W + 0.5 * h * k1)
                    k3 = rhs(W + 0.5 * h * k2)
                    k4 = rhs(W + h * k3)
                    W += (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        row += 1

    loss_stream, loss_direct = closed(R_stream), closed(R_direct)
    return dict(
        time=start[shown, None] + tau[shown], K=K, n_f=n_f, n_b=n_b, q=float(q),
        shown=shown, R_int=R_int, R_stream=R_stream, R_direct=R_direct,
        loss_int=loss_int, loss_stream=loss_stream, loss_direct=loss_direct,
        box=0.5 * (R_direct[shown] ** 2).sum(1),
        flow_gap=float(np.abs(loss_int - loss_stream).max()
                       / loss_stream[:, :1].max()),
        box_gap=float(np.abs(R_int - R_direct).max() / np.abs(R_direct).max()),
        solve_gap=float(np.abs(R_stream - R_direct).max() / np.abs(R_direct).max()),
    )


def _flow_end_gaps(res, idx):
    """The three agreements of figure 1, worst task of `idx`, each task to itself.

    Returns (integrated residual vs eq. (27), protocol residual vs eq. (27),
    integrated loss vs the closed form over the window), each relative to that
    task's own residual or height, so a task late in the stream is held to the
    same standard as the first one whatever the stream has grown to.
    """
    Ri, Rd, Rs = res["R_int"][idx], res["R_direct"][idx], res["R_stream"][idx]
    nd = rownorm(Rd)
    rows = np.searchsorted(res["shown"], idx)
    li, lc = res["loss_int"][rows], res["loss_stream"][rows]
    return (float(np.max(rownorm(Ri - Rd) / nd)),
            float(np.max(rownorm(Rs - Rd) / nd)),
            float(np.max(np.abs(li - lc).max(1) / lc[:, 0])))


def fig_flow(cfg, flows, path, show, per_q=False):
    """Figure 1: the stream as it actually runs, its first and last `show` tasks.

    One row per entry of `flows`, each a `flow_case` of one stream of K tasks,
    integrated through every one of them from the seed's W^0 and recorded at the
    two ends.  The two panels are its first and its last `show` tasks, with a break
    in the task axis between: the left says the three routes agree where the
    stream starts, and the right that they still agree once it has run -- by then
    the integration has inherited every window's leftover, e^-DECAY of each
    residual, and the solve has been carried through the whole triangular system.
    A stream of no more than 2 show tasks is drawn whole across the two panels,
    with no break.

    The original set draws one row, K_FLOW tasks long with FLOW_SHOW at each end.
    --sim draws one row per task similarity (`per_q`, through `fig_sim_flow`), all
    on the same seed, so the rows differ in q and in nothing else.  Returns
    [(q, (first, last)), ...], each end's `_flow_end_gaps` or None, for the log.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    nrow = len(flows)
    show = int(show)
    K = flows[0]["K"]
    # the original figure's one row gets a taller panel; --sim keeps its own rows,
    # however many values of q it has
    row_h = 2.3 if per_q or nrow > 1 else 3.0
    h = 1.9 + row_h * nrow
    fig, axes = plt.subplots(nrow, 2, figsize=(13.0, h), squeeze=False,
                             sharey="row")
    fig.subplots_adjust(left=0.085, right=0.985, top=1 - 1.05 / h,
                        bottom=1.05 / h, hspace=0.42, wspace=0.035)
    lines = []
    for ri, res in enumerate(flows):
        shown = res["shown"]
        # the two ends never overlap: a stream of no more than 2 show tasks is drawn
        # whole, its second panel taking whatever the first leaves
        ends = (shown[shown < min(show, K)], shown[shown >= max(K - show, show)])
        raw = res["time"]
        # equal slots per task: the windows are DECAY time constants of their own
        # task and a thin write gate makes those vary, so raw flow time would give
        # one slow task half the axis.  Within a slot the coordinate is the
        # fraction of that window, so every tooth descends at the same slope and
        # the picture compares heights.  Task t (from 0) fills [t, t + 1)
        slot = shown[:, None] + (raw - raw[:, :1]) / (raw[:, -1:] - raw[:, :1])
        lo, hi = res["box"].min() * 1e-15, res["box"].max() * 10.0
        gaps = []
        for ci, idx in enumerate(ends):
            ax = axes[ri][ci]
            if not idx.size:
                ax.set_visible(False)
                gaps.append(None)
                continue
            rows = np.searchsorted(shown, idx)
            for j in rows:
                ax.semilogy(slot[j], res["loss_int"][j], color=MUTED, linewidth=3.2,
                            alpha=0.5, solid_capstyle="round")
                ax.semilogy(slot[j], res["loss_stream"][j], color=SERIES[0],
                            linewidth=1.3)
                ax.semilogy(slot[j], res["loss_direct"][j], color=SERIES[1],
                            linewidth=1.0, linestyle=(0, (4, 2)))
            ax.semilogy(idx.astype(float), res["box"][rows], linestyle="none",
                        marker="s", markersize=8, markerfacecolor="none",
                        markeredgecolor=SERIES[1], markeredgewidth=1.4)
            ax.set_ylim(lo, hi)
            ax.set_xlim(idx[0] - 0.55, idx[-1] + 1.25)
            ax.set_xticks(idx.astype(float))
            ax.set_xticklabels([str(int(t) + 1) for t in idx])
            _chrome(ax)
            g = _flow_end_gaps(res, idx)
            gaps.append(g)
            ax.set_title(f"tasks {int(idx[0]) + 1}-{int(idx[-1]) + 1}:   integrated "
                         f"vs eq. (27) {g[0]:.1e}   protocol vs eq. (27) "
                         f"{g[1]:.1e}", fontsize=8, color=MUTED, pad=4, loc="left")
        left, right = axes[ri]
        loss = r"$\frac{1}{2}\|\hat\theta_t-\theta_t\|^2$"
        left.set_ylabel((f"$q={res['q']:g}$\n\n" if per_q else "") + loss,
                        fontsize=9.5, color=INK2)
        # the break: the right panel shares the left one's vertical axis and
        # carries no spine of its own, and two slashes mark the cut in the task axis
        right.spines["left"].set_visible(False)
        right.tick_params(axis="y", left=False, labelleft=False)
        cut = dict(marker=[(-1, -2.4), (1, 2.4)], markersize=9, linestyle="none",
                   color=INK2, markeredgecolor=INK2, markeredgewidth=1.0,
                   clip_on=False, zorder=10)
        if K > 2 * show:
            left.plot([1.0], [0.0], transform=left.transAxes, **cut)
            right.plot([0.0], [0.0], transform=right.transAxes, **cut)
        lines.append((res["q"], gaps))

    handles = [Line2D([], [], color=MUTED, lw=3.2, alpha=0.5,
                      label="gradient flow, integrated"),
               Line2D([], [], color=SERIES[0], lw=1.3,
                      label="closed form, stream residuals (16)"),
               Line2D([], [], color=SERIES[1], lw=1.0, ls=(0, (4, 2)),
                      label="closed form, direct residuals (27)"),
               Line2D([], [], ls="none", marker="s", markersize=8,
                      markerfacecolor="none", markeredgecolor=SERIES[1],
                      markeredgewidth=1.4, label=r"direct solve, $\frac{1}{2}\|r_t\|^2$")]
    fig.legend(handles=handles, frameon=False, fontsize=9, ncol=4,
               loc="upper center", bbox_to_anchor=(0.5, 1 - 0.42 / h),
               labelcolor=INK2)
    fig.suptitle("the stream as it actually runs"
                 + (", one row per task similarity" if per_q else "") + "   "
                 f"($N={cfg['N']}$, $N_{{in}}={cfg['D']}$, $n_f={flows[0]['n_f']}$, "
                 f"$n_b={flows[0]['n_b']}$, {_w0_tag(cfg)}, $K={K}$, "
                 f"seed {cfg['FLOW_SEED']})", fontsize=11, color=INK,
                 y=1 - 0.08 / h)
    mid = K - 2 * show
    decay, z = float(cfg["DECAY"]), float(cfg["MAX_RATE_STEP"])
    # RK4 at c_t h = z misses e^(-z) by z^5/120 a step; over the DECAY/z steps of a
    # window that is DECAY z^4/120 of what is left, and what is left is e^-DECAY
    handed = decay * z ** 4 / 120.0 * np.exp(-decay)
    window = (f"task, and the fraction of its window (${decay:g}/c_t$ of flow time "
              "each)")
    same = ("every row is the same seed -- the same $\\tilde\\theta_t$, "
            "$\\theta_0$, readout, gates and $W^0$ -- so the rows differ in $q$ "
            "alone.  " if per_q else "")
    if mid > 0:
        foot = (f"{window}: the first {show} and the last {show} of one stream of "
                f"{K}, with the {mid} between integrated and not drawn.\n{same}"
                "the tasks between are integrated through the same"
                + ("\n" if same else " ")
                + f"windows at the step {z:g}$/c_t$, which moves what they hand on "
                f"by about {handed:.0e} of it, far below the $e^{{-{decay:g}}}$ = "
                f"{np.exp(-decay):.0e} every window leaves"
                + (".  " if same else ".\n")
                + "each panel names its worst task, relative to its own residual.")
    else:
        foot = (f"{window}: all {K} tasks of one stream.\n{same}each panel names "
                "its worst task, relative to its own residual.")
    fig.text(0.5, 0.012, foot, ha="center", va="bottom", fontsize=8, color=MUTED,
             linespacing=1.5)
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)
    return lines


# ============================================================ figures 2 and 3

def case_vs_tasks(cfg, d_f, rho=1.0, stream=TASKS, q=0.0):
    """One (d_f, rho, q), scored at logarithmically spaced states, by both routes.

    Every arm is read at the plotted point and nowhere else.  Forgetting and the
    displacement belong to state K and are read out of it; interference and the
    step belong to task K, the arrival that trained into that state, and are that
    one task's own numbers -- nothing is pooled over the tasks around it.  The only
    average is over seeds, taken geometrically with the standard error of that mean
    alongside, and the walk and the closed form see the same seeds.

    The size of the weights, "mag" = ||W^K||_F, belongs to state K as the
    displacement does, and it has a state before any task, W^0: "mag0" is
    ||W^0||_F, the point every magnitude trace starts from at K = 0.  Its route gap
    is kept apart, as `gap_mag`, so that `gap` is what the figures that do not draw
    the magnitude always reported.
    """
    N, D, K = cfg["N"], cfg["D"], cfg["K_TASKS"]
    n_f, n_b = counts(N, d_f, rho)
    pts = log_points(K, cfg["N_POINTS_DECADE"])
    shape = (cfg["SEEDS"], pts.size)
    raw = {k: np.full(shape, np.nan) for k in
           ("ret", "tr", "dw", "disp", "ret_x", "tr_x", "dw_x", "disp_x",
            "mag", "mag_x")}
    mag0 = np.full(cfg["SEEDS"], np.nan)
    gap = gap_mag = 0.0

    def at(d):
        """A per-state quantity on the evaluated states, in order."""
        return np.array([d[int(k)] for k in pts])

    for s in range(cfg["SEEDS"]):
        v, F, B, Theta, W0 = draw_seed(cfg, K, n_f, n_b, stream, s, q=q)
        mag, mag_x = {}, {}
        tr, ret, dw, disp = run_stream(v, F, B, Theta, pts, W0=W0, mag=mag)
        tr_x, ret_x, dw_x, disp_x = exact_scores(v, F, B, Theta, pts, W0=W0,
                                                 mag=mag_x)
        gap = max(gap, _route_gap(tr, tr_x), _route_gap(dw, dw_x),
                  _route_gap(at(ret), at(ret_x)),
                  _route_gap(at(disp), at(disp_x)))
        gap_mag = max(gap_mag, _route_gap(at(mag), at(mag_x)))
        raw["ret"][s], raw["ret_x"][s] = at(ret), at(ret_x)
        raw["disp"][s], raw["disp_x"][s] = at(disp), at(disp_x)
        raw["mag"][s], raw["mag_x"][s] = at(mag), at(mag_x)
        mag0[s] = fro(W0)
        # task K is row K - 1: it is read out of state K - 1 and trains into state K
        raw["tr"][s], raw["tr_x"][s] = tr[pts - 1], tr_x[pts - 1]
        raw["dw"][s], raw["dw_x"][s] = dw[pts - 1], dw_x[pts - 1]
    out = {}
    for k, a in raw.items():
        agg = [geo_sem(a[:, q]) for q in range(pts.size)]
        out[k] = np.array([g[0] for g in agg])
        out[k + "_sem"] = np.array([g[1] for g in agg])
        out[k + "_n"] = np.array([g[2] for g in agg])
    out["mag0"], out["mag0_sem"], out["mag0_n"] = geo_sem(mag0)
    out["x"], out["n_b"], out["gap"] = pts.astype(float), n_b, gap
    out["gap_mag"] = gap_mag
    out["d_f"], out["q"] = d_f, float(q)
    return out


def _route_gap(a, b):
    """Largest relative disagreement between the two routes, ignoring exact zeros."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 0)
    if not m.any():
        return 0.0
    return float(np.max(np.abs(a[m] - b[m]) / np.abs(b[m])))


def _two(ax, x, sim, exact, colour, label=None, sem=None, err="band", marker=None):
    """One series, two ways: the simulation and the closed form.

    The walk is drawn broad and translucent and the closed form thin on top of it,
    because the two are the same numbers: what the reader should see is a dashed
    line riding inside a solid band, not two curves to compare.  No figure draws
    the mean field any more: Gamma -> d_f is right over the first few tasks and
    wrong almost everywhere after, so it misled more than it explained.

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
    return ln, ((float(x[ok][-1]), float(sim[ok][-1])) if ok.any() else None)


def _route_legend(colour=None, err=None):
    """The two line styles, and the error bar, named once per figure."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    c = colour or MUTED
    out = [Line2D([], [], color=c, lw=2.8, alpha=0.40, label="simulation"),
           Line2D([], [], color=c, lw=1.2, ls=(0, (5, 2)),
                  label="whole stream, eq. (27) & (30)")]
    if err == "band":
        out.append(Patch(facecolor=c, alpha=0.16, edgecolor="none",
                         label="$\\pm1$ s.e.m. over seeds"))
    elif err == "bars":
        out.append(Line2D([], [], color=c, lw=1.0, marker="_", markersize=7,
                          label="$\\pm1$ s.e.m. over seeds"))
    return out


def _slots(n, what="traces"):
    """The first n categorical slots, refusing a fifth rather than cycling."""
    if n > len(SERIES):
        raise ValueError(
            f"{n} {what} and {len(SERIES)} categorical slots: the slots are "
            f"assigned in a fixed order and never cycled, so a fifth trace would "
            f"repeat a colour.  split the figure instead")
    return SERIES[:n]


def density_traces(cfg, cases):
    """The `a` panels: one trace per density, d_f = d_b, as figure 2 always was.

    Each trace is (case key, legend label, end label, d_f, d_b, q, colour); q is 0
    here, and the colours are the categorical slots in order.
    """
    ds = [d for d in cfg["DENSITIES"] if d in cases]
    return [(d, f"$d_f=d_b={d}$", f"{d}", d, d, 0.0, c)
            for d, c in zip(ds, _slots(len(ds), "densities"))]


def split_traces(cfg, cases, d_f=None):
    """The `b` panels: one trace per split ratio at one density, as figure 3's rows.

    The realised write density comes from `counts`, not from d_f/rho: the counts are
    rounded onto the 1/N grid once and everything downstream uses the realised pair.
    """
    d_f = cfg["FIG2B_DF"] if d_f is None else d_f
    rhos = [rho for rho in cfg["SPLIT_RHOS"] if (d_f, rho) in cases]
    out = [((d_f, rho), f"$d_f/d_b={rho:g}$", f"{rho:g}",
            d_f, cases[(d_f, rho)]["n_b"] / cfg["N"], 0.0, c)
           for rho, c in zip(rhos, _slots(len(rhos), "split ratios"))]
    if not out:
        raise ValueError(
            f"no case at d_f={d_f}: FIG2B_DF has to be one of SPLIT_DENSITIES "
            f"{tuple(cfg['SPLIT_DENSITIES'])}, which is what the split sweep walks")
    return out


def fig_vs_tasks(cfg, cases, path, arms, traces, title, note, cap=None, ylog=False,
                 plain=False, split_legend=False):
    """One panel per arm against task index, one trace per entry of `traces`.

    One function for figures 2a, 2b, 8a and 8b, and for figures 2 and 8 of the
    task-similarity set.  `arms` is ARMS or ARMS_W -- the two scores, or the three
    weight metrics -- and `traces` is
    [(case_key, legend label, end label, d_f, d_b, q, colour), ...] in drawing
    order, so the caller decides whether the traces run over the density, over the
    split ratio at one density, or over q, and this function only draws them.

    `cap` is (floor, ceiling) for the panels that need one: past the threshold of
    eq. (40) a split trace runs to 1e68 by the end of the stream and an axis that
    followed it would flatten the others into a single line, so the axis stops and a
    trace that leaves the panel is labelled with the decade it reached.

    `plain` labels a short logarithmic axis in plain numbers (`_plain_log_ticks`),
    and `split_legend` puts the traces and the line styles on two legend rows,
    for the five traces of q; both are off for figures 2a, 2b, 8a and 8b.

    ARMS_W carries a third arm, the magnitude of the weights, which starts at
    K = 0 from W^0: when it is drawn every panel gets the task axis that has room
    for K = 0 (`_task_axis`), so the tasks line up.
    """
    import matplotlib.pyplot as plt

    if not traces:
        print(f"  no cases for {path.name}; skipped")
        return
    n = len(arms)
    has_mag = any(a[0] == "mag" for a in arms)
    # the two-panel figures keep the 5-inch layout they always had; a wider one
    # carries a longer note, so it grows by a line of footnote per line of note
    lines_ = 2 + (note.count("\n") + 1 if note else 0)
    h = 5.0 if n == 2 else 4.55 + 0.17 * lines_
    fig, axes = plt.subplots(1, n, figsize=(12.6 if n == 2 else 5.8 * n, h),
                             sharex=True)

    def from_top(y):
        """A height the 5-inch layout put at y, kept as far below the top."""
        return y if n == 2 else 1.0 - (1.0 - y) * 5.0 / h
    handles = []
    worst = max(max(cases[t[0]]["gap"], cases[t[0]]["gap_mag"] if has_mag else 0.0)
                for t in traces)
    for ci, (arm, ylab, sub) in enumerate(arms):
        ax = axes[ci]
        ends, res = [], None
        for k, (key, label, tag, d_f, d_b, q, colour) in enumerate(traces):
            res = cases[key]
            x, sim, sem, exact = _mag_series(res, arm)
            ln, end = _two(ax, x, sim, exact, colour, label if ci == 0 else None,
                           sem=sem, err="band")
            if ci == 0:
                handles.append(ln)
            if end:
                over = cap is not None and end[1] > cap[1]
                ends.append((end[0], min(end[1], cap[1]) if cap else end[1],
                             f"{tag} $\\to10^{{{int(np.log10(end[1]))}}}$"
                             if over else tag))
        if has_mag:
            _task_axis(ax, cfg["K_TASKS"])
        else:
            ax.set_xscale("log")
        if ylog or cap:
            ax.set_yscale("log")
        if cap:
            ax.set_ylim(*cap)
        if plain and (ylog or cap):
            _plain_log_ticks(ax, *ax.get_ylim())
        if arm in ("ret", "tr"):
            _baseline(ax)
        ax.set_title(sub, fontsize=10, color=INK, pad=8)
        ax.set_ylabel(ylab, fontsize=10, color=INK2)
        ax.set_xlabel("tasks trained, $K$", fontsize=9.5, color=INK2)
        _chrome(ax)
        _end_labels(ax, ends)

    top = from_top(0.9)
    routes = _route_legend(err="band")
    if not split_legend:
        fig.legend(handles=handles + routes, frameon=False,
                   fontsize=8.5, ncol=len(handles) + len(routes), loc="upper center",
                   bbox_to_anchor=(0.5, from_top(0.935)), labelcolor=INK2)
    else:
        # five traces on one row with the line styles would run off the figure:
        # the traces get a row of their own and the line styles the row beneath
        fig.legend(handles=handles, frameon=False, fontsize=8.5,
                   ncol=len(handles), loc="upper center",
                   bbox_to_anchor=(0.5, from_top(0.945)), labelcolor=INK2)
        fig.legend(handles=routes, frameon=False,
                   fontsize=8.5, ncol=len(routes), loc="upper center",
                   bbox_to_anchor=(0.5, from_top(0.9)), labelcolor=INK2)
        top = from_top(0.855)
    fig.suptitle(f"{title}   ($N={cfg['N']}$, $N_{{in}}={cfg['D']}$, "
                 f"$K={cfg['K_TASKS']}$, {_w0_tag(cfg)}, {cfg['SEEDS']} seeds)",
                 fontsize=11, color=INK, y=from_top(0.995))
    fig.text(0.5, 0.012,
             "the dashed line is the closed form on the same seeds -- the coupling "
             "eq. (19), one triangular solve eq. (27) and the band eq. (30), with no "
             "state ever formed.\nit rides inside the solid band because the two are "
             f"the same numbers; here they agree to {worst:.0e} relative, worst case "
             "over every point drawn." + (f"\n{note}" if note else ""),
             ha="center", va="bottom", fontsize=8, color=MUTED, linespacing=1.5)
    bottom = (0.105 + 0.022 * (note.count("\n") + 1 if note else 0) if n == 2
              else (0.12 + 0.172 * lines_) / h)
    fig.tight_layout(rect=(0, bottom, 1, top))
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


NOTE_MAG = ("the magnitude starts from $\\|W^0\\|_F$ at $K=0$, before any task has "
            "trained, on a task axis linear below one task to make room for it.")


def fig_vs_tasks_split(cfg, cases, path):
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
                ln, end = _two(ax, res["x"], res[arm], res[arm + "_x"], SERIES[k],
                               f"$d_f/d_b={rho:g}$" if ri + ci == 0 else None,
                               sem=res[arm + "_sem"], err="band")
                if ri + ci == 0:
                    handles.append(ln)
                if end:
                    last = end[1]
                    text = (f"{rho:g}" if last <= hi else
                            f"{rho:g} $\\to10^{{{int(np.log10(last))}}}$")
                    ends.append((end[0], min(last, hi), text))
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

    routes = _route_legend(err="band")
    fig.legend(handles=handles + routes, frameon=False,
               fontsize=8.5, ncol=len(handles) + len(routes), loc="upper center",
               bbox_to_anchor=(0.5, 0.96), labelcolor=INK2)
    fig.suptitle("the raw residual against task index, at three read densities   "
                 f"($N={cfg['N']}$, $N_{{in}}={cfg['D']}$, $K={cfg['K_TASKS']}$, "
                 f"{_w0_tag(cfg)}, {cfg['SEEDS']} seeds;  1 = no better than "
                 "$W=0$)",
                 fontsize=11, color=INK, y=0.995)
    fig.text(0.5, 0.008,
             "the vertical axis stops at $10^{6}$: past the threshold of eq. (40) the "
             "state grows like $\\exp(\\lambda K)$ and $d_f/d_b=5$ ends between "
             "$10^{61}$ and $10^{86}$, so a trace that leaves\nthe panel is labelled "
             "with the decade it reached.  the dashed closed form rides inside the "
             f"solid band, agreeing to {worst:.0e} relative at worst -- eighty decades "
             "of growth\ncost the triangular solve nothing.",
             ha="center", va="bottom", fontsize=8, color=MUTED, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.055, 1, 0.945))
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


# ==================================================================== figure 4

def case_vs_density(cfg, d_f, rho, q=0.0):
    """Both scores at the end of a stream of K_DENSITY, by both routes.

    Forgetting is the average over every task the stream trained, read out of the
    final state.  Interference is the last task alone, ||r_K||, read out of the
    state just before it trains; the only average is over seeds.
    """
    N, D, K = cfg["N"], cfg["D"], cfg["K_DENSITY"]
    n_f, n_b = counts(N, d_f, rho)
    acc = {k: [] for k in ("ret", "tr", "ret_x", "tr_x")}
    gap = 0.0
    for s in range(cfg["SEEDS"]):
        v, F, B, Theta, W0 = draw_seed(cfg, K, n_f, n_b, DENSITY, s, q=q)
        tr, ret, _, _ = run_stream(v, F, B, Theta, (K,), W0=W0)
        tr_x, ret_x, _, _ = exact_scores(v, F, B, Theta, (K,), W0=W0)
        gap = max(gap, _route_gap(tr, tr_x),
                  _route_gap([ret[K]], [ret_x[K]]))
        acc["ret"].append(ret[K])
        acc["ret_x"].append(ret_x[K])
        acc["tr"].append(tr[K - 1])
        acc["tr_x"].append(tr_x[K - 1])
    out = {}
    for k, x in acc.items():
        out[k], out[k + "_sem"], out[k + "_n"] = geo_sem(x)
    out["n_b"], out["K"], out["gap"] = n_b, K, gap
    out["d_f"], out["q"] = d_f, float(q)
    return out


def fig_vs_density(cfg, grid, path):
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
            ln, end = _two(ax, xs, sim, exa, SERIES[k],
                           f"$d_f/d_b={rho:g}$" if ci == 0 else None,
                           sem=sem, err="bars", marker="o")
            if ci == 0:
                handles.append(ln)
            if end:
                ends.append((end[0], end[1], f"{rho:g}"))
        ax.set_yscale("log")
        _baseline(ax)
        ax.set_title(sub, fontsize=10, color=INK, pad=8)
        ax.set_ylabel(ylab, fontsize=10, color=INK2)
        ax.set_xlabel("read density $d_f$", fontsize=9.5, color=INK2)
        _chrome(ax)
        _end_labels(ax, ends)

    routes = _route_legend(err="bars")
    fig.legend(handles=handles + routes, frameon=False,
               fontsize=8.5, ncol=len(handles) + len(routes), loc="upper center",
               bbox_to_anchor=(0.5, 0.935), labelcolor=INK2)
    fig.suptitle("the raw residual at the end of the stream, against read density   "
                 f"($N={cfg['N']}$, $N_{{in}}={cfg['D']}$, $K={K}$, "
                 f"{_w0_tag(cfg)}, {cfg['SEEDS']} seeds)", fontsize=11, color=INK,
                 y=0.995)
    fig.text(0.5, 0.012,
             "the dashed closed form rides inside the solid band, agreeing to "
             f"{worst:.0e} relative at worst.  at $d_f=1$ the read gate is the "
             "identity and $\\Gamma$ is exactly all-ones, so the split\ncannot touch "
             "the residual there: the three ratios meet.  interference is one task per "
             "seed, and scatters more than forgetting does.",
             ha="center", va="bottom", fontsize=8, color=MUTED, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.1, 1, 0.9))
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


# ==================================================================== figure 5

def case_heat(cfg, d_f, rho, q=0.0):
    """One run to K_HEAT at one cell, scored at the three snapshots.

    Forgetting at a snapshot is the average over every task trained by then, so
    every cell of the grid has one -- there is no lag that can reach back past the
    start of the stream and no cell to leave blank.  Interference at a snapshot is
    task k alone, read out of the state just before it trains; the only average is
    over seeds.
    """
    N, D, K = cfg["N"], cfg["D"], cfg["K_HEAT"]
    n_f, n_b = counts(N, d_f, rho)
    snaps = tuple(int(s) for s in cfg["SNAPSHOTS"])
    out = {k: {"ret": [], "tr": []} for k in snaps}
    for s in range(cfg["SEEDS"]):
        v, F, B, Theta, W0 = draw_seed(cfg, K, n_f, n_b, HEAT, s, q=q)
        tr, ret, _, _ = run_stream(v, F, B, Theta, snaps, W0=W0)
        for k in snaps:
            out[k]["ret"].append(ret[k])
            out[k]["tr"].append(tr[k - 1])
    return {k: {m: geo(x) for m, x in d.items()} for k, d in out.items()}, n_b


def fig_heatmaps(cfg, cells, path):
    """Two rows, forgetting and interference, at three snapshots of one run.

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
    rows = (("ret", r"forgetting,  $\log_{10}$"),
            ("tr", r"interference,  $\log_{10}$"))

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
                 f"{_w0_tag(cfg)}, {cfg['SEEDS']} seeds per cell)", fontsize=11,
                 color=INK, y=0.965)
    fig.text(0.5, 0.012,
             "forgetting at $k$ averages every task trained by then, so no cell is "
             "missing.  interference is task $k$ alone, read out of the state before it "
             "trains.\nthe scale stops at "
             f"$10^{{{ceil:g}}}$ and the arrow marks cells past it: at the split corner "
             "$\\|r\\|$ reaches $10^{31}$, which no scale resolves against a band of "
             "width one.\nthe unclipped numbers are in the .txt.",
             ha="center", va="bottom", fontsize=8, color=MUTED, linespacing=1.5)
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


# ============================================================ figures 6 and 7

DW_METRICS = {
    "step": (r"$\frac{1}{T}\sum_{t\leq T}\|W^t-W^{t-1}\|_F$",
             "the average step: how far the state moves on a task"),
    "disp": (r"$\|W^T-W^0\|_F$",
             "the net displacement: how far the state has moved from $W^0$"),
}


def case_delta_w(cfg, d_f, rho, exact=True, q=0.0, at=None):
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

    `at` is a state s, 1 <= s < K_DW, at which five metrics are read besides, all
    off the one state W^s (`_at_state`): forgetting over the s tasks it has
    trained, interference -- task s + 1 read out of it before that task trains --
    how far it has moved, ||W^s - W^0||_F, the step that made it,
    ||W^s - W^{s-1}||_F, and the size of the weights, ||W^s||_F.  Keys `ret@s`,
    `tr@s`, `disp@s`, `dw@s`, `mag@s` and their `_x` twins from the closed form,
    with `gap_at` their route gap, and `mag0`, ||W^0||_F before any task; --sim
    figure 6a reads them at the penultimate task, s = K_DW - 1.  None reads
    nothing more, and either way the snapshot keys and `gap` are what they always
    were.
    """
    N, D, K = cfg["N"], cfg["D"], cfg["K_DW"]
    n_f, n_b = counts(N, d_f, rho)
    snaps = tuple(int(t) for t in cfg["DW_SNAPSHOTS"])
    if max(snaps) > K:
        raise ValueError(f"snapshot {max(snaps)} is past the stream length {K}")
    if at is not None:
        at = int(at)
        if not 1 <= at < K:
            raise ValueError(
                f"state {at}: interference reads task {at + 1} out of it, so it "
                f"needs 1 <= s < K_DW = {K}")
    states = snaps + ((at,) if at is not None and at not in snaps else ())
    # every key up front, so that exact=False still leaves the _x twins, as nan
    acc = {f"{m}@{t}": [] for m in ("step", "disp", "step_x", "disp_x")
           for t in snaps}
    if at is not None:
        acc.update({f"{m}{x}@{at}": [] for x in ("", "_x") for m in AT_KEYS})
        acc["mag0"] = []
    gap = gap_at = 0.0
    for sd in range(cfg["SEEDS"]):
        v, F, B, Theta, W0 = draw_seed(cfg, K, n_f, n_b, DELTAW, sd, q=q)
        mag = {} if at is not None else None
        tr, ret, dw, disp = run_stream(v, F, B, Theta, states, W0=W0, mag=mag)
        cum = np.cumsum(dw)
        seed = {}
        for t in snaps:
            seed[f"step@{t}"] = cum[t - 1] / t
            seed[f"disp@{t}"] = disp[t]
        if at is not None:
            mine = _at_state(tr, ret, dw, disp, mag, at)
            seed.update(mine)
            seed["mag0"] = fro(W0)
        if exact:
            mag_x = {} if at is not None else None
            tr_x, ret_x, dw_x, disp_x = exact_scores(
                v, F, B, Theta, states,
                retention=(at,) if at is not None else False, W0=W0, mag=mag_x)
            cum_x = np.cumsum(dw_x)
            gap = max(gap, _route_gap(dw, dw_x),
                      _route_gap([disp[t] for t in snaps],
                                 [disp_x[t] for t in snaps]))
            for t in snaps:
                seed[f"step_x@{t}"] = cum_x[t - 1] / t
                seed[f"disp_x@{t}"] = disp_x[t]
            if at is not None:
                theirs = _at_state(tr_x, ret_x, dw_x, disp_x, mag_x, at)
                gap_at = max(gap_at, _route_gap(list(mine.values()),
                                                list(theirs.values())))
                seed.update({k.replace("@", "_x@"): x for k, x in theirs.items()})
        for k, x in seed.items():
            acc[k].append(x)
    out = {"n_f": n_f, "n_b": n_b, "gap": gap, "d_f": d_f, "q": float(q)}
    if at is not None:
        out["at"], out["gap_at"] = at, gap_at
    for k, x in acc.items():
        out[k], out[k + "_sem"], out[k + "_n"] = geo_sem(x)
    return out


AT_KEYS = ("ret", "tr", "disp", "dw", "mag")    # what `_at_state` reads off W^s


def _at_state(tr, ret, dw, disp, mag, s):
    """The five metrics off one state W^s, from one route's per-task arrays.

    `tr` and `dw` are per task, row t (from 0) the task that reads state t and
    trains into state t + 1; `ret`, `disp` and `mag` are per state.  So row s of
    `tr` is task s + 1 read out of W^s -- the interference that state hands the
    next task -- and row s - 1 of `dw` is task s's own step, the one that made W^s.
    """
    return {f"ret@{s}": ret[s], f"tr@{s}": tr[s], f"disp@{s}": disp[s],
            f"dw@{s}": dw[s - 1], f"mag@{s}": mag[s]}


def _dw_series(cells, metric, axis, held, T):
    """One trace: x, and the two routes with the error of the first.

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
            np.array([cells[k][f"{metric}_x@{int(T)}"] for k in keys]))


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

    ylab, sub = DW_METRICS[metric]
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
                    cells, metric, axis, held, T)
    pool = np.concatenate([np.concatenate([a[2], a[4]])
                           for a in series.values()])
    pool = pool[np.isfinite(pool) & (pool > 0)]
    hi_lim = min(float(pool.max()) * 2.0, float(cfg["DW_CEIL"]))
    lo_lim = max(float(pool.min()) / 2.0, hi_lim * 1e-12)

    fig, axes = plt.subplots(3, 3, figsize=(13.2, 12.4))
    # a wide gutter: where every trace of a panel is clipped, its end labels all
    # carry the decade they reached and the stack is wide enough to reach the next
    # panel's tick labels at the default spacing
    fig.subplots_adjust(left=0.085, right=0.955, top=0.905, bottom=0.19,
                        hspace=0.34, wspace=0.32)
    handles = [[], []]
    for ri, (axis, held_all, xlab, fmt) in enumerate(rows):
        for ci, T in enumerate(snaps):
            ax = axes[ri][ci]
            ends = []
            for hi, held in enumerate(held_all):
                if (ri, ci, hi) not in series:
                    continue
                h, xs, sim, sem, exa = series[(ri, ci, hi)]
                ln, end = _two(ax, xs, sim, exa, SERIES[hi],
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

    cax = fig.add_axes([0.345, 0.128, 0.31, 0.011])
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
    routes = _route_legend(err="bars")
    fig.legend(handles=routes, frameon=False, fontsize=8.5,
               ncol=len(routes), loc="upper center", bbox_to_anchor=(0.5, 0.968),
               labelcolor=INK2)
    fig.suptitle(f"{sub}   "
                 f"($N={cfg['N']}$, $N_{{in}}={cfg['D']}$, {_w0_tag(cfg)}, "
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


# ================================================= task similarity: --sim figures
# Every figure here is one of the eight above with q brought in, and each reuses
# that figure's snapshots, grid step and caps, so the two sets are read on the
# same terms.  The seeds come from the same named streams, so at q = 0 a case is,
# number for number, the original's at the same (d_f, rho) and stream length:
# figures 2, 3a and 8 are figures 2a and 8a (the TASKS stream, which is not
# figure 3's), figure 3b is figure 3 wherever its densities meet SPLIT_DENSITIES,
# and figures 4 to 7 are the original cells.  Figure 1's q = 0 row is the original
# figure 1, number for number, while SIM_K_FLOW and SIM_FLOW_SHOW equal K_FLOW and
# FLOW_SHOW, as they do by default: `draw` spends the stream's gates before its
# teachers, so two stream lengths see different teachers, and a task drawn is
# integrated at a finer step than one that is not.  Figure 6a is the one that is
# not a rework with q added: its columns are five metrics off one state, not three
# stream lengths.

def _plain_log_ticks(ax, lo, hi):
    """Label a short logarithmic axis with plain numbers rather than one decade.

    A log axis spanning a decade or two shows a single 10^k label, which leaves a
    reader nothing to read a value off; this labels 1, 2 and 5 of every decade
    (more steps under one decade) in plain figures.  Longer axes are left alone.
    """
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0 or hi / lo > 300.0:
        return
    subs = (1.0, 2.0, 5.0) if hi / lo > 10.0 else (1.0, 1.5, 2.0, 3.0, 5.0, 7.0)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=subs))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _q_grid(step, q_max):
    """0, step, 2 step, ... up to q_max inclusive: a q axis."""
    n = int(np.floor(q_max / step + 1e-9)) + 1
    return [round(i * step, 9) for i in range(n)]


def q_traces(cfg, cases, d_f, rho=1.0):
    """One trace per q at one (d_f, rho), on the ordinal ramp, for fig_vs_tasks.

    `cases` is keyed (d_f, rho, q); the write density in the trace is the realised
    n_b/N, as `split_traces` has it.
    """
    qs = [float(q) for q in cfg["SIM_QS"]]
    col = q_colours(qs)
    return [((d_f, rho, q), f"$q={q:g}$", f"{q:g}", d_f,
             cases[(d_f, rho, q)]["n_b"] / cfg["N"], q, col[q])
            for q in qs if (d_f, rho, q) in cases]


def fig_sim_flow(cfg, flows, path):
    """Figure 1 once per task similarity, one row each, over a long stream.

    Each row is one stream of SIM_K_FLOW tasks and its first and last SIM_FLOW_SHOW
    tasks are drawn, as `fig_flow` draws the original figure 1 from K_FLOW and
    FLOW_SHOW.  Every row is the same seed, so the rows differ in q and in nothing
    else.
    """
    return fig_flow(cfg, flows, path, cfg["SIM_FLOW_SHOW"], per_q=True)


def _log_span(values, floor, ceil):
    """A decade-aligned vertical range for a log axis, inside [floor, ceil]."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if not v.size:
        return floor, ceil, False
    lo = max(floor, 10.0 ** np.floor(np.log10(v.min())))
    hi = min(ceil, 10.0 ** np.ceil(np.log10(v.max())))
    if hi <= lo:
        hi = lo * 10.0
    return lo, hi, bool(v.max() > hi)


def _two_row_legend(fig, traces, y, err):
    """The traces on one row and the line styles beneath, above the panels."""
    fig.legend(handles=traces, frameon=False, fontsize=8.5, ncol=len(traces),
               loc="upper center", bbox_to_anchor=(0.5, y), labelcolor=INK2)
    routes = _route_legend(err=err)
    fig.legend(handles=routes, frameon=False, fontsize=8.5,
               ncol=len(routes), loc="upper center", bbox_to_anchor=(0.5, y - 0.03),
               labelcolor=INK2)


def fig_sim_rows(cfg, cases, path, rows, title, note, ylog=False, arms=ARMS):
    """Figure 3's layout with q as the trace: one row per density or split ratio.

    `rows` is [(row label, traces), ...], each trace as `q_traces` builds it, and
    `cases` is keyed (d_f, rho, q).

    The vertical axis is linear and shared by every panel when every trace stays
    near 1, as figure 2's is (3a: d_f = d_b is stable).  With `ylog` each row gets
    a logarithmic range of its own, capped at FIG3_FLOOR and FIG3_CEIL (3b: past
    the split threshold one row runs away and would flatten the rows that do
    not); a trace that leaves a panel is labelled with the decade it reached.

    `arms` are the columns: ARMS, the two scores, for 3b, and ARMS_ALL for 3a --
    the scores and then the three weight metrics, update magnitude, weight
    displacement and weight magnitude.  The scores keep the axis above; each
    weight column gets a logarithmic axis of its own, shared down the column so the
    rows compare, and the columns go by their short names (ARM_NAME).  The weight
    magnitude starts at K = 0, from W^0, so when it is drawn every panel takes the
    task axis that has room for it (`_task_axis`).
    """
    import matplotlib.pyplot as plt

    rows = [(lab, trs) for lab, trs in rows if trs]
    if not rows:
        print(f"  no cases for {path.name}; skipped")
        return
    every = [t for _, trs in rows for t in trs]
    ncol = len(arms)
    wide = ncol > 2
    has_mag = any(a[0] == "mag" for a in arms)
    worst = max(max(cases[t[0]]["gap"], cases[t[0]]["gap_mag"] if has_mag else 0.0)
                for t in every)

    def pooled(trs):
        # forgetting at K = 1 is zero to rounding, ~1e-16: it would set the floor
        v = np.concatenate([np.asarray(cases[t[0]][a], dtype=float) for t in trs
                            for a in ("ret", "tr")])
        return v[np.isfinite(v) & (v > 1e-12)]

    if ylog:
        spans = []
        for _, trs in rows:
            v = pooled(trs)
            lo_ = max(cfg["FIG3_FLOOR"], float(v.min()) / 1.5)
            hi_ = min(cfg["FIG3_CEIL"], float(v.max()) * 1.5)
            spans.append((lo_, hi_, bool(v.max() > hi_)))
    else:
        spans = [(0.0, float(pooled(every).max()) * 1.06, False)] * len(rows)
    clipped = any(sp[2] for sp in spans)
    # one logarithmic range per weight column, over every row: the traces, their
    # error bands and the closed form
    wspan = {}
    for arm in (a[0] for a in arms if a[0] in ("dw", "disp", "mag")):
        parts = []
        for t in every:
            _, sim, sem, exact = _mag_series(cases[t[0]], arm)
            e = np.where(np.isfinite(sem), sem, 0.0)
            parts += [sim * np.exp(e), sim * np.exp(-e), exact]
        v = np.concatenate(parts)
        v = v[np.isfinite(v) & (v > 0)]
        hi_ = min(float(v.max()) * 1.3, float(cfg["FIG3_CEIL"]))
        wspan[arm] = (max(float(v.min()) / 1.3, hi_ * 1e-12), hi_)
    nrow = len(rows)
    height = 2.25 + 2.8 * nrow
    width = 12.6 if not wide else 1.4 + 3.75 * ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(width, height), sharex=True,
                             squeeze=False)
    handles = []
    for ri, (rlab, traces) in enumerate(rows):
        for ci, (arm, ylab, sub) in enumerate(arms):
            ax = axes[ri][ci]
            weight = arm in wspan
            lo, hi = wspan[arm] if weight else spans[ri][:2]
            ends = []
            for key, label, tag, d_f, d_b, q, colour in traces:
                x, sim, sem, exact = _mag_series(cases[key], arm)
                ln, end = _two(ax, x, sim, exact, colour,
                               label if ri + ci == 0 else None,
                               sem=sem, err="band")
                if ri + ci == 0:
                    handles.append(ln)
                if end:
                    last = end[1]
                    text = (tag if last <= hi else
                            f"{tag} $\\to10^{{{int(np.log10(last))}}}$")
                    ends.append((end[0], min(last, hi), text))
            if has_mag:
                _task_axis(ax, cfg["K_TASKS"])
            else:
                ax.set_xscale("log")
            if ylog or weight:
                ax.set_yscale("log")
                _plain_log_ticks(ax, lo, hi)
            ax.set_ylim(lo, hi)
            if not weight:
                # named once in the wide layout, where interference crowds it
                _baseline(ax, note=(ri == 0 and (not wide or ci == 0)))
            if ri == 0:
                ax.set_title(ARM_NAME[arm] if wide else sub, fontsize=10,
                             color=INK, pad=8)
            if ri == nrow - 1:
                ax.set_xlabel("tasks trained, $K$", fontsize=9.5, color=INK2)
            yl = ARM_FORMULA[arm] if wide else ylab
            ax.set_ylabel(f"{rlab}\n\n{yl}" if ci == 0 else yl,
                          fontsize=9.5, color=INK2)
            _chrome(ax)
            _end_labels(ax, ends, gap=0.075)

    _two_row_legend(fig, handles, 1 - 0.42 / height, "band")
    fig.suptitle(f"{title}   ($N={cfg['N']}$, $N_{{in}}={cfg['D']}$, "
                 f"$K={cfg['K_TASKS']}$, {_w0_tag(cfg)}, {cfg['SEEDS']} seeds)",
                 fontsize=11, color=INK, y=1 - 0.06 / height)
    scale = ("each row has a logarithmic range of its own"
             + (f", stopping at {cfg['FIG3_CEIL']:.0e}: a trace that leaves a panel "
                "is labelled with the decade it reached" if clipped else "")
             + ".  " if ylog else "")
    if wide:
        scale += ("the two scores share one vertical axis and each weight column "
                  "has a logarithmic one of its own, shared down the column.  ")
    fig.text(0.5, 0.008,
             scale + "the dashed closed form rides inside the solid band, agreeing "
             f"to {worst:.0e} relative at worst." + (f"\n{note}" if note else ""),
             ha="center", va="bottom", fontsize=8, color=MUTED, linespacing=1.5)
    extra = note.count("\n") + 1 if note else 0
    fig.tight_layout(rect=(0, (0.38 + 0.17 * extra) / height, 1,
                           1 - 0.95 / height))
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


def fig_sim_vs_q(cfg, grid, path, traces, title, note):
    """Figure 4 with q on the abscissa: both scores at the end of the stream.

    `grid` is keyed (trace value, q) and `traces` is [(trace value, legend label,
    end label, colour, d_f, d_b), ...].
    """
    import matplotlib.pyplot as plt

    K = cfg["K_DENSITY"]
    qs = sorted({q for _, q in grid})
    worst = max(c["gap"] for c in grid.values())
    pool = np.array([grid[k][a] for k in grid for a in ("ret", "tr")])
    lo, hi, clipped = _log_span(pool, 1e-300, cfg["FIG3_CEIL"])
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), sharex=True)
    handles = []
    for ci, (arm, ylab, sub) in enumerate(ARMS):
        ax = axes[ci]
        ends = []
        for y, label, tag, colour, d_f, d_b in traces:
            xs = np.array([q for q in qs if (y, q) in grid])
            sim = np.array([grid[(y, q)][arm] for q in xs])
            exa = np.array([grid[(y, q)][arm + "_x"] for q in xs])
            sem = np.array([grid[(y, q)][arm + "_sem"] for q in xs])
            ln, end = _two(ax, xs, sim, exa, colour, label if ci == 0 else None,
                           sem=sem, err="bars", marker="o")
            if ci == 0:
                handles.append(ln)
            if end:
                last = end[1]
                text = (tag if not clipped or last <= hi else
                        f"{tag} $\\to10^{{{int(np.log10(last))}}}$")
                ends.append((end[0], min(last, hi), text))
        ax.set_yscale("log")
        if clipped:
            ax.set_ylim(top=hi)
        v = pool[np.isfinite(pool) & (pool > 0)]
        _plain_log_ticks(ax, float(v.min()) / 1.2, float(v.max()) * 1.2)
        _baseline(ax)
        ax.set_title(sub, fontsize=10, color=INK, pad=8)
        ax.set_ylabel(ylab, fontsize=10, color=INK2)
        ax.set_xlabel("task similarity  $q$", fontsize=9.5, color=INK2)
        _chrome(ax)
        _end_labels(ax, ends)

    routes = _route_legend(err="bars")
    fig.legend(handles=handles + routes, frameon=False,
               fontsize=8.5, ncol=len(handles) + len(routes), loc="upper center",
               bbox_to_anchor=(0.5, 0.935), labelcolor=INK2)
    fig.suptitle(f"{title}   ($N={cfg['N']}$, $N_{{in}}={cfg['D']}$, $K={K}$, "
                 f"{_w0_tag(cfg)}, {cfg['SEEDS']} seeds)", fontsize=11, color=INK,
                 y=0.995)
    cap = (f"the vertical axis stops at $10^{{{int(np.log10(hi))}}}$ and a trace "
           "that leaves it is labelled with the decade it reached.  "
           if clipped else "")
    fig.text(0.5, 0.012,
             cap + "the dashed closed form rides inside the solid band, agreeing to "
             f"{worst:.0e} relative at worst." + (f"\n{note}" if note else ""),
             ha="center", va="bottom", fontsize=8, color=MUTED, linespacing=1.5)
    extra = note.count("\n") + 1 if note else 0
    fig.tight_layout(rect=(0, 0.07 + 0.03 * extra, 1, 0.9))
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


def fig_sim_heatmaps(cfg, cells, path, ys, ylab, yfmt, title, note):
    """Figure 5 with q as the abscissa and a density or a split ratio as the rows.

    `cells` is keyed (y, q) -> case_heat's (snapshots, n_b).  The colour is
    log10||r|| on figure 5's diverging ramp -- ||r|| = 1, the state worth what
    W = 0 is worth, is the neutral grey -- with its ceiling, and both rows share
    one scale.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    qs = sorted({q for _, q in cells})
    ys = [float(y) for y in ys]
    snaps = list(cfg["SNAPSHOTS"])
    rows = (("ret", r"forgetting,  $\log_{10}$"),
            ("tr", r"interference,  $\log_{10}$"))
    planes = {}
    for met, _ in rows:
        for k in snaps:
            Z = np.full((len(ys), len(qs)), np.nan)
            for i, y in enumerate(ys):
                for j, q in enumerate(qs):
                    if (y, q) in cells:
                        val = cells[(y, q)][0][k][met]
                        if np.isfinite(val) and val > 0:
                            Z[i, j] = np.log10(val)
            planes[(met, k)] = Z

    cmap = diverging_cmap()
    cmap.set_bad((0.0, 0.0, 0.0, 0.0))
    vals = np.concatenate([planes[(m, k)][np.isfinite(planes[(m, k)])]
                           for m, _ in rows for k in snaps])
    ceil = cfg["HEAT_CEIL"]
    norm = TwoSlopeNorm(vmin=float(min(np.nanmin(vals), -0.05)), vcenter=0.0,
                        vmax=max(ceil, 0.05))
    clipped = bool(np.nanmax(vals) > ceil)
    dq = qs[1] - qs[0] if len(qs) > 1 else 0.1
    dy = ys[1] - ys[0] if len(ys) > 1 else 1.0

    fig, axes = plt.subplots(2, 3, figsize=(11.6, 7.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.115, right=0.865, top=0.865, bottom=0.145,
                        hspace=0.17, wspace=0.09)
    for ri, (met, rtitle) in enumerate(rows):
        for ci, k in enumerate(snaps):
            ax = axes[ri][ci]
            im = ax.imshow(np.clip(planes[(met, k)], None, ceil), origin="lower",
                           aspect="auto", cmap=cmap, norm=norm,
                           extent=(qs[0] - dq / 2, qs[-1] + dq / 2,
                                   ys[0] - dy / 2, ys[-1] + dy / 2))
            ax.set_xticks(qs)
            ax.set_xticklabels([f"{q:g}" for q in qs])
            ax.set_yticks(ys)
            ax.set_yticklabels([yfmt.format(y) for y in ys])
            if ri == 0:
                ax.set_title(f"$k={k}$ tasks", fontsize=10, color=INK, pad=6)
            if ri == 1:
                ax.set_xlabel("task similarity  $q$", fontsize=9.5, color=INK2)
            if ci == 0:
                ax.set_ylabel(f"{rtitle}\n\n{ylab}", fontsize=9.5, color=INK2)
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
    fig.suptitle(f"{title}   ($N={cfg['N']}$, $N_{{in}}={cfg['D']}$, "
                 f"$T={cfg['K_HEAT']}$, {_w0_tag(cfg)}, {cfg['SEEDS']} seeds per "
                 "cell)", fontsize=11, color=INK, y=0.965)
    fig.text(0.5, 0.012,
             "forgetting at $k$ averages every task trained by then, so no cell is "
             "missing.  interference is task $k$ alone, read out of the state "
             "before it trains.\n"
             + (f"the scale stops at $10^{{{ceil:g}}}$ and the arrow marks cells "
                "past it; the unclipped numbers are in the .txt.  " if clipped
                else "the unclipped numbers are in the .txt.  ")
             + note, ha="center", va="bottom", fontsize=8, color=MUTED,
             linespacing=1.5)
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


def fig_sim_delta_w(cfg, cells, path, metric, axis, qheat, title):
    """Figure 6 or 7 with q brought in: q against a density or a split ratio.

    `cells` is keyed (y, q) -> case_delta_w, and `axis` names y: a dict with
    `grid` (row 1's abscissa and row 3's rows), `traces` (row 2's traces, from
    SIM_DENSITIES or SIM_RHOS), `label` (its axis label), `legend` (a format for
    one value of it) and `tick` (a tick format).  `qheat` is the q axis of rows 2
    and 3, the SIM_HEAT_Q_STEP grid.  Row 1 runs over y with one trace
    per q, on the ordinal ramp; row 2 runs over q with one trace per value of
    `traces`, in the categorical slots; row 3 is the whole grid, q across and y up.
    As in figures 6 and 7, rows 1 and 2 are slices of what row 3 shows whole, and
    they share one vertical axis across the three columns, capped at DW_CEIL.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    ylab, sub = DW_METRICS[metric]
    snaps = [int(t) for t in cfg["DW_SNAPSHOTS"]]
    ygrid = [float(y) for y in axis["grid"]]
    qgrid = [float(q) for q in qheat]
    qs = [float(q) for q in cfg["SIM_QS"]]
    qcol = q_colours(qs)
    ytr = [float(y) for y in axis["traces"]]
    ycol = dict(zip(ytr, _slots(len(ytr), axis["label"])))

    def line(keys, xs, T):
        col = f"{metric}@{int(T)}"
        return (np.asarray(xs, dtype=float),
                np.array([cells[k][col] for k in keys]),
                np.array([cells[k][col + "_sem"] for k in keys]),
                np.array([cells[k][f"{metric}_x@{int(T)}"] for k in keys]))

    # every trace of rows 1 and 2 first, so the two rows can share one axis
    series = {}
    for ci, T in enumerate(snaps):
        for q in qs:
            ys_ = [y for y in ygrid if (y, q) in cells]
            if ys_:
                series[(0, ci, q)] = (qcol[q], f"$q={q:g}$", f"{q:g}") + line(
                    [(y, q) for y in ys_], ys_, T)
        for y in ytr:
            qs_ = [q for q in qgrid if (y, q) in cells]
            if qs_:
                series[(1, ci, y)] = (ycol[y], axis["legend"].format(y),
                                      f"{y:g}") + line([(y, q) for q in qs_], qs_, T)
    pool = np.concatenate([np.concatenate([a[4], a[6]])
                           for a in series.values()])
    pool = pool[np.isfinite(pool) & (pool > 0)]
    hi_lim = min(float(pool.max()) * 2.0, float(cfg["DW_CEIL"]))
    lo_lim = max(float(pool.min()) / 2.0, hi_lim * 1e-12)

    fig, axes = plt.subplots(3, 3, figsize=(13.2, 12.4))
    fig.subplots_adjust(left=0.085, right=0.955, top=0.885, bottom=0.19,
                        hspace=0.34, wspace=0.32)
    handles = [[], []]
    for ci, T in enumerate(snaps):
        for ri in (0, 1):
            ax = axes[ri][ci]
            ends = []
            for key, (colour, legend, tag, xs, sim, sem, exa) in series.items():
                if key[0] != ri or key[1] != ci:
                    continue
                ln, end = _two(ax, xs, sim, exa, colour,
                               legend if ci == 0 else None,
                               sem=sem, err="bars", marker="o")
                if ci == 0:
                    handles[ri].append(ln)
                if end:
                    last = end[1]
                    text = (tag if last <= hi_lim else
                            f"{tag} $\\to10^{{{int(np.log10(last))}}}$")
                    ends.append((end[0], min(last, hi_lim), text))
            ax.set_yscale("log")
            ax.set_ylim(lo_lim, hi_lim)
            _plain_log_ticks(ax, lo_lim, hi_lim)
            if ri == 0 and axis.get("integer"):
                ax.set_xticks(ygrid)
                ax.set_xticklabels([axis["tick"].format(y) for y in ygrid])
            ax.set_title(f"$T={T}$ tasks", fontsize=10, color=INK, pad=6)
            ax.set_xlabel(axis["label"] if ri == 0 else "task similarity  $q$",
                          fontsize=9.5, color=INK2)
            if ci == 0:
                ax.set_ylabel(ylab, fontsize=10, color=INK2)
            _chrome(ax)
            _end_labels(ax, ends, gap=0.075)
    # the two rows run over different variables, so each names its own traces
    for ri in (0, 1):
        axes[ri][0].legend(handles=handles[ri], frameon=False, fontsize=8.2,
                           labelcolor=INK2, handlelength=1.6, borderpad=0.2,
                           loc="upper right" if ri == 0 else "upper left")

    # row 3: the grid, q across and y up, one sequential scale over the snapshots
    planes, ceil = {}, float(cfg["DW_HEAT_CEIL"])
    qh = qgrid
    for T in snaps:
        Z = np.full((len(ygrid), len(qh)), np.nan)
        for i, y in enumerate(ygrid):
            for j, q in enumerate(qh):
                val = cells.get((y, q), {}).get(f"{metric}@{T}", np.nan)
                if np.isfinite(val) and val > 0:
                    Z[i, j] = np.log10(val)
        planes[T] = Z
    allv = np.concatenate([planes[T][np.isfinite(planes[T])] for T in snaps])
    norm = Normalize(vmin=float(allv.min()), vmax=min(float(allv.max()), ceil))
    clipped = bool(allv.max() > ceil)
    cmap = sequential_cmap()
    cmap.set_bad((0.0, 0.0, 0.0, 0.0))
    dq = qh[1] - qh[0] if len(qh) > 1 else 0.1
    dy = ygrid[1] - ygrid[0] if len(ygrid) > 1 else 1.0
    for ci, T in enumerate(snaps):
        ax = axes[2][ci]
        im = ax.imshow(np.clip(planes[T], None, ceil), origin="lower",
                       aspect="auto", cmap=cmap, norm=norm,
                       extent=(qh[0] - dq / 2, qh[-1] + dq / 2,
                               ygrid[0] - dy / 2, ygrid[-1] + dy / 2))
        ax.set_xticks(qh)
        ax.set_xticklabels([f"{q:g}" for q in qh])
        ax.set_yticks(ygrid)
        ax.set_yticklabels([axis["tick"].format(y) for y in ygrid])
        ax.set_xlabel("task similarity  $q$", fontsize=9.5, color=INK2)
        ax.set_title(f"$T={T}$ tasks", fontsize=10, color=INK, pad=6)
        if ci == 0:
            ax.set_ylabel(axis["label"], fontsize=9.5, color=INK2)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.set_facecolor("#f3f2ee")
        for side in ax.spines:
            ax.spines[side].set_color(AXIS)
            ax.spines[side].set_linewidth(0.8)

    cax = fig.add_axes([0.345, 0.128, 0.31, 0.011])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal",
                      extend="max" if clipped else "neither")
    cb.set_label(r"$\log_{10}$ " + ylab, fontsize=8.5, color=INK2)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_edgecolor(AXIS)

    worst = max((c["gap"] for c in cells.values()), default=0.0)
    thin = sum(1 for c in cells.values()
               if min(c[f"{metric}@{T}_n"] for T in snaps) < cfg["SEEDS"])
    # nominal ratios that realise the same pair of counts give flat stretches
    same = {}
    for y in ygrid:
        c = cells.get((y, qh[0]))
        if c is not None:
            same.setdefault((c["n_f"], c["n_b"]), []).append(y)
    dup = [ys_ for ys_ in same.values() if len(ys_) > 1]
    routes = _route_legend(err="bars")
    fig.legend(handles=routes, frameon=False, fontsize=8.5,
               ncol=len(routes), loc="upper center", bbox_to_anchor=(0.5, 0.958),
               labelcolor=INK2)
    fig.suptitle(f"{sub}, {title}   ($N={cfg['N']}$, $N_{{in}}={cfg['D']}$, "
                 f"{_w0_tag(cfg)}, {cfg['SEEDS']} seeds per cell)",
                 fontsize=11, color=INK, y=0.995)
    on_grid = (all(q in qh for q in qs)
               and all(any(abs(y - g) < 1e-9 for g in ygrid) for y in ytr))
    note = [
        ("rows 1 and 2 are two slices of the grid row 3 shows whole, so the figure "
         "is one set of numbers read three ways." if on_grid else
         "rows 1 and 2 are slices of the sweep row 3 shows, plus the trace values "
         "that fall off its grid, which are run for rows 1 and 2 alone."),
        ("rows 1 and 2 share one vertical axis across all three columns, capped at "
         f"{hi_lim:.0e}; a trace that leaves a panel is labelled with the decade it "
         f"reached, and row 3's colour stops at $10^{{{ceil:g}}}$."
         if hi_lim >= float(cfg["DW_CEIL"]) else
         "rows 1 and 2 share one vertical axis across all three columns"
         + (f", and row 3's colour stops at $10^{{{ceil:g}}}$." if clipped
            else ".")),
        "the dashed closed form rides inside the solid band -- the coupling eq. (19) "
        "and one triangular solve eq. (27), with no state ever formed -- agreeing to "
        f"{worst:.0e} relative at worst.",
        ("the split ratio is nominal: " + ";  ".join(
            f"{', '.join(f'{y:g}' for y in ys_)} realise the same pool"
            for ys_ in dup) + ", so the traces are flat there by construction.")
        if dup else "",
        f"{thin} of {len(cells)} cells lost a seed to overflow." if thin else "",
    ]
    fig.text(0.5, 0.012, "\n".join(ln for ln in note if ln), ha="center",
             va="bottom", fontsize=7.8, color=MUTED, linespacing=1.55)
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


# --sim figure 6a's columns: (key, column title, the metric off state K - 1, and
# its kind).  All five are read off the one state the penultimate task leaves,
# W^{K-1} (`case_delta_w`'s `at`), so the last task is the one interference reads.
# The residual pair is read on figure 5's colour map, diverging about ||r|| = 1,
# the state worth what W = 0 is worth; the three weight columns on figure 6's
# sequential one, a scale each, since a size has no such point to diverge about.
AT_METRICS = (
    ("ret", "forgetting", r"$\frac{1}{K-1}\sum_{i<K}\|r_i^{(K-1)}\|$", "resid"),
    ("tr", "interference", r"$\|\theta_K-\hat\theta_K^{(K-1)}\|$", "resid"),
    ("disp", "weight displacement", r"$\|W^{K-1}-W^0\|_F$", "weight"),
    ("dw", "update magnitude", r"$\|W^{K-1}-W^{K-2}\|_F$", "weight"),
    ("mag", "weight magnitude", r"$\|W^{K-1}\|_F$", "weight"),
)


def fig_sim_metrics(cfg, cells, path, axis, qheat, title):
    """--sim figure 6a: five metrics off the penultimate state, q against density.

    The columns are the metrics, not stream lengths: forgetting over the K - 1 tasks
    W^{K-1} has trained, interference -- task K read out of it before it trains --
    how far it has moved from W^0, the step that made it, and the size of the
    weights themselves, all off the one state the penultimate task of a K_DW
    stream leaves.  The rows are figure 6's: row 1 runs over y (axis["grid"]) with
    one trace per q, on the ordinal ramp; row 2 runs over q (`qheat`) with one
    trace per value of axis["traces"], in the categorical slots; row 3 is the whole
    grid, q across and y up.  Rows 1 and 2 are slices of what row 3 shows whole.

    `cells` is keyed (y, q) -> case_delta_w with `at` = K_DW - 1.  Each column has a
    vertical axis of its own, shared by its rows 1 and 2: the five are in
    different units, so one axis would flatten four of them, and the two rows are
    one set of numbers.  In row 3 the residual pair shares one diverging scale --
    figure 5's colour map about log10||r|| = 0, but symmetric about it, so that a
    step of colour is the same factor either side, and spanning only what these
    cells reach -- and each weight column has a sequential scale of its own.

    The weight magnitude has no task axis to start from K = 0 on, as it does in
    figures 3a and 8, so its K = 0 value is drawn as a reference line instead:
    ||W^0||_F, the same for every cell, since W^0 is drawn once per seed.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm

    K = int(cfg["K_DW"])
    ats = {int(c["at"]) for c in cells.values() if "at" in c}
    if ats != {K - 1} or any("at" not in c for c in cells.values()):
        raise ValueError(f"figure 6a reads every cell at the penultimate state "
                         f"K_DW - 1 = {K - 1}; these cells were read at {ats}")
    s = K - 1
    ygrid = [float(y) for y in axis["grid"]]
    qgrid = [float(q) for q in qheat]
    qs = [float(q) for q in cfg["SIM_QS"]]
    qcol = q_colours(qs)
    ytr = [float(y) for y in axis["traces"]]
    ycol = dict(zip(ytr, _slots(len(ytr), axis["label"])))
    ncol = len(AT_METRICS)
    # W^0 is one draw per seed index, so every cell holds the same ||W^0||_F
    init = [c["mag0"] for c in cells.values()
            if np.isfinite(c.get("mag0", np.nan)) and c["mag0"] > 0]
    mag0 = float(np.median(init)) if init else None

    def line(keys, xs, m):
        return (np.asarray(xs, dtype=float),
                np.array([cells[k][f"{m}@{s}"] for k in keys]),
                np.array([cells[k][f"{m}@{s}_sem"] for k in keys]),
                np.array([cells[k][f"{m}_x@{s}"] for k in keys]))

    # every trace of rows 1 and 2 first, so each column's two rows share an axis
    series = {}
    for ci, (m, _, _, _) in enumerate(AT_METRICS):
        for q in qs:
            ys_ = [y for y in ygrid if (y, q) in cells]
            if ys_:
                series[(0, ci, q)] = (qcol[q], f"$q={q:g}$", f"{q:g}") + line(
                    [(y, q) for y in ys_], ys_, m)
        for y in ytr:
            qs_ = [q for q in qgrid if (y, q) in cells]
            if qs_:
                series[(1, ci, y)] = (ycol[y], axis["legend"].format(y),
                                      f"{y:g}") + line([(y, q) for q in qs_], qs_, m)
    lims = []
    for ci, (m, _, _, _) in enumerate(AT_METRICS):
        parts = [np.array([mag0])] if m == "mag" and mag0 else []
        for key, a in series.items():
            if key[1] != ci:
                continue
            sim, sem = a[4], np.where(np.isfinite(a[5]), a[5], 0.0)
            parts += [sim * np.exp(sem), sim * np.exp(-sem), a[6]]
        pool = np.concatenate(parts)
        pool = pool[np.isfinite(pool) & (pool > 0)]
        hi = min(float(pool.max()) * 1.3, float(cfg["DW_CEIL"]))
        lims.append((max(float(pool.min()) / 1.3, hi * 1e-12), hi))

    fig, axes = plt.subplots(3, ncol, figsize=(4.1 * ncol, 12.8))
    fig.subplots_adjust(left=0.05, right=0.97, top=0.885, bottom=0.22,
                        hspace=0.36, wspace=0.36)
    handles = [[], []]
    for ci, (m, head, ylab, kind) in enumerate(AT_METRICS):
        lo, hi = lims[ci]
        for ri in (0, 1):
            ax = axes[ri][ci]
            ends = []
            for key, (colour, legend, tag, xs, sim, sem, exa) in series.items():
                if key[0] != ri or key[1] != ci:
                    continue
                ln, end = _two(ax, xs, sim, exa, colour,
                               legend if ci == 0 else None,
                               sem=sem, err="bars", marker="o")
                if ci == 0:
                    handles[ri].append(ln)
                if end:
                    last = end[1]
                    text = (tag if last <= hi else
                            f"{tag} $\\to10^{{{int(np.log10(last))}}}$")
                    ends.append((end[0], min(last, hi), text))
            ax.set_yscale("log")
            ax.set_ylim(lo, hi)
            _plain_log_ticks(ax, lo, hi)
            if kind == "resid":
                # unannotated: the traces cross it, and the footnote names it
                _baseline(ax, note=False)
            if m == "mag" and mag0:
                # the size before any task, named at its right-hand end with the
                # traces so that the nudging keeps it clear of theirs
                ax.axhline(mag0, color=MUTED, linewidth=1.0, zorder=1,
                           linestyle=(0, (6, 3)))
                ends.append(((ygrid if ri == 0 else qgrid)[-1], mag0, "$W^0$"))
            if ri == 0:
                ax.set_title(head, fontsize=10.5, color=INK, pad=8)
            ax.set_xlabel(axis["label"] if ri == 0 else "task similarity  $q$",
                          fontsize=9.5, color=INK2)
            ax.set_ylabel(ylab, fontsize=10, color=INK2)
            _chrome(ax)
            _end_labels(ax, ends, gap=0.075)
    # the two rows run over different variables, so each names its own traces, in
    # the displacement's panel: d_f = d_b = 1 telescopes to about 1/||v||, which
    # stretches that column's shared axis down a decade below every other trace and
    # leaves its lower left empty in both rows
    free = next(ci for ci, a in enumerate(AT_METRICS) if a[0] == "disp")
    for ri in (0, 1):
        axes[ri][free].legend(handles=handles[ri], frameon=False, fontsize=8.2,
                              labelcolor=INK2, handlelength=1.6, borderpad=0.4,
                              loc="lower left")

    # row 3: the grid, q across and y up, in log10 of each metric
    planes = {}
    for m, _, _, _ in AT_METRICS:
        Z = np.full((len(ygrid), len(qgrid)), np.nan)
        for i, y in enumerate(ygrid):
            for j, q in enumerate(qgrid):
                val = cells.get((y, q), {}).get(f"{m}@{s}", np.nan)
                if np.isfinite(val) and val > 0:
                    Z[i, j] = np.log10(val)
        planes[m] = Z

    def finite(ms):
        return np.concatenate([planes[m][np.isfinite(planes[m])] for m in ms])

    resid = [m for m, _, _, k in AT_METRICS if k == "resid"]
    rv = finite(resid)
    half = min(max(float(np.abs(rv).max()), 0.05), float(cfg["HEAT_CEIL"]))
    scales = {m: (diverging_cmap(), TwoSlopeNorm(vmin=-half, vcenter=0.0,
                                                  vmax=half),
                  ("both" if rv.min() < -half and rv.max() > half else
                   "max" if rv.max() > half else
                   "min" if rv.min() < -half else "neither"))
              for m in resid}
    ceil = float(cfg["DW_HEAT_CEIL"])
    for m, _, _, k in AT_METRICS:
        if k == "weight":
            wv = finite([m])
            scales[m] = (sequential_cmap(),
                         Normalize(vmin=float(wv.min()),
                                   vmax=min(float(wv.max()), ceil)),
                         "max" if wv.max() > ceil else "neither")
    dq = qgrid[1] - qgrid[0] if len(qgrid) > 1 else 0.1
    dy = ygrid[1] - ygrid[0] if len(ygrid) > 1 else 1.0
    images = {}
    for ci, (m, head, ylab, kind) in enumerate(AT_METRICS):
        ax = axes[2][ci]
        cmap, norm, _ = scales[m]
        cmap.set_bad((0.0, 0.0, 0.0, 0.0))
        images[m] = ax.imshow(np.clip(planes[m], -half if kind == "resid" else None,
                                      half if kind == "resid" else ceil),
                              origin="lower", aspect="auto", cmap=cmap, norm=norm,
                              extent=(qgrid[0] - dq / 2, qgrid[-1] + dq / 2,
                                      ygrid[0] - dy / 2, ygrid[-1] + dy / 2))
        ax.set_xticks(qgrid)
        ax.set_xticklabels([f"{q:g}" for q in qgrid])
        ax.set_yticks(ygrid)
        ax.set_yticklabels([axis["tick"].format(y) for y in ygrid])
        ax.set_xlabel("task similarity  $q$", fontsize=9.5, color=INK2)
        if ci == 0:
            ax.set_ylabel(axis["label"], fontsize=9.5, color=INK2)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.set_facecolor("#f3f2ee")
        for side in ax.spines:
            ax.spines[side].set_color(AXIS)
            ax.spines[side].set_linewidth(0.8)

    # one colour bar under the residual pair, one under each weight column
    def bar(m, cols, label):
        x0 = axes[2][cols[0]].get_position().x0
        x1 = axes[2][cols[-1]].get_position().x1
        y0 = axes[2][cols[0]].get_position().y0
        w = (x1 - x0) * (0.7 if len(cols) > 1 else 0.86)
        cax = fig.add_axes([(x0 + x1 - w) / 2, y0 - 0.068, w, 0.0105])
        cb = fig.colorbar(images[m], cax=cax, orientation="horizontal",
                          extend=scales[m][2])
        cb.set_label(label, fontsize=8.5, color=INK2)
        cb.ax.tick_params(colors=MUTED, labelsize=8)
        cb.outline.set_edgecolor(AXIS)

    ri_cols = [ci for ci, a in enumerate(AT_METRICS) if a[3] == "resid"]
    bar(resid[0], ri_cols, "$\\log_{10}\\|r\\|$      $0$ = no better than $W=0$,  "
                           "below = better,  above = worse")
    for ci, (m, _, ylab, k) in enumerate(AT_METRICS):
        if k == "weight":
            bar(m, [ci], r"$\log_{10}$ " + ylab)

    worst = max((c["gap_at"] for c in cells.values()), default=0.0)
    thin = sum(1 for c in cells.values()
               if min(c[f"{m}@{s}_n"] for m, _, _, _ in AT_METRICS) < cfg["SEEDS"])
    routes = _route_legend(err="bars")
    fig.legend(handles=routes, frameon=False, fontsize=8.5,
               ncol=len(routes), loc="upper center", bbox_to_anchor=(0.5, 0.958),
               labelcolor=INK2)
    fig.suptitle(f"five metrics at the penultimate task, {title}   "
                 f"($N={cfg['N']}$, $N_{{in}}={cfg['D']}$, {_w0_tag(cfg)}, "
                 f"state $K-1={s}$ of $K={K}$, {cfg['SEEDS']} seeds per cell)",
                 fontsize=11, color=INK, y=0.995)
    on_grid = (all(q in qgrid for q in qs)
               and all(any(abs(y - g) < 1e-9 for g in ygrid) for y in ytr))
    note = [
        "all five columns are read off one state, $W^{K-1}$, the one the "
        f"penultimate task leaves: forgetting averages the {s} tasks it has "
        f"trained, interference is the last task, {K}, read out of it before it "
        "trains,",
        "and the three weight columns are how far it has moved from $W^0$, the step "
        "that moved it last, and the size of the weights themselves.  "
        + ("rows 1 and 2 are two slices of the grid row 3 shows whole."
           if on_grid else "rows 1 and 2 are slices of the sweep row 3 shows, plus "
           "the trace values that fall off its grid."),
        "each column has a vertical axis of its own, shared by its rows 1 and 2.  in "
        "the residual columns the grey line at 1 is a state worth no more than "
        "$W=0$, and in row 3 the two share one diverging scale, symmetric about "
        "$\\|r\\|=1$;",
        "each weight column has a sequential scale of its own.  interference and the "
        "step are one task's own numbers, not pooled over the tasks around it, so "
        "they scatter across seeds more than forgetting does.",
        "the dashed closed form rides inside the solid band -- the coupling eq. (19), "
        "one triangular solve eq. (27) and the band eq. (30), with no state ever "
        f"formed -- agreeing to {worst:.0e} relative at worst.",
        ("the dashed grey line marked $W^0$ in the weight magnitude's column is "
         "$\\|W^0\\|_F$, the size before any task, the same in every cell since "
         "$W^0$ is drawn once per seed." if mag0 else
         "$W^0=0$, so the weight magnitude has no starting size to draw."),
        f"{thin} of {len(cells)} cells lost a seed to overflow." if thin else "",
    ]
    fig.text(0.5, 0.012, "\n".join(ln for ln in note if ln), ha="center",
             va="bottom", fontsize=7.8, color=MUTED, linespacing=1.55)
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


NOTE_SIM_SCORE = (
    "the stream clears most of the shared teacher $\\sqrt{q}\\,\\theta_0$ within "
    "about $1/d_f$ tasks, then keeps about $\\sqrt{2q(1-d_f)/n_f}$ of it for "
    "thousands;\nlate in the stream both scores return to $\\sqrt{1-q}$ of their "
    "$q=0$ values.")
NOTE_SIM_SCORE_SPLIT = (
    "in a stable row both scores return late to $\\sqrt{1-q}$ of their $q=0$ "
    "values;\npast the threshold the shared teacher and $W^0$ are amplified with "
    "everything else, so a runaway row keeps more than that: no $q$ escapes the "
    "instability of eq. (17).")
NOTE_SIM_WEIGHT = (
    "the step shrinks with $q$ as the residual does; the displacement shrinks less: "
    "reading $\\theta_0$ out of every pool means writing it into the neurons\nthe "
    "readout barely weights, which takes large weights, and the stream keeps doing "
    "that for thousands of tasks.\n" + NOTE_MAG)
NOTE_SIM_ALL = NOTE_SIM_SCORE + "\n" + NOTE_SIM_WEIGHT


def _sim_table(lines, head, rows_):
    lines.append(head)
    lines.extend(rows_)


def main_sim(cfg, here, want):
    """The task-similarity figures, into `here`; `want` selects them by number."""
    qs = [float(q) for q in cfg["SIM_QS"]]
    q_colours(qs)                                     # refuse a sixth q up front
    dens = [float(d) for d in cfg["SIM_DENSITIES"]]
    rhos = [float(r) for r in cfg["SIM_RHOS"]]
    _slots(len(dens), "entries of SIM_DENSITIES")
    _slots(len(rhos), "entries of SIM_RHOS")
    d_sim, d_split = float(cfg["SIM_DF"]), float(cfg["SIM_SPLIT_DF"])

    def on(*figs):
        return not want or bool(want & set(figs))

    lines = ["task similarity: theta_t = sqrt(q) theta_0 + sqrt(1-q) thetatilde_t,",
             "  theta_0 once per seed, thetatilde_t every task, both unit vectors, so",
             "  E||theta_t||^2 = 1 and E[theta_t . theta_u] = q for t != u.",
             f"  q traces {tuple(qs)};  density traces (d_f = d_b) {tuple(dens)};",
             f"  split traces {tuple(rhos)} at d_f = {d_split:g};  figures 2 and 8 at "
             f"d_f = d_b = {d_sim:g}",
             f"initial state W^0: iid N(0, s^2/N_in), s = W0_SCALE = "
             f"{cfg['W0_SCALE']:g}",
             "scores are the RAW residual, ||r||, higher is worse:",
             "  interference(K) = ||theta_K - thetahat_K^(K-1)||, task K's own",
             "  forgetting(K)   = (1/K) sum_{i<=K} ||theta_i - thetahat_i^(K)||",
             "0 = solved, 1 = no better than W = 0.  seeds combined geometrically.",
             ""]
    head = (f"{'forgetting':>14s} {'interference':>14s}"
            f"{'forget eq27/30':>15s} {'interf eq27/30':>15s} {'route gap':>11s}")
    head_w = (f"{'dW':>12s} {'disp':>12s} {'|W|':>12s}"
              f"{'dW, eq.27':>13s} {'disp, eq.27':>13s} {'|W|, eq.27':>13s} "
              f"{'|W^0|':>9s}{'route gap':>11s}")

    def end_row(lab, r, a, b):
        return (f"  {lab} {r['n_b']:4d} {r[a][-1]:14.4g} {r[b][-1]:14.4g}"
                f"{r[a + '_x'][-1]:15.4g} {r[b + '_x'][-1]:15.4g}{r['gap']:11.1e}")

    def end_row_w(lab, r):
        """The three weight metrics at the end of the stream, and ||W^0||_F."""
        return (f"  {lab} {r['n_b']:4d} " + " ".join(
                    f"{r[a][-1]:12.4g}" for a in ("dw", "disp", "mag"))
                + "".join(f"{r[a + '_x'][-1]:13.4g}" if i == 0 else
                          f" {r[a + '_x'][-1]:13.4g}"
                          for i, a in enumerate(("dw", "disp", "mag")))
                + f" {r['mag0']:9.4g}{max(r['gap'], r['gap_mag']):11.1e}")

    if on("1"):
        print("\ntask similarity, figure 1: the flow over a long stream, per q ...")
        flows = [flow_case(cfg, q=q, K=cfg["SIM_K_FLOW"], show=cfg["SIM_FLOW_SHOW"])
                 for q in qs]
        gaps = fig_sim_flow(cfg, flows, here / "fig1_flow.png")
        lines.append(f"fig1  K={cfg['SIM_K_FLOW']}, the first and last "
                     f"{cfg['SIM_FLOW_SHOW']} tasks drawn; worst task of each end, "
                     "relative to its own residual")
        lines.append(f"  {'q':>5s} {'end':>6s} {'integrated vs (27)':>19s} "
                     f"{'protocol vs (27)':>17s} {'flow vs closed':>15s}")
        for q, ends_ in gaps:
            for end, g in zip(("first", "last"), ends_):
                if g is not None:
                    lines.append(f"  {q:5.2f} {end:>6s} {g[0]:19.2e} {g[1]:17.2e} "
                                 f"{g[2]:15.2e}")
        lines.append("")

    tasks, split = {}, {}
    if on("2", "3", "8"):
        need = ({d_sim} if on("2", "8") else set()) | (set(dens) if on("3") else set())
        print("task similarity: the density sweep against task index "
              f"(figures 2, 3a, 8), {len(need) * len(qs)} cases ...")
        for d in sorted(need, reverse=True):
            for q in qs:
                tasks[(d, 1.0, q)] = case_vs_tasks(cfg, d, 1.0, TASKS, q)
    if on("3"):
        print(f"task similarity: the split sweep at d_f = {d_split:g} (figure 3b), "
              f"{len(rhos) * len(qs)} cases ...")
        for rho in rhos:
            for q in qs:
                try:
                    split[(d_split, rho, q)] = case_vs_tasks(cfg, d_split, rho,
                                                             SPLIT, q)
                except ValueError:
                    pass

    if on("2"):
        tr = q_traces(cfg, tasks, d_sim)
        fig_vs_tasks(cfg, tasks, here / "fig2_vs_tasks.png", ARMS, tr,
                     "forgetting and interference against task index, one trace per "
                     f"task similarity, $d_f=d_b={d_sim:g}$", NOTE_SIM_SCORE,
                     split_legend=True)
        lines.append(f"fig2  at the end of the stream (K={cfg['K_TASKS']}), "
                     f"d_f = d_b = {d_sim:g}")
        _sim_table(lines, f"  {'q':>5s} {'n_b':>4s} " + head,
                   [end_row(f"{t[5]:5.2f}", tasks[t[0]], "ret", "tr") for t in tr])
        lines.append("")

    if on("8"):
        tr = q_traces(cfg, tasks, d_sim)
        fig_vs_tasks(cfg, tasks, here / "fig8_weight_vs_tasks.png", ARMS_W, tr,
                     "update magnitude, weight displacement and weight magnitude, "
                     "one trace per task similarity, "
                     f"$d_f=d_b={d_sim:g}$", NOTE_SIM_WEIGHT, ylog=True,
                     plain=True, split_legend=True)
        lines.append(f"fig8  at the end of the stream (K={cfg['K_TASKS']}), "
                     f"d_f = d_b = {d_sim:g}.  dW = ||W^K - W^(K-1)||_F, "
                     "disp = ||W^K - W^0||_F, |W| = ||W^K||_F;")
        lines.append("      |W^0| = ||W^0||_F, where every |W| trace starts at K = 0 "
                     "(unless W^0 = 0); the route gap covers all three")
        _sim_table(lines, f"  {'q':>5s} {'n_b':>4s} " + head_w,
                   [end_row_w(f"{t[5]:5.2f}", tasks[t[0]]) for t in tr])
        lines.append("")

    if on("3"):
        print("task similarity, figures 3a and 3b ...")
        fig_sim_rows(cfg, tasks, here / "fig3a_vs_tasks_density.png",
                     [(f"$d_f=d_b={d:g}$", q_traces(cfg, tasks, d)) for d in dens],
                     "the two scores and the three weight metrics against task "
                     "index, one row per density, one trace per task similarity",
                     NOTE_SIM_ALL, arms=ARMS_ALL)
        fig_sim_rows(cfg, split, here / "fig3b_vs_tasks_split.png",
                     [(f"$d_f/d_b={rho:g}$", q_traces(cfg, split, d_split, rho))
                      for rho in rhos],
                     "the same, one row per split ratio at "
                     f"$d_f={d_split:g}$", NOTE_SIM_SCORE_SPLIT, ylog=True)
        lines.append(f"fig3  at the end of the stream (K={cfg['K_TASKS']})")
        lines.append("  3a, one row per density (d_f = d_b): the two scores")
        _sim_table(lines, f"  {'d_f':>5s} {'q':>5s} {'n_b':>4s} " + head,
                   [end_row(f"{d:5.2f} {q:5.2f}", tasks[(d, 1.0, q)], "ret", "tr")
                    for d in dens for q in qs if (d, 1.0, q) in tasks])
        lines.append("  3a, the three weight metrics: dW = ||W^K - W^(K-1)||_F, "
                     "disp = ||W^K - W^0||_F, |W| = ||W^K||_F, from |W^0| at K = 0")
        _sim_table(lines, f"  {'d_f':>5s} {'q':>5s} {'n_b':>4s} " + head_w,
                   [end_row_w(f"{d:5.2f} {q:5.2f}", tasks[(d, 1.0, q)])
                    for d in dens for q in qs if (d, 1.0, q) in tasks])
        lines.append(f"  3b, one row per split ratio at d_f = {d_split:g}")
        _sim_table(lines, f"  {'rho':>5s} {'q':>5s} {'n_b':>4s} " + head,
                   [end_row(f"{rho:5.1f} {q:5.2f}", split[(d_split, rho, q)],
                            "ret", "tr")
                    for rho in rhos for q in qs if (d_split, rho, q) in split])
        lines.append("")

    if on("4"):
        q4 = _q_grid(cfg["SIM_Q_STEP"], cfg["SIM_Q_MAX"])
        print(f"task similarity, figure 4: against q, "
              f"{(len(dens) + len(rhos)) * len(q4)} cases ...")
        grid_a, grid_b = {}, {}
        for d in dens:
            for q in q4:
                grid_a[(d, q)] = case_vs_density(cfg, d, 1.0, q)
        for rho in rhos:
            for q in q4:
                try:
                    grid_b[(rho, q)] = case_vs_density(cfg, d_split, rho, q)
                except ValueError:
                    pass
        tr_a = [(d, f"$d_f=d_b={d:g}$", f"{d:g}", c, d, grid_a[(d, q4[0])]["n_b"]
                 / cfg["N"]) for d, c in zip(dens, _slots(len(dens)))]
        tr_b = [(rho, f"$d_f/d_b={rho:g}$", f"{rho:g}", c, d_split,
                 grid_b[(rho, q4[0])]["n_b"] / cfg["N"])
                for rho, c in zip(rhos, _slots(len(rhos))) if (rho, q4[0]) in grid_b]
        fig_sim_vs_q(cfg, grid_a, here / "fig4a_vs_q_density.png", tr_a,
                     "the raw residual at the end of the stream, against task "
                     "similarity, one trace per density",
                     "at $q\\to1$ every task is $\\theta_0$ and the new part "
                     "$\\sqrt{1-q}\\,\\tilde\\theta_t$ is all that is left to "
                     "interfere.")
        fig_sim_vs_q(cfg, grid_b, here / "fig4b_vs_q_split.png", tr_b,
                     "the same at $d_f=" f"{d_split:g}$, one trace per split ratio",
                     "")
        lines.append(f"fig4  at the end of the stream (K={cfg['K_DENSITY']})")
        for tag, grid, var in (("4a, one trace per density (d_f = d_b)", grid_a,
                                "d_f"),
                               (f"4b, one trace per split ratio at d_f = "
                                f"{d_split:g}", grid_b, "rho")):
            lines.append(f"  {tag}")
            lines.append(f"  {var:>5s} {'q':>5s} {'n_b':>4s} " + head)
            for (y, q), r in sorted(grid.items()):
                lines.append(f"  {y:5.2f} {q:5.2f} {r['n_b']:4d} "
                             f"{r['ret']:14.4g} {r['tr']:14.4g}"
                             f"{r['ret_x']:15.4g} {r['tr_x']:15.4g}"
                             f"{r['gap']:11.1e}")
        lines.append("")

    if on("5"):
        qh = _q_grid(cfg["SIM_HEAT_Q_STEP"], cfg["SIM_Q_MAX"])
        ya = _grid(cfg["D"] / cfg["N"], 1.0, cfg["HEAT_DF_STEP"])
        yb = [float(r) for r in cfg["HEAT_RHOS"]]
        print(f"task similarity, figure 5: {len(qh) * (len(ya) + len(yb))} cells ...")
        cells_a, cells_b = {}, {}
        for d in ya:
            for q in qh:
                try:
                    cells_a[(d, q)] = case_heat(cfg, d, 1.0, q)
                except ValueError:
                    pass
        for rho in yb:
            for q in qh:
                try:
                    cells_b[(rho, q)] = case_heat(cfg, d_split, rho, q)
                except ValueError:
                    pass
        fig_sim_heatmaps(cfg, cells_a, here / "fig5a_heatmaps_density.png",
                         ya, "read density  $d_f=d_b$",
                         "{:.1f}", "task similarity against density, three "
                         "snapshots of one run", "every cell writes as densely as "
                         "it reads, $d_f=d_b$.")
        fig_sim_heatmaps(cfg, cells_b, here / "fig5b_heatmaps_split.png",
                         yb, "splitness  $d_f/d_b$",
                         "{:g}", f"task similarity against splitness at "
                         f"$d_f={d_split:g}$", "the split's instability does not "
                         "care what the teachers share: past the threshold every "
                         "$q$ runs away.")
        lines.append(f"fig5  snapshots {cfg['SNAPSHOTS']}, unclipped (the figure "
                     f"shows log10, capped at {cfg['HEAT_CEIL']:g})")
        for tag, cells, var in (("5a, q against density (d_f = d_b)", cells_a,
                                 "d_f"),
                                (f"5b, q against splitness at d_f = {d_split:g}",
                                 cells_b, "rho")):
            lines.append(f"  {tag}")
            lines.append(f"  {var:>5s} {'q':>5s} {'n_b':>4s} {'k':>6s} "
                         f"{'forgetting':>14s} {'interference':>14s}")
            for (y, q), (snap, n_b) in sorted(cells.items()):
                for k in cfg["SNAPSHOTS"]:
                    lines.append(f"  {y:5.2f} {q:5.2f} {n_b:4d} {k:6d} "
                                 f"{snap[k]['ret']:14.4g} {snap[k]['tr']:14.4g}")
        lines.append("")

    if on("6", "7"):
        qh = _q_grid(cfg["SIM_HEAT_Q_STEP"], cfg["SIM_Q_MAX"])
        ya = _grid(cfg["DW_DF_STEP"], 1.0, cfg["DW_DF_STEP"])
        yb = [float(r) for r in cfg["DW_RHOS"]]
        # figure 6a reads its five metrics off the penultimate state of the same
        # streams figure 7a's snapshots come from, so the density sweep carries both
        pen = int(cfg["K_DW"]) - 1
        at = {"a": pen if on("6") else None, "b": None}
        sweeps = {}
        for tag, yg, ytr in (("a", ya, dens), ("b", yb, rhos)):
            # the grid with every q trace on it, and every y trace across the q grid
            keys = ({(y, q) for y in yg for q in set(qh) | set(qs)}
                    | {(y, q) for y in ytr for q in qh})
            print(f"task similarity, figures 6{tag} and 7{tag}: {len(keys)} cells ...")
            cells = {}
            for y, q in sorted(keys):
                try:
                    cells[(y, q)] = (case_delta_w(cfg, y, 1.0, q=q, at=at[tag])
                                     if tag == "a"
                                     else case_delta_w(cfg, d_split, y, q=q))
                except ValueError:
                    pass
            sweeps[tag] = cells
        axes_ = {"a": dict(grid=ya, traces=dens, label="read density  $d_f=d_b$",
                           legend="$d_f=d_b={:g}$", tick="{:.1f}"),
                 "b": dict(grid=yb, traces=rhos, label="splitness  $d_f/d_b$",
                           legend="$d_f/d_b={:g}$", tick="{:g}", integer=True)}
        what = {"a": "against task similarity and density ($d_f=d_b$)",
                "b": f"against task similarity and splitness ($d_f={d_split:g}$)"}

        def snapshot_table(which, tag, metric, extra=""):
            var = "d_f" if tag == "a" else "rho"
            lines.append(f"  {which}{tag}, {what[tag]}{extra}"
                         .replace("$", "").replace("\\", ""))
            lines.append(f"  {var:>5s} {'q':>5s} {'n_b':>4s} {'T':>6s} "
                         f"{'simulation':>14s} {'eq. (27)':>14s} {'s.e.m.':>9s} "
                         f"{'seeds':>6s} {'route gap':>11s}")
            for (y, q), c in sorted(sweeps[tag].items()):
                for T in cfg["DW_SNAPSHOTS"]:
                    k = f"{metric}@{T}"
                    lines.append(f"  {y:5.2f} {q:5.2f} {c['n_b']:4d} {T:6d} "
                                 f"{c[k]:14.4g} {c[f'{metric}_x@{T}']:14.4g} "
                                 f"{c[k + '_sem']:9.3f} {c[k + '_n']:6d} "
                                 f"{c['gap']:11.1e}")

        if on("6"):
            print("task similarity, figures 6a and 6b ...")
            fig_sim_metrics(cfg, sweeps["a"], here / "fig6a_metrics_density.png",
                            axes_["a"], qh, what["a"])
            fig_sim_delta_w(cfg, sweeps["b"], here / "fig6b_step_split.png", "step",
                            axes_["b"], qh, what["b"])
            K_ = int(cfg["K_DW"])
            lines.append(f"fig6  6a: five metrics off the penultimate state, "
                         f"W^{pen} of a stream of K = {K_};  6b: the mean step per "
                         "task")
            lines.append(f"  6a, {what['a']}".replace("$", "").replace("\\", ""))
            lines.append(f"      forgetting = (1/{pen}) sum_(i<{K_}) "
                         f"||theta_i - thetahat_i^({pen})||;  interference = "
                         f"||theta_{K_} - thetahat_{K_}^({pen})||, the last task "
                         "read out of it")
            lines.append(f"      disp = ||W^{pen} - W^0||_F;  step = "
                         f"||W^{pen} - W^{pen - 1}||_F;  |W| = ||W^{pen}||_F, "
                         "from |W^0| = ||W^0||_F before any task")
            lines.append("      s.e.m. in logs; seeds is the fewest any of the five "
                         "kept")
            five = [m for m, _, _, _ in AT_METRICS]
            names = {"ret": "forgetting", "tr": "interference", "disp": "disp",
                     "dw": "step", "mag": "|W|"}
            lines.append(f"  {'d_f':>5s} {'q':>5s} {'n_b':>4s}  {'':<13s}"
                         + " ".join(f"{names[m]:>13s}" for m in five)
                         + f" {'|W^0|':>9s} {'seeds':>6s} {'route gap':>10s}")
            for (y, q), c in sorted(sweeps["a"].items()):
                pad = " " * 16                    # under "d_f     q  n_b"
                lines.append(f"  {y:5.2f} {q:5.2f} {c['n_b']:4d}  {'simulation':<13s}"
                             + " ".join(f"{c[f'{m}@{pen}']:13.4g}" for m in five)
                             + f" {c['mag0']:9.4g}"
                             + f" {min(c[f'{m}@{pen}_n'] for m in five):6d}"
                             f" {c['gap_at']:10.1e}")
                lines.append(f"  {pad}  {'s.e.m.':<13s}"
                             + " ".join(f"{c[f'{m}@{pen}_sem']:13.3f}" for m in five))
                lines.append(f"  {pad}  {'eq. (27)':<13s}"
                             + " ".join(f"{c[f'{m}_x@{pen}']:13.4g}" for m in five))
            snapshot_table("6", "b", "step",
                           f", the mean step per task at T={cfg['DW_SNAPSHOTS']}")
            lines.append("")
        if on("7"):
            print("task similarity, figures 7a and 7b ...")
            for tag in ("a", "b"):
                fig_sim_delta_w(cfg, sweeps[tag],
                                here / f"fig7{tag}_displacement_"
                                       f"{'density' if tag == 'a' else 'split'}.png",
                                "disp", axes_[tag], qh, what[tag])
            lines.append(f"fig7  T={cfg['DW_SNAPSHOTS']}, net displacement "
                         "||W^T - W^0||_F")
            for tag in ("a", "b"):
                snapshot_table("7", tag, "disp")
            lines.append("")

    _write_log(here / "neuronal_curves.txt", lines, want)
    print(f"\nwritten to {here}")
    return 0


# ================================================================== self-test

def solve_check(cfg, d_f=0.3, rho=2.0, index=11, q=0.0):
    """The walk against the closed form, on exactly the path the figures draw.

    All four metrics, not just the two scores: the step of eq. (16) and the
    displacement go through the same comparison, the walk measuring them off the
    state it forms and the closed form reading them out of one triangular solve.
    Both start from the seed's W^0: the walk forms the state from it, the solve
    sees only the teachers less what it reads out.  `q` is the task similarity.
    """
    N, D, K = cfg["N"], cfg["D"], cfg["K_CHECK"]
    n_f, n_b = counts(N, d_f, rho)
    states = (K // 4, K // 2, K)
    v, F, B, Theta, W0 = draw_seed(cfg, K, n_f, n_b, TEST, index, q=q)
    tr, ret, dw, disp = run_stream(v, F, B, Theta, states, W0=W0)
    tr_x, ret_x, dw_x, disp_x = exact_scores(v, F, B, Theta, states, W0=W0)
    return max(_route_gap(tr, tr_x),
               _route_gap(dw, dw_x),
               _route_gap([ret[s] for s in states], [ret_x[s] for s in states]),
               _route_gap([disp[s] for s in states], [disp_x[s] for s in states]))


def self_test(cfg, verbose=True):
    """Routes that share no code, checked against each other.

    Every check holds from any initial state and runs from the configured one,
    W0_SCALE; at W0_SCALE = 0 each is the zero-start check it replaced.
    """
    fails = 0
    s0 = float(cfg["W0_SCALE"])
    q_t = 0.6          # the task similarity the q checks run at

    def _raises(fn):
        try:
            fn()
        except ValueError:
            return True
        return False

    def rep(name, got, tol):
        nonlocal fails
        ok = np.all(np.abs(got) <= tol)
        fails += not ok
        if verbose:
            print(f"  [{'ok  ' if ok else 'FAIL'}] {name:<58s} "
                  f"{float(np.max(np.abs(got))):.3e}")

    def mc_drive(rng, M, K, D, d_f, N=20, q=0.0):
        """M replicates of the drive the mean field sees, theta_t - v^T F_t W^0.

        Unit teachers, less the readout of a W^0 at the configured scale through
        real gates -- n_f of N neurons per task, drawn afresh -- with v of unit
        norm, so that E||v||^2 = 1 holds exactly, as the mean field assumes.  At
        scale 0 it is the teachers alone, drawn exactly as they were before W^0.
        At q > 0 the teachers share a direction, through `similar_teachers`.
        """
        Th = rng.standard_normal((M, K, D))
        Th /= np.linalg.norm(Th, axis=2, keepdims=True)
        if q:
            t0 = rng.standard_normal((M, D))
            t0 /= np.linalg.norm(t0, axis=1, keepdims=True)
            Th = np.stack([similar_teachers(Th[m], t0[m], q) for m in range(M)])
        if s0 == 0.0:
            return Th
        n_f = int(round(d_f * N))
        u = rng.standard_normal((M, N))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        W0 = rng.standard_normal((M, N, D)) * (s0 / np.sqrt(D))
        for t in range(K):
            f = np.argsort(np.argsort(rng.random((M, N)), axis=1), axis=1) < n_f
            Th[:, t] -= ((f * u)[:, None, :] @ W0)[:, 0]
        return Th

    if verbose:
        print(f"-- the stream from W^0 (scale {s0:g}) against the exact "
              "triangular solve --")
    for d_f, rho in ((0.5, 1.0), (0.3, 2.0), (0.2, 3.0)):
        rep(f"d_f={d_f}, rho={rho}: streamed scores == eq. (27)/(30)",
            solve_check(cfg, d_f, rho), 1e-9)

    if verbose:
        print("-- the initial state --")
    v, F, B, Theta, W0 = draw_seed(cfg, 40, 20, 10, TEST, 5)
    ref = draw(cfg["N"], cfg["D"], 40, 20, 10, rng_for(cfg, TEST, 5))
    rep("draw_seed leaves v, the gates and Theta as draw makes them",
        [float(np.abs(a - b).max()) for a, b in zip((v, F, B, Theta), ref)], 0)
    rep("W^0 is one draw per seed index, whatever K, n_f and n_b",
        float(np.abs(W0 - draw_seed(cfg, 80, 60, 20, TEST, 5)[4]).max()), 0)
    if s0 == 0.0:
        rep("W0_SCALE = 0 is the zero start, exactly", float(np.abs(W0).max()), 0)
    else:
        big = initial_state(500, 40, s0, rng_for(cfg, TEST, 41, W0_PART))
        rep(f"W^0 entries have variance s^2/N_in, s = {s0:g}",
            float(big.var()) * 40 / s0 ** 2 - 1.0, 0.05)
    tr, ret, dw, disp = run_stream(v, F, B, Theta, (1, 2, 40), W0=W0)
    tr0, ret0, dw0, disp0 = run_stream(v, F, B,
                                       Theta - initial_readout(v, F, W0),
                                       (1, 2, 40))
    rep("walk from W^0 == walk from 0 on Theta - thetahat^(0)",
        [_route_gap(tr, tr0), _route_gap(dw, dw0),
         _route_gap([ret[s] for s in (2, 40)], [ret0[s] for s in (2, 40)]),
         _route_gap([disp[s] for s in (1, 2, 40)],
                    [disp0[s] for s in (1, 2, 40)])], 1e-10)

    if verbose:
        print("-- the two scores at their reference states --")
    rep("task 1 reads W^0: ||r_1|| == ||theta_1 - v^T F_1 W^0||",
        tr[0] - float(np.linalg.norm(Theta[0] - (F[0] * v) @ W0)), 1e-12)
    rep("forgetting(1) = 0: the only task trained is solved", ret[1], 1e-12)
    # two steps replayed by hand from W^0: task 2 is solved, so forgetting(2) is
    # half of what is left of task 1
    W1 = W0 + np.outer(B[0] * v, Theta[0] - (F[0] * v) @ W0) / float(B[0] @ v ** 2)
    W2 = W1 + np.outer(B[1] * v, Theta[1] - (F[1] * v) @ W1) / float(B[1] @ v ** 2)
    rep("forgetting(2) == ||r_1^(2)|| / 2, hand-replayed",
        ret[2] - 0.5 * float(rownorm((Theta[0] - (F[0] * v) @ W2)[None, :])[0]),
        1e-12)
    rep("the residual is raw: ||theta_t|| == 1, nothing is divided out",
        np.linalg.norm(Theta, axis=1) - 1.0, 1e-12)

    if verbose:
        print("-- the integrated flow from W^0 against eq. (16) --")
    res = flow_case({**cfg, "K_FLOW": 6, "PER_TASK": 60})
    rep("integrated residuals == direct solve", res["box_gap"], 1e-7)
    rep("integrated loss == closed form, per task height", res["flow_gap"], 1e-6)
    rep("stream residuals == direct solve", res["solve_gap"], 1e-12)

    if verbose:
        print("-- the mean field against its own recursion, driven from W^0 --")
    rng = rng_for(cfg, TEST, 3)
    for d_f in (0.5, 0.2):
        K, D, M = 200, 32, 3000
        p = 1.0 - d_f
        Th = mc_drive(rng, M, K, D, d_f)
        S = np.zeros((M, D))
        Ss = np.empty((M, K, D))
        Rm = np.empty((M, K, D))
        for t in range(K):
            Rm[:, t] = Th[:, t] - d_f * S
            S = p * S + Th[:, t]
            Ss[:, t] = S
        for Delta in (-1, 6, 20):
            for t in (1, 3, 40, 120):
                if Delta >= 0 and t + Delta >= K:
                    continue
                if Delta == -1:
                    got = (Rm[:, t - 1] ** 2).sum(1).mean()
                else:
                    got = (d_f ** 2 * ((Ss[:, t + Delta - 1]
                                        - Ss[:, t - 1]) ** 2).sum(1)).mean()
                want = float(mf_resid(d_f, t, Delta, s0))
                rep(f"  d_f={d_f}, Delta={Delta:3d}, t={t:3d}: ||r|| = {want:.4f}",
                    np.sqrt(got) / want - 1.0, 0.02)
        # the forgetting score is the same formula averaged term by term
        for Ks in (20, 150):
            per = np.sqrt((d_f ** 2 * ((Ss[:, Ks - 1][:, None, :]
                                        - Ss[:, :Ks]) ** 2).sum(2)).mean(0))
            rep(f"  d_f={d_f}, forgetting({Ks}) = {mf_retention(d_f, Ks, s0):.4f}",
                per.mean() / mf_retention(d_f, Ks, s0) - 1.0, 0.02)

    if verbose:
        print("-- the mean field is exact at d_f = 1, where Gamma is all-ones --")
    v, F, B, Theta, W0 = draw_seed(cfg, 400, cfg["N"], cfg["N"], TEST, 9)
    rep("d_f = 1 gives Gamma == 1 everywhere",
        np.asarray(coupling(v, F, B)) - 1.0, 1e-12)
    tr, ret, dw, disp = run_stream(v, F, B, Theta, (400,), W0=W0)
    rep("  and r_t == theta_t - theta_{t-1}: W^0 is gone after task 1",
        tr[1:] - np.linalg.norm(Theta[1:] - Theta[:-1], axis=1), 1e-12)
    rep("  the one task that reads it: r_1 == theta_1 - v^T W^0",
        tr[0] - float(np.linalg.norm(Theta[0] - v @ W0)), 1e-12)
    rep("  r.m.s. of it == the mean field, sqrt(2)",
        np.sqrt((tr[1:] ** 2).mean()) / float(mf_resid(1.0, 200, -1, s0)) - 1.0,
        0.04)
    rep("  the geometric mean sits below it by exp(-1/4D)",
        geo(tr[1:]) / (np.sqrt(2.0) * np.exp(-1.0 / (4 * cfg["D"]))) - 1.0, 0.02)

    if verbose:
        print("-- the weight change: the step of eq. (16) is rank one --")
    v, F, B, Theta, W0 = draw_seed(cfg, 120, 24, 12, TEST, 21)
    cw = B @ v ** 2
    tr, ret, dw, disp = run_stream(v, F, B, Theta, (1, 40, 120), W0=W0)
    rep("||W^t-W^{t-1}||_F == ||r_t||/sqrt(c_t), eqs. (15) and (16)",
        dw / (tr / np.sqrt(cw)) - 1.0, 1e-12)
    rep("the first step is ||theta_1 - v^T F_1 W^0||/sqrt(c_1)",
        dw[0] * np.sqrt(cw[0])
        / float(np.linalg.norm(Theta[0] - (F[0] * v) @ W0)) - 1.0, 1e-12)
    rep("the displacement never exceeds the steps that made it",
        max(0.0, (disp[120] - dw.sum()) / dw.sum()), 1e-12)
    rep("mf_step is mf_step_at averaged over the stream",
        mf_step(0.3, 0.1, 50, s0)
        / mf_step_at(0.3, 0.1, np.arange(1, 51), s0).mean() - 1.0, 1e-12)
    G = coupling(v, F, B)
    drive = Theta - initial_readout(v, F, W0)
    R = np.asarray(solve_stream(G, drive))
    rep("the solve is causal: rows <= k of the K-solve are the k-solve",
        _route_gap(np.asarray(solve_stream(G[:40, :40], drive[:40])).ravel(),
                   R[:40].ravel()), 1e-12)
    rep("blocked forward substitution == the row-at-a-time reference",
        _route_gap(R.ravel(), forward_substitution(np.asarray(G), drive).ravel()),
        1e-10)
    rep("read_back == the band eq. (30) the scores are read through",
        _route_gap(np.asarray(read_back(G, R, 40)).ravel(),
                   np.asarray(-(np.triu(np.asarray(G)[:40, :40], 1)
                                @ R[:40])).ravel()), 1e-12)

    if verbose:
        print("-- at d_f = d_b = 1 the sum telescopes and both are exact --")
    v, F, B, Theta, W0 = draw_seed(cfg, 200, cfg["N"], cfg["N"], TEST, 22)
    nv = float(np.linalg.norm(v))
    tr, ret, dw, disp = run_stream(v, F, B, Theta, (100, 200), W0=W0)
    rep("  ||W^t-W^0||_F == ||theta_t - v^T W^0||/||v||, every t",
        [disp[s] * nv / float(np.linalg.norm(Theta[s - 1] - v @ W0)) - 1.0
         for s in (100, 200)], 1e-12)
    rep("  and the step is ||theta_t - theta_{t-1}|| / ||v||",
        dw[1:] * nv - np.linalg.norm(Theta[1:] - Theta[:-1], axis=1), 1e-12)
    rep("  mf_displacement(1, 1, T) == sqrt(1 + s^2) exactly",
        [mf_displacement(1.0, 1.0, T, s0) - np.sqrt(1.0 + s0 ** 2)
         for T in (100, 200)], 1e-12)
    rep("  mf_step(1, 1, T) == the measured step, bar the estimator gap",
        dw.mean() * nv / mf_step(1.0, 1.0, 200, s0) - 1.0, 0.05)
    # W^t = (I - v v^T/||v||^2) W^0 + v theta_t^T/||v||^2, two orthogonal parts:
    # the one direction every task reads is replaced and the rest of W^0 stays put
    mag, mag_x = {}, {}
    run_stream(v, F, B, Theta, (100, 200), W0=W0, mag=mag)
    exact_scores(v, F, B, Theta, (100, 200), W0=W0, mag=mag_x)
    kept = float(np.linalg.norm(W0 - np.outer(v, v @ W0) / nv ** 2))
    rep("  ||W^t||_F^2 == ||P W^0||^2 + ||theta_t||^2/||v||^2, both routes",
        [np.sqrt(kept ** 2 + float(Theta[s - 1] @ Theta[s - 1]) / nv ** 2)
         / m[s] - 1.0 for m in (mag, mag_x) for s in (100, 200)], 1e-12)

    if verbose:
        print("-- the weight-change mean field against its own recursion --")
    rng = rng_for(cfg, TEST, 23)
    for d_f, d_b in ((0.5, 0.5), (0.4, 0.1), (0.2, 0.05)):
        K_, D_, M = 120, 24, 2000
        p = 1.0 - d_f
        Th = mc_drive(rng, M, K_, D_, d_f)
        S = np.zeros((M, D_))
        Rm = np.empty((M, K_, D_))
        for t in range(K_):
            Rm[:, t] = Th[:, t] - d_f * S
            S = p * S + Th[:, t]
        for T in (20, 120):
            want_ = mf_displacement(d_f, d_b, T, s0)
            got = np.sqrt(((1.0 / d_b - 1.0) * (Rm[:, :T] ** 2).sum((1, 2))
                           + (Rm[:, :T].sum(1) ** 2).sum(1)).mean())
            rep(f"  d_f={d_f}, d_b={d_b}, T={T:3d}: ||W^T-W^0|| = {want_:8.3f}",
                got / want_ - 1.0, 0.02)
            want_ = mf_step(d_f, d_b, T, s0)
            got = np.sqrt((Rm[:, :T] ** 2).sum(2).mean(0)).mean() / np.sqrt(d_b)
            rep(f"  d_f={d_f}, d_b={d_b}, T={T:3d}: step       = {want_:8.3f}",
                got / want_ - 1.0, 0.02)

    if verbose:
        print("-- one cell of the grid against a replay of the same seeds --")
    small = {**cfg, "K_DW": 80, "DW_SNAPSHOTS": (10, 80), "SEEDS": 3}
    cell = case_delta_w(small, 0.5, 2.0)
    n_f, n_b = counts(cfg["N"], 0.5, 2.0)
    hand = {10: [], 80: []}
    handd = {10: [], 80: []}
    for sd in range(small["SEEDS"]):
        v, F, B, Theta, W0 = draw_seed(small, 80, n_f, n_b, DELTAW, sd)
        W = W0.copy()
        cw = B @ v ** 2
        tot = 0.0
        for t in range(80):
            r = Theta[t] - (F[t] * v) @ W
            stp = np.outer(B[t] * v, r) / cw[t]
            tot += float(np.linalg.norm(stp))
            W += stp
            if t + 1 in hand:
                hand[t + 1].append(tot / (t + 1))
                handd[t + 1].append(float(np.linalg.norm(W - W0)))
    for T in (10, 80):
        rep(f"  step@{T} == the hand replay", cell[f"step@{T}"] / geo(hand[T]) - 1.0,
            1e-12)
        rep(f"  disp@{T} == the hand replay", cell[f"disp@{T}"] / geo(handd[T]) - 1.0,
            1e-12)

    if verbose:
        print("-- --sim figure 6a: five metrics off the penultimate state --")
    # the cell again with the five read at s = 79, the last state with a task after
    # it: nothing the snapshots carry may move, and each of the five is checked
    # against a walk by hand, at q_t, where figure 6a spends most of its cells
    keys = [f"{m}@{T}{sfx}" for m in ("step", "disp", "step_x", "disp_x")
            for T in (10, 80) for sfx in ("", "_sem", "_n")]
    cell_at = case_delta_w(small, 0.5, 2.0, at=79)
    rep("  reading them at s = 79 moves no snapshot, and not the route gap",
        [cell_at[k] - cell[k] for k in keys] + [cell_at["gap"] - cell["gap"]], 0)
    rep("  and a state with no task after it is refused",
        int(not _raises(lambda: case_delta_w(small, 0.5, 2.0, at=80))), 0)
    at_q = case_delta_w(small, 0.5, 1.0, q=q_t, at=79)
    n_f, n_b = counts(cfg["N"], 0.5, 1.0)
    five = {k: [] for k in ("ret", "tr", "disp", "dw", "mag", "mag0")}
    for sd in range(small["SEEDS"]):
        v, F, B, Theta, W0 = draw_seed(small, 80, n_f, n_b, DELTAW, sd, q=q_t)
        W, cw = W0.copy(), B @ v ** 2
        for t in range(79):
            before = W.copy()
            W = W + np.outer(B[t] * v, Theta[t] - (F[t] * v) @ W) / cw[t]
        # W is W^79: tasks 1..79 read back out of it, task 80 read before it trains
        five["ret"].append(float(np.mean([np.linalg.norm(Theta[i] - (F[i] * v) @ W)
                                          for i in range(79)])))
        five["tr"].append(float(np.linalg.norm(Theta[79] - (F[79] * v) @ W)))
        five["disp"].append(float(np.linalg.norm(W - W0)))
        five["dw"].append(float(np.linalg.norm(W - before)))
        five["mag"].append(float(np.linalg.norm(W)))
        five["mag0"].append(float(np.linalg.norm(W0)))
    for k, what in (("ret", "forgetting over tasks 1..79"),
                    ("tr", "interference, task 80 read out of W^79"),
                    ("disp", "||W^79 - W^0||_F"), ("dw", "||W^79 - W^78||_F"),
                    ("mag", "||W^79||_F")):
        rep(f"  q={q_t:g}: {what} == the hand replay",
            [at_q[f"{k}@79"] / geo(five[k]) - 1.0,
             at_q[f"{k}_x@79"] / geo(five[k]) - 1.0], 1e-9)
    if s0 > 0:
        rep("  and ||W^0||_F, the size before any task, the same in every cell",
            [at_q["mag0"] / geo(five["mag0"]) - 1.0,
             cell_at["mag0"] / at_q["mag0"] - 1.0], 1e-12)

    if verbose:
        print("-- one figure-2/8 case against a replay of the same seeds --")
    # all five arms at once, which pins the one convention figures 2, 3 and 8 rest
    # on: every arm is read at the plotted point and nowhere else -- the per-task
    # pair at task K, the per-state pair at state K after that task has trained --
    # with nothing pooled over the tasks around it.  Once dissimilar, once at q_t.
    small = {**cfg, "K_TASKS": 60, "N_POINTS_DECADE": 2, "SEEDS": 2}
    pts = log_points(small["K_TASKS"], small["N_POINTS_DECADE"])
    n_f, n_b = counts(cfg["N"], 0.3, 1.0)
    arms = ("tr", "dw", "ret", "disp", "mag")
    for qv in (0.0, q_t):
        res = case_vs_tasks(small, 0.3, q=qv)
        hand = {k: [[] for _ in pts] for k in arms}
        start = []
        for sd in range(small["SEEDS"]):
            v, F, B, Theta, W0 = draw_seed(small, small["K_TASKS"], n_f, n_b,
                                           TASKS, sd, q=qv)
            W = W0.copy()
            cw = B @ v ** 2
            tr_, dw_, state = [], [], {}
            for t in range(small["K_TASKS"]):
                r = Theta[t] - (F[t] * v) @ W
                tr_.append(float(np.linalg.norm(r)))
                stp = np.outer(B[t] * v, r) / cw[t]
                dw_.append(float(np.linalg.norm(stp)))
                W += stp
                if t + 1 in set(int(p) for p in pts):
                    state[t + 1] = (
                        float(np.mean([np.linalg.norm(Theta[i] - (F[i] * v) @ W)
                                       for i in range(t + 1)])),
                        float(np.linalg.norm(W - W0)), float(np.linalg.norm(W)))
            start.append(float(np.linalg.norm(W0)))
            for j, p in enumerate(pts):
                hand["tr"][j].append(tr_[int(p) - 1])
                hand["dw"][j].append(dw_[int(p) - 1])
                hand["ret"][j].append(state[int(p)][0])
                hand["disp"][j].append(state[int(p)][1])
                hand["mag"][j].append(state[int(p)][2])
        for k in arms:
            rep(f"  q={qv:g}: {k} == the hand replay, at all {pts.size} points",
                [res[k][j] / geo(hand[k][j]) - 1.0 for j in range(pts.size)], 1e-12)
        rep(f"  q={qv:g}: the magnitude's closed form == its walk, and gap_mag says so",
            [_route_gap(res["mag"], res["mag_x"]), res["gap_mag"]], 1e-9)
        if s0 > 0:
            rep(f"  q={qv:g}: its trace starts from ||W^0||_F at K = 0",
                [res["mag0"] / geo(start) - 1.0,
                 _mag_series(res, "mag")[0][0], _mag_series(res, "mag")[1][0]
                 / res["mag0"] - 1.0], 1e-12)

    if verbose:
        print("-- figures 4 and 5: interference is the one task, not a window --")
    small = {**cfg, "K_DENSITY": 30, "K_HEAT": 30, "SNAPSHOTS": (3, 30),
             "SEEDS": 2}
    n_f, n_b = counts(cfg["N"], 0.3, 2.0)

    def hand_tr(stream, sd, K, qv=0.0):
        """||r_t|| for every task of one seed, walked by hand from W^0."""
        v, F, B, Theta, W0 = draw_seed(small, K, n_f, n_b, stream, sd, q=qv)
        W, cw, out = W0.copy(), B @ v ** 2, []
        for t in range(K):
            r = Theta[t] - (F[t] * v) @ W
            out.append(float(np.linalg.norm(r)))
            W += np.outer(B[t] * v, r) / cw[t]
        return out

    for qv in (0.0, q_t):
        want = geo([hand_tr(DENSITY, sd, 30, qv)[29]
                    for sd in range(small["SEEDS"])])
        rep(f"  q={qv:g}: fig 4's interference is ||r_K|| of the last task alone",
            case_vs_density(small, 0.3, 2.0, q=qv)["tr"] / want - 1.0, 1e-12)
        heat = case_heat(small, 0.3, 2.0, q=qv)[0]
        for k in (3, 30):
            want = geo([hand_tr(HEAT, sd, 30, qv)[k - 1]
                        for sd in range(small["SEEDS"])])
            rep(f"  q={qv:g}: fig 5's interference at k={k:2d} is task k alone",
                heat[k]["tr"] / want - 1.0, 1e-12)

    if verbose:
        print(f"-- task similarity, at q_t = {q_t:g} --")
    D_ = cfg["D"]
    v, F, B, Theta, W0 = draw_seed(cfg, 40, 20, 10, TEST, 5)
    vq, Fq, Bq, Thq, W0q = draw_seed(cfg, 40, 20, 10, TEST, 5, q=q_t)
    th0 = shared_teacher(D_, rng_for(cfg, TEST, 5, THETA0_PART))
    rep("q moves nothing but the teachers: v, gates and W^0 as at q = 0",
        [float(np.abs(a - b).max()) for a, b in ((v, vq), (F, Fq), (B, Bq),
                                                 (W0, W0q))], 0)
    rep("theta_t == sqrt(q) theta_0 + sqrt(1-q) thetatilde_t, thetatilde = draw's",
        float(np.abs(Thq - (np.sqrt(q_t) * th0[None, :]
                            + np.sqrt(1.0 - q_t) * Theta)).max()), 0)
    Th2 = draw_seed(cfg, 80, 60, 20, TEST, 5, q=q_t)[3]
    Tt2 = draw_seed(cfg, 80, 60, 20, TEST, 5)[3]
    rep("theta_0 is one draw per seed index, whatever K, n_f and n_b",
        float(np.abs((Th2 - np.sqrt(1.0 - q_t) * Tt2) / np.sqrt(q_t)
                     - th0[None, :]).max()), 1e-12)
    rep("  and the shared teacher's stream is none of the others'",
        int(np.allclose(rng_for(cfg, TEST, 30, THETA0_PART).standard_normal(8),
                        rng_for(cfg, TEST, 30, W0_PART).standard_normal(8))), 0)
    rng = rng_for(cfg, TEST, 52)
    M, Kq = 4000, 6
    Tt = rng.standard_normal((M, Kq, D_))
    Tt /= np.linalg.norm(Tt, axis=2, keepdims=True)
    t0 = rng.standard_normal((M, D_))
    t0 /= np.linalg.norm(t0, axis=1, keepdims=True)
    Tq = np.stack([similar_teachers(Tt[m], t0[m], q_t) for m in range(M)])
    G_ = np.einsum("mtd,mud->mtu", Tq, Tq)
    off = ~np.eye(Kq, dtype=bool)
    rep("E||theta_t||^2 == 1: the scale the scores are read on stays put",
        float(np.einsum("mtt->", G_)) / (M * Kq) - 1.0, 0.01)
    rep(f"E[theta_t . theta_u] == q = {q_t:g} for t != u",
        float(G_[:, off].mean()) - q_t, 0.01)
    rep("q = 0 hands the teachers back untouched, and q outside [0, 1] is refused",
        [float(np.abs(similar_teachers(Tt[0], t0[0], 0.0) - Tt[0]).max()),
         int(not _raises(lambda: similar_teachers(Tt[0], t0[0], 1.2)))], 0)
    for d_f, rho in ((0.5, 1.0), (0.3, 2.0)):
        rep(f"d_f={d_f}, rho={rho}: streamed scores == eq. (27)/(30) at q_t",
            solve_check(cfg, d_f, rho, q=q_t), 1e-9)

    rng = rng_for(cfg, TEST, 53)
    for d_f in (0.5, 0.2):
        K, D, M = 150, 32, 3000
        p = 1.0 - d_f
        Th = mc_drive(rng, M, K, D, d_f, q=q_t)
        S = np.zeros((M, D))
        Ss = np.empty((M, K, D))
        Rm = np.empty((M, K, D))
        for t in range(K):
            Rm[:, t] = Th[:, t] - d_f * S
            S = p * S + Th[:, t]
            Ss[:, t] = S
        for Delta in (-1, 6):
            for t in (1, 3, 40):
                if Delta == -1:
                    got = (Rm[:, t - 1] ** 2).sum(1).mean()
                else:
                    got = (d_f ** 2 * ((Ss[:, t + Delta - 1]
                                        - Ss[:, t - 1]) ** 2).sum(1)).mean()
                want = float(mf_resid(d_f, t, Delta, s0, q_t))
                rep(f"  d_f={d_f}, Delta={Delta:3d}, t={t:3d}: ||r|| = {want:.4f}",
                    np.sqrt(got) / want - 1.0, 0.02)
        per = np.sqrt((d_f ** 2 * ((Ss[:, 119][:, None, :]
                                    - Ss[:, :120]) ** 2).sum(2)).mean(0))
        rep(f"  d_f={d_f}, forgetting(120) = {mf_retention(d_f, 120, s0, q_t):.4f}",
            per.mean() / mf_retention(d_f, 120, s0, q_t) - 1.0, 0.02)
        d_b = d_f / 4
        for T in (20, 120):
            want_ = mf_displacement(d_f, d_b, T, s0, q_t)
            got = np.sqrt(((1.0 / d_b - 1.0) * (Rm[:, :T] ** 2).sum((1, 2))
                           + (Rm[:, :T].sum(1) ** 2).sum(1)).mean())
            rep(f"  d_f={d_f}, d_b={d_b:g}, T={T:3d}: ||W^T-W^0|| = {want_:7.3f}",
                got / want_ - 1.0, 0.02)
        want_ = mf_step(d_f, d_b, 120, s0, q_t)
        got = np.sqrt((Rm[:, :120] ** 2).sum(2).mean(0)).mean() / np.sqrt(d_b)
        rep(f"  d_f={d_f}, d_b={d_b:g}, T=120: step = {want_:.3f}",
            got / want_ - 1.0, 0.02)

    v, F, B, Theta, W0 = draw_seed(cfg, 400, cfg["N"], cfg["N"], TEST, 9, q=q_t)
    Tt = draw_seed(cfg, 400, cfg["N"], cfg["N"], TEST, 9)[3]
    tr, ret, dw, disp = run_stream(v, F, B, Theta, (400,), W0=W0)
    rep("d_f = 1: r_t == sqrt(1-q)(thetatilde_t - thetatilde_{t-1}), theta_0 gone",
        tr[1:] - np.sqrt(1.0 - q_t) * np.linalg.norm(Tt[1:] - Tt[:-1], axis=1),
        1e-12)
    rep("  r.m.s. of it == the mean field, sqrt(2(1-q))",
        np.sqrt((tr[1:] ** 2).mean()) / float(mf_resid(1.0, 200, -1, s0, q_t))
        - 1.0, 0.04)

    fl = {**cfg, "PER_TASK": 60}
    both = flow_case(fl, q=q_t, K=30, show=4)
    every = flow_case(fl, q=q_t, K=30)
    rep("figure 1 over 30 tasks at q_t: integrated residuals == direct solve",
        both["box_gap"], 1e-7)
    rep("  the protocol == direct solve, every task", both["solve_gap"], 1e-12)
    rep("  only the two ends are recorded, four tasks each",
        [both["loss_int"].shape[0] - 8, int(both["shown"][3]) - 3,
         int(both["shown"][4]) - 26], 0)
    last = both["shown"][both["shown"] >= 26]
    rep("  the last window's worst task, to its own residual",
        _flow_end_gaps(both, last)[0], 1e-7)
    rep("  the coarse step between the ends moves the last residuals by < 1e-10",
        float(np.abs(both["R_int"][26:] - every["R_int"][26:]).max()
              / np.abs(every["R_int"][26:]).max()), 1e-10)
    lum = [0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
           for c in (tuple(int(h[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
                     for h in Q_RAMP)]
    rep("the q ramp runs light to dark, one step per q, and refuses a sixth",
        [int(not all(np.diff(lum) < 0)),
         int(list(q_colours((0.0, 0.3, 0.5, 0.7, 0.9)).values()) != list(Q_RAMP)),
         int(not _raises(lambda: q_colours(range(6))))], 0)

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
    rep("  and the initial state's stream is not the draw's",
        int(np.allclose(rng_for(cfg, TEST, 30).standard_normal(8),
                        rng_for(cfg, TEST, 30, W0_PART).standard_normal(8))), 0)
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
    """An inclusive grid anchored at `hi`, so that d_f = 1 is always on it.

    Anchored at the top and not at the bottom.  The lower end of the density grids
    is D/N, which is not in general a whole number of steps below 1, and d_f = 1 is
    the calibration point several captions in this file lean on: the read gate is
    the identity there and Gamma is exactly all-ones, so the split cannot touch the
    residual.  Anchoring at the bottom drops it whenever 1 - D/N is not a
    multiple of the step, and worse, walks past 1.0 -- at N = 200, D = 12 the old
    form ended at 1.01, where `counts` raises and the cell was silently swallowed by
    the caller's `except ValueError`.  It happened to be right at N = 60, where
    D/N = 0.2 sits a whole 16 steps below 1.

    0.2..1.0 by 0.1 still has exactly nine points, and every grid this file built at
    N = 60 or N = 120 is unchanged.
    """
    n = int(np.floor((hi - lo) / step + 1e-9)) + 1
    return [round(hi - i * step, 9) for i in range(n)][::-1]


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
    cfg = CONFIG
    sim = "--sim" in argv
    here = Path(__file__).resolve().parent / (
        (cfg["SIM_OUTDIR"] if sim else cfg["OUTDIR"]) or "")
    want = {a for a in argv if a in {"1", "2", "3", "4", "5", "6", "7", "8"}}

    print("=" * 76)
    print("  neuronal split gating -- experiments and figures"
          + ("  (task similarity)" if sim else ""))
    print(f"  N={cfg['N']}  N_in={cfg['D']}  seeds={cfg['SEEDS']}"
          f"  root seed={cfg['ROOT_SEED']}  W0 scale={cfg['W0_SCALE']:g}")
    if sim:
        print(f"  q in {tuple(cfg['SIM_QS'])}")
    if "--jax" in argv:
        print(f"  the coupling and the solve on {use_jax(True)}")
    print("=" * 76)
    if self_test(cfg) and "--test" not in argv:
        print("\n  self-test failed; not drawing anything")
        return 1
    if "--test" in argv:
        return 0
    here.mkdir(parents=True, exist_ok=True)
    if sim:
        return main_sim(cfg, here, want)

    gap = solve_check(cfg)
    lines = [f"initial state W^0: iid N(0, s^2/N_in), s = W0_SCALE = "
             f"{cfg['W0_SCALE']:g}"
             + ("  -- the zero start" if cfg["W0_SCALE"] == 0 else ""),
             f"stream vs the exact solve (27) at K={cfg['K_CHECK']}: {gap:.3e}",
             "scores are the RAW residual, ||r||, with no normalising and no squaring.",
             "  interference(K) = ||theta_K - thetahat_K^(K-1)||, task K's own: nothing",
             "                    is pooled over the tasks around it",
             "  forgetting(K)   = (1/K) sum_{i<=K} ||theta_i - thetahat_i^(K)||",
             "  both are named for what a large value means: higher is worse.",
             "the weight change, figures 6-8, is the step of eq. (16) and its sum:",
             "  step(T)         = (1/T) sum_{t<=T} ||W^t - W^{t-1}||_F = "
             "(1/T) sum ||r_t||/sqrt(c_t)",
             "  disp(T)         = ||W^T - W^0||_F, the same increments summed first:",
             "                    how far the state has moved from W^0, not divided by T",
             "0 = solved, 1 = no better than W = 0.  seeds combined geometrically,",
             "with +-1 s.e.m. of that mean, in logs -- the figures draw exp(+-sem).",
             "the eq.27/30 columns are the same numbers from the closed form, with",
             "nothing trained; 'route gap' is the largest relative disagreement.",
             ""]
    head = (f"{'forgetting':>14s} {'interference':>14s}"
            f"{'forget eq27/30':>15s} {'interf eq27/30':>15s} {'route gap':>11s}")
    def rows(sub_cases, keys, arms):
        """One table: the two arms at the end of the stream, for each trace."""
        a, b = arms[0][0], arms[1][0]
        out = []
        for key, lab in keys:
            r = sub_cases[key]
            out.append(f"  {lab:>9s} {r['n_b']:4d} "
                       f"{r[a][-1]:14.4g} {r[b][-1]:14.4g}"
                       f"{r[a + '_x'][-1]:15.4g} {r[b + '_x'][-1]:15.4g}"
                       f"{r['gap']:11.1e}")
        return out

    head_w = (f"{'dW':>12s} {'disp':>12s} {'|W|':>12s}"
               f"{'dW, eq.27':>13s} {'disp, eq.27':>13s} {'|W|, eq.27':>13s} "
               f"{'|W^0|':>9s}{'route gap':>11s}")

    def rows_w(sub_cases, keys):
        """Figure 8's table: the three weight metrics at the end, and ||W^0||_F."""
        out = []
        for key, lab in keys:
            r = sub_cases[key]
            out.append(f"  {lab:>9s} {r['n_b']:4d} "
                       + " ".join(f"{r[a][-1]:12.4g}" for a in ("dw", "disp", "mag"))
                       + f"{r['dw_x'][-1]:13.4g} {r['disp_x'][-1]:13.4g} "
                       f"{r['mag_x'][-1]:13.4g} {r['mag0']:9.4g}"
                       f"{max(r['gap'], r['gap_mag']):11.1e}")
        return out

    if not want or "1" in want:
        print("\nfigure 1: the flow over the whole stream, drawn at its two ends ...")
        res = flow_case(cfg, K=cfg["K_FLOW"], show=cfg["FLOW_SHOW"])
        (_, ends), = fig_flow(cfg, [res], here / "fig1_flow.png", cfg["FLOW_SHOW"])
        lines += [f"fig1  n_f={res['n_f']} n_b={res['n_b']}  K={res['K']}, the first "
                  f"and last {cfg['FLOW_SHOW']} tasks drawn.  over the whole stream: "
                  f"integrated residuals vs eq. (27) {res['box_gap']:.2e}  "
                  f"stream vs eq. (27) {res['solve_gap']:.2e}",
                  f"  worst task of each end, relative to its own residual; the "
                  f"drawn tasks' flow vs the closed form {res['flow_gap']:.2e} of "
                  "the tallest",
                  f"  {'end':>6s} {'integrated vs (27)':>19s} "
                  f"{'protocol vs (27)':>17s} {'flow vs closed':>15s}"]
        lines += [f"  {end:>6s} {g[0]:19.2e} {g[1]:17.2e} {g[2]:15.2e}"
                  for end, g in zip(("first", "last"), ends) if g is not None]
        lines.append("")

    # figures 2a/8a share one sweep and figures 2b/3/8b share another, so each
    # case is walked and solved once however many of the four are asked for
    dens, split = {}, {}
    if not want or (want & {"2", "8"}):
        print("the density sweep, d_f = d_b (figures 2a and 8a) ...")
        dens = {d: case_vs_tasks(cfg, d) for d in cfg["DENSITIES"]}
    if not want or (want & {"2", "3", "8"}):
        print("the split sweep (figures 2b, 3 and 8b) ...")
        for d in cfg["SPLIT_DENSITIES"]:
            for rho in cfg["SPLIT_RHOS"]:
                try:
                    split[(d, rho)] = case_vs_tasks(cfg, d, rho, stream=SPLIT)
                except ValueError:
                    pass

    if not want or "2" in want:
        print("figures 2a and 2b: the residual against task index ...")
        t_d, t_s = density_traces(cfg, dens), split_traces(cfg, split)
        fig_vs_tasks(cfg, dens, here / "fig2a_vs_tasks.png", ARMS, t_d,
                     "the raw residual against task index, one trace per density"
                     ";  0 = solved, 1 = no better than $W=0$", "")
        fig_vs_tasks(cfg, split, here / "fig2b_vs_split.png", ARMS, t_s,
                     f"the same at $d_f={cfg['FIG2B_DF']:g}$, one trace per split "
                     "ratio;  1 = no better than $W=0$", "",
                     cap=(cfg["FIG3_FLOOR"], cfg["FIG3_CEIL"]))
        lines.append(f"fig2  at the end of the stream (K={cfg['K_TASKS']})")
        lines.append("  2a, one trace per density (d_f = d_b)")
        lines.append(f"  {'d_f':>9s} {'n_b':>4s} " + head)
        lines += rows(dens, [(t[0], f"{t[3]:g}") for t in t_d], ARMS)
        lines.append(f"  2b, one trace per split ratio at d_f={cfg['FIG2B_DF']:g}")
        lines.append(f"  {'rho':>9s} {'n_b':>4s} " + head)
        lines += rows(split, [(t[0], f"{t[0][1]:g}") for t in t_s], ARMS)
        lines.append("")

    if not want or "8" in want:
        print("figures 8a and 8b: the weights against task index ...")
        t_d, t_s = density_traces(cfg, dens), split_traces(cfg, split)
        fig_vs_tasks(cfg, dens, here / "fig8a_weight_vs_tasks.png", ARMS_W, t_d,
                     "update magnitude, weight displacement and weight magnitude, "
                     "one trace per density", NOTE_MAG, ylog=True)
        fig_vs_tasks(cfg, split, here / "fig8b_weight_vs_split.png", ARMS_W, t_s,
                     f"the same at $d_f={cfg['FIG2B_DF']:g}$, one trace per split "
                     "ratio", NOTE_MAG,
                     cap=(cfg["FIG8_FLOOR"], cfg["FIG8_CEIL"]))
        lines.append(f"fig8  at the end of the stream (K={cfg['K_TASKS']}).  "
                     "dW = ||W^K - W^{K-1}||_F, the step task K took;")
        lines.append("      disp = ||W^K - W^0||_F, how far the state has moved "
                     "from W^0 once it trained;  |W| = ||W^K||_F,")
        lines.append("      the weights themselves, whose trace starts at K = 0 from "
                     "|W^0| = ||W^0||_F (unless W^0 = 0).  the route gap covers "
                     "all three.")
        lines.append("  8a, one trace per density (d_f = d_b)")
        lines.append(f"  {'d_f':>9s} {'n_b':>4s} " + head_w)
        lines += rows_w(dens, [(t[0], f"{t[3]:g}") for t in t_d])
        lines.append(f"  8b, one trace per split ratio at d_f={cfg['FIG2B_DF']:g}")
        lines.append(f"  {'rho':>9s} {'n_b':>4s} " + head_w)
        lines += rows_w(split, [(t[0], f"{t[0][1]:g}") for t in t_s])
        lines.append("")

    if not want or "3" in want:
        print("figure 3: against task index, one trace per split ratio ...")
        cases = split
        fig_vs_tasks_split(cfg, cases, here / "fig3_vs_tasks_split.png")
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
        fig_vs_density(cfg, grid, here / "fig4_vs_density.png")
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
                ("7", "disp", "fig7_displacement.png",
                 "net displacement ||W^T - W^0||_F")):
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
