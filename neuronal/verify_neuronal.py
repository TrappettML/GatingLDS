"""Independent numerical check of the neuronal split-gating note.

Nothing is imported from any other file in this repository.  The model is
written down once, in `flow_rhs`, exactly as the note states it:

    read gate   F_t = diag(f^(t)_1, ..., f^(t)_N),  f in {0,1}
    write gate  B_t = diag(b^(t)_1, ..., b^(t)_N),  b in {0,1},  b_i <= f_i
    student     yhat_t(x) = v^T F_t W x / sqrt(D)
    flow        Wdot = -B_t v (v^T F_t W - theta_t)

Every closed form in the note is then checked against one of three things that
do not use it: a fourth-order integration of that flow, a replay of the
protocol from W = 0, or a Monte Carlo average.

    python3 verify_neuronal.py          # checks + the two measured tables
    python3 verify_neuronal.py --quick  # checks only
"""

import sys
import numpy as np

TOL = 1e-11


# ----------------------------------------------------------------- the draw

def draw(N, D, K, n_f, n_b, seed=0, frozen=False):
    """Readout, nested per-task neuron gates, teachers.

    One permutation of the N neurons per task: the first n_f places are read,
    the first n_b written, so B_t sits inside F_t by construction.  The gate is
    a property of the neuron, so all D input coordinates share it -- that is
    what makes the gating neuronal.  `frozen` reuses one draw for every task.
    """
    if not 1 <= n_b <= n_f <= N:
        raise ValueError(f"need 1 <= n_b <= n_f <= N, got {n_b}, {n_f}, {N}")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(N) / np.sqrt(N)            # eq. (1)
    draws = 1 if frozen else K
    rank = np.argsort(np.argsort(rng.random((draws, N)), axis=-1), axis=-1)
    f = (rank < n_f).astype(float)
    b = (rank < n_b).astype(float)
    if frozen:
        f = np.repeat(f, K, axis=0)
        b = np.repeat(b, K, axis=0)
    Theta = rng.standard_normal((K, D))
    Theta /= np.linalg.norm(Theta, axis=1, keepdims=True)
    return v, f, b, Theta


def masses(v, f, b):
    """Write mass c_t = v^T B_t v and read mass m_t = v^T F_t v, per task."""
    w = v ** 2
    return b @ w, f @ w


# ------------------------------------------------- the model, stated once

def flow_rhs(W, v, f_t, b_t, theta_t):
    """Wdot = -B_t v (v^T F_t W - theta_t).  The only statement of the model."""
    return -np.outer(b_t * v, (f_t * v) @ W - theta_t)


def integrate_task(W0, v, f_t, b_t, theta_t, decay=40.0, steps=1200):
    """Run the flow to (numerical) convergence with fourth-order steps."""
    c = float((b_t * v) @ v)
    h = decay / c / steps
    W = W0.copy()
    for _ in range(steps):
        k1 = flow_rhs(W, v, f_t, b_t, theta_t)
        k2 = flow_rhs(W + 0.5 * h * k1, v, f_t, b_t, theta_t)
        k3 = flow_rhs(W + 0.5 * h * k2, v, f_t, b_t, theta_t)
        k4 = flow_rhs(W + h * k3, v, f_t, b_t, theta_t)
        W = W + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return W


def replay(v, f, b, Theta):
    """The protocol of the note, run from W = 0 with the closed-form endpoint.

    Returns every state W^0..W^K and the inherited residuals r_t.
    """
    K, N = f.shape
    D = Theta.shape[1]
    W = np.zeros((N, D))
    states = [W.copy()]
    R = np.empty((K, D))
    for t in range(K):
        r = Theta[t] - (f[t] * v) @ W                       # eq. (14)
        c = float((b[t] * v) @ v)                           # eq. (17)
        W = W + np.outer(b[t] * v, r) / c                   # eq. (18)
        states.append(W.copy())
        R[t] = r
    return np.array(states), R


# ------------------------------------------------------------- the algebra

def coupling(v, f, b):
    """Gamma_kt = v^T F_k B_t v / v^T B_t v, a single K x K matrix."""
    w = v ** 2
    num = f @ (w[:, None] * b.T)        # num[k, t] = sum_i f_ki w_i b_ti
    return num / (b @ w)[None, :]


def stream(Gamma, Theta):
    """R = (I + L)^{-1} Theta by forward substitution, L = strict lower Gamma."""
    K, D = Theta.shape
    L = np.tril(Gamma, -1)
    R = np.empty((K, D))
    for t in range(K):
        R[t] = Theta[t] - L[t, :t] @ R[:t]
    return R


def lagged(Gamma, R, Delta):
    """R^(Delta) = -U(Delta) R, U(Delta) the first Delta superdiagonals."""
    K = R.shape[0]
    U = np.triu(Gamma, 1) - np.triu(Gamma, Delta + 1)
    del K
    return -U @ R


def eps(Rlag, Theta):
    """Loss ratio against the null state W = 0, per task."""
    return (Rlag ** 2).sum(1) / (Theta ** 2).sum(1)


# --------------------------------------------------------- the homogeneous map

def growth_rate(v, f, b, burn=200, x0_seed=1):
    """Top growth rate per task of prod_t (I - Pi_t), by a renormalised walk."""
    K, N = f.shape
    w = v ** 2
    x = np.random.default_rng(x0_seed).standard_normal(N)
    x /= np.linalg.norm(x)
    burn = 0 if burn >= K else burn
    total, used = 0.0, 0
    for t in range(K):
        a, z = b[t] * v, f[t] * v
        x = x - a * ((z @ x) / (w @ b[t]))         # x <- (I - Pi_t) x
        n = np.linalg.norm(x)
        if n == 0.0:
            return -np.inf
        if t >= burn:
            total += np.log(n)
            used += 1
        x /= n
    return total / used


# ================================================================== checks

class Report:
    def __init__(self):
        self.fails = 0
        self.n = 0

    def __call__(self, name, got, tol=TOL, want=0.0):
        self.n += 1
        ok = np.all(np.abs(np.asarray(got) - want) <= tol)
        self.fails += not ok
        flag = "ok  " if ok else "FAIL"
        worst = float(np.max(np.abs(np.asarray(got) - want)))
        print(f"  [{flag}] {name:<62s} {worst:.3e}")

    def close(self):
        print(f"\n  {self.n - self.fails}/{self.n} checks passed")
        return self.fails


def checks(rep):
    N, D, K = 40, 7, 60
    n_f, n_b = 20, 8
    v, f, b, Theta = draw(N, D, K, n_f, n_b, seed=3)
    c, m = masses(v, f, b)
    states, R = replay(v, f, b, Theta)

    print("\n-- the within-task flow, eqs. (7)-(18) --")
    # the rank-one ansatz and the endpoint, against an integration of eq. (7)
    err_end, err_ans, err_sol, err_alpha = [], [], [], []
    for t in [0, 1, 17, K - 1]:
        W0 = states[t]
        Wf = integrate_task(W0, v, f[t], b[t], Theta[t])
        err_end.append(np.abs(Wf - states[t + 1]).max())
        # trajectory at a finite time: W(s) - W^{t-1} = B v u(s), u = (r/c)(1-e^{-cs})
        s = 0.7 / c[t]
        Ws = integrate_task(W0, v, f[t], b[t], Theta[t], decay=0.7, steps=900)
        r = Theta[t] - (f[t] * v) @ W0
        u = (r / c[t]) * (1.0 - np.exp(-c[t] * s))
        err_ans.append(np.abs(Ws - W0 - np.outer(b[t] * v, u)).max())
        err_alpha.append(abs((1.0 - np.exp(-c[t] * s)) - 0.5034146962085906))
        err_sol.append(np.abs((f[t] * v) @ states[t + 1] - Theta[t]).max())
    rep("endpoint of the integrated flow == eq. (18)", err_end, tol=1e-9)
    rep("trajectory == W^{t-1} + B_t v u(s), eq. (15)", err_ans, tol=1e-9)
    rep("alpha_t = 1 - exp(-c_t s) at s = 0.7/c_t", err_alpha, tol=1e-12)
    rep("task solved at its endpoint: v^T F_t W^t = theta_t", err_sol)
    rep("nesting B_t F_t = B_t", (b * f - b))
    rep("write mass c_t > 0", -np.minimum(c, 0.0))

    print("\n-- the closed linear map, eqs. (19)-(20) --")
    err_lds, err_idem, err_tr, err_fro, err_op, err_sec = [], [], [], [], [], []
    for t in [0, 5, 31, K - 1]:
        Pi = np.outer(b[t] * v, f[t] * v) / c[t]
        lhs = (np.eye(N) - Pi) @ states[t] + np.outer(b[t] * v, Theta[t]) / c[t]
        err_lds.append(np.abs(lhs - states[t + 1]).max())
        err_idem.append(np.abs(Pi @ Pi - Pi).max())
        err_tr.append(abs(np.trace(Pi) - 1.0))
        rho_t = m[t] / c[t]
        err_fro.append(abs((Pi ** 2).sum() - rho_t))
        sv = np.linalg.svd(Pi, compute_uv=False)
        err_op.append(abs(sv[0] - np.sqrt(rho_t)))
        err_op.append(abs(np.linalg.norm(np.eye(N) - Pi, 2) - np.sqrt(rho_t)))
        cosb = ((b[t] * v) @ (f[t] * v)) / (
            np.linalg.norm(b[t] * v) * np.linalg.norm(f[t] * v))
        err_sec.append(abs(1.0 / cosb ** 2 - rho_t))
    rep("W^t = (I - Pi_t) W^{t-1} + B_t v theta_t / c_t", err_lds)
    rep("Pi_t^2 = Pi_t", err_idem)
    rep("tr Pi_t = 1", err_tr)
    rep("||Pi_t||_F^2 = rho_t = m_t / c_t", err_fro)
    rep("||Pi_t||_2 = ||I - Pi_t||_2 = sqrt(rho_t)", err_op)
    rep("rho_t = sec^2(angle between B_t v and F_t v)", err_sec)
    rep("rho_t >= 1", -np.minimum(m / c - 1.0, 0.0))

    # rho = 1 forces B = F and an orthogonal projection
    v1, f1, b1, T1 = draw(N, D, 12, 14, 14, seed=5)
    Pi1 = np.outer(b1[0] * v1, f1[0] * v1) / float((b1[0] * v1) @ v1)
    rep("rho = 1: Pi symmetric, ||I - Pi||_2 = 1",
        [np.abs(Pi1 - Pi1.T).max(), abs(np.linalg.norm(np.eye(N) - Pi1, 2) - 1.0)])

    print("\n-- the coupling, eqs. (21)-(22) --")
    Gamma = coupling(v, f, b)
    err_cpl = []
    for t in range(K):
        d_hat = (f * v) @ (states[t + 1] - states[t])      # all k at once
        err_cpl.append(np.abs(d_hat - np.outer(Gamma[:, t], R[t])).max())
    rep("hat theta^(t)_k - hat theta^(t-1)_k = Gamma_kt r_t, all k", err_cpl)
    rep("Gamma_tt = 1", np.diag(Gamma) - 1.0)
    rep("Gamma in [0, 1]", [-min(Gamma.min(), 0.0), max(Gamma.max() - 1.0, 0.0)])
    G2 = coupling(v, f, b)
    rep("Gamma is one K x K array (no input-coordinate index)",
        float(Gamma.shape != (K, K)) + np.abs(Gamma - G2).max())

    print("\n-- the whole stream, eqs. (23)-(29) --")
    Rs = stream(Gamma, Theta)
    rep("triangular solve == replayed residuals", np.abs(Rs - R).max())
    L = np.tril(Gamma, -1)
    rep("(I + L) R = Theta", np.abs((np.eye(K) + L) @ R - Theta).max())
    rep("det(I + L) = 1", abs(np.linalg.det(np.eye(K) + L) - 1.0))
    rep("L^K = 0", np.abs(np.linalg.matrix_power(L, K)).max())
    powers = [np.linalg.matrix_power(-L, j) for j in range(K)]
    scale = max(np.abs(P).max() for P in powers)      # the series cancels; the
    neu = sum(powers)                                 # check has to be relative
    rep("(I + L)^{-1} = sum_{j<K} (-L)^j  (relative to the largest term)",
        np.abs(neu @ (np.eye(K) + L) - np.eye(K)).max() / scale, tol=1e-14)
    acc = np.array([np.cumsum(Gamma[k] * R.T, axis=1).T for k in range(3)])
    for k in range(3):
        rep(f"hat theta^(t)_{k} = sum_{{s<=t}} Gamma_ks r_s",
            np.abs(acc[k] - np.array([(f[k] * v) @ states[t + 1] for t in range(K)])).max())

    print("\n-- retention and the lag curve, eqs. (30)-(34) --")
    err_lag, err_rec = [], []
    prev = np.zeros((K, D))
    for Delta in range(0, 9):
        Rl = lagged(Gamma, R, Delta)
        direct = np.array([Theta[t] - (f[t] * v) @ states[min(t + Delta, K - 1) + 1]
                           for t in range(K - Delta)])
        err_lag.append(np.abs(Rl[:K - Delta] - direct).max())
        Dd = np.diag(np.diag(Gamma, Delta), Delta) if Delta > 0 else np.zeros((K, K))
        err_rec.append(np.abs(Rl - (prev - Dd @ R)).max())
        prev = Rl
    rep("R^(Delta) = -U(Delta) R == replayed protocol", err_lag)
    rep("R^(Delta) = R^(Delta-1) - D_Delta R", err_rec)
    rep("R^(0) = 0", np.abs(lagged(Gamma, R, 0)).max())
    rep("Gamma = L + I + U(K-1)",
        np.abs(Gamma - (L + np.eye(K) + np.triu(Gamma, 1))).max())
    rep("eps^(0) = 0", eps(lagged(Gamma, R, 0), Theta))
    rep("eps^(-1) = ||r_t||^2 / ||theta_t||^2",
        np.abs(eps(R, Theta) - (R ** 2).sum(1) / (Theta ** 2).sum(1)).max())
    rep("eps_t(W=0) = 1", np.abs(eps(Theta, Theta) - 1.0).max())

    Delta = 5
    S = -np.triu(Gamma, 1) + np.triu(Gamma, Delta + 1)
    S = S @ np.linalg.inv(np.eye(K) + L)
    band = np.array([[S[t, p] for p in range(t + Delta + 1, K)] for t in range(K - Delta - 1)],
                    dtype=object)
    rep("S(Delta)_{tp} = 0 for p > t + Delta",
        max((np.abs(np.array(row, dtype=float)).max() if len(row) else 0.0) for row in band))
    rep("S(Delta)_{t,t+Delta} = -Gamma_{t,t+Delta}",
        np.abs(np.diag(S, Delta) + np.diag(Gamma, Delta)).max())
    rep("last row of S(Delta) is zero", np.abs(S[-1]).max())

    print("\n-- the averages, eqs. (22), (37), (44) --")
    # E[Gamma_kt] = d_f for independently drawn masks, k != t
    Ns, reps_ = 60, 4000
    acc_g = []
    for s in range(reps_):
        vv, ff, bb, _ = draw(Ns, 1, 2, 30, 12, seed=1000 + s)
        acc_g.append(coupling(vv, ff, bb)[0, 1])
    rep("E[Gamma_kt] = d_f = 0.5 (independent gates)",
        abs(np.mean(acc_g) - 0.5), tol=4 * np.std(acc_g) / np.sqrt(reps_))

    # E[rho_t] = (n_f - 2) / (n_b - 2)
    for (Nv, nf, nb) in [(60, 30, 12), (120, 60, 20), (240, 120, 61)]:
        rng = np.random.default_rng(7)
        g = rng.standard_normal((40000, Nv)) ** 2
        idx = np.argsort(rng.random((40000, Nv)), axis=1)
        fm = np.zeros((40000, Nv)); bm = np.zeros((40000, Nv))
        np.put_along_axis(fm, idx[:, :nf], 1.0, axis=1)
        np.put_along_axis(bm, idx[:, :nb], 1.0, axis=1)
        rho = (fm * g).sum(1) / (bm * g).sum(1)
        pred = (nf - 2) / (nb - 2)
        se = np.std(rho) / np.sqrt(40000)
        rep(f"E[rho_t] = (n_f-2)/(n_b-2) = {pred:.4f}  [n_f={nf}, n_b={nb}]",
            abs(rho.mean() - pred), tol=4 * se)
        p, q = nf - nb, nb
        var = p * (p + 2) / ((q - 2) * (q - 4)) - p ** 2 / (q - 2) ** 2
        rep(f"  var[rho_t] = p(p+2)/((q-2)(q-4)) - p^2/(q-2)^2 = {var:.4f}",
            abs(rho.var() / var - 1.0), tol=0.06)
        rep(f"  E[rho_t] >= rho = {nf/nb:.4f}", -min(pred - nf / nb, 0.0))

    # one step on an error pointing nowhere in particular
    Nx, trials = 50, 200000
    vv, ff, bb, _ = draw(Nx, 1, 1, 25, 10, seed=11)
    cc, mm = masses(vv, ff, bb)
    Pi = np.outer(bb[0] * vv, ff[0] * vv) / cc[0]
    rng = np.random.default_rng(2)
    X = rng.standard_normal((Nx, trials))
    Y = X - Pi @ X
    ratio = (Y ** 2).sum() / (X ** 2).sum()
    pred = 1.0 + (mm[0] / cc[0] - 2.0) / Nx
    rep(f"E||(I-Pi)x||^2 / E||x||^2 = 1 + (rho_t - 2)/N = {pred:.6f}",
        abs(ratio - pred), tol=3e-3)
    rep("  same ratio in the trace form (N - 2 tr Pi + ||Pi||_F^2)/N",
        abs(np.trace((np.eye(Nx) - Pi).T @ (np.eye(Nx) - Pi)) / Nx - pred))
    # the same statement for the full N x D state, which is what the map acts on
    Xm = rng.standard_normal((Nx, 9, 20000))
    Ym = np.einsum("ij,jkl->ikl", np.eye(Nx) - Pi, Xm)
    rep("  and for the whole N x D state, all D columns at once",
        abs((Ym ** 2).sum() / (Xm ** 2).sum() - pred), tol=3e-3)

    # early stopping: Pi -> alpha Pi, threshold alpha < 2 / rho_t
    for alpha in (0.3, 0.7, 1.0):
        Pa = alpha * Pi
        rep(f"  early stop alpha={alpha}: ratio = 1 + (a^2 rho - 2a)/N",
            abs(np.trace((np.eye(Nx) - Pa).T @ (np.eye(Nx) - Pa)) / Nx
                - (1.0 + (alpha ** 2 * mm[0] / cc[0] - 2 * alpha) / Nx)))

    print("\n-- the large-d limit of the metric --")
    Nl, Kl, Dl, Dld = 60, 200, 10, 4000
    tl = Kl // 2
    lim, big = [], []
    for sd in range(30):
        vv, ff, bb, TT = draw(Nl, Dld, Kl, 30, 10, seed=900 + sd)
        G = coupling(vv, ff, bb)
        big.append(eps(lagged(G, stream(G, TT), Dl), TT)[tl])
        lim.append(limit_eps(vv, ff, bb, Dl, tl))
    rep("eps^(Delta)_t -> sum_p S(Delta)_tp^2 as d grows (rel., d = 4000)",
        np.abs(np.array(big) / np.array(lim) - 1.0).max(), tol=0.12)

    print("\n-- a finite training budget, remark (ii) --")
    Ne, De, Ke = 40, 6, 40
    v2, f2, b2, T2 = draw(Ne, De, Ke, 20, 8, seed=3)
    c2, _ = masses(v2, f2, b2)
    for budget in (0.4, 0.9, 3.0):
        al = 1.0 - np.exp(-c2 * budget)
        W2 = np.zeros((Ne, De))
        R2 = np.empty((Ke, De))
        st2 = [W2.copy()]
        for t in range(Ke):
            r = T2[t] - (f2[t] * v2) @ W2
            W2 = W2 + al[t] * np.outer(b2[t] * v2, r) / c2[t]
            R2[t] = r
            st2.append(W2.copy())
        G2 = coupling(v2, f2, b2)
        La = np.tril(G2 * al[None, :], -1)
        rep(f"  budget s={budget}: (I + L_alpha) R = Theta, L_ts = alpha_s Gamma_ts",
            np.abs(np.linalg.solve(np.eye(Ke) + La, T2) - R2).max())
        dh = np.array([(f2 * v2) @ (st2[t + 1] - st2[t]) for t in range(Ke)])
        rep(f"  budget s={budget}: delta theta_k = alpha_t Gamma_kt r_t",
            np.abs(dh - np.array([al[t] * np.outer(G2[:, t], R2[t])
                                  for t in range(Ke)])).max())
        errs = []
        for Delta in range(6):
            U2 = (np.triu(G2, 1) - np.triu(G2, Delta + 1)) * al[None, :]
            Rl = (1 - al)[:, None] * R2 - U2 @ R2
            direct = np.array([T2[t] - (f2[t] * v2) @ st2[min(t + Delta, Ke - 1) + 1]
                               for t in range(Ke - Delta)])
            errs.append(np.abs(Rl[:Ke - Delta] - direct).max())
        rep(f"  budget s={budget}: r^(D) = (1-alpha_t) r_t - sum alpha_u Gamma_tu r_u",
            max(errs))

    print("\n-- the pigeonhole bound on the free array, eq. (12) --")
    lo, hi = [], []
    for (Nn, nf2, nb2) in [(100, 80, 40), (60, 45, 30), (60, 30, 15)]:
        rng = np.random.default_rng(5)
        ov = []
        for _ in range(3000):
            pk = rng.permutation(Nn)
            pt = rng.permutation(Nn)
            ov.append(len(set(pk[:nf2]) & set(pt[:nb2])) / nb2)
        bound = max(0.0, 1.0 + (nf2 - Nn) / nb2)
        lo.append(bound - min(ov))                       # must be <= 0
        hi.append(abs(np.mean(ov) - nf2 / Nn))           # mean is d_f
    rep("realised overlap never falls below max{0, 1+(n_f-N)/n_b}",
        max(max(lo), 0.0))
    rep("  and averages to d_f for independent gates", max(hi), tol=0.01)

    print("\n-- the walk below the threshold, remark (i) --")
    for Nw, want in ((60, 0.99), (120, 0.90)):
        vw, fw, bw, _ = draw(Nw, 1, 200000, Nw // 2, Nw // 2, seed=100)
        ww = vw ** 2
        x = np.random.default_rng(1).standard_normal(Nw)
        x /= np.linalg.norm(x)
        for t in range(200000):
            x = x - (bw[t] * vw) * (((fw[t] * vw) @ x) / (ww @ bw[t]))
            x /= np.linalg.norm(x)
        share = (x ** 2)[np.argsort(ww)][:3].sum()
        rep(f"  N={Nw}: 3 smallest v_i^2 carry >= {want} of ||x||^2 (got {share:.4f})",
            max(0.0, want - share))

    print("\n-- the frozen gate, remark (iii) --")
    vf, ff_, bf, _ = draw(60, 1, 800, 30, 6, seed=4, frozen=True)
    Pif = np.outer(bf[0] * vf, ff_[0] * vf) / float((bf[0] * vf) @ vf)
    prod = np.linalg.matrix_power(np.eye(60) - Pif, 25)
    rep("one gate held for the whole stream: product telescopes to I - Pi",
        np.abs(prod - (np.eye(60) - Pif)).max())
    rep("  and its growth rate is 0 at rho = 5",
        abs(growth_rate(vf, ff_, bf, burn=100)), tol=1e-12)

    return rep


# =========================================================== measured tables

def table_growth(K=4000, seeds=6, Ns=(60, 120, 240, 480),
                 rhos=(1.0, 1.5, 2.0, 2.5, 3.0)):
    """2 N lambda against the prediction rho_bar - 2, the table of remark (i)."""
    print(f"\n-- growth rate per task, 2 N lambda: K = {K}, {seeds} draws, n_f = N/2 --")
    print(f"    {'rho':>5s}" + "".join(f" | {('N=' + str(N)):>18s}" for N in Ns))
    print(f"    {'':>5s}" + "".join(f" | {'pred.':>8s} {'meas.':>9s}" for _ in Ns))
    rows, worst = {}, 0.0
    for rho in rhos:
        line = f"    {rho:5.1f}"
        for N in Ns:
            n_f = int(round(0.5 * N))
            n_b = int(round(n_f / rho))
            pr, me = [], []
            for s in range(seeds):
                v, f, b, _ = draw(N, 1, K, n_f, n_b, seed=100 + s)
                c, m = masses(v, f, b)
                pr.append((m / c).mean() - 2.0)
                me.append(2 * N * growth_rate(v, f, b))
            se = float(np.std(me, ddof=1) / np.sqrt(seeds))
            worst = max(worst, se)
            rows[(rho, N)] = (float(np.mean(pr)), float(np.mean(me)), se)
            line += f" | {np.mean(pr):+8.2f} {np.mean(me):+6.2f}({se:.2f})"
        print(line)
    print(f"    (standard error over the {seeds} draws in brackets; "
          f"largest {worst:.3f})")
    return rows


def table_drift(Ns=(60, 240, 480), Ks=(1000, 4000, 16000, 64000)):
    """Below the threshold a fitted rate drifts toward zero: it is not a rate."""
    print(f"\n-- rho = 1, where every step is nonexpansive: fitted 2 N lambda --")
    print("       N |" + "".join(f" {('K=' + str(K)):>10s}" for K in Ks))
    out = {}
    for N in Ns:
        line = f"    {N:4d} |"
        for K in Ks:
            v, f, b, _ = draw(N, 1, K, N // 2, N // 2, seed=100)
            out[(N, K)] = x = 2 * N * growth_rate(v, f, b)
            line += f" {x:+10.3f}"
        print(line)
    return out


def table_crossing(K=20000, seeds=8, Ns=(60, 120, 240, 480)):
    """Where the sign turns over: at n_b* with n_f = 2 n_b* - 2, i.e. E[rho] = 2."""
    print(f"\n-- the sign of lambda against n_b, K = {K}, {seeds} draws, n_f = N/2 --")
    print(f"    {'N':>4s} {'n_f':>4s} {'n_b':>4s} | {'rho':>6s} {'E[rho]':>7s} "
          f"{'rho_bar':>8s} | {'2N lambda':>16s}")
    out = {}
    for N in Ns:
        n_f = N // 2
        nb_star = (n_f + 2) // 2                 # n_f = 2 n_b - 2
        for n_b in (nb_star - 1, nb_star):
            pr, me = [], []
            for s in range(seeds):
                v, f, b, _ = draw(N, 1, K, n_f, n_b, seed=200 + s)
                c, m = masses(v, f, b)
                pr.append((m / c).mean())
                me.append(2 * N * growth_rate(v, f, b))
            se = float(np.std(me, ddof=1) / np.sqrt(seeds))
            out[(N, n_b)] = (float(np.mean(pr)), float(npem := np.mean(me)), se)
            star = " *" if n_b == nb_star else "  "
            print(f"    {N:4d} {n_f:4d} {n_b:4d}{star}| {n_f/n_b:6.3f} "
                  f"{(n_f-2)/(n_b-2):7.3f} {np.mean(pr):8.3f} | "
                  f"{npem:+8.4f} +- {se:.4f}")
    print("    (* is the predicted crossing, rho = 2 - 2/n_b)")
    return out


def limit_eps(v, f, b, Delta, t):
    """The d -> infinity limit of eps^(Delta)_t: sum_p S(Delta)_{tp}^2."""
    K = f.shape[0]
    G = coupling(v, f, b)
    U = np.triu(G, 1) - np.triu(G, Delta + 1)
    S = -U @ np.linalg.inv(np.eye(K) + np.tril(G, -1))
    return float((S[t] ** 2).sum())


def table_spread(N=60, K=300, Delta=10, seeds=120, Ds=(1, 2, 4, 8, 16, 32, 64, 128),
                 rhos=(1.0, 3.0)):
    """eps does not concentrate as d grows: one gate draw serves every coordinate."""
    print(f"\n-- spread of log eps over draws, against d: N = {N}, K = {K}, "
          f"Delta = {Delta}, {seeds} draws --")
    print(f"    {'rho':>4s} |" + "".join(f" {('d=' + str(D)):>7s}" for D in Ds)
          + f" | {'d -> inf':>9s}")
    out = {}
    for rho in rhos:
        n_f = N // 2
        n_b = int(round(n_f / rho))
        t = K // 2
        line, sds = f"    {rho:4.1f} |", []
        for D in Ds:
            vals = []
            for s in range(seeds):
                v, f, b, Theta = draw(N, D, K, n_f, n_b, seed=500 + s)
                G = coupling(v, f, b)
                vals.append(np.log(eps(lagged(G, stream(G, Theta), Delta), Theta)[t]))
            sd = float(np.std(vals, ddof=1))
            out[(rho, D)] = sd
            line += f" {sd:7.2f}"
        lim = [np.log(limit_eps(*draw(N, 1, K, n_f, n_b, seed=500 + s)[:3], Delta, t))
               for s in range(seeds)]
        out[(rho, "inf")] = float(np.std(lim, ddof=1))
        print(line + f" | {np.std(lim, ddof=1):9.2f}")
    print("    (last column: the limit sum_p S(Delta)_tp^2, which has no d in it)")
    return out


def main():
    quick = "--quick" in sys.argv
    print("=" * 78)
    print("  splitgating_neuronal.tex -- independent numerical check")
    print("=" * 78)
    rep = checks(Report())
    fails = rep.close()
    if not quick:
        table_growth()
        table_drift()
        table_crossing()
        table_spread()
    print()
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
