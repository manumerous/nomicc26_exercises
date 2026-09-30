"""Throw a ball as far as possible in a crosswind (Hermite-Simpson collocation, solved with Uno).

Same problem and notation as ball_throw.py / ball_throw.tex, but discretized in the
full space: the node states x_0..x_N and the midpoint states x_{i+1/2} are NLP
variables, and the dynamics are imposed as Hermite-Simpson collocation equations,
eq. (hs). The NLP is eq. (nlp-hs):

  z = (theta, psi, T, x_0, ..., x_N, x_{1/2}, ..., x_{N-1/2}),  n = 3 + 6(2N+1)
  min  f(z) = -(r_{E,N}^2 + r_{N,N}^2)
  s.t. x_0 - x_0(theta, psi) = 0                                        (6 rows)
       x_{i+1/2} - (x_i + x_{i+1})/2 - h/8 (F_i - F_{i+1}) = 0          (6 rows per i)
       x_{i+1} - x_i - h/6 (F_i + 4 F_{i+1/2} + F_{i+1}) = 0            (6 rows per i)
       0 <= r_{U,i} <= h_max (i = 1..N-1),  r_{U,N} = 0,  p_L <= p <= p_U   (bounds)

Derivatives are exact and sparse: the Jacobian and the Lagrangian Hessian
sigma * grad^2 f + sum_j lambda_j grad^2 c_j (Uno's MULTIPLIER_POSITIVE convention).
"""
import argparse
import io

import numpy as np
import unopy

from ball_throw import Ball, rk4_shoot, no_drag_flight_time, ivp_range, solve_one, ShootingNLP

Inf = float("inf")
NS = 6  # state dimension

# nonzero pattern of a 6x6 block  c1 I + c2 dF/dx: diagonal, dr'/dv = I and the 3x3 dv'/dv block
_BLK = sorted({(a, a) for a in range(6)} | {(a, a + 3) for a in range(3)}
              | {(3 + a, 3 + b) for a in range(3) for b in range(3)})
BLK_R = np.array([r for r, _ in _BLK])
BLK_C = np.array([c for _, c in _BLK])
TRIL3 = [(a, b) for a in range(3) for b in range(a + 1)]  # lower triangle of a 3x3 block


def rhs_hess_contract(ball, x, mu):
    """sum_j mu_j d^2 a_j / dv^2 (3x3), a(v) = -k |v_rel| v_rel, eq. (d2F)."""
    vrel = x[3:] - ball.w
    nv = np.linalg.norm(vrel)
    if nv == 0:
        return np.zeros((3, 3))
    mv = mu @ vrel
    return -ball.k * ((np.outer(mu, vrel) + np.outer(vrel, mu) + mv * np.eye(3)) / nv
                      - mv * np.outer(vrel, vrel) / nv**3)


class CollocationNLP:
    """Uno callbacks for the Hermite-Simpson NLP (nlp-hs).

    Variable layout: z[0:3] = p = (theta, psi, T), then the nodes x_0..x_N, then the
    midpoints x_{1/2}..x_{N-1/2}. Constraint layout: rows 0..5 are the initial
    condition; interval i owns rows 6+12i..6+12i+5 (Hermite midpoint) and
    6+12i+6..6+12i+11 (Simpson).
    """

    def __init__(self, ball, N):
        self.ball, self.N = ball, N
        self.n = 3 + NS * (2 * N + 1)
        self.n_con = NS + 2 * NS * N
        self._build_jac_structure()
        self._build_hess_structure()

    # ---- indexing
    def ix(self, i):
        """first index of node state x_i in z"""
        return 3 + NS * i

    def im(self, i):
        """first index of midpoint state x_{i+1/2} in z"""
        return 3 + NS * (self.N + 1) + NS * i

    def unpack(self, z):
        z = np.asarray(z[:self.n], dtype=float)
        X = z[3:3 + NS * (self.N + 1)].reshape(self.N + 1, NS)  # nodes x_i
        M = z[3 + NS * (self.N + 1):].reshape(self.N, NS)       # midpoints x_{i+1/2}
        return z[:3], X, M

    def bounds(self):
        """p_L <= p <= p_U as in ball_throw.solve_one; path and landing constraints as bounds."""
        lb = np.full(self.n, -Inf)
        ub = np.full(self.n, Inf)
        lb[:3] = [0.0, 0.0, 0.1]
        ub[:3] = [np.pi / 2, 2 * np.pi, 5.0]
        for i in range(1, self.N):
            lb[self.ix(i) + 2], ub[self.ix(i) + 2] = 0.0, self.ball.hmax  # 0 <= r_{U,i} <= h_max
        lb[self.ix(self.N) + 2] = ub[self.ix(self.N) + 2] = 0.0         # landing r_{U,N} = 0
        return lb, ub

    # ---- objective
    def objective(self, z):
        """f(z) = -(r_{E,N}^2 + r_{N,N}^2)."""
        j = self.ix(self.N)
        return -(z[j]**2 + z[j + 1]**2)

    def gradient(self, z, grad):
        j = self.ix(self.N)
        grad[:] = 0.0
        grad[j] = -2 * z[j]
        grad[j + 1] = -2 * z[j + 1]

    # ---- constraints, eq. (hs)
    def constraints(self, z, c):
        (theta, psi, T), X, M = self.unpack(z)
        h = T / self.N
        F = self.ball.rhs
        c[:NS] = X[0] - self.ball.initial_state(theta, psi)[0]
        FX = [F(x) for x in X]
        for i in range(self.N):
            r = NS + 2 * NS * i
            c[r:r + NS] = M[i] - 0.5 * (X[i] + X[i + 1]) - h / 8 * (FX[i] - FX[i + 1])
            c[r + NS:r + 2 * NS] = X[i + 1] - X[i] - h / 6 * (FX[i] + 4 * F(M[i]) + FX[i + 1])

    def _build_jac_structure(self):
        rows, cols = [], []
        # initial condition: d/dx_0 = I, d/d(theta, psi) = -d x_0/du (velocity rows only)
        rows += list(range(NS)); cols += [self.ix(0) + a for a in range(NS)]
        for j in (0, 1):
            rows += [3, 4, 5]; cols += [j] * 3
        for i in range(self.N):
            for r in (NS + 2 * NS * i, NS + 2 * NS * i + NS):  # Hermite rows, Simpson rows
                for col0 in (self.ix(i), self.im(i), self.ix(i + 1)):
                    rows += list(r + BLK_R); cols += list(col0 + BLK_C)
                rows += list(range(r, r + NS)); cols += [2] * NS  # d/dT
        self.jac_rows, self.jac_cols = rows, cols

    def jacobian(self, z, vals):
        (theta, psi, T), X, M = self.unpack(z)
        N, h, I6 = self.N, T / self.N, np.eye(NS)
        F, A = self.ball.rhs, self.ball.rhs_jac
        _, S0 = self.ball.initial_state(theta, psi)
        out = [np.ones(NS), -S0[3:, 0], -S0[3:, 1]]
        FX = [F(x) for x in X]
        AX = [A(x) for x in X]
        for i in range(N):
            Fm, Am = F(M[i]), A(M[i])
            # Hermite midpoint row block: d/dx_i, d/dx_{i+1/2}, d/dx_{i+1}, d/dT
            out += [(-0.5 * I6 - h / 8 * AX[i])[BLK_R, BLK_C], I6[BLK_R, BLK_C],
                    (-0.5 * I6 + h / 8 * AX[i + 1])[BLK_R, BLK_C],
                    -(FX[i] - FX[i + 1]) / (8 * N)]
            # Simpson row block
            out += [(-I6 - h / 6 * AX[i])[BLK_R, BLK_C], (-4 * h / 6 * Am)[BLK_R, BLK_C],
                    (I6 - h / 6 * AX[i + 1])[BLK_R, BLK_C],
                    -(FX[i] + 4 * Fm + FX[i + 1]) / (6 * N)]
        vals[:] = np.concatenate(out)

    # ---- Lagrangian Hessian (lower triangle), eq. (d2F)
    def _build_hess_structure(self):
        rows, cols = [0, 1, 1], [0, 0, 1]  # (theta, psi) block from x_0(u)
        # one 3x3 velocity block + the T cross terms for every node and midpoint
        self._pts = [self.ix(i) for i in range(self.N + 1)] + [self.im(i) for i in range(self.N)]
        for j0 in self._pts:
            v = j0 + 3
            rows += [v + a for a, b in TRIL3]; cols += [v + b for a, b in TRIL3]
            rows += [v, v + 1, v + 2]; cols += [2, 2, 2]
        j = self.ix(self.N)
        rows += [j, j + 1]; cols += [j, j + 1]  # objective: r_{E,N}, r_{N,N}
        self.hess_rows, self.hess_cols = rows, cols

    def hessian(self, z, sigma, lam, vals):
        (theta, psi, T), X, M = self.unpack(z)
        N, h, ball = self.N, T / self.N, self.ball
        lam = np.asarray(lam, dtype=float)
        lam0 = lam[:NS]
        lH = lam[NS:].reshape(N, 2, NS)[:, 0]  # Hermite multipliers per interval
        lS = lam[NS:].reshape(N, 2, NS)[:, 1]  # Simpson multipliers per interval
        # nu: weight of h F(x) in lambda^T c, per point (nodes collect from both neighbours)
        nuX = np.zeros((N + 1, NS))
        nuX[:-1] += -lH / 8 - lS / 6
        nuX[1:] += lH / 8 - lS / 6
        nuM = -4 * lS / 6
        # (theta, psi) block: -lambda_0^T d^2 x_0 / du^2 (velocity rows only)
        ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(psi), np.sin(psi)
        lv, v0 = lam0[3:], ball.v0
        out = [[-v0 * lv @ [-ct * sp, -ct * cp, -st],
                -v0 * lv @ [-st * cp, st * sp, 0.0],
                -v0 * lv @ [-ct * sp, -ct * cp, 0.0]]]
        for x, nu in zip(list(X) + list(M), list(nuX) + list(nuM)):
            Hvv = h * rhs_hess_contract(ball, x, nu[3:])          # d^2/dv^2 of h nu^T F(x)
            HTv = (ball.rhs_jac(x).T @ nu)[3:] / N                 # d^2/dT dv of h nu^T F(x)
            out.append([Hvv[a, b] for a, b in TRIL3])
            out.append(HTv)
        out.append([-2 * sigma, -2 * sigma])
        vals[:] = np.concatenate([np.asarray(o, dtype=float) for o in out])

    # ---- initial guess
    def initial_guess(self, p0):
        """z_0 for a start p0 = (theta0, psi0, T0), without a solve.

        Nodes: one RK4 pass of N steps, h = T0/N. Midpoints: the Hermite equation of
        (hs) solved explicitly, so c^H(z_0) = 0 and c^S(z_0) = O(h^5). Not clipped to
        the bounds (r_{U,N} < 0 when T0 is too long).
        """
        X, _ = rk4_shoot(self.ball, p0, self.N, sens=False)
        h = p0[2] / self.N
        FX = np.array([self.ball.rhs(x) for x in X])
        M = 0.5 * (X[:-1] + X[1:]) + h / 8 * (FX[:-1] - FX[1:])
        return np.concatenate([p0, X.ravel(), M.ravel()])


def solve_colloc(nlp, p0, preset="ipopt", verbose=False):
    """Solve the NLP (nlp-hs) with Uno from the start p0 = (theta0, psi0, T0)."""
    lb, ub = nlp.bounds()
    model = unopy.Model(unopy.PROBLEM_NONLINEAR, nlp.n, unopy.ZERO_BASED_INDEXING)
    model.set_variables_lower_bounds(lb.tolist())
    model.set_variables_upper_bounds(ub.tolist())
    model.set_objective(unopy.MINIMIZE, nlp.objective, nlp.gradient)
    zeros = [0.0] * nlp.n_con
    model.set_constraints(nlp.n_con, nlp.constraints, zeros, zeros,
                          len(nlp.jac_rows), nlp.jac_rows, nlp.jac_cols, nlp.jacobian)
    model.set_lagrangian_hessian(len(nlp.hess_rows), unopy.LOWER_TRIANGLE,
                                 nlp.hess_rows, nlp.hess_cols, nlp.hessian)
    model.set_lagrangian_sign_convention(unopy.MULTIPLIER_POSITIVE)
    model.set_initial_primal_iterate(nlp.initial_guess(np.asarray(p0, dtype=float)).tolist())

    solver = unopy.UnoSolver()
    if not verbose:
        solver.set_logger_stream(io.StringIO())
    solver.set_preset(preset)
    result = solver.optimize(model)
    ok = int(result.optimization_status) == unopy.SUCCESS
    z = np.asarray(result.primal_solution[:nlp.n], dtype=float)
    return ok, z, result


def multistart(nlp, preset, verbose=False, shooting=None):
    """Same start grid as ball_throw.multistart; optionally also solve the shooting NLP."""
    best = None
    head = f"{'psi0':>5} {'th0':>4} | {'status':<8} {'it':>4} {'cpu[s]':>7} {'psi':>8} {'theta':>7} {'T':>6} {'range':>8}"
    print(head + (f" | {'shoot it':>8} {'cpu[s]':>7} {'range':>8}" if shooting else ""))
    for psi0 in (0, 90, 180, 270):
        for th0 in (30, 45, 60):
            theta0 = np.radians(th0)
            p0 = [theta0, np.radians(psi0), no_drag_flight_time(nlp.ball, theta0)]
            ok, z, res = solve_colloc(nlp, p0, preset, verbose)
            p, X, _ = nlp.unpack(z)
            rng = np.hypot(X[-1, 0], X[-1, 1])
            status = "SUCCESS" if ok else str(res.optimization_status).split(".")[-1]
            line = (f"{psi0:5d} {th0:4d} | {status:<8} {res.number_iterations:4d} {res.cpu_time:7.3f} "
                    f"{np.degrees(p[1]):8.3f} {np.degrees(p[0]):7.3f} {p[2]:6.3f} {rng:8.4f}")
            if shooting:
                sok, sp_, sres = solve_one(shooting, p0)
                straj, _ = rk4_shoot(shooting.ball, sp_, shooting.N, sens=False)
                line += (f" | {sres.number_iterations:8d} {sres.cpu_time:7.3f} "
                         f"{np.hypot(straj[-1, 0], straj[-1, 1]):8.4f}")
            print(line)
            if ok and (best is None or rng > best[1]):
                best = (z, rng)
    return best


# ---------------------------------------------------------------- verification

def check_derivs(ball, N=6, seed=0):
    """Central finite differences of grad f, the Jacobian and the Lagrangian Hessian (small N)."""
    rng = np.random.default_rng(seed)
    nlp = CollocationNLP(ball, N)
    z = nlp.initial_guess(np.array([0.7, 2.0, 1.4])) + 0.05 * rng.standard_normal(nlp.n)
    lam, sigma, eps = rng.standard_normal(nlp.n_con), 0.7, 1e-6

    def dense_jac(zz):
        v = np.zeros(len(nlp.jac_rows))
        nlp.jacobian(zz, v)
        J = np.zeros((nlp.n_con, nlp.n))
        np.add.at(J, (nlp.jac_rows, nlp.jac_cols), v)
        return J

    def grad_lag(zz):  # sigma grad f + J^T lambda
        g = np.zeros(nlp.n)
        nlp.gradient(zz, g)
        return sigma * g + dense_jac(zz).T @ lam

    g = np.zeros(nlp.n)
    nlp.gradient(z, g)
    J = dense_jac(z)
    hv = np.zeros(len(nlp.hess_rows))
    nlp.hessian(z, sigma, lam, hv)
    H = np.zeros((nlp.n, nlp.n))
    np.add.at(H, (nlp.hess_rows, nlp.hess_cols), hv)
    H = H + np.tril(H, -1).T
    gfd, Jfd, Hfd = np.zeros(nlp.n), np.zeros_like(J), np.zeros_like(H)
    for j in range(nlp.n):
        dz = np.zeros(nlp.n)
        dz[j] = eps
        cp, cm = np.zeros(nlp.n_con), np.zeros(nlp.n_con)
        nlp.constraints(z + dz, cp)
        nlp.constraints(z - dz, cm)
        Jfd[:, j] = (cp - cm) / (2 * eps)
        gfd[j] = (nlp.objective(z + dz) - nlp.objective(z - dz)) / (2 * eps)
        Hfd[:, j] = (grad_lag(z + dz) - grad_lag(z - dz)) / (2 * eps)

    def rel(a, b):
        return np.max(np.abs(a - b)) / max(1.0, np.max(np.abs(b)))
    print(f"check-derivs (N={N}, n={nlp.n}, m={nlp.n_con}): gradient rel err {rel(g, gfd):.2e}, "
          f"Jacobian rel err {rel(J, Jfd):.2e}, Hessian rel err {rel(H, Hfd):.2e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--N", type=int, default=200, help="number of collocation intervals")
    ap.add_argument("--v0", type=float, default=10.0, help="release speed [m/s]")
    ap.add_argument("--h0", type=float, default=2.0, help="release height [m]")
    ap.add_argument("--hmax", type=float, default=10.0, help="height cap [m]")
    ap.add_argument("--wind", type=float, default=5.0, help="wind speed from the North [m/s]")
    ap.add_argument("--mass", type=float, default=0.156, help="[kg]")
    ap.add_argument("--diam", type=float, default=0.071, help="[m]")
    ap.add_argument("--cd", type=float, default=0.47,
                    help="drag coefficient (smooth sphere, subcritical Re ~ 7e4)")
    ap.add_argument("--rho", type=float, default=1.225, help="air density [kg/m^3]")
    ap.add_argument("--g", type=float, default=9.81, help="[m/s^2]")
    ap.add_argument("--preset", default="ipopt", help="Uno preset (ipopt or filtersqp)")
    ap.add_argument("--check-derivs", action="store_true", help="finite-difference derivative check")
    ap.add_argument("--compare", action="store_true",
                    help="also solve the single-shooting NLP of ball_throw.py and re-simulate")
    ap.add_argument("--verbose", action="store_true", help="show the Uno log")
    args = ap.parse_args()

    ball = Ball(args)
    nlp = CollocationNLP(ball, args.N)
    print(f"Uno {unopy.current_uno_version()}, preset {args.preset}; Hermite-Simpson, N = {args.N}: "
          f"n = {nlp.n} variables, m = {nlp.n_con} equality constraints, "
          f"Jacobian nnz = {len(nlp.jac_rows)}, Hessian nnz (lower) = {len(nlp.hess_rows)}")
    if args.check_derivs:
        check_derivs(ball)

    shooting = ShootingNLP(ball, args.N) if args.compare else None
    best = multistart(nlp, args.preset, args.verbose, shooting)
    if best is None:
        print("No start converged.")
        return
    z, rng = best
    (theta, psi, T), X, _ = nlp.unpack(z)
    print(f"\nOptimal heading   : {np.degrees(psi):.4f} deg (compass, 0 = N, 90 = E)")
    print(f"Optimal elevation : {np.degrees(theta):.4f} deg")
    print(f"Flight time       : {T:.4f} s")
    print(f"Landing point     : E {X[-1, 0]:+.4f} m, N {X[-1, 1]:+.4f} m")
    print(f"Range             : {rng:.4f} m")
    print(f"Peak height       : {X[:, 2].max():.4f} m (cap {ball.hmax:g} m)")

    if args.compare:
        r, t, _ = ivp_range(ball, theta, psi)
        print(f"\nre-simulation (solve_ivp): range {r:.6f} m, flight time {t:.6f} s")
        print(f"  collocation minus solve_ivp: range {rng - r:+.2e} m, T {T - t:+.2e} s")


if __name__ == "__main__":
    main()
