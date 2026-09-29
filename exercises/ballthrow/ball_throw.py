"""Throw a ball as far as possible in a crosswind (single shooting + RK4, solved with Uno).

A cricket ball (156 g, 71 mm, smooth leather) leaves the hand at a fixed speed of
10 m/s from shoulder height. A wind blows from the North. Choose the elevation
theta, the compass heading psi and the flight time T that maximize the horizontal
distance of the landing point, while the ball stays between the ground and 10 m.

Axes: x = East, y = North, z = Up. Heading psi is a compass bearing (0 = N,
90 deg = E, clockwise). The only NLP variables are [theta, psi, T]; the trajectory
comes from N fixed RK4 steps of size h = T/N, with exact derivatives of the
discrete RK4 map (forward sensitivities).
"""
import argparse
import io

import numpy as np
import unopy

Inf = float("inf")
NP = 3  # NLP variables: theta, psi, T


class Ball:
    """Physical data and the dynamics s' = F(s), s = (x, y, z, vx, vy, vz)."""

    def __init__(self, args):
        self.m = args.mass
        self.d = args.diam
        self.cd = args.cd
        self.rho = args.rho
        self.g = args.g
        self.v0 = args.v0
        self.h0 = args.h0
        self.hmax = args.hmax
        # wind *from* the North blows toward the South
        self.w = np.array([0.0, -args.wind, 0.0])
        area = np.pi * self.d**2 / 4
        self.k = self.rho * self.cd * area / (2 * self.m)

    def rhs(self, s):
        u = s[3:] - self.w
        a = -self.k * np.linalg.norm(u) * u
        a[2] -= self.g
        return np.concatenate([s[3:], a])

    def rhs_jac(self, s):
        """dF/ds (6x6)."""
        u = s[3:] - self.w
        nu = np.linalg.norm(u)
        J = np.zeros((6, 6))
        J[:3, 3:] = np.eye(3)
        if nu > 0:
            J[3:, 3:] = -self.k * (nu * np.eye(3) + np.outer(u, u) / nu)
        return J

    def initial_state(self, theta, psi):
        ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(psi), np.sin(psi)
        s0 = np.array([0.0, 0.0, self.h0,
                       self.v0 * ct * sp, self.v0 * ct * cp, self.v0 * st])
        S0 = np.zeros((6, NP))  # d s0 / d(theta, psi, T)
        S0[3:, 0] = self.v0 * np.array([-st * sp, -st * cp, ct])
        S0[3:, 1] = self.v0 * np.array([ct * cp, -ct * sp, 0.0])
        return s0, S0


def rk4_shoot(ball, p, N, sens=True):
    """Integrate N RK4 steps from p = (theta, psi, T).

    Returns the states s_0..s_N (N+1 x 6) and, if sens, the sensitivities
    S_i = d s_i / d p (N+1 x 6 x 3) of the discrete RK4 map.
    """
    theta, psi, T = p
    h = T / N
    e = np.array([0.0, 0.0, 1.0 / N])  # dh/dp
    s, S = ball.initial_state(theta, psi)
    traj = np.empty((N + 1, 6))
    sens_all = np.empty((N + 1, 6, NP)) if sens else None
    traj[0] = s
    if sens:
        sens_all[0] = S
    F, A = ball.rhs, ball.rhs_jac
    for i in range(N):
        s1 = s
        k1 = F(s1)
        s2 = s + 0.5 * h * k1
        k2 = F(s2)
        s3 = s + 0.5 * h * k2
        k3 = F(s3)
        s4 = s + h * k3
        k4 = F(s4)
        ksum = k1 + 2 * k2 + 2 * k3 + k4
        if sens:
            dk1 = A(s1) @ S
            dk2 = A(s2) @ (S + 0.5 * h * dk1 + 0.5 * np.outer(k1, e))
            dk3 = A(s3) @ (S + 0.5 * h * dk2 + 0.5 * np.outer(k2, e))
            dk4 = A(s4) @ (S + h * dk3 + np.outer(k3, e))
            S = S + h / 6 * (dk1 + 2 * dk2 + 2 * dk3 + dk4) + np.outer(ksum, e) / 6
            sens_all[i + 1] = S
        s = s + h / 6 * ksum
        traj[i + 1] = s
    return traj, sens_all


class ShootingNLP:
    """Uno callbacks for min -(x_N^2 + y_N^2).

    Constraint rows: 0 is landing, z_N = 0; rows 1..N-1 are 0 <= z_i <= hmax.
    One RK4 pass is cached per iterate and shared by all callbacks.
    """

    def __init__(self, ball, N):
        self.ball, self.N = ball, N
        self.m = N
        self.cl = np.concatenate([[0.0], np.zeros(N - 1)])
        self.cu = np.concatenate([[0.0], np.full(N - 1, ball.hmax)])
        # dense (m x 3) Jacobian, row-major triplets
        self.jac_rows = np.repeat(np.arange(self.m), NP).tolist()
        self.jac_cols = np.tile(np.arange(NP), self.m).tolist()
        self._key = None

    def _eval(self, x):
        p = np.asarray(x[:NP], dtype=float)

        key = p.tobytes()
        if key != self._key:
            self._traj, self._sens = rk4_shoot(self.ball, p, self.N)
            self._key = key
        return self._traj, self._sens

    def objective(self, x):
        traj, _ = self._eval(x)
        return -(traj[-1, 0]**2 + traj[-1, 1]**2)

    def gradient(self, x, g):
        traj, S = self._eval(x)
        g[:] = -2 * (traj[-1, 0] * S[-1, 0] + traj[-1, 1] * S[-1, 1])

    def constraints(self, x, c):
        traj, _ = self._eval(x)
        c[0] = traj[-1, 2]
        c[1:] = traj[1:-1, 2]

    def jacobian(self, x, vals):
        _, S = self._eval(x)
        J = np.vstack([S[-1, 2][None, :], S[1:-1, 2]])
        vals[:] = J.ravel()


def solve_one(nlp, p0, verbose=False):
    ball = nlp.ball
    lb = [0.0, 0.0, 0.1]
    ub = [np.pi / 2, 2 * np.pi, 5.0]
    model = unopy.Model(unopy.PROBLEM_NONLINEAR, NP, unopy.ZERO_BASED_INDEXING)
    model.set_variables_lower_bounds(lb)
    model.set_variables_upper_bounds(ub)
    model.set_objective(unopy.MINIMIZE, nlp.objective, nlp.gradient)
    model.set_constraints(nlp.m, nlp.constraints, nlp.cl.tolist(), nlp.cu.tolist(),
                          len(nlp.jac_rows), nlp.jac_rows, nlp.jac_cols, nlp.jacobian)
    model.set_initial_primal_iterate(list(p0))

    solver = unopy.UnoSolver()
    if not verbose:
        solver.set_logger_stream(io.StringIO())
    solver.set_preset("filtersqp")
    solver.set_option("QP_solver", "BQPD")
    result = solver.optimize(model)
    ok = int(result.optimization_status) == unopy.SUCCESS
    p = np.asarray(result.primal_solution[:NP], dtype=float)
    return ok, p, result


def no_drag_flight_time(ball, theta):
    vz = ball.v0 * np.sin(theta)
    return (vz + np.sqrt(vz**2 + 2 * ball.g * ball.h0)) / ball.g


def multistart(nlp, verbose=False):
    best = None
    print(f"{'psi0':>6} {'theta0':>6} | {'status':<22} {'psi':>8} {'theta':>7} {'T':>6} {'range':>8}")
    for psi0 in (0, 90, 180, 270):
        for th0 in (30, 45, 60):
            theta0 = np.radians(th0)
            p0 = [theta0, np.radians(psi0), no_drag_flight_time(nlp.ball, theta0)]
            ok, p, res = solve_one(nlp, p0, verbose)
            traj, _ = rk4_shoot(nlp.ball, p, nlp.N, sens=False)
            rng = np.hypot(traj[-1, 0], traj[-1, 1])
            status = str(res.optimization_status).split(".")[-1]
            print(f"{psi0:6d} {th0:6d} | {status:<22} {np.degrees(p[1]):8.3f} "
                  f"{np.degrees(p[0]):7.3f} {p[2]:6.3f} {rng:8.4f}")
            if ok and (best is None or rng > best[1]):
                best = (p, rng)
    return best


# ---------------------------------------------------------------- verification

def check_jac(nlp, seed=0):
    rng = np.random.default_rng(seed)
    p = np.array([rng.uniform(0.2, 1.2), rng.uniform(0, 2 * np.pi), rng.uniform(0.8, 2.0)])
    g = np.zeros(NP)
    nlp.gradient(p, g)
    J = np.zeros(nlp.m * NP)
    nlp.jacobian(p, J)
    J = J.reshape(nlp.m, NP)
    gfd = np.zeros(NP)
    Jfd = np.zeros((nlp.m, NP))
    for j in range(NP):
        dp = np.zeros(NP)
        dp[j] = 1e-6
        cp, cm = np.zeros(nlp.m), np.zeros(nlp.m)
        nlp.constraints(p + dp, cp)
        nlp.constraints(p - dp, cm)
        Jfd[:, j] = (cp - cm) / 2e-6
        gfd[j] = (nlp.objective(p + dp) - nlp.objective(p - dp)) / 2e-6
    eg = np.max(np.abs(g - gfd)) / max(1.0, np.max(np.abs(gfd)))
    eJ = np.max(np.abs(J - Jfd)) / max(1.0, np.max(np.abs(Jfd)))
    print(f"check-jac at p={np.round(p, 4)}: gradient rel err {eg:.2e}, Jacobian rel err {eJ:.2e}")


def simulate_ivp(ball, theta, psi):
    """Reference trajectory with an adaptive integrator and a ground-hit event."""
    from scipy.integrate import solve_ivp
    s0, _ = ball.initial_state(theta, psi)

    def ground(t, s):
        return s[2]
    ground.terminal, ground.direction = True, -1
    return solve_ivp(lambda t, s: ball.rhs(s), (0, 10), s0, events=ground,
                     rtol=1e-11, atol=1e-11, dense_output=True)


def ivp_range(ball, theta, psi):
    sol = simulate_ivp(ball, theta, psi)
    s = sol.y_events[0][0]
    return np.hypot(s[0], s[1]), sol.t_events[0][0], np.max(sol.y[2])


def verify(ball, p, rng_nlp, N):
    theta, psi, T = p
    r, t, zmax = ivp_range(ball, theta, psi)
    print(f"\nre-simulation (solve_ivp): range {r:.6f} m, flight time {t:.6f} s, peak {zmax:.4f} m")
    print(f"  NLP (RK4, N={N}) minus solve_ivp: range {rng_nlp - r:+.2e} m, T {T - t:+.2e} s")
    for n in (25, 50, 100, 200, 400):
        # RK4 range with (theta, psi) fixed and T from the ground crossing (Newton on z_N(T) = 0)
        TT = T
        for _ in range(20):
            traj, S = rk4_shoot(ball, (theta, psi, TT), n)
            TT -= traj[-1, 2] / S[-1, 2, 2]
        traj, _ = rk4_shoot(ball, (theta, psi, TT), n, sens=False)
        print(f"  N={n:4d}: RK4 range error {np.hypot(traj[-1, 0], traj[-1, 1]) - r:+.3e} m")

    # brute-force grid: coarse over the whole (theta, psi) box, then 1-degree refine
    best = (-1, 0, 0)
    for th in np.arange(5, 90, 5):
        for ps in np.arange(0, 360, 10):
            rr = ivp_range(ball, np.radians(th), np.radians(ps))[0]
            best = max(best, (rr, th, ps))
    _, th_c, ps_c = best
    for th in np.arange(th_c - 5, th_c + 5.01, 1):
        for ps in np.arange(ps_c - 10, ps_c + 10.01, 1):
            rr = ivp_range(ball, np.radians(th), np.radians(ps))[0]
            best = max(best, (rr, th, ps))
    print(f"grid search: best range {best[0]:.6f} m at theta={best[1]:.0f} deg, psi={best[2] % 360:.0f} deg "
          f"({'Uno is at least as good' if r >= best[0] - 1e-9 else 'GRID BEATS UNO'})")


def plot(ball, p, N, fname):
    import matplotlib.pyplot as plt
    traj, _ = rk4_shoot(ball, p, N, sens=False)
    fig = plt.figure(figsize=(11, 4.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="tab:blue")
    r = max(1.0, np.abs(traj[:, :2]).max())
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_zlim(0, max(ball.hmax, traj[:, 2].max()))
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_zlabel("Up [m]")
    ax.set_title("Trajectory")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(traj[:, 0], traj[:, 1], color="tab:blue", label="ground track")
    ax2.plot(0, 0, "ko", label="thrower")
    ax2.plot(traj[-1, 0], traj[-1, 1], "rx", label="landing")
    ax2.annotate("", xy=(0.9, 0.55), xytext=(0.9, 0.85), xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="->", color="gray", lw=2))
    ax2.text(0.88, 0.87, f"wind {-ball.w[1]:g} m/s", transform=ax2.transAxes,
             ha="right", color="gray")
    ax2.set_xlabel("East [m]")
    ax2.set_ylabel("North [m]")
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left")
    ax2.set_title(f"Top view: heading {np.degrees(p[1]):.2f} deg")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--N", type=int, default=200, help="number of RK4 steps")
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
    ap.add_argument("--check-jac", action="store_true", help="finite-difference derivative check")
    ap.add_argument("--verify", action="store_true", help="re-simulate with solve_ivp and grid-search")
    ap.add_argument("--plot", action="store_true", help="save ball_throw.png")
    ap.add_argument("--verbose", action="store_true", help="show the Uno log")
    args = ap.parse_args()

    ball = Ball(args)
    nlp = ShootingNLP(ball, args.N)
    print(f"Uno {unopy.current_uno_version()}; k = rho Cd A / (2m) = {ball.k:.4e} 1/m")
    if args.check_jac:
        check_jac(nlp)

    best = multistart(nlp, args.verbose)
    if best is None:
        print("No start converged.")
        return
    p, rng = best
    traj, _ = rk4_shoot(ball, p, args.N, sens=False)
    print(f"\nOptimal heading   : {np.degrees(p[1]):.4f} deg (compass, 0 = N, 90 = E)")
    print(f"Optimal elevation : {np.degrees(p[0]):.4f} deg")
    print(f"Flight time       : {p[2]:.4f} s")
    print(f"Landing point     : E {traj[-1, 0]:+.4f} m, N {traj[-1, 1]:+.4f} m")
    print(f"Range             : {rng:.4f} m")
    print(f"Peak height       : {traj[:, 2].max():.4f} m (cap {ball.hmax:g} m)")

    if args.verify:
        verify(ball, p, rng, args.N)
    if args.plot:
        plot(ball, p, args.N, "ball_throw.png")
        print("Saved ball_throw.png")


if __name__ == "__main__":
    main()
