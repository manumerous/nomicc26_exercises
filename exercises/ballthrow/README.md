# Exercise 05: Longest throw in a crosswind (single shooting and RK4 with Uno)

A full derivation is in `ball_throw.pdf` (source: `ball_throw.tex`, build with `latexmk -pdf ball_throw.tex`). It covers the continuous optimal control problem, the discretized NLP passed to Uno, the solution with the constants used, and a table that maps each symbol to its name and line in `ball_throw.py`.

The ball is a cricket ball, smooth leather, 156 g and 71 mm across. It is thrown at a fixed speed of 10 m/s from 2 m. A 5 m/s wind blows from the North, and the ball must stay below 10 m. The question: which heading gives the longest throw?

## Model

Axes: x = East, y = North, z = Up. The heading ψ is a compass bearing (0 = N, 90° = E).

$$
\dot p = v,\qquad \dot v = -k\,\|v-w\|\,(v-w) - g e_z,\qquad
k = \frac{\rho C_d A}{2m},\quad w = (0,-5,0)
$$

The ball starts at p(0) = (0, 0, 2) with v(0) = 10 (cosθ sinψ, cosθ cosψ, sinθ).

$$
\max_{\theta,\psi,T}\; x_N^2 + y_N^2 \quad\text{s.t.}\quad z_N = 0,\;\; 0 \le z_i \le 10\;(i=1..N-1)
$$

- The state s_i is computed with N fixed RK4 steps of size h = T/N (single shooting), so the NLP has only 3 variables.
- Gradients are exact derivatives of the discrete RK4 map, from forward sensitivities. Uno runs with the `filtersqp` preset, BQPD, and the L-BFGS Hessian.

Assumptions (each can be changed with a CLI flag):

- C_d = 0.47, the value for a smooth sphere. The Reynolds number is about 7×10⁴, which is subcritical.
- ρ = 1.225 kg/m³.
- g = 9.81 m/s².

## Running

    conda activate optimization
    python ball_throw.py [--N 200] [--check-jac] [--verify] [--plot] [--verbose]

- `--check-jac` compares the RK4 sensitivities with finite differences.
- `--verify` re-simulates the result with `solve_ivp`, shows RK4 convergence in N, and runs a (θ, ψ) grid search.
- `--cd 0 --wind 0` reproduces the closed-form drag-free optimum.

A full-space version is in `python ball_throw_colloc.py [--N 200] [--preset ipopt|filtersqp] [--check-derivs] [--compare]`. It uses Hermite–Simpson collocation, so the states are NLP variables and the dynamics are constraints, and it supplies an exact sparse Hessian. `--compare` also solves the single-shooting NLP from each start.

## Result (default run)

The best throw is due **South (downwind)**: heading 180°, elevation 40.19°, T = 1.568 s, range 11.875 m and peak height 4.09 m, so the 10 m cap is inactive.

The upwind throw (heading 0°, elevation 37.8°, range 10.57 m) is also a KKT point. That is why the solver uses multistart.
