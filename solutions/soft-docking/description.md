# Exercise: Spacecraft Soft Docking

## 1. Learning objectives

In this exercise, we formulate and solve an optimal-control problem for a spacecraft approaching a docking site.

The problem is implemented in three variants:

1. a standard nonlinear program without soft-docking constraints
2. a mathematical program with complementarity constraints
3. a mixed-integer nonlinear program.

The students will learn how to:

- formulate a discretized optimal-control problem
- model spacecraft translation dynamics
- define a docking approach region
- formulate logical implications
- implement implications using binary variables and big-$M$ constraints
- implement the same logic with complementarity constraints
- compare NLP, MPCC, and MINLP formulations
- study the effect of variable bounds and big-$M$ constants
- verify whether a computed trajectory satisfies the intended physical constraints.

---

## 2. Problem description

A spacecraft must approach a docking site located at

$$
p_d =
\begin{bmatrix}
0 & 0 & 0
\end{bmatrix}^{\mathsf T}.
$$

The initial spacecraft position is

$$
p_0 =
\begin{bmatrix}
1000 & 1000 & 500
\end{bmatrix}^{\mathsf T}\ \mathrm{m},
$$

and its initial velocity is

$$
v_0 =
\begin{bmatrix}
50 & 10 & -5
\end{bmatrix}^{\mathsf T}\ \mathrm{m/s}.
$$

The spacecraft is controlled by a three-dimensional thrust vector. Its objective is to approach the docking site while limiting control effort.

When the spacecraft enters a spherical trigger region around the docking site, two additional requirements become active:

1. the spacecraft velocity must decrease as it approaches the docking site
2. the spacecraft must remain inside an approach cone aligned with the docking-site orientation.

These requirements form the logical implication

$$
\text{inside trigger region}
\quad\Longrightarrow\quad
\text{safe speed and valid line of sight}.
$$

The exercise compares different formulations of this implication.

---

## 3. Discretization

The time horizon length $T>0$ is divided into $N$ control intervals with a sampling time $\frac{T}{N}$.
The discrete-time stages are indexed by $k = 0,\ldots,N-1$, with peicewise constant controls

---

## 4. State and control variables

The spacecraft state consists of position and velocity:

$$
x =
\begin{bmatrix}
p \\
v
\end{bmatrix}
\in\mathbb{R}^6,
$$

where

$$
p =
\begin{bmatrix}
p_x & p_y & p_z
\end{bmatrix}^{\mathsf T}
$$

is the position, and

$$
v =
\begin{bmatrix}
v_x & v_y & v_z
\end{bmatrix}^{\mathsf T}
$$

is the velocity.

The control input is the thrust vector

$$
u =
\begin{bmatrix}
F_x & F_y & F_z
\end{bmatrix}^{\mathsf T}
\in\mathbb{R}^3.
$$

Each thrust component is bounded by

$$
-F_{\max}
\le
F_i
\le
F_{\max},
$$

with

$$
F_{\max} = 5000\ \mathrm{N}.
$$

---

## 5. Spacecraft dynamics

The spacecraft is represented by a double-integrator model. For simplicity, gravity, orbital dynamics, rotational motion, fuel-mass variation, and disturbances are neglected.

The continuous-time dynamics are

$$
\dot p = v,
$$

and

$$
\dot v = \frac{1}{m}u,
$$

where the spacecraft mass is

$$
m = 1500\ \mathrm{kg}.
$$

In state-space form,

$$
\dot x =
\begin{bmatrix}
v \\
u/m
\end{bmatrix}.
$$

The implementation discretizes these equations using a fourth-order Runge--Kutta method:

```python
F = integrate_rk4(x, u, xdot, data.dt)
```

The discrete dynamic constraint is

$$
x_{k+1} =
\Phi_{\mathrm{RK4}}(x_k,u_k,\Delta t).
$$

---

## 6. State bounds

The componentwise position bounds are

$$
p_{\min}
\le
p_k
\le
p_{\max},
$$

with

$$
p_{\min} =
\begin{bmatrix}
-10 & -10 & -10
\end{bmatrix}^{\mathsf T}\ \mathrm{m},
$$

and

$$
p_{\max} =
\begin{bmatrix}
2000 & 2000 & 2000
\end{bmatrix}^{\mathsf T}\ \mathrm{m}.
$$

The velocity bounds are

$$
-v_{\max}
\le
v_k
\le
v_{\max},
$$

where

$$
v_{\max} =
\begin{bmatrix}
500 & 500 & 500
\end{bmatrix}^{\mathsf T}\ \mathrm{m/s}.
$$


---

## 7. Objective function

Define the relative position

$$
q_k = p_k-p_d.
$$

The stage objective penalizes distance from the docking site and control effort:

$$
\ell_k =
q_k^{\mathsf T}Q_pq_k
+
\rho_u\lVert u_k\rVert_2^2.
$$

The supplied values are

$$
Q_p = I_3
$$

and

$$
\rho_u = 0.1.
$$

The accumulated stage objective is scaled by the horizon length:

$$
J_{\mathrm{stage}} =
\frac{1}{T}
\sum_{k=0}^{N-1}
\ell_k.
$$

If terminal equality constraints are disabled, terminal penalties are added:

$$
J_{\mathrm{terminal}} =
10\lVert p_N-p_d\rVert_2^2
+
100\lVert v_N\rVert_2^2.
$$

The full objective is

$$
J =
J_{\mathrm{stage}}
+
J_{\mathrm{terminal}}.
$$

If exact terminal constraints are enabled, the intended conditions are

$$
p_N = p_d
$$

and

$$
v_N = 0.
$$

---

## 8. Soft-docking conditions

The docking requirements become active near the docking site.

Define the distance to the docking site as

$$
d_k = \lVert q_k\rVert_2.
$$

The trigger radius is

$$
r_{\mathrm{cone}} = 100\ \mathrm{m}.
$$

The intended trigger condition is

$$
d_k \le r_{\mathrm{cone}}.
$$

Once this condition is active, the spacecraft must satisfy a speed condition and a line-of-sight condition.

---

## 9. Soft-docking speed condition

The speed must decrease with distance from the docking site:

$$
\lVert v_k\rVert_2
\le
\alpha d_k.
$$

The parameter is

$$
\alpha = 0.1\ \mathrm{s}^{-1}.
$$

Define the speed-constraint residual

$$
G_{\mathrm{speed},k} = \lVert v_k\rVert_2 - \alpha d_k.
$$

The speed condition is equivalent to

$$
G_{\mathrm{speed},k}\le0.
$$

Examples are:

- at $d_k=100\ \mathrm{m}$, the allowed speed is at most $10\ \mathrm{m/s}$
- at $d_k=20\ \mathrm{m}$, the allowed speed is at most $2\ \mathrm{m/s}$
- at $d_k=0$, the allowed speed is $0$.

This enforces a maximal docking speed profile.

---

## 10. Approach-cone condition

The docking-site orientation is constructed from

$$
n_{\mathrm{raw}} =
\begin{bmatrix}
1 & 1 & 0
\end{bmatrix}^{\mathsf T}.
$$

The normalized orientation is

$$
n_d = \frac{n_{\mathrm{raw}}}{\lVert n_{\mathrm{raw}}\rVert_2} =
\frac{1}{\sqrt{2}}
\begin{bmatrix}
1 & 1 & 0
\end{bmatrix}^{\mathsf T}.
$$

The cone half-angle is

$$
\theta_{\max}=45^\circ.
$$

A point lies inside the approach cone if

$$
q_k^{\mathsf T}n_d
\ge
d_k\cos\theta_{\max}.
$$

Define the line-of-sight residual

$$
G_{\mathrm{los},k} = d_k\cos\theta_{\max} -
q_k^{\mathsf T}n_d.
$$

The cone condition is

$$
G_{\mathrm{los},k}\le0.
$$

---

## 11. Logical soft-docking requirement

The desired implication at every stage is

$$
d_k\le r_{\mathrm{cone}}
\quad\Longrightarrow\quad
G_{\mathrm{speed},k}\le0
\ \text{and}\
G_{\mathrm{los},k}\le0.
$$

The three implementations handle this logic differently:

1. the NLP formulation omits the implication
2. the MINLP formulation uses a binary trigger and big-$M$ constraints
3. the MPCC formulation uses complementarity constraints and nonnegative slack variables.

---
