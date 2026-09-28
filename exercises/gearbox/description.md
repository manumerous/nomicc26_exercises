# Exercise: Optimal Gear Selection for a Truck

## 1. Learning objectives

In this exercise, we formulate and solve a dynamic **mixed-integer nonlinear program (MINLP)** for a truck traveling over a road with varying slope.

The students will learn how to:

- model longitudinal vehicle dynamics
- represent gear selection using binary variables
- formulate gear-dependent engine constraints
- activate constraints using big-$M$ formulations
- model and penalize gear shifts
- understand the effects of parameter choices on an optimal trajectory
- compare different MINLP formulations and solver behavior.

The model is implemented with CasADi and the CAMINO `Description` interface. It can be solved either:

- directly with Bonmin through CasADi; or
- with a CAMINO decomposition method and an external mixed-integer solver.

---

## 2. Problem description

Consider a truck traveling over a road with a piecewise-constant slope. The truck must complete a prescribed trip in a fixed amount of time.

At every time step, the optimizer chooses:

1. the wheel traction force
2. one of four gears
3. whether to shift relative to the previous time step.

The objective is to minimize an approximate fuel-consumption model while avoiding unnecessary speed variation and excessive shifting.

The decisions must respect:

- longitudinal vehicle dynamics
- vehicle-speed bounds
- engine-speed bounds
- a nonlinear maximum-torque curve
- exactly one selected gear per time step
- restrictions on successive gear changes
- prescribed terminal speed
- prescribed total traveled distance.

---

## 3. Discretization and notation

The fixed time horizon $T$ is divided into $N$ intervals of duration

$
\Delta t=\frac{T}{N}.
$

For the provided instance,

$
T=40\ \mathrm{s},
\qquad
N=40,
\qquad
\Delta t=1\ \mathrm{s}.
$

The stage index is

$
k=0,\ldots,N-1.
$

There are $G=4$ available gears, indexed by

$
g=1,\ldots,G.
$

### Continuous variables

For every stage $k$:

- $v_k$: vehicle speed in $\mathrm{m/s}$
- $F_k$: wheel traction force in $\mathrm{kN}$.

### Binary variables

For every stage $k$ and gear $g$:

$
z_{k,g}=
\begin{cases}
1, & \text{if gear }g\text{ is selected at stage }k,\\
0, & \text{otherwise}.
\end{cases}
$

For $k=1,\ldots,N-1$, a shift indicator is introduced:

$
q_k=
\begin{cases}
1, & \text{if the gear changes between stages }k-1\text{ and }k,\\
0, & \text{otherwise}.
\end{cases}
$

---

## 4. Vehicle model

The truck is modeled as a point mass moving along the road.

The longitudinal force balance is $m \dot{v}=F_{\mathrm{trac}}-F_{\mathrm{res}}$

where $F_{\mathrm{res}}$ is the sum of rolling resistance, grade resistance, and aerodynamic drag.

Because the optimization model measures forces in $\mathrm{kN}$, while the vehicle mass is in $\mathrm{kg}$, the discrete dynamics are

$ v_{k+1} = v_k +
\Delta t\frac{1000}{m}
\left(F_k-F_{\mathrm{res},k}\right).
$

The factor $1000$ converts $\mathrm{kN}$ to $\mathrm{N}$.


The total resistive force is

$
F_{\mathrm{res},k} =
F_{\mathrm{roll},k}
+
F_{\mathrm{grade},k}
+
F_{\mathrm{aero},k}.
$

### 4.1 Rolling resistance

The rolling-resistance force is

$
F_{\mathrm{roll},k} =
\frac{mgC_r\cos\alpha_k}{1000}.
$

### 4.2 Grade resistance

The grade-resistance force is

$
F_{\mathrm{grade},k} =
\frac{mg\sin\alpha_k}{1000}.
$

We use the small angle approximation in order to linearize the problem.

The default profile alternates between flat and 6% uphill sections:

$
\gamma_k=
\begin{cases}
0, & k=0,\ldots,4,\\
0.06, & k=5,\ldots,9,\\
0, & k=10,\ldots,14,\\
0.06, & k=15,\ldots,19,\\
\vdots&
\end{cases}
$

### 4.3 Aerodynamic drag

The aerodynamic-drag force is

$
F_{\mathrm{aero},k} =
\frac{
\frac12\rho C_d A_f v_k^2
}{1000}.
$

---

## 5. Gearbox model

The transmission ratios are

$
r=
\begin{bmatrix}
14.0 & 10.0 & 7.5 & 4.8
\end{bmatrix}.
$

Exactly one gear must be selected at every stage:

$
\sum_{g=1}^{G}z_{k,g}=1.
$

This is known as a **one-hot encoding** of the gear decision.

---

## 6. Engine operating point

For a gear $g$, the engine speed is

$
\omega_{k,g} =
\frac{r_g}{R_w}v_k,
$

where $R_w$ is the wheel radius.

The engine torque required to produce wheel force $F_k$ is

$
\tau_{k,g} =
\frac{F_kR_w}{\eta r_g},
$

where:

- $\eta$ is the driveline efficiency
- $F_k$ is measured in $\mathrm{kN}$
- $\tau_{k,g}$ is consequently measured in $\mathrm{kN\,m}$.

The engine power is

$
P_{k,g} =
\omega_{k,g}\tau_{k,g},
$

is measured in $\mathrm{kW}$.

Notice that

$
P_{k,g} =
\frac{r_gv_k}{R_w}
\frac{F_kR_w}{\eta r_g} =
\frac{F_kv_k}{\eta}.
$

Thus, ideal transmitted power is independent of the selected gear. Gear-dependent fuel consumption is introduced through engine-speed-dependent friction terms.

---

## 7. Engine constraints

### 7.1 Engine-speed limits

The engine must operate between

$
\omega_{\min} =
700\frac{2\pi}{60}\ \mathrm{rad/s}
$

and

$
\omega_{\max} =
2200\frac{2\pi}{60}\ \mathrm{rad/s}.
$

The limits should apply only to the selected gear.

For each gear, the lower engine-speed constraint is formulated as

$
\omega_{\min} -
M^{\omega,\mathrm{low}}_g(1-z_{k,g}) \le
\omega_{k,g}.
$

The upper constraint is

$
\omega_{k,g} \le
\omega_{\max}
+
M^{\omega,\mathrm{high}}_g(1-z_{k,g}).
$

If $z_{k,g}=1$, these reduce to the physical engine-speed constraints. If $z_{k,g}=0$, the big-$M$ terms relax them.

One task in the exercise is to replace these generic constants with tighter, gear-dependent values.

### 7.2 Maximum-torque curve

The engine's maximum torque is approximated by

$
\tau_{\max}(\omega) =
\tau_{\mathrm{peak}} -
c_\tau(\omega-\omega_{\mathrm{peak}})^2.
$

The parameter values are

$
\tau_{\mathrm{peak}} =
1.6\ \mathrm{kN\,m},
$

$
\omega_{\mathrm{peak}} =
1400\frac{2\pi}{60}\ \mathrm{rad/s},
$

and

$
c_\tau=8.0\cdot 10^{-5}.
$

The torque must satisfy

$
\tau_{k,g} \le
\tau_{\max}(\omega_{k,g})
+
M^\tau_g(1-z_{k,g}).
$

Again, this constraint is active when $z_{k,g}=1$ and relaxed otherwise.

Because the torque curve is quadratic in engine speed, this is a nonlinear constraint.

---

## 8. Fuel-consumption model

The approximate fuel rate associated with gear $g$ is

$
\dot m_{f,k,g} =
a_0 +
a_{\omega,1}\bar\omega_{k,g} + a_{\omega,2}\bar\omega_{k,g}^2 + a_{P,1}P_{k,g} + a_{P,2}P_{k,g}^2,
$

where

$
\bar\omega_{k,g} =
\frac{\omega_{k,g}}{\omega_{\mathrm{scale}}}.
$

The fuel-model coefficients are:

| Parameter      |     Value |
|----------------|----------:|
| $a_0$          |    $0.45$ |
| $a_{\omega,1}$ |    $0.10$ |
| $a_{\omega,2}$ |    $0.15$ |
| $a_{P,1}$      |   $0.055$ |
| $a_{P,2}$      | $10^{-5}$ |

Only the selected gear contributes to the stage fuel consumption:

$
\dot m_{f,k} =
\sum_{g=1}^{G}
z_{k,g}\dot m_{f,k,g}.
$

This expression contains products between binary variables and nonlinear continuous expressions, making the model an MINLP.

The total approximate fuel cost is

$
J_{\mathrm{fuel}} =
\sum_{k=0}^{N-1}
\Delta t\,\dot m_{f,k}.
$

---

## 9. Speed-variation penalty

The objective includes a penalty on speed changes and discourages acceleration and deceleration:

$
J_{\Delta v} =
\rho_v
\sum_{k=0}^{N-1}
\Delta t\,(v_{k+1}-v_k)^2.
$

---

## 10. Shift detection and shift penalty
### Task
Penalize gear shifts to discourage too frequent gear changes.

### Hint
If the selected gear changes, at least one component of the one-hot vector of gears changes.
Devise a way to identify this occuring and penalize excessive gear changes.

---

## 11. Successive-shifting constraint
### Task
Implement constraints which enforce the truck may only shift up or down by one gear at a time.

### Hint
The numerical gear number at stage $k$ can be written as

$
j_k =
\sum_{g=1}^{G} g z_{k,g}.
$

---

## 12. Terminal conditions

### 12.1 Terminal speed

The initial and terminal speeds are

$
v_0=8\ \mathrm{m/s},
\qquad
v_N=20\ \mathrm{m/s}.
$

The final-speed condition is imposed as

$
v_N=v_{\mathrm{terminal}}.
$

### 12.2 Traveled distance

Distance is approximated with trapezoidal integration:

$
s_N =
\sum_{k=0}^{N-1}
\frac{\Delta t}{2}
(v_k+v_{k+1}).
$

The target distance is chosen as

$
s_{\mathrm{target}} =
\frac{v_0+v_N}{2}T.
$

For the default data,

$
s_{\mathrm{target}} =
\frac{8+20}{2}\cdot40 =
560\ \mathrm{m}.
$

The optimization imposes

$
s_N=s_{\mathrm{target}}.
$

This is the distance traveled by a vehicle whose speed changes linearly from the initial speed to the terminal speed.

---
