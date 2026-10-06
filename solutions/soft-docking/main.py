#!/usr/bin/env python
import argparse
from dataclasses import dataclass
from sys import argv

import casadi as ca
from camino.utils.integrators import integrate_rk4
from camino.utils.conversion import to_0d
from camino.settings import GlobalSettings as GS
from camino.problems.dsc import Description
import numpy as np

from utils import plot_docking_trajectory, plot_state_time_series, plot_control_time_series, plot_constraint_satisfaction

GS.CASADI_VAR = ca.SX


@dataclass
class SpacecraftData():
    mass = 1500  # (kg)

    N_horizon = 45  # horizon legth
    T = 150  # seconds
    dt = T/N_horizon  # sampling time (sec)

    docking_site_position = np.array([0, 0, 0])
    spacecraft_initial_position = np.array([1000, 1000, 500])
    spacecraft_initial_velocity = np.array([50, 10, -5])

    max_force = 5000  # max thrust (N)
    max_position = np.array([2000, 2000, 2000])
    min_position = np.array([-10, -10, -10])
    max_velocity = np.array([500, 500, 500])

    epsilon = 1e-2

    Q_position = np.diag([1.0,1.0,1.0])
    rho_control = 1e-1

    use_terminal_constraint=False

    # Data for soft-docking
    r_cone = 100.0  # (meters)
    alpha = 0.1  # slope for the soft-docking velocity
    theta_max = np.radians(45)  # half-angle of the approach cone
    orientation_setter = np.array([1, 1, 0])
    docking_site_orientation = orientation_setter / np.linalg.norm(orientation_setter)

def spacecraft_model_linear_dynamic(data: SpacecraftData):
    p = ca.SX.sym("p", 3)
    v = ca.SX.sym("v", 3)
    x = ca.vertcat(p, v)
    u = ca.SX.sym("F", 3)

    xdot = ca.vertcat(v, 1/data.mass * u)

    return x, u, xdot

def spacecraft_ocp_without_softdocking():
    data = SpacecraftData()
    dsc = Description()

    x, u, xdot = spacecraft_model_linear_dynamic(data)
    nx = x.shape[0]
    nu = u.shape[0]
    F = integrate_rk4(x, u, xdot, data.dt)

    initial_state = np.concatenate((data.spacecraft_initial_position, data.spacecraft_initial_velocity))
    max_state = np.concatenate((data.max_position, data.max_velocity))
    min_state = np.concatenate((data.min_position, -data.max_velocity))
    Xk = dsc.add_parameters("X0", nx, initial_state)

    for k in range(data.N_horizon):
        Uk = dsc.sym("Uk", nu, lb=-data.max_force, ub=data.max_force, w0=0)
        # Integrate till the end of the interval
        Xk_end = F(Xk, Uk)
        q = Xk[:3] - data.docking_site_position
        dsc.f += ca.bilin(data.Q_position,q) + data.rho_control * ca.norm_2(Uk)**2

        # New NLP variable for state at end of interval
        Xk = dsc.sym("Xk", nx, lb=min_state, ub=max_state, w0=initial_state)
        dsc.eq(Xk_end, Xk)
    dsc.f *= 1/data.T
    if data.use_terminal_constraint:
        dsc.add_g(0, Xk[:3] - (data.docking_site_position), 0)
        dsc.add_g(0, Xk[3:], 0)
    else:
        dsc.f += 1e1* ca.norm_2(Xk[:3] - data.docking_site_position) ** 2 + 1e2 * ca.norm_2(Xk[3:] - np.zeros(3)) ** 2

    problem = dsc.get_problem()
    data = dsc.get_data()
    idx_state = dsc.get_indices("Xk")
    idx_control = dsc.get_indices("Uk")
    return problem, data, idx_state, idx_control

def spacecraft_ocp_with_softdocking_mpcc():
    data = SpacecraftData()
    dsc = Description()

    x, u, xdot = spacecraft_model_linear_dynamic(data)
    nx = x.shape[0]
    nu = u.shape[0]
    F = integrate_rk4(x, u, xdot, data.dt)

    initial_state = np.concatenate((data.spacecraft_initial_position, data.spacecraft_initial_velocity))
    max_state = np.concatenate((data.max_position, data.max_velocity))
    min_state = np.concatenate((data.min_position, -data.max_velocity))
    Xk = dsc.add_parameters("X0", nx, initial_state)

    idx_g = 0
    idx_w = 0
    cc_pairs = []
    cc_types = []  # cc_types from libMad: VARVAR 0 VARCON 1 CONVAR 2 CONCON 3
    for k in range(data.N_horizon):
        Uk = dsc.sym("Uk", nu, lb=-data.max_force, ub=data.max_force, w0=0)
        idx_w += nu

        # Integrate to the end of the interval
        Xk_end = F(Xk, Uk)

        # New NLP variable for state at end of interval
        Xk = dsc.sym("Xk", nx, lb=min_state, ub=max_state, w0=initial_state)
        idx_w += nx
        dsc.eq(Xk_end, Xk)
        idx_g += nx

        q = Xk[:3] - data.docking_site_position
        dsc.f += ca.bilin(data.Q_position,q) + data.rho_control * ca.norm_2(Uk)**2
        distance = ca.norm_2(q)

        H = data.r_cone + data.epsilon - distance  # trigger condition
        G_speed = ca.norm_2(Xk[3:6]) - data.alpha * distance  # consequence 1
        G_los = (distance * ca.cos(data.theta_max) - q.T @ data.docking_site_orientation) # consequence 2

        # A = [y, y_speed, s_speed, y_los, s_los]
        A = dsc.sym("A", 5, lb=0, ub=ca.inf, w0=[1, 1, 1, 1, 1])
        idx_A = idx_w
        idx_w += 5

        y = A[0]
        y_speed = A[1]
        s_speed = A[2]
        y_los = A[3]
        s_los = A[4]

        # Trigger: 0 <= y _|_ y-H >= 0
        idx_yH = dsc.add_g(0, data.r_cone*y - H, ca.inf)
        cc_pairs.append((idx_A, idx_yH))
        cc_types.append(1)

        # Relaxed consequences
        dsc.add_g(-ca.inf, G_speed - s_speed, 0)
        dsc.add_g(-ca.inf, G_los - s_los, 0)

        # Slacks vanish when the trigger is active: 0 <= y_speed _|_ s_speed >= 0
        cc_pairs.append((idx_A + 1, idx_A + 2))
        cc_types.append(0)

        # Slacks vanish when the trigger is active: 0 <= y_los _|_ s_los >= 0
        cc_pairs.append((idx_A + 3, idx_A + 4))
        cc_types.append(0)

        dsc.add_g(0, y_speed - y_los, 0)
        dsc.add_g(0, y - y_speed, 0)

    dsc.f *= 1/data.T
    if data.use_terminal_constraint:
        dsc.add_g(0, Xk[:3] - (data.docking_site_position + data.epsilon*data.orientation_setter), 0)
        dsc.add_g(0, Xk[3:], 0)
    else:
        dsc.f += 1e1* ca.norm_2(Xk[:3] - data.docking_site_position) ** 2 + 1e2 * ca.norm_2(Xk[3:] - np.zeros(3)) ** 2

    problem = dsc.get_problem()
    data = dsc.get_data()
    idx_state = dsc.get_indices("Xk")
    idx_control = dsc.get_indices("Uk")
    return problem, data, idx_state, idx_control, {"cc_pairs": cc_pairs, "cc_types": cc_types, "idx_aux": dsc.get_indices("A")}

def spacecraft_ocp_with_softdocking_minlp():
    data = SpacecraftData()
    dsc = Description()

    x, u, xdot = spacecraft_model_linear_dynamic(data)
    nx = x.shape[0]
    nu = u.shape[0]
    F = integrate_rk4(x, u, xdot, data.dt)

    initial_state = np.concatenate((data.spacecraft_initial_position, data.spacecraft_initial_velocity))
    max_state = np.concatenate((data.max_position, data.max_velocity))
    min_state = np.concatenate((data.min_position, -data.max_velocity))
    Xk = dsc.add_parameters("X0", nx, initial_state)

    # BigM computation
    M_dist = data.r_cone + data.epsilon
    M_speed = np.linalg.norm(data.max_velocity)
    pd = data.docking_site_position
    q_abs_max = np.maximum(
        np.abs(data.min_position - pd),
        np.abs(data.max_position - pd),
    )
    d_max = np.linalg.norm(q_abs_max)

    M_los = d_max * (1 + np.cos(data.theta_max))

    for k in range(data.N_horizon):
        Uk = dsc.sym("Uk", nu, lb=-data.max_force, ub=data.max_force, w0=0)

        # Integrate till the end of the interval
        Xk_end = F(Xk, Uk)

        # New NLP variable for state at end of interval
        Xk = dsc.sym("Xk", nx, lb=min_state, ub=max_state, w0=initial_state)
        dsc.eq(Xk_end, Xk)

        # Binary trigger variable
        zk = dsc.sym("zk", 1, lb=0, ub=1, w0=0, discrete=True)

        q = Xk[:3] - data.docking_site_position
        distance = ca.norm_2(q)
        dsc.f += ca.bilin(data.Q_position,q) + data.rho_control * ca.norm_2(Uk)**2

        G_speed = ca.norm_2(Xk[3:6]) - data.alpha * distance  # consequence 1
        G_los = (distance * ca.cos(data.theta_max) - q.T @ data.docking_site_orientation)  # consequence 2

        # z = 0 is allowed only outside the trigger region
        # distance - r >= eps - M_dist*z
        dsc.add_g(data.epsilon, distance - data.r_cone + M_dist * zk, ca.inf)

        # Activate speed constraint when z = 1
        dsc.add_g(-ca.inf, G_speed - M_speed * (1 - zk), 0)

        # Activate LOS constraint when z = 1
        dsc.add_g( -ca.inf, G_los - M_los * (1 - zk), 0)
    dsc.f *= 1/data.T
    if data.use_terminal_constraint:
        dsc.add_g(0, Xk[:3] - data.docking_site_position, 0)
        dsc.add_g(0, Xk[3:], 0)
    else:
        dsc.f += 1e1* ca.norm_2(Xk[:3] - data.docking_site_position) ** 2 + 1e2 * ca.norm_2(Xk[3:] - np.zeros(3)) ** 2

    problem = dsc.get_problem()
    data = dsc.get_data()
    idx_state = dsc.get_indices("Xk")
    idx_control = dsc.get_indices("Uk")
    return problem, data, idx_state, idx_control

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("formulation", choices=["nlp", "mpcc", "minlp"],
                        help="Choose one of ['nlp', 'mpcc', 'minlp'] to solve using the corresponding formulation")
    parser.add_argument("--show-plots", action="store_true",
                        help="Show plots along with saving them")
    args = parser.parse_args()

    spacecraft_data = SpacecraftData()

    if args.formulation == "nlp":
        problem, data, idx_state, idx_control = spacecraft_ocp_without_softdocking()
        mysolver = ca.nlpsol(
            "mynlp",
            "ipopt",
            {"f": problem.f, "g": problem.g, "x": problem.x, "p": problem.p},
        )

    if args.formulation == "mpcc":
        problem, data, idx_state, idx_control, cc_info = spacecraft_ocp_with_softdocking_mpcc()
        # cc_types from libmad: VARVAR 0 VARCON 1 CONVAR 2 CONCON 3
        mysolver = ca.nlpsol(
            "mympcc",
            "ccopt",
            {"f": problem.f, "g": problem.g, "x": problem.x, "p": problem.p,},
            {
                "cc_pairs": cc_info["cc_pairs"],
                "cc_types": cc_info["cc_types"],
                "madnlp.bound_relax_factor": 0,
                "ccopt.relaxation_update.TYPE": "RolloffRelaxationUpdate",
                #"ccopt.q_regularization": "critical_rho",
             }
        )
        #breakpoint()

    if args.formulation == "minlp":
        problem, data, idx_state, idx_control = spacecraft_ocp_with_softdocking_minlp()
        is_discrete = [1 if i in np.array(problem.idx_x_integer).flatten() else 0 for i in range(problem.x.shape[0])]
        mysolver = ca.nlpsol(
            "mynlp",
            "bonmin",
            {"f": problem.f, "g": problem.g, "x": problem.x, "p": problem.p},
            {"discrete": is_discrete}
        )

    solution = mysolver(x0=data.x0, lbx=data.lbx, ubx=data.ubx, lbg=data.lbg, ubg=data.ubg, p=data.p)
    print(f"solution: x={solution['x']}, objective value={solution['f']}")
    x_sol = to_0d(solution['x'])
    state = np.vstack((data.p, x_sol[idx_state]))
    control = x_sol[idx_control]
    if args.formulation == "mpcc":
        aux_vars = x_sol[cc_info["idx_aux"]]

    time = np.linspace(0, spacecraft_data.dt*spacecraft_data.N_horizon, spacecraft_data.N_horizon+1)
    plot_docking_trajectory(state, config=spacecraft_data, subtitle=args.formulation, show=args.show_plots)
    plot_state_time_series(state, time, config=spacecraft_data, subtitle=args.formulation, show=args.show_plots)
    plot_control_time_series(control, time[:-1], subtitle=args.formulation, show=args.show_plots)
    plot_constraint_satisfaction(state, time, config=spacecraft_data, subtitle=args.formulation, show=args.show_plots)
