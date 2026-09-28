#!/usr/bin/env python3

import argparse
from dataclasses import dataclass, field

import casadi as ca
import numpy as np

from camino.problems.dsc import Description
from camino.settings import GlobalSettings as GS
from camino.utils.conversion import to_0d
from camino.solver import MinlpSolver, MinlpProblem, MinlpData, Settings, Stats

GS.CASADI_VAR = ca.SX


@dataclass
class TruckData:
    # ---------------------------------------------------------------
    # Discretization
    # ---------------------------------------------------------------
    N: int = 40
    T: float = 40.0                 # s
    dt: float = 1.0                 # overwritten in __post_init__

    # ---------------------------------------------------------------
    # Vehicle
    # ---------------------------------------------------------------
    mass: float = 18000.0          # kg
    gravity: float = 9.81            # m/s^2
    rolling_coefficient: float = 0.006
    air_density: float = 1.225       # kg/m^3
    drag_coefficient: float = 0.65
    frontal_area: float = 8.0        # m^2
    wheel_radius: float = 0.50       # m

    # ---------------------------------------------------------------
    # Gearbox
    #
    # Ratios are effective total engine-to-wheel ratios, including
    # the final drive. They are ordered from low to high gear.
    # ---------------------------------------------------------------
    gear_ratios: np.ndarray = field(
        default_factory=lambda: np.array([14.0, 10.0, 7.5, 4.8])
    )
    driveline_efficiency: float = 0.94

    # ---------------------------------------------------------------
    # Engine operating region
    # 700, 2200 and 1400 rpm converted to rad/s.
    # ---------------------------------------------------------------
    omega_min: float = 700.0 * 2.0 * np.pi / 60.0
    omega_max: float = 2200.0 * 2.0 * np.pi / 60.0
    omega_peak: float = 1400.0 * 2.0 * np.pi / 60.0
    omega_scale: float = 1000.0 * 2.0 * np.pi / 60.0

    # Maximum torque curve in kN m:
    #
    # T_max(w) = torque_peak
    #            - torque_curve_coefficient*(w - omega_peak)^2
    #
    torque_peak: float = 1.6
    torque_curve_coefficient: float = 8.0e-5

    # ---------------------------------------------------------------
    # Bounds
    # ---------------------------------------------------------------
    speed_min: float = 8.0          # m/s
    speed_max: float = 22.0          # m/s
    traction_force_max: float = 60.0 # kN

    initial_speed: float = 8.0       # m/s
    terminal_speed: float = 20.0

    # Piecewise-constant grade for each interval.
    # Positive values denote uphill grade.
    road_grade: np.ndarray = field(default_factory=lambda: np.array(
        [0.00] * 5
        + [0.06] * 5
        + [0.00] * 5
        + [0.06] * 5
        + [0.00] * 5
        + [0.06] * 5
        + [0.00] * 5
        + [0.06] * 5
    ))

    # ---------------------------------------------------------------
    # Fuel model
    #
    # Approximate fuel rate [g/s]:
    #
    # fuel_rate = fuel_idle
    #             + fuel_power_linear * P
    #             + fuel_power_quadratic * P^2
    #             + fuel_friction_linear * omega
    #             + fuel_friction_quadratic * omega^2
    # where P = omega*T is engine power in kW when omega is rad/s
    # and T is kN m, since 1 kN m/s = 1 kW.
    # ---------------------------------------------------------------
    fuel_idle: float = 0.45              # g/s
    fuel_power_linear: float = 0.055     # g/(kW s)
    fuel_power_quadratic: float = 1.0e-5 # g/(kW^2 s)
    fuel_friction_linear: float = 0.10
    fuel_friction_quadratic: float = 0.15

    # Cost weights
    shift_penalty: float = 0.05        # fuel-equivalent cost per shift
    speed_tracking_weight: float = 0.50

    # Big M constant
    bigM: float = 100.0

    def __post_init__(self):
        self.dt = self.T / self.N

        if len(self.road_grade) != self.N:
            raise ValueError("road_grade must have N entries")

    @property
    def n_gears(self):
        return len(self.gear_ratios)


def initial_gear_guess(cfg: TruckData, k: int):
    """
    Select a gear whose engine speed is close to omega_peak at the
    initial vehicle speed.
    """
    omega_candidates = (
        cfg.gear_ratios * cfg.initial_speed / cfg.wheel_radius
    )
    gear = int(np.argmin(np.abs(omega_candidates - cfg.omega_peak)))

    guess = np.zeros(cfg.n_gears)
    guess[gear] = 1.0
    return guess.tolist()


def engine_speed_big_m(cfg: TruckData, ratio: float):
    """
    Compute separate, relatively tight big-M constants for

        omega_min <= omega
        omega <= omega_max.

    Candidate engine speed is affine in vehicle speed.
    """
    # Implement the big M calculations here.
    return cfg.bigM, cfg.bigM


def torque_big_m(cfg: TruckData, ratio: float):
    """
    Conservative bound for relaxation of

        T <= T_max(omega).

    It maximizes T - T_max over the rectangular bounds of speed
    and traction force. For this quadratic torque curve, the lowest
    T_max over a speed interval occurs at one of its endpoints.
    """

    # Implement the big M calculations here.
    return cfg.bigM

def truck_gear_selection_minlp():
    cfg = TruckData()
    dsc = Description()

    G = cfg.n_gears
    dt = cfg.dt

    # Initial speed is represented as a parameter.
    vk = dsc.add_parameters("v0", 1, cfg.initial_speed)

    # Indices useful for plotting/extraction.
    # Description also stores these internally by variable name.
    v_variables = []
    force_variables = []
    gear_variables = []
    shift_variables = []
    distance = 0

    for k in range(cfg.N):
        # Wheel traction force in kN.
        Fk = dsc.sym("F", 1, lb=0.0, ub=cfg.traction_force_max, w0=20.0)

        # One-hot gear selection.
        zk = dsc.sym("z", G, lb=0.0, ub=1.0, w0=initial_gear_guess(cfg, k), discrete=True)

        # Exactly one active gear.
        dsc.eq(ca.sum1(zk), 1.0, is_linear=1, is_discrete=1)

        grade = float(cfg.road_grade[k])

        # Resistive forces in kN.
        F_roll = (cfg.mass * cfg.gravity * cfg.rolling_coefficient * np.cos(np.arctan(grade)) / 1000.0)
        F_grade = (cfg.mass * cfg.gravity * np.sin(np.arctan(grade)) / 1000.0)
        F_aero = (0.5 * cfg.air_density * cfg.drag_coefficient * cfg.frontal_area * vk**2 / 1000.0)
        F_resistive = F_roll + F_grade + F_aero

        # m*dv/dt = 1000*(Ftraction - Fresistance),
        # because force variables are measured in kN.
        vk_end = vk + dt * 1000.0 / cfg.mass * (Fk - F_resistive)
        vk_next = dsc.sym("v", 1, lb=cfg.speed_min, ub=cfg.speed_max, w0=cfg.initial_speed)
        dsc.eq(vk_next, vk_end)
        distance += 0.5 * dt * (vk + vk_next)

        # -----------------------------------------------------------
        # Conditional engine constraints and stage fuel consumption
        # -----------------------------------------------------------
        stage_fuel_rate = 0

        for g in range(G):
            ratio = float(cfg.gear_ratios[g])

            # Candidate operating point if gear g is selected.
            omega_kg = ratio * vk / cfg.wheel_radius        # rad/s
            torque_kg = (
                Fk * cfg.wheel_radius
                / (cfg.driveline_efficiency * ratio)
            )                                               # kN m

            torque_max_kg = (
                cfg.torque_peak
                - cfg.torque_curve_coefficient
                * (omega_kg - cfg.omega_peak) ** 2
            )

            # Tight big-M constants based on the variable bounds.
            M_omega_low, M_omega_high = 500, 500
            M_torque = torque_big_m(cfg, ratio)

            # Active when z[k,g] = 1:
            #
            # omega_min <= omega_kg <= omega_max
            dsc.leq(
                cfg.omega_min - M_omega_low * (1.0 - zk[g]),
                omega_kg,
            )
            dsc.leq(
                omega_kg,
                cfg.omega_max + M_omega_high * (1.0 - zk[g]),
            )

            # Active maximum-torque constraint:
            #
            # torque_kg <= torque_max(omega_kg)
            dsc.leq(
                torque_kg,
                torque_max_kg + M_torque * (1.0 - zk[g]),
            )

            # omega [rad/s] * torque [kN m] = power [kW].
            engine_power_kg = omega_kg * torque_kg
            omega_scaled = omega_kg / cfg.omega_scale

            fuel_rate_kg = (
                cfg.fuel_idle
                + cfg.fuel_friction_linear * omega_scaled
                + cfg.fuel_friction_quadratic * omega_scaled**2
                + cfg.fuel_power_linear * engine_power_kg
                + cfg.fuel_power_quadratic * engine_power_kg**2
            )

            # Only the selected gear contributes to fuel use.
            stage_fuel_rate += zk[g] * fuel_rate_kg

        dsc.f += dt * stage_fuel_rate

        # A small regularization discourages unnecessary speed changes.
        dsc.f += (
            cfg.speed_tracking_weight
            * dt
            * (vk_next - vk) ** 2
        )

        # -----------------------------------------------------------
        # Shift indication
        # -----------------------------------------------------------
        if k > 0:
            pass
            # Implement shift indication and sequential gear constraints here!

        force_variables.append(Fk)
        gear_variables.append(zk)
        v_variables.append(vk_next)

        z_previous = zk
        vk = vk_next

    # Recover the initial speed at the end of the horizon.
    dsc.eq(vk, cfg.terminal_speed, is_linear=1)
    target_distance = (cfg.initial_speed + cfg.terminal_speed)/2 * cfg.T
    dsc.eq(distance, target_distance)

    dsc.check()

    problem = dsc.get_problem()
    minlp_data = dsc.get_data()

    indices = {
        "speed": dsc.get_indices("v"),
        "force": dsc.get_indices("F"),
        "gear": dsc.get_indices("z"),
    }

    return problem, minlp_data, indices, cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("d", choices=["minlp"],
                        help="Choose one of ['nlp', 'mpcc', 'minlp'] to solve using the corresponding formulation")
    parser.add_argument("with_camino", choices=["true", "false"],
                        help="boolean to solve the problem using CAMINO or directly using CasADi")
    parser.add_argument("--show-plots", action="store_true",
                        help="Show plots along with saving them")
    args = parser.parse_args()

    problem, data, indices, cfg = truck_gear_selection_minlp()

    integer_indices = set(np.asarray(problem.idx_x_integer, dtype=int).flatten().tolist())
    is_integer = [ 1 if i in integer_indices else 0 for i in range(problem.x.shape[0])]

    if args.with_camino == 'true':
        settings = Settings()
        settings.MIP_SOLVER = "gurobi"
        stats = Stats("s-b-miqp", "truck-gear-shift")
        settings.MIP_SETTINGS_ALL["gurobi"]["gurobi.Presolve"] = 0  # Presolve makes Hessian MIQP nonconvex
        settings.BRMIQP_GAP = 1e-1
        settings.LBMILP_GAP = 1e-1

        solver = MinlpSolver("s-b-miqp", problem, data, stats, settings)
        result = solver.solve(data)
        solver.stats.print()

        x_sol = np.asarray(result.x_sol).reshape(-1)
        obj_sol = result.obj_val

    elif args.with_camino == 'false':
        solver = ca.nlpsol(
            "truck_minlp",
            "bonmin",
            { "f": problem.f, "g": problem.g, "x": problem.x, "p": problem.p},
            {
                "discrete": is_integer,
                "bonmin.algorithm": "B-BB",
                "bonmin.time_limit": 120.0,
                "bonmin.integer_tolerance": 1e-6,
                "print_time": True,
            },
        )

        solution = solver(x0=data.x0, lbx=data.lbx, ubx=data.ubx, lbg=data.lbg, ubg=data.ubg, p=data.p)

        x_sol = np.asarray(solution["x"]).reshape(-1)
        obj_sol = solution['f']

    speed = np.concatenate((
        [cfg.initial_speed],
        x_sol[np.asarray(indices["speed"]).reshape(-1)],
    ))

    force = x_sol[np.asarray(indices["force"]).reshape(-1)]

    # Each call to dsc.sym("z", G, ...) produces a list of G indices.
    gear_binary = np.vstack([
        x_sol[np.asarray(stage_indices, dtype=int)]
        for stage_indices in indices["gear"]
    ])

    selected_gear = np.argmax(gear_binary, axis=1) + 1

    if "shift" in indices and len(indices["shift"]) > 0:
        shift = np.array([
            x_sol[int(stage_indices[0])]
            for stage_indices in indices["shift"]
        ])
    else:
        shift = np.array([])

    print(f"Objective: {float(obj_sol):.4f}")
    print("Speed [km/h]:")
    print(np.round(3.6 * speed, 2))
    print("Traction force [kN]:")
    print(np.round(force, 2))
    print("Selected gears:")
    print(selected_gear)
    print("Shift indicators:")
    print(np.round(shift).astype(int))
