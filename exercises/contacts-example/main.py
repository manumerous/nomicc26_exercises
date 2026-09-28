#!/usr/bin/env python
from dataclasses import dataclass, field

import casadi as ca
from scipy.interpolate import make_interp_spline
import numpy as np

from nosnoc import MPCC
from nosnoc.mpccsol.plugins.ccopt import CCOptOptions
from vdx.vartypes import *

from utils import animate

@dataclass
class ManipulationExampleOptions():
    T:float = 20.0
    N:int = 50
    N_fe:int = 2

    # Radii
    r1:float = 1.0
    r2:float = 1.0
    r3:float = 1.0

    # Masses
    m1:float = 1.0
    m2:float = 1.0
    m3:float = 1.0
    m_wall:float = 1.0

    # Viscous friction:
    mu1:float = 0.01
    mu2:float = 0.01
    mu3:float = 1e-3
    mu_wall:float = 1e-4

    # Walld dynamics
    k_wall:float = 2.0
    wall_rest_position:float = -5


    # State bounds ball 1
    lbq1:np.ndarray = field(default_factory=lambda:np.array([-10.,-2.0]))
    ubq1:np.ndarray = field(default_factory=lambda:np.array([-2.0,10.]))
    lbv1:np.ndarray = field(default_factory=lambda:np.array([-10.,-10.]))
    ubv1:np.ndarray = field(default_factory=lambda:np.array([10.,10.]))

    # State bounds ball 2
    lbq2:np.ndarray = field(default_factory=lambda:np.array([2.0,-2.0]))
    ubq2:np.ndarray = field(default_factory=lambda:np.array([10.,10.]))
    lbv2:np.ndarray = field(default_factory=lambda:np.array([-10.,-10.]))
    ubv2:np.ndarray = field(default_factory=lambda:np.array([10.,10.]))

    # State bounds ball 3
    lbq3:np.ndarray = field(default_factory=lambda:np.array([-10.,-10.]))
    ubq3:np.ndarray = field(default_factory=lambda:np.array([10.,10.]))
    lbv3:np.ndarray = field(default_factory=lambda:np.array([-np.inf,-np.inf]))
    ubv3:np.ndarray = field(default_factory=lambda:np.array([np.inf,np.inf]))

    # State bounds wall
    lbq_wall:np.ndarray = field(default_factory=lambda:np.array([-100.]))
    ubq_wall:np.ndarray = field(default_factory=lambda:np.array([100.]))
    lbv_wall:np.ndarray = field(default_factory=lambda:np.array([-np.inf]))
    ubv_wall:np.ndarray = field(default_factory=lambda:np.array([np.inf]))

    # Control bound ball 1
    lbu1:np.ndarray = field(default_factory=lambda:np.array([-3, -3]))
    ubu1:np.ndarray = field(default_factory=lambda:np.array([3, 3]))

    # Control bound ball 2
    lbu2:np.ndarray = field(default_factory=lambda:np.array([-3, -3]))
    ubu2:np.ndarray = field(default_factory=lambda:np.array([3, 3]))

    # initial state
    q1_init:np.ndarray = field(default_factory=lambda:np.array([-5, 2]))
    q2_init:np.ndarray = field(default_factory=lambda:np.array([5, 2]))
    q3_init:np.ndarray = field(default_factory=lambda:np.array([4, 0]))
    q_wall_init:np.ndarray = field(default_factory=lambda:np.array([-5]))

    v1_init:np.ndarray = field(default_factory=lambda:np.array([0, 0]))
    v2_init:np.ndarray = field(default_factory=lambda:np.array([0, 0]))
    v3_init:np.ndarray = field(default_factory=lambda:np.array([0, 0]))
    v_wall_init:np.ndarray = field(default_factory=lambda:np.array([0]))

    # Objective:
    R:np.ndarray = field(default_factory=lambda:np.diag([1e-3, 1e-3, 1e-3, 1e-3]))

    q1_target:np.ndarray = field(default_factory=lambda:np.array([-5,0]))
    q2_target:np.ndarray = field(default_factory=lambda:np.array([5,0]))
    q3_target:np.ndarray = field(default_factory=lambda:np.array([-4,-3]))
    q_wall_target:np.ndarray = field(default_factory=lambda:np.array([-25]))

    q_traj:list = field(default_factory=
                        lambda:[
                            (0.0,np.array([ # initial
                                -5,2, # q1
                                5,2,  # q2
                                4,0,  # q3
                                -5,    # wall
                            ])),
                            (5.0,np.array([ # bounce
                                -5,2, # q1
                                5,2,  # q2
                                0,-4, # q3
                                -5,    # wall
                            ])),
                            (10.0,np.array([ # return position
                                -5,2, # q1
                                5,2,  # q2
                                -3,0, # q3
                                -5,    # wall
                            ])),
                            (15.0,np.array([ # bounce 2
                                -5,2, # q1
                                5,2,  # q2
                                0,-4, # q3
                                -5,    # wall
                            ])),
                            (20.0,np.array([ # final position
                                -5,2, # q1
                                5,2,  # q2
                                4,0, # q3
                                -5,    # wall
                            ])),
                        ])

    Qq:np.ndarray = field(default_factory=lambda:np.diag([
        0.0,0.0,
        0.0,0.0,
        100.0,100.0,
        0.0,
    ]))

    Qv:np.ndarray = field(default_factory=lambda:np.diag([
        1e-5,1e-5,
        1e-5,1e-5,
        1e-5,1e-5,
        0.0,
    ]))

    Qq_T:np.ndarray = field(default_factory=lambda:np.diag([
        1e-1,1e-1,
        1e-1,1e-1,
        10.0,10.0,
        0.0,
    ]))

    Qv_T:np.ndarray = field(default_factory=lambda:np.diag([
        1e-1,1e-1,
        1e-1,1e-1,
        100.0,100.0,
        0.0,
    ]))


def build_manipulation(opts):
    mpcc = MPCC()
    h = opts.T/(opts.N)
    h_fe = opts.T/(opts.N*opts.N_fe)
    # First ball data
    q1 = ca.SX.sym("q1", 2)
    v1 = ca.SX.sym("v1", 2)
    u1 = ca.SX.sym("u1", 2)

    # Second ball data
    q2 = ca.SX.sym("q2", 2)
    v2 = ca.SX.sym("v2", 2)
    u2 = ca.SX.sym("u2", 2)

    # Third ball data
    q3 = ca.SX.sym("q3", 2)
    v3 = ca.SX.sym("v3", 2)

    # Wall data
    q_wall = ca.SX.sym("q_wall", 1)
    v_wall = ca.SX.sym("q_wall", 1)

    q = ca.vertcat(q1, q2, q3, q_wall)
    v = ca.vertcat(v1, v2, v3, v_wall)
    u = ca.vertcat(u1, u2)
    x = ca.vertcat(q,v)

    # Forces (non-contact)
    f1 = u1 - opts.mu1 * v1
    f2 = u2 - opts.mu2 * v2
    f3 =  - opts.mu3 * v3
    f_wall = -opts.k_wall*(q_wall-opts.wall_rest_position) - opts.mu_wall * v_wall # Viscous friction!

    # inertia matrix:
    M = ca.diag([opts.m1,opts.m1,opts.m2,opts.m2,opts.m3,opts.m3,opts.m_wall])

    # contacts
    c12 = ca.norm_2(q1-q2) - (opts.r1 + opts.r2)
    c13 = ca.norm_2(q1-q3) - (opts.r1 + opts.r3)
    c23 = ca.norm_2(q2-q3) - (opts.r2 + opts.r3)
    c1g_l = (q1[1] - q_wall) - opts.r1
    c2g_l = (q2[1] - q_wall) - opts.r2
    c3g_l = (q3[1] - q_wall) - opts.r3

    c = ca.vertcat(c12, c13, c23, c1g_l, c2g_l, c3g_l)

    jac_c = ca.jacobian(c,q)

    # dims
    nc = c.shape[0]
    nx = x.shape[0]
    nq = q.shape[0]
    nu = u.shape[0]

    # Contact forces
    lam = ca.SX.sym("lambda", nc)

    # dynamics
    f_all = ca.vertcat(f1, f2, f3, f_wall) + ca.transpose(jac_c) @ lam
    xdot = ca.vertcat(v, ca.inv(M) @ f_all)
    x0 = ca.SX.sym("x0", nx)

    # target
    q_target = ca.SX.sym("q_target", nq)

    t_interp = np.hstack([[t] for (t,_) in opts.q_traj])
    q_interp = np.vstack([q for (_,q) in opts.q_traj])
    interp = make_interp_spline(t_interp, q_interp, k=1)

    # path constraints
    g_path = ca.vertcat(
        (q3[0]**2 - 4.0) - q3[1], # keepout region
    )

    # functions
    dyn_fun = ca.Function("dynamics", [x0, x, lam, u], [x - (x0 + h_fe*xdot)]) # Implicit Euler!
    g_path_fun = ca.Function("path_constraints", [x,u], [g_path])
    c_fun = ca.Function("c_fun", [x], [c])
    obj_fun = ca.Function(
        "obj",
        [x,u,q_target],
        [0.5*ca.bilin(opts.R,u) + 0.5*ca.bilin(opts.Qq,q-q_target) + 0.5*ca.bilin(opts.Qv,v)],
    )
    obj_fun_T = ca.Function(
        "obj_T",
        [x, q_target],
        [0.5*ca.bilin(opts.Qq_T,q-q_target) + 0.5*ca.bilin(opts.Qv_T,v)],
    )

    # bounds
    lbq = np.concatenate([
        opts.lbq1, opts.lbq2, opts.lbq3, opts.lbq_wall
    ])
    lbv = np.concatenate([
        opts.lbv1, opts.lbv2, opts.lbv3, opts.lbv_wall
    ])
    lbx = np.concatenate([
        lbq, lbv
    ])

    ubq = np.concatenate([
        opts.ubq1, opts.ubq2, opts.ubq3, opts.ubq_wall
    ])
    ubv = np.concatenate([
        opts.ubv1, opts.ubv2, opts.ubv3, opts.ubv_wall
    ])
    ubx = np.concatenate([
        ubq, ubv
    ])

    lbu = np.concatenate([
        opts.lbu1, opts.lbu2
    ])
    ubu = np.concatenate([
        opts.ubu1, opts.ubu2
    ])

    q_init = np.concatenate([
        opts.q1_init, opts.q2_init, opts.q3_init, opts.q_wall_init
    ])
    v_init = np.concatenate([
        opts.v1_init, opts.v2_init, opts.v3_init, opts.v_wall_init
    ])
    x_init=np.concatenate([q_init,v_init])

    mpcc.w.x[0,0] = Primal("x0", nx, lb=x_init, ub=x_init, init=x_init)
    mpcc.w.x[range(1,opts.N+1),range(1,opts.N_fe+1)] = Primal("x", nx, lb=lbx, ub=ubx, init=x_init)
    mpcc.w.c[range(1,opts.N+1),range(1,opts.N_fe+1)] = Primal("c", nc, lb=0.0, ub=np.inf, init=0.5)
    mpcc.w.lam[range(1,opts.N+1),range(1,opts.N_fe+1)] = Primal("lam", nc, lb=0.0, ub=np.inf, init=0.5)
    mpcc.w.u[range(1,opts.N+1)] = Primal("u", nu, lb=lbu, ub=ubu)
    mpcc.p.q_target[range(1,opts.N+1)] = Parameter("q_target", nq)

    x_prev = mpcc.w.x[0,0].sym
    t_curr = 0.0
    for ii in range(1,opts.N+1):
        u_ii = mpcc.w.u[ii]
        for jj in range(1,opts.N_fe+1):
            t_curr += h_fe
            x_ii_jj = mpcc.w.x[ii,jj].sym
            c_ii_jj = mpcc.w.c[ii,jj].sym
            lam_ii_jj = mpcc.w.lam[ii,jj].sym
            mpcc.g.dynamics[ii,jj] = Constraint(dyn_fun(x_prev, x_ii_jj, lam_ii_jj, u_ii))
            mpcc.g.c_lift[ii,jj] = Constraint(c_ii_jj - c_fun(x_ii_jj))
            mpcc.G.contact[ii,jj] = CConstraint(c_ii_jj)
            mpcc.H.contact[ii,jj] = CConstraint(lam_ii_jj)
            # Set x init
            mpcc.w.x[ii, jj](init=np.hstack([interp(t_curr), np.zeros(nq)]))
            x_prev = x_ii_jj
        mpcc.g.path[ii] = Constraint(g_path_fun(x_prev, u_ii),lb=0, ub=np.inf)
        # Set q_target (and x_init
        q_target_ii = mpcc.p.q_target[ii].sym
        mpcc.p.q_target[ii](val=interp(t_curr))
        mpcc.w.x[ii, opts.N_fe](init=np.hstack([interp(t_curr), np.zeros(nq)]))
        if ii < opts.N:
            mpcc.f += obj_fun(x_prev, u_ii, q_target_ii)

    q_target_end = mpcc.p.q_target[opts.N].sym
    mpcc.f = mpcc.f*h_fe + obj_fun_T(x_prev, q_target_end)
    ccopt_options = CCOptOptions()
    ccopt_options.ccopt_opts['relaxation_update.TYPE'] = "RolloffUpdate"
    #ccopt_options.madnlp_opts['max_iter'] = 50
    mpcc.solve(casadi_opts=ccopt_options, plugin="ccopt")
    mpcc.solve(casadi_opts=ccopt_options, plugin="ccopt")

    x_res = mpcc.w.x[:,:].res
    u_res = mpcc.w.u[:].res
    q_target = np.vstack([q_init, mpcc.p.q_target[:].val])
    c_res = mpcc.w.c[:,:].res
    lam_res = mpcc.w.lam[:,:].res

    animate(x_res, q_target, opts, save_path="pushing.gif")





if __name__ == "__main__":
    opts = ManipulationExampleOptions()
    build_manipulation(opts)
