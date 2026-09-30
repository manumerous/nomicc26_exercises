import numpy as np
from scipy.sparse import random_array, eye_array
import casadi as ca
from utils import sprandsym
# TODO(@anton): maybe include nosnoc/vdx
# import nosnoc as ns

class SparsePortfolioOptimization():

    def __init__(self, N, Q=None, mu=None, beta=1.0, rho=1.0, M=100, density=0.05):
        if Q is None:
            MQ = sprandsym(N, density)
            Q = MQ @ MQ.T + eye_array(N)*np.random.rand(N)
        if mu is None:
            mu = np.random.rand(N)
        assert Q.shape == (N,N)
        assert mu.shape == (N,) or mu.shape == (N,1)

        self.N = N
        self.Q = ca.sparsify(ca.DM(Q.toarray()))
        self.mu = ca.DM(mu)
        self.beta = beta
        self.rho = rho
        self.M = M

        self._build_common()

        self.ccopt_res = None
        self.daqp_res = None
        self.bonmin_res = None
        self.gurobi_res = None
        self.ccopt_solver = None
        self.daqp_solver = None
        self.bonmin_solver = None
        self.gurobi_solver = None

    def solve_ccopt(self, x0=None):
        if self.ccopt_solver is None:
            self._build_ccopt()
        res = self.ccopt_solver(
            x0=np.zeros(5*self.N) if x0 is None else x0,
            lbx=self.lbw,
            ubx=self.ubw,
            lbg=self.lbg,
            ubg=self.ubg,
            p=np.array([self.rho,self.beta,self.M]),
        )
        self.ccopt_res = res
        return res

    def solve_daqp(self, x0=None):
        if self.daqp_solver is None:
            self._build_daqp()
        res = self.daqp_solver(
            x0=np.zeros(2*self.N) if x0 is None else x0,
            lbx=self.lbw_daqp,
            ubx=self.ubw_daqp,
            lbg=self.lbg_daqp,
            ubg=self.ubg_daqp,
            p=np.array([self.rho,self.beta,self.M]),
        )
        self.daqp_res = res
        return res

    def solve_bonmin(self, x0=None):
        if self.bonmin_solver is None:
            self._build_bonmin()
        res = self.bonmin_solver(
            x0=np.zeros(2*self.N) if x0 is None else x0,
            lbx=self.lbw_bonmin,
            ubx=self.ubw_bonmin,
            lbg=self.lbg_bonmin,
            ubg=self.ubg_bonmin,
            p=np.array([self.rho,self.beta,self.M]),
        )
        self.bonmin_res = res
        return res

    def solve_gurobi(self, x0=None):
        if self.gurobi_solver is None:
            self._build_gurobi()
        res = self.gurobi_solver(
            x0=np.zeros(2*self.N) if x0 is None else x0,
            lbx=self.lbw_gurobi,
            ubx=self.ubw_gurobi,
            lbg=self.lbg_gurobi,
            ubg=self.ubg_gurobi,
            p=np.array([self.rho,self.beta,self.M]),
        )
        self.gurobi_res = res
        return res

    def extract_ccopt_solution(self):
        if self.ccopt_res is None:
            return None
        x = self.ccopt_res['x'][:self.N]
        y = np.round(1-self.ccopt_res['x'][-self.N:])

        return np.concatenate([x.full(),y])


    def _build_common(self):
        self.x = ca.SX.sym("x", self.N)
        self.g_common = ca.sum1(self.x)
        self.p_rho = ca.SX.sym("rho")
        self.p_beta = ca.SX.sym("beta")
        self.p_M = ca.SX.sym("M")
        self.p = ca.vertcat(self.p_rho, self.p_beta, self.p_M)
        self.f_common = 0.5*ca.bilin(self.Q, self.x) - self.p_beta*ca.dot(self.mu, self.x)

    def _build_ccopt(self):
        # Build auxiliary variables for the ccopt formulation
        self.x_p = ca.SX.sym("x_p", self.N)
        self.x_n = ca.SX.sym("x_n", self.N)
        self.xi = ca.SX.sym("xi", self.N)
        self.x_abs = ca.SX.sym("x_abs", self.N)

        self.w = ca.vertcat(
            self.x,
            self.x_p,
            self.x_n,
            self.x_abs,
            self.xi
        )

        self.ccopt_obj = self.f_common + self.p_rho*ca.sum1(1 - self.xi)
        self.g = ca.vertcat(
            self.x - (self.x_p - self.x_n),
            self.x_abs - (self.x_p + self.x_n),
            self.g_common
            )
        self.lbg = np.zeros(2*self.N + 1)
        self.lbg[-1] = 1.0
        self.ubg = np.zeros(2*self.N + 1)
        self.ubg[-1] = 1.0

        self.lbw = np.concatenate([
            -np.inf*np.ones(self.N), # lbx
            np.zeros(self.N), # lbx_p
            np.zeros(self.N), # lbx_n
            np.zeros(self.N), # lbx_abs
            np.zeros(self.N), # lbxi
        ])

        self.ubw = np.concatenate([
            np.inf*np.ones(self.N), # ubx
            np.inf*np.ones(self.N), # ubx_p
            np.inf*np.ones(self.N), # ubx_n
            np.inf*np.ones(self.N), # ubx_abs
            np.ones(self.N), # ubxi
        ])

        self.cc_types = np.zeros(2*self.N,dtype=int)

        left_cc = np.concatenate([
            np.arange(self.N, 2*self.N),
            np.arange(2*self.N, 3*self.N),
        ])
        right_cc = np.concatenate([
            np.arange(3*self.N, 4*self.N),
            np.arange(4*self.N, 5*self.N),
        ])
        self.cc_pairs = np.vstack([
            left_cc,
            right_cc,
        ]).T

        casadi_solver_opts = {
            "cc_pairs": self.cc_pairs.tolist(),
            "cc_types": self.cc_types.tolist(),
            "print_time": False,
        }
        casadi_solver_opts["ccopt"] = {
            "relaxation_update.TYPE": "RolloffRelaxationUpdate",
            "q_regularization": "critical_rho"
        }
        casadi_solver_opts["madnlp"] = {
            "bound_relax_factor": 0.0
        }


        nlp = {
            "x": self.w,
            "p": self.p,
            "f": self.ccopt_obj,
            "g": self.g,
        }

        self.ccopt_solver = ca.nlpsol("sparse_portfolio_ccopt", "ccopt", nlp, casadi_solver_opts)

    def _build_daqp(self):
        self.y = ca.SX.sym("y", self.N)
        self.p_M = 500.0
        self.y_eps = 1e-7
        self.obj_daqp = self.f_common + self.p_rho*ca.sum1(self.y) + self.y_eps*ca.sum1(self.y**2)

        self.w_daqp = ca.vertcat(
            self.x,
            self.y,
        )
        self.g_daqp = ca.vertcat(
            self.x + self.y*self.p_M,
            -self.x + self.y*self.p_M,
            self.g_common,
        )

        self.lbw_daqp = np.concatenate([
            -np.inf*np.ones(self.N),
            np.zeros(self.N)
        ])

        self.ubw_daqp = np.concatenate([
            np.inf*np.ones(self.N),
            np.ones(self.N)
        ])

        self.lbg_daqp = np.concatenate([
            np.zeros(self.N),
            np.zeros(self.N),
            np.ones(1)
        ])


        self.ubg_daqp = np.concatenate([
            np.inf*np.ones(self.N),
            np.inf*np.ones(self.N),
            np.ones(1)
        ])

        daqp = {
            "f": self.obj_daqp,
            "p": self.p,
            "x": self.w_daqp,
            "g": self.g_daqp
        }

        daqp_opts = {
            #'discrete': np.concatenate([np.zeros(self.N,dtype=int),np.ones(self.N,dtype=int)]),
            'discrete': [0] * self.N + [1] * self.N,
            'error_on_fail': False,
            'daqp.iter_limit': 100,
            #'daqp.time_limit': 10,
        }
        self.daqp_solver = ca.qpsol('solver', 'daqp', daqp, daqp_opts)

    def _build_bonmin(self):
        self.y = ca.SX.sym("y", self.N)
        self.p_M = 500.0
        self.obj_bonmin = self.f_common + self.p_rho*ca.sum1(self.y)

        self.w_bonmin = ca.vertcat(
            self.x,
            self.y,
        )
        self.g_bonmin = ca.vertcat(
            self.x + self.y*self.p_M,
            -self.x + self.y*self.p_M,
            self.g_common,
        )

        self.lbw_bonmin = np.concatenate([
            -np.inf*np.ones(self.N),
            np.zeros(self.N)
        ])

        self.ubw_bonmin = np.concatenate([
            np.inf*np.ones(self.N),
            np.ones(self.N)
        ])

        self.lbg_bonmin = np.concatenate([
            np.zeros(self.N),
            np.zeros(self.N),
            np.ones(1)
        ])


        self.ubg_bonmin = np.concatenate([
            np.inf*np.ones(self.N),
            np.inf*np.ones(self.N),
            np.ones(1)
        ])


        bonmin = {
            "f": self.obj_bonmin,
            "p": self.p,
            "x": self.w_bonmin,
            "g": self.g_bonmin
        }

        bonmin_opts = {
            #'discrete': np.concatenate([np.zeros(self.N,dtype=int),np.ones(self.N,dtype=int)]),
            'discrete': [0] * self.N + [1] * self.N,
            'error_on_fail': False,
            #'bonmin.algorithm': "B-OA",
            'bonmin.print_level': 0,
            'bonmin.bb_log_level': 0,
            #'bonmin.oa_log_level': 0,
            #'bonmin.oa_cuts_log_level': 0,
            #'bonmin.milp_log_level': 0,
            #'bonmin.nlp_log_level': 0,
            #'bonmin.fp_log_level': 0,
            #'bonmin.lp_log_level': 0,
        }
        self.bonmin_solver = ca.nlpsol('bonmin_portfolio', 'bonmin', bonmin, bonmin_opts)

    def _build_gurobi(self):
        self.y = ca.SX.sym("y", self.N)
        self.obj_gurobi = self.f_common + self.p_rho*ca.sum1(self.y)

        self.w_gurobi = ca.vertcat(
            self.x,
            self.y,
        )
        self.g_gurobi = ca.vertcat(
            self.x + self.y*self.p_M,
            -self.x + self.y*self.p_M,
            self.g_common,
        )

        self.lbw_gurobi = np.concatenate([
            -np.inf*np.ones(self.N),
            np.zeros(self.N)
        ])

        self.ubw_gurobi = np.concatenate([
            np.inf*np.ones(self.N),
            np.ones(self.N)
        ])

        self.lbg_gurobi = np.concatenate([
            np.zeros(self.N),
            np.zeros(self.N),
            np.ones(1)
        ])


        self.ubg_gurobi = np.concatenate([
            np.inf*np.ones(self.N),
            np.inf*np.ones(self.N),
            np.ones(1)
        ])


        gurobi = {
            "f": self.obj_gurobi,
            "p": self.p,
            "x": self.w_gurobi,
            "g": self.g_gurobi
        }

        gurobi_opts = {
            #'discrete': np.concatenate([np.zeros(self.N,dtype=int),np.ones(self.N,dtype=int)]),
            'discrete': [0] * self.N + [1] * self.N,
            'error_on_fail': False,
            'gurobi.Presolve': 0,
            'gurobi.Threads': 1,
        }
        self.gurobi_solver = ca.qpsol('gurobi_portfolio', 'gurobi', gurobi, gurobi_opts)
