import numpy as np
from scipy.sparse import random_array, eye_array
import casadi as ca
from utils import sprandsym
# TODO(@anton): maybe include nosnoc/vdx
# import nosnoc as ns


class SparsePortfolioOptimization:
    def __init__(self, N, Q=None, mu=None, beta=1.0, rho=1.0, M=100, density=0.05):
        if Q is None:
            MQ = sprandsym(N, density)
            Q = MQ @ MQ.T + eye_array(N) * np.random.rand(N)
        if mu is None:
            mu = np.random.rand(N)
        assert Q.shape == (N, N)
        assert mu.shape == (N,) or mu.shape == (N, 1)

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
            x0=np.zeros(5 * self.N) if x0 is None else x0,
            lbx=self.lbw_ccopt,
            ubx=self.ubw_ccopt,
            lbg=self.lbg_ccopt,
            ubg=self.ubg_ccopt,
            p=np.array([self.rho, self.beta, self.M]),
        )
        self.ccopt_res = res
        return res

    def solve_daqp(self, x0=None):
        if self.daqp_solver is None:
            self._build_daqp()
        res = self.daqp_solver(
            x0=np.zeros(2 * self.N) if x0 is None else x0,
            lbx=self.lbw_daqp,
            ubx=self.ubw_daqp,
            lbg=self.lbg_daqp,
            ubg=self.ubg_daqp,
            p=np.array([self.rho, self.beta, self.M]),
        )
        self.daqp_res = res
        return res

    def solve_bonmin(self, x0=None):
        if self.bonmin_solver is None:
            self._build_bonmin()
        res = self.bonmin_solver(
            x0=np.zeros(2 * self.N) if x0 is None else x0,
            lbx=self.lbw_bonmin,
            ubx=self.ubw_bonmin,
            lbg=self.lbg_bonmin,
            ubg=self.ubg_bonmin,
            p=np.array([self.rho, self.beta, self.M]),
        )
        self.bonmin_res = res
        return res

    def solve_gurobi(self, x0=None):
        if self.gurobi_solver is None:
            self._build_gurobi()
        res = self.gurobi_solver(
            x0=np.zeros(2 * self.N) if x0 is None else x0,
            lbx=self.lbw_gurobi,
            ubx=self.ubw_gurobi,
            lbg=self.lbg_gurobi,
            ubg=self.ubg_gurobi,
            p=np.array([self.rho, self.beta, self.M]),
        )
        self.gurobi_res = res
        return res

    def _build_common(self):
        self.x = ca.SX.sym("x", self.N)
        self.g_common = ca.sum1(self.x)
        self.p_rho = ca.SX.sym("rho")
        self.p_beta = ca.SX.sym("beta")
        self.p_M = ca.SX.sym("M")
        self.p = ca.vertcat(self.p_rho, self.p_beta, self.p_M)
        self.f_common = 0.5 * ca.bilin(self.Q, self.x) - self.p_beta * ca.dot(
            self.mu, self.x
        )

    def _build_ccopt(self):
        """
        Build the `self.ccopt_solver` object using the 'ccopt' CasADi nlpsol plugin.

        This should be done by populating the `*_ccopt` variables in the `mpcc` object,
        along with the `cc_pairs` and `cc_types` options.

        Additionally populate the `self.lbg_ccopt`, `self.ubg_ccopt`, `self.lbw_ccopt`,
        and `self.ubw_ccopt` data vectors.
        """

        self.x_p = ca.SX.sym("x_p", self.N)
        self.x_n = ca.SX.sym("x_n", self.N)
        self.x_abs = ca.SX.sym("x_abs", self.N)
        self.z = ca.SX.sym("z", self.N)

        self.w_ccopt = ca.vertcat(self.x, self.x_p, self.x_n, self.x_abs, self.z)

        self.obj_ccopt = self.f_common + self.p_rho * ca.sum1(1 - self.z)
        self.g_ccopt = ca.vertcat(
            self.g_common,
            self.x - self.x_p + self.x_n,
            self.x_abs - self.x_p - self.x_n,
        )

        self.cc_pairs_list = []
        self.cc_types_list = []
        for i in range(self.N):
            self.cc_pairs_list.append((self.N + i, 2 * self.N + i))
            self.cc_types_list.append(0)  # VARVAR 0 VARCON 1 CONVAR 2 CONCON 3
            self.cc_pairs_list.append((3 * self.N + i, 4 * self.N + i))
            self.cc_types_list.append(0)  # VARVAR 0 VARCON 1 CON

        self.lbg_ccopt = ca.vertcat(1.0, ca.DM.zeros(self.g_ccopt.shape[0] - 1))
        self.ubg_ccopt = ca.vertcat(1.0, ca.DM.zeros(self.g_ccopt.shape[0] - 1))
        self.lbw_ccopt = ca.vertcat(
            ca.DM.zeros(self.N),
            ca.DM.zeros(self.N),
            ca.DM.zeros(self.N),
            ca.DM.zeros(self.N),
            ca.DM.zeros(self.N),
        )

        self.lbw_ccopt = ca.vertcat(
            -np.inf * ca.DM.ones(self.N),
            ca.DM.zeros(self.N),
            ca.DM.zeros(self.N),
            ca.DM.zeros(self.N),
            ca.DM.zeros(self.N),
        )

        self.ubw_ccopt = ca.vertcat(
            np.inf * ca.DM.ones(self.N),
            np.inf * ca.DM.ones(self.N),
            np.inf * ca.DM.ones(self.N),
            np.inf * ca.DM.ones(self.N),
            ca.DM.ones(self.N),
        )

        casadi_solver_opts = {
            "cc_pairs": self.cc_pairs_list,
            "cc_types": self.cc_types_list,  # cc_types from libMad: VARVAR 0 VARCON 1 CONVAR 2 CONCON 3
            "print_time": False,
        }

        casadi_solver_opts["ccopt"] = {
            "relaxation_update.TYPE": "RolloffRelaxationUpdate",
            "q_regularization": "critical_rho"
        }

        casadi_solver_opts["madnlp"] = {"bound_relax_factor": 0.0}

        mpcc = {
            "x": self.w_ccopt,
            "p": self.p,
            "f": self.obj_ccopt,
            "g": self.g_ccopt,
        }

        self.ccopt_solver = ca.nlpsol(
            "sparse_portfolio_ccopt", "ccopt", mpcc, casadi_solver_opts
        )

    def _build_daqp(self):
        """
        Build the `self.daqp_solver` object using the 'daqp' CasADi qpsol plugin.

        This should be done by populating the `*_daqp` variables in the `daqp` object,
        along with the `discrete` option.

        Additionally populate the `self.lbg_daqp`, `self.ubg_daqp`, `self.lbw_daqp`,
        and `self.ubw_daqp` data vectors.
        """

        raise NotImplementedError(
            "Please implement the daqp portfolio optimization solver"
        )
        daqp = {"f": self.obj_daqp, "p": self.p, "x": self.w_daqp, "g": self.g_daqp}

        daqp_opts = {
            "discrete": [],
            "error_on_fail": False,
            "daqp.iter_limit": 100,
        }
        self.daqp_solver = ca.qpsol("solver", "daqp", daqp, daqp_opts)

    def _build_bonmin(self):
        """
        Build the `self.bonmin_solver` object using the 'bonmin' CasADi nlpsol plugin.

        This should be done by populating the `*_bonmin` variables in the `bonmin` object,
        along with the `discrete` option.

        Additionally populate the `self.lbg_bonmin`, `self.ubg_bonmin`, `self.lbw_bonmin`,
        and `self.ubw_bonmin` data vectors.
        """

        raise NotImplementedError(
            "Please implement the bonmin portfolio optimization solver"
        )
        bonmin = {
            "f": self.obj_bonmin,
            "p": self.p,
            "x": self.w_bonmin,
            "g": self.g_bonmin,
        }

        bonmin_opts = {
            "discrete": [],
        }
        self.bonmin_solver = ca.nlpsol(
            "bonmin_portfolio", "bonmin", bonmin, bonmin_opts
        )

    def _build_gurobi(self):
        """
        Build the `self.gurobi_solver` object using the 'gurobi' CasADi qpsol plugin.

        This should be done by populating the `*_gurobi` variables in the `gurobi` object,
        along with the `discrete` option.

        Additionally populate the `self.lbg_gurobi`, `self.ubg_gurobi`, `self.lbw_gurobi`,
        and `self.ubw_gurobi` data vectors.
        """

        raise NotImplementedError(
            "Please implement the gurobi portfolio optimization solver"
        )
        gurobi = {
            "f": self.obj_gurobi,
            "p": self.p,
            "x": self.w_gurobi,
            "g": self.g_gurobi,
        }

        gurobi_opts = {
            "discrete": [],
            "error_on_fail": False,
            #'gurobi.Presolve': 0,
            #'gurobi.Threads': 1,
        }
        self.gurobi_solver = ca.qpsol("gurobi_portfolio", "gurobi", gurobi, gurobi_opts)
