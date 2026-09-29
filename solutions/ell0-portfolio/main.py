#!/usr/bin/env python
import argparse

import numpy as np

from model import SparsePortfolioOptimization

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--num-variables", type=int, default=10, help="Number of primal decision variables to use")
    parser.add_argument("--ccopt", action="store_true", help="Locally solve the MPCC reformulation via CCOpt.jl.")
    parser.add_argument("--warmstart-miqp", action="store_true", help="Warm start MIQP methods with CCOpt")
    parser.add_argument("--daqp", action="store_true", help="Globally solve using DAQP, only recommended for small problems, N<=25.")
    parser.add_argument("--bonmin", action="store_true", help="Globally solve using bonmin: this may take a long time for N>25 or so.")
    parser.add_argument("--gurobi", action="store_true", help="Globally solve using gurobi: this may take a long time for N>25 or so.")
    parser.add_argument("--seed", type=int, help="seed to use for random problem generation")
    parser.add_argument("--rho", type=float, default=0.1, help="ell_0 penalty parameter")
    parser.add_argument("--density", type=float, default=0.05, help="density of the random matrix M where Q=M @ M.T + diag(np.random.rand(N))")
    args = parser.parse_args()

    N = args.num_variables

    if args.seed is not None:
        np.random.seed(args.seed)

    args.warmstart_miqp = args.warmstart_miqp and args.ccopt

    model1 = SparsePortfolioOptimization(N,rho=args.rho, density=args.density)

    # CCOpt.jl
    if args.ccopt:
        res_ccopt = model1.solve_ccopt()
        res_ccopt = model1.solve_ccopt()
        ccopt_stats = model1.ccopt_solver.stats()
        print(f"x_ccopt= {res_ccopt['x'][0:N]}")
        print(f"y_ccopt= {np.round(1-res_ccopt['x'][-N:].full()).T}")
        print(f"f_ccopt= {res_ccopt['f']}")

    # bonmin
    if args.bonmin:
        res_bonmin = model1.solve_bonmin(x0=model1.extract_ccopt_solution() if args.warmstart_miqp else None)
        bonmin_stats = model1.bonmin_solver.stats()
        print(f"x_bonmin= {res_bonmin['x'][0:N]}")
        print(f"y_bonmin= {np.round(res_ccopt['x'][-N:].full()).T}")
        print(f"f_bonmin= {res_bonmin['f']}")

    # daqp
    if args.daqp:
        res_daqp = model1.solve_daqp(x0=model1.extract_ccopt_solution() if args.warmstart_miqp else None)
        daqp_stats = model1.daqp_solver.stats()
        print(f"x_daqp= {res_daqp['x'][0:N]}")
        print(f"y_daqp= {np.round(res_daqp['x'][-N:].full()).T}")
        print(f"f_daqp= {res_daqp['f']}")

    # gurobi
    if args.gurobi:
        res_gurobi = model1.solve_gurobi(x0=model1.extract_ccopt_solution() if args.warmstart_miqp else None)
        gurobi_stats = model1.gurobi_solver.stats()
        print(f"x_gurobi= {res_gurobi['x'][0:N]}")
        print(f"y_gurobi= {np.round(res_gurobi['x'][-N:].full()).T}")
        print(f"f_gurobi= {res_gurobi['f']}")
