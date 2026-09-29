# Water network tutorial tasks

We consider a water network design problem, and will use claude code or a similar coding agent to work through the exercise. In this exercise we will use two sets of solvers: **Casadi** and **UNO**, which we will install as a prerequisite from sources:

- **UNO/UnoPy** — See <https://github.com/cvanaret/Uno>. First check whether a
  **Python** environment with Uno exists, otherwise install one.
- **Casadi/CAMINO** — See <https://github.com/minlp-toolbox/CAMINO>. Install it and all dependencies.

Our goal is to develop and refine the MINLP formulation of the water network design problem, and use it to explore differences
between MINLP and MPEC formulations and solvers. See the latex file [WaterNetwork.tex](WaterNetwork.tex) for more details.

1. Create a model that can be called from **Python** and solved using **Casadi**
   and **UNO**, using the **AMPL** models (or the latex description above) as a reference. The **AMPL** models are in the directory [ampl/](ampl/) *Initially, we will treat the pipe diameter as continuous.*

2. There are two ways to re-formulate the binary restrictions ($z_{ij}$) as an MPEC:

   1. Replace the binary constraint (eq:minlp-bin) by a set of complementarity constraints:

      $$0 \leq z_{ij} \perp z_{ij} \leq 1, \quad (i,j)\in\mathcal{A}. \tag{mpec1}$$

   2. Remove the binary variables, and write the complementarity on the arc-flow variables:
      
      $$0\leq q^+_{ij}\perp q^-_{ij}\geq0,\quad (i,j)\in\mathcal{A}.$$

   Experiment with both formulations, using **UNO** and the MPEC solvers in **CASADI**. To write the problem as a nonlinear optimization problem, we replace the complementarity constraints by a nonlinear constraint. There are two ways to formulate the
   complementarity:

   $$\sum_{(i,j)\in\mathcal{A}} q^+_{ij} q^-_{ij} \leq 0, \quad \text{and} \quad
   \sum_{(i,j)\in\mathcal{A}} q^+_{ij} q^-_{ij} = 0,$$

   in addition to $q^+_{ij}, q^-_{ij} \geq 0$. For the binary variables, the aggregate form is

   $$\sum_{(i,j)\in\mathcal{A}} z_{ij} (1 - z_{ij}) \leq 0, \quad 0 \leq z_{ij} \leq 1,$$

   started from $z_{ij} = \tfrac12$.
   Which works best? Can you formulate a smooth exact penalty function?

3. As an alternative to the MPECs, we can also formulate an NLP problem (with reduced smoothness). Given that $Q_{\text{pow}} \geq 2$, we can also remove the disjunction on $q_{ij}^+, q_{ij}^-$, and use just $q_{ij}$ using the equivalent terms (eq:PL) in the pressure-flow equations. Solve the resulting NLP using **UNO**, and compare to the MPEC formulations in terms of speed and solution quality.

4. Extend the MINLP model (and the MPEC models) to include a discrete set of diameters, $d_{ij} \in \mathcal{D}$,
   where the data for $\mathcal{D}$ is defined in Table (tab:diameters).
   Add a special-ordered-set of Type 1 constraint to model the discrete choices. Can the SOS-1 constraint be formulated as a complementarity constraint?

5. Finally investigate, which model gives the best solution in terms of CPU time and objective function value for solving the final MINLP (with discrete pipe diameters). Is it the full MINLP (with binaries $z_{ij}$ and SOS-1 constraints), or can you get better results with the MPEC formulations, or the (risky?) NLP formulation using (eq:PL)?
