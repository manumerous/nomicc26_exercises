# Planning Manipulation through contacts
One application where modelling optimal control through contacts.
Simple contacts can be modelled via the Signorini condition:

$$
0 \le c(x) \perp \lambda_{\mathrm{normal}} \ge 0,
$$

where $c(x)$ is the contact gap function and $\lambda_{\mathrm{normal}}$ is the normal force corresponding to that contact.
This model can be further extended with various models for friction.

## Trampoline example
We have implemented for you a fairly basic two dimensional manipulation problem consisting of two actuated discs, one unactuated disc, and a trampoline, modelled as a simple mass-spring-damper system. 
The system is discretized using the implicit Euler method.

The state space consists of the positions of the three discs $i=1,2,3$ in Cartesian space: $q_i \in \mathbb{R}^2$, their Cartesian velocities: $v_i \in \mathbb{R}^2$, and the one dimensional position and velocity of the trampoline: $q_{\mathrm{wall}} \in \mathbb{R}$, $v_{\mathrm{wall}} \in \mathbb{R}$.
We implement pairwise contacts between each of the three discs, and between the discs and the trampoline.
The disc contacts are defined as the euclidian distance between the $q_i$ and $q_j$ minus the sum of their radii.
The the two actuated discs are directly controlled by an actuation force $u_i \in \mathbb{R}^2$.

The goal of the current control problem is to pass the unactuated disc under a parabolic obstacle using the trampoline.
To this end we provide a time varying trajectory for the third disc, the deviation from which is quadratically penalized.

## Extensions
In this exercise you are tasked with extending the basic two dimensional manipulation example as you see fit! Some ideas include:

- Adding a two dimensional friction model, to include [frictional contacts/impacts](https://publications.syscop.de/Nurkanovic2024.pdf). 
- Implementing implicit signed distance functions to [model (union-of-)polyhedral objects](https://publications.syscop.de/Dietz2025.pdf), or [(union-of-)ellipsoid objects](https://publications.syscop.de/Pozharskiy2024a.pdf).
- Use the `nosnoc` [complementarity-Lagrangian system model]((https://publications.syscop.de/Nurkanovic2024.pdf)) and use the FESD-J discretization for higher order accuracy.
- Extending the example to three dimensions, adding more objects, etc.

