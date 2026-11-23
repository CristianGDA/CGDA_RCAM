# Nonlinear 6DoF trim, linearization and flying qualities assessment for the GARTEUR RCAM

Flight dynamics toolkit for the twin engine Research Civil Aircraft Model (RCAM), which was developed by the Group for Aeronautical Research and Technology in Europe (GARTEUR[1][2]).

## OVERVIEW
This program computes and evaluates the aircraft dynamics in a steady, straight and level flight condition through the following implementations:

1. **Nonlinear 6DoF RCAM dynamics**: Implements the GARTEUR RCAM equations of motion including aerodynamic forces and moments, and twin engine propulsion.
2. **Actuator & Engine Dynamics:** Models first order actuator with position and rate saturations for control inputs ($\delta_a$, $\delta_e$, $\delta_r$, $\delta_{th1}$, $\delta_{th2}$).
3. **trim solver**: Computes the steady state, straight and level flight condition via constrained numerical optimization (SLSQP).
4. **Numerical linearization**: Extracts decoupled longitudinal and lateral/directional state space matrices ($A$, $B$) via central finite differences.
5. **Flying qualities assessment**: Evaluates flight modes against **MIL-HDBK-1797A (Class III, Category B)** Level 1 criteria[4].
6. **Nonlinear time domain response**: Comparing ideal (instantaneous) control inputs against first order actuator and engine dynamics with rate and position limits under control surface and engine throttle doublet inputs.
