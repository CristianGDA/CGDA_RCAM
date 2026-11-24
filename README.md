# Nonlinear 6DoF trim, linearization and flying qualities assessment for the GARTEUR RCAM

Flight dynamics toolkit for the twin engine Research Civil Aircraft Model (RCAM), which was developed by the Group for Aeronautical Research and Technology in Europe (GARTEUR[1][2]).

> **Theoretical background:** The derivation of the 6 DoF equations of motion, reference frames, and RCAM physical, geometry and aerodynamic parameters is detailed in [Chapter 2](docs/chapter_2_rcam_model.pdf) (in Spanish) of my thesis. The trim computation and numerical linearization algorithms (originally developed in MATLAB/Simulink) are documented in [Chapter 3](docs/chapter_3_trim_linearization.pdf) (in Spanish).

## OVERVIEW
This program computes and evaluates the aircraft dynamics in a steady, straight and level flight condition through the following implementations:

1. **Nonlinear 6DoF RCAM dynamics**: Implements the GARTEUR RCAM equations of motion including mass, inertia, and geometric parameters, aerodynamic forces and moments, and twin engine propulsion.
2. **Actuator & Engine Dynamics:** Models first order actuators with position and rate saturations for control inputs ($\delta_a$, $\delta_e$, $\delta_r$, $\delta_{th1}$, $\delta_{th2}$).
3. **Trim solver**: Computes the steady state, straight and level flight condition via constrained numerical optimization (SLSQP).
4. **Numerical linearization**: Extracts decoupled longitudinal and lateral/directional state space matrices ($A$, $B$) via central finite differences.
5. **Flying qualities assessment**: Evaluates flight modes against **MIL-HDBK-1797A (Class III, Category B)** Level 1 criteria[4].
6. **Nonlinear time domain response**: Compares ideal (instantaneous) control inputs against first order actuator and engine dynamics with rate and position limits under control surface and engine throttle doublet inputs.

## RESULTS

## REFERENCES
[1] P.F. Lambrechts et al. Robust flight control design challenge problem formulation and manual: the research civil aircraft (RCAM). Technical Publication TP-088-3, Group for aeronautical research and technology in Europe GARTEUR-FM(AG08), 1995.

[2] J.-F. Magni, S. Bennani, and J. Terlouw, Robust Flight Control, A Design Challenge (GARTEUR). Berlin, Germany: Springer-Verlag, 1997, vol. 224, Lecture Notes in Control and Information Sciences.

[3] Brian L. Stevens, Frank L. Lewis, Eric N. Johnson, Aircraft control and simulation, Controls Design and Autonomous Systems, 3rd edition Wiley 2016.

[4] U.S. Department of Defense, "Flying Qualities of Piloted Aircraft", Military Handbook MIL-HDBK-1797A, Dec. 19, 1997.
