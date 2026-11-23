from dataclasses import dataclass, field
import numpy as np
from scipy.integrate import odeint
import sys
from scipy.optimize import minimize
from tabulate import tabulate
from colorama import Back, Fore, Style, init
import matplotlib.pyplot as plt
import os
init(autoreset=True)


"""
*** Project title: 
    Nonlinear 6DoF trim, linearization and flying qualities assessment for the GARTEUR RCAM.

*** Description:
    Analytical flight dynamics script for the twin engine Research Civil Aircraft Model (RCAM),
    which was developed by the Group for Aeronautical Research and Technology in Europe (GARTEUR[1][2])
    
    This program computes and evaluates the aircraft dynamics in a steady, straight and level flight
    condition through the following implementations:

    1. Nonlinear 6DoF equations of motion (flat earth model [3])
    2. Steady state, straight and level trim computation via constrained numerical optimization (SLSQP)
    3. Numerical linearization around the trim condition using the central difference method
    4. Longitudinal and lateral-directional dynamic stability analysis and verification 
        of flying qualities using MIL-HDBK-1797A [4]
    5. Nonlinear time domain response to control surface and engine throttle doublet inputs, 
        comparing ideal (instantaneous) inputs against first order actuator and engine dynamics
        with rate and position limits

[1] P.F. Lambrechts et al. Robust flight control design challenge problem formulation and manual: the
research civil aircraft (RCAM). Technical Publication TP-088-3, Group for aeronautical research and
technology in Europe GARTEUR-FM(AG08), 1995.
[2] J.-F. Magni, S. Bennani, and J. Terlouw, Robust Flight Control, A Design Challenge (GARTEUR).
Berlin, Germany: Springer-Verlag, 1997, vol. 224, Lecture Notes in Control and Information Sciences.
[3] Brian L. Stevens, Frank L. Lewis, Eric N. Johnson, Aircraft control and simulation, Controls Design
and Autonomous Systems, 3rd edition Wiley 2016.
[4] U.S. Department of Defense, "Flying Qualities of Piloted Aircraft", 
Military Handbook MIL-HDBK-1797A, Dec. 19, 1997.
"""

from rcam_parameters import RCAM_parameters 
from aircraft_model import Aircraft_model
from trim_straight_level import trim_straight_level_flight 
from linearization import linearization                         
from stability_qualities import stability_analysis
from doublet_response import response_doublet_input


def main():
    """
    Main execution block, it instantiates RCAM parameters and dynamics, requests for the steady state, straight
    and level flight condition (cruise speed and altitude) and runs trim condition, linearization, 
    MIL-HDBK-1797A stability assessment and doublet response plots.

    Inputs:
    =========================
    None

    Outputs:
    =========================
    - Trim Condition summary and optimization results
            * Table of steady state values
            * Table of inputs required to maintain steady states
    - Linearization around the trim condition
            * Decoupled state matrix A
            * Decoupled control input matrix B
    - Longitudinal and lateral/directional stability and flying qualities
            * Table of longitudinal dynamic stability characteristics
            * Table of longitudinal flying qualities compliance (MIL-HDBK-1797A)
            * Table of lateral/directional dynamic stability characteristics
            * Table of lateral/directional flying qualities compliance (MIL-HDBK-1797A)
    - Doublet response plots comparing ideal(commanded) vs dynamic actuator models
    """

    print(Style.BRIGHT + Fore.LIGHTBLACK_EX + "="*90)
    print(" NONLINEAR 6 DOF TRIM, LINEARIZATION AND FLYING QUALITIES ASSESSMENT FOR THE GARTEUR RCAM ")
    print(Style.BRIGHT + Fore.LIGHTBLACK_EX + "="*90)

    print(Style.BRIGHT + Fore.CYAN + "Author: ", end="")
    print("Cristian Gómez Daza Argumedo")
    print(Style.BRIGHT + Fore.CYAN + "Aircraft: ", end="")
    print("Research Civil Aircraft Model (RCAM)")
    print(Style.BRIGHT + Fore.CYAN + "Aircraft designer: ", end="")
    print("Group for Aeronautical Research and Technology in Europe (GARTEUR FM AG08)")
    print(Style.BRIGHT + Fore.CYAN + "Design envelope: ", end="")
    print("Approach | Valid velocity(airspeed) Va: 60-140 m/s")
    print(Style.BRIGHT + Fore.CYAN + "Trim condition: ", end="")
    print("Steady state, straight and level flight")
    print(Style.BRIGHT + Fore.CYAN + "Assessment: ", end="")
    print("MIL-HDBK-1797A (Class III, Category B)")
    print(Style.BRIGHT + Fore.LIGHTBLACK_EX + "-"*90)
    

    # Step 1. Instantiate parameters and aircraft model
    param_profile1 = RCAM_parameters()
    rcam_model1 = Aircraft_model(param_profile1)
    
    # Step 2. User inputs
    # Validate Desired Cruise Speed [m/s]
    while True:

        try:
            Va = float(input("Desired trim velocity Va [m/s] = "))

            if Va < 60.0: 
                print(f"Invalid. Va = {Va} must be >= 60 m/s\n")
            elif Va > 140.0:
                print(f"Invalid. Va = {Va} must be <= 140 m/s\n")
            else: 
                break

        except ValueError:
            print(f"Invalid. (Va) must be a number\n")

    # Altitude only sets the initial NED position for plots (atmosphere is constant sea level)
    while True:

        try:
            h = float(input("Initial Altitude h = "))

            if h < 0.0: 
                print(f"Invalid: h = {h} m must be >= 0 m.\n")
            else: 
                break

        except ValueError:
            print(f"Invalid. (h) is not a number\n")

    # Step 3. Compute trim condition
    print("\n")
    z0 = trim_straight_level_flight(Va, h, rcam_model1)

    # Step 4. Linearize around trim condition and decouple state and control matrices
    A, B, A_transformed, B_transformed = linearization(z0, rcam_model1)

    # Step 5. Dynamic stability and MIL-HDBK-1797A flying qualities tables
    stability_analysis(A_transformed)

    # Step 6. Interactive menu for doublet response plots
    while True:

        try:
            actuator = int(input("Select a number to plot a response\n    " \
            "1. RCAM response to aileron doublet input (Roll subsidence and spiral modes)\n    " \
            "2. RCAM response to elevator doublet input (Short and phugoid modes)\n    " \
            "3. RCAM response to rudder doublet input (Dutch Roll (Coupled Lateral-Directional)\n    " \
            "4. RCAM response to engines doublet input (Short and phugoid modes)\n    " \
            "5. Finish the program\n"))

            if actuator < 1 and actuator > 5: 
                print(f"Invalid. Selection '{actuator}' must be between 1 and 5\n")
            if actuator == 5: break

            response_doublet_input(z0, param_profile1, rcam_model1, actuator-1, h)

        except ValueError:
            print(f"Invalid. selection '{actuator}' is not a number\n")

# Required for Pytest
if __name__ == "__main__":
    main()