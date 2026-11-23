import numpy as np
from scipy.optimize import minimize
from colorama import Back, Fore, Style, init
from tabulate import tabulate


def trim_straight_level_flight(V_des, h, aircraft_model_object):
    """
    This function computes the trim condition required for steady state, straight and level flight
    for the nonlinear RCAM model. 

    This function requires as input:
        - V_des: Desired airspeed [m/s] given by the user
        - h: Initial flight altitude [m] given by the user
        - aircraft_model_object: Instance of the Aircraft_model class

    This function returns:
        - z0 = [x0, ue]: the feasible decision vector that satisfies the trim condition, where:
            x_0 in R^n: are the states values that satisfy the trim condition 
            u_e in R^m: are the control values required to maintain the trim condition

    Summary.
    The general equation of motion are composed of 9 nonlinear differential equations, in the RCAM model
    there are 9 states and 5 inputs, resulting in 14 decision variables. 

    The RCAM equations of motion form an undetermined system, therefore, multiple solutions exist. 
    The analitycal approaches, such as substituting known variables, allow to solve the system. 
    However, due to the complexity of the force and moment equations, obtaining analitycal solutions 
    becomes a challenging problem.
    
    An alternative approach to solve the equations of motion is the use of numerical 
    optimization methods such as Nelder-mead (Simplex method), least squares, or sequential least 
    squares quadratic programming also known as (SLSQP), among others. 
    The aim is to find a feasible set z0 = [x0,ue] such that x_dot = f(x0,ue) ≈ 0, instead of 
    minimizing a cost function.

    Numerical optimization methods allow to define a feasibility problem for x_dot = f(x,u) subjected 
    to equality constraints (such as steady state constraints, flight condition constraints, 
    and physical symmetry constraints) to find a feasible set to find an equilibrium point.

    Due to the complexity of the constraints, it is possible to convert a contrainted feasibility
    problem into an uncostrainted feasibility problem using the Penalty Method. This method adds 
    the squared constraints residuals to the objective function to penalize constraint violations.
    
    The expresion for the uncontraint cost function is given as:  f_unc(z) = f0 + (c^T)Hc
    Where:
        - f0: Original cost function (since we are not interested in minimizing (feseability problem), it is zero)
        - c:  Constraint residuals vector evaluated with "z" decision variables
        - H:  Diagonal weighting matrix containing the penalty parameters

    The aim of this function is to find the decision variables z0=[x0,ue] that satisfy x_dot = f(x0,ue) ≈ 0
    
    The following steps are implemented:

    1. Define the uncounstrained penalty cost function
    2. Define an initial guess for the decision variables. 
    For straight level flight the following initial values are proposed:
    z0_guess = [V_des, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0, 0.05, 0.05]
    3. Select a numerical solver for the feasibility problem:
    Scipy provides several local minimization solvers. Nelder-mead, which is a direct search method (simplex), 
    is useful when derivatives are unavailable. It doesn't require computing the gradient; however, 
    convergence can be slow and there is no guarantee of convergence. Also oscillatory behavior may occur 
    if simplex parameters are not properly tuned.

    In this script, the SLSQP algorithm is used, due to its performance to handle nonlinear objective functions 
    using gradient information and explicit bounds (defined in the Aircraft_model class)
    """

    # Step 1. Define the uncounstrained cost function
    def unc_cost_function(z,V_des,aircraft_model_object):
        """
        This nested function evaluates the uncountrained cost function using the quadratic penalty method

        Input:
            - z in R^(n+m): Decision variables vector [x,u]
            - V_des: Desired airspeed [m/s] given by the user
            - aircraft_model_object: Instance of the Aircraft_model class
        
        Output:
            - f_obj: Scalar quadratic penalty cost (higher value means larger constraint violations)
        """

        # Step 1.1 Assign the decision variables (state and input vectors) to the aircraft object
        aircraft_model_object.x = z[:9]
        aircraft_model_object.u = z[9:]
        
        # Step 1.2. Evaluate the constraints  (obtain residual vector "c_vector" from 
        # the "straight_level_flight_constraints" method)
        c_vector = aircraft_model_object.straight_level_flight_constraints(V_des)

        # Step 1.3. Create the penalty weighting matrix "H"
        # (each element of the diagonal contains the penalty weight for the corresponding constraint)
        H = np.eye(len(c_vector))

        # Penalty parameters can be adjusted manually to improve convergence
        H = 1.0*H

        # Step 1.4. Compute the scalar quadratic cost 
        f_obj = 0.0 + np.transpose(c_vector)@H@c_vector
        return f_obj

    # Step 2. Define the initial guess vector
    # where z = [u, v, w, p, q, r, phi, theta, psi, u_a, u_e, u_r, u_th1, u_th2]
    # Small non zero values are assigned to u_e, u_th1, u_th2 to aid gradient estimation
    z0_guess = [V_des,0,0,0,0,0,0,0,0,0,0.01,0,0.05,0.05] 

    # Step 3. Solver the feasiblity problem using the SLSQP optimization method
    print("="*45)
    print(Style.BRIGHT + "*** COMPUTATION OF FLIGHT TRIM CONDITION ***")
    print("="*45)
    print(Style.BRIGHT + "- FLIGHT CONDITION:" + Style.NORMAL + " Steady state, straight and level flight")
    print(Style.BRIGHT + "- SOLVER ALGORITHM:" + Style.NORMAL + " SLSQP (Sequential Least Squares Quadratic Programming)")
    print(Style.BRIGHT + f"- DESIRED CRUISE SPEED:" + Style.NORMAL + f" {V_des} m/s")
    print(Style.BRIGHT + f"- FLIGHT ALTITUDE:" + Style.NORMAL + f" {h} m")

    # Solver settings
    sol_z = minimize(unc_cost_function, 
                   z0_guess, 
                   args = (V_des,aircraft_model_object), 
                   method = 'slsqp',
                   tol = 1e-12, 
                   options = {"maxiter":1000, "ftol": 1e-12}, 
                   bounds = aircraft_model_object.aircraft_model_bounds())
    
    # Filter numerical noise: substitute with 0.0 when |z_i| <= 1e-4
    # Round for display
    z = []
    for i in sol_z.x:
        if np.absolute(i) <= 1e-4:
            z.append(0.0)
        else:
            z.append(f"{i:.5f}")

    z = np.array(z, dtype=float) 

    # Convert angular variables (inidices 3 to 13) from radians to degrees and filter numerical noise 
    z_rad = []

    for i in sol_z.x[3:]:
        if np.absolute(i) <= 1e-4:
            z_rad.append(0.0)
        else:
            z_rad.append(f"{np.rad2deg(i):.5f}")

    z_rad = np.array(z_rad, dtype=float)


    print("")
    print(f"-"*30)
    print(Style.BRIGHT + "*** OPTIMIZATION RESULTS ***")
    print(f"-"*30)

    # TRIM CONDITION REPORT
    if sol_z.success:

        print(Style.BRIGHT + "- RESULT: " + Style.NORMAL + Fore.GREEN + f"{sol_z.message}")
        print(Style.BRIGHT + "- Final unc_cost_function residual value: "  + Style.NORMAL + f"{sol_z.fun:.5e}")
        print("")
        
        print(Fore.CYAN +"="*40 + " STEADY STATES VALUES " + "="*40)

        header1 = ["Variable", "State id.", "Description", "Value (SI)", "Value (deg)"]

        table1 = [["u", "x[0]", "Axial velocity (in Body frame Fb)", f"{z[0]} m/s", "-----"],
                 ["v", "x[1]", "Lateral velocity (in Body frame Fb)", f"{z[1]} m/s" ,"-----"],
                 ["w", "x[2]", "Vertical velocity (in Body frame Fb)", f"{z[2]} m/s" ,"-----"],
                 ["p", "x[3]", "Roll rate (in Body frame Fb)", f"{z[3]} rad/s", f"{z_rad[0]} deg/s"],
                 ["q", "x[4]", "Pitch rate (in Body frame Fb)", f"{z[4]} rad/s", f"{z_rad[1]} deg/s"],
                 ["r", "x[5]", "Yaw rate (in Body frame Fb)", f"{z[5]} rad/s", f"{z_rad[2]} deg/s"],
                 ["phi", "x[6]", "Roll angle (Euler angle)", f"{z[6]} rad", f"{z_rad[3]} deg"],
                 ["theta", "x[7]", "Pitch angle (Euler angle)", f"{z[7]} rad", f"{z_rad[4]} deg"],
                 ["psi", "x[8]", "Yaw angle (Euler angle)", f"{z[8]} rad", f"{z_rad[5]} deg"]]

        print(tabulate(table1,header1, tablefmt="fancy_grid"))

        print("\n")
        print(Fore.CYAN +"="*25 + " INPUT VALUES REQUIRED TO MAINTAIN STEADY STATES " + "="*25)
        header2 = ["Symbol", "Input id", "Description", "Value (SI)", "Value (deg)"]
        
        table2 = [["delta_aileron", "u[0]", "Aileron deflection", f"{z[9]} rad", f"{z_rad[6]} deg"],
                 ["delta_elevator", "u[1]", "Elevator deflection", f"{z[10]} rad", f"{z_rad[7]} deg"],
                 ["delta_rudder", "u[2]", "Rudder deflection", f"{z[11]} rad", f"{z_rad[8]} deg"],
                 ["delta_engine_1", "u[3]", "Throttle position of engine 1", f"{z[12]} rad", f"{z_rad[9]} deg"],
                 ["delta_engine_2", "u[4]", "Throttle position of engine 2", f"{z[13]} rad", f"{z_rad[10]} deg"]]

        print(tabulate(table2,header2, tablefmt="fancy_grid"))
    
        return sol_z.x
    else:
        
        print(Style.BRIGHT + "- RESULT: " + Style.NORMAL + Fore.RED + f"{sol_z.message}")
        print(Style.BRIGHT + "- Final unc_cost_function residual value: "  + Style.NORMAL + f"{sol_z.fun:.5e}")
        return sol_z.success
