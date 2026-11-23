import numpy as np
from tabulate import tabulate
from colorama import Fore, Style

def linearization(z0, aircraft_model_object, step_size=1e-6):
    """
    This function linearizes the nonlinear aircraft model around the trim condition
    using numerical differentiation to enable linear stability analysis and flight control design.

    Background
    =========================
    The linear representation is based on small perturbation theory. 
    It approximates the nonlinear system using a first order multivariable Taylor expansion:

        delta_x_dot = A*delta_x + B*delta_u 
        (Linear time-invariant system representation)

    The Jacobian matrices (A,B) are computed using the Central difference scheme,
    which provides higher accuracy compared to forward and backward difference schemes.

    Formula used to approximate column "j" of the state matrix "A":
        
        A[:, j] ≈ ( f(x0 + h_j, ue) - f(x0 - h_j, ue) ) / ( 2 * h_j )

    Formula used to approximate column "j" of the control input matrix "B":

        B[:, j] ≈ ( f(x0, ue + h_j) - f(x0, ue - h_j) ) / ( 2 * h_j )
    
    Inputs:
    =========================

        - z0 = [x0, ue]: Trim vector containing steady state values and equilibrium inputs
        - aircraft_model_object: Instance of the Aircraft_model class
        - step_size: Perturbation step size for numerical differentiation (1e-6 as default)

    Returns:
    =========================
        - A: (n x n) State jacobian matrix (original state ordering)
        - B: (n x m) Control jacobian matrix (original state ordering)
        - A_transformed: (n x n) Reordered state matrix decoupling longitudinal and lateral-directional dynamics
        - B_transformed: (n x m) Reordered control input matrix matching A_transformed
    """

    # ----- Step 1. Slice the trim vector into steady state values (x0) and equilibrium control input (ue) -----
    x0 = z0[:9]
    ue = z0[9:]

    n_states = len(x0)
    m_inputs = len(ue)


    # ----- Step 2. Perturbation settings -----
    # Define the perturbation step size "h" for states and control inputs
    steps_x0 = np.full(n_states, step_size, dtype=float)
    steps_ue = np.full(m_inputs, step_size, dtype=float)

    # STATE VARIABLES PERTURBATION MATRIX
    # Diagonal matrix where each diagonal entry represents the perturbation magnitude for state  "j".
    h_x0 = np.diag(steps_x0)

    # CONTROL INPUT VARIABLES PERTURBATION MATRIX
    # Diagonal matrix where each diagonal entry represents the perturbation magnitude for input control "j".
    h_ue = np.diag(steps_ue)


    # ----- Step 3. Compute Jacobian "A" (state matrix) -----
    # Initialize (n x n) matrix
    A = np.zeros((n_states, n_states))

    # Loop through each state variable (columns of "A")
    for j in range(n_states):
        
        # Extract perturbation vector "h_x0[:,j]" and magnitude for state "j" 
        pv_hx0 = h_x0[:,j] 
        pmag_hx0 = h_x0[j,j]

        # ----- Forward perturbation -----
        # Evaluate dynamics at (x0 + h_x0[:,j]) keeping control input constant
        aircraft_model_object.x = x0 + pv_hx0
        aircraft_model_object.u = ue

        f_forward = aircraft_model_object.equations_of_motion_model()
        
        # ----- Backward perturbation -----
        # Evaluate dynamics at (x0 - h_x0[:,j]) keeping controls constant     
        aircraft_model_object.x = x0 - pv_hx0
        aircraft_model_object.u = ue
        
        f_backward= aircraft_model_object.equations_of_motion_model()


        # --- Central difference computation ---
        # Compute the partial derivatives for the current column
        A[:,j] = (f_forward - f_backward)/(2*pmag_hx0)


    # ----- Step 4. Compute Jacobian "B" (control input matrix)
    # Initialize (n x m) matrix
    B = np.zeros((n_states, m_inputs))

    # Loop through each control input (columns of "B")
    for j in range(m_inputs):
        
        # Extract perturbation vector "h_ue[:,j]" and magnitude for the "j" input  
        pv_hue = h_ue[:,j] 
        pmag_hue = h_ue[j,j]


        # ----- Forward perturbation -----
        # Evaluate dynamics at (ue + h_ue[:,j]) keeping states constant
        aircraft_model_object.x = x0 
        aircraft_model_object.u = ue + pv_hue

        f_forward = aircraft_model_object.equations_of_motion_model()
        
        # ----- Backward perturbation -----
        # Evaluate dynamics at (ue - h_x0[:,j]) keeping states constant     
        aircraft_model_object.x = x0 
        aircraft_model_object.u = ue - pv_hue
        
        f_backward= aircraft_model_object.equations_of_motion_model()


        # --- Central difference computation ---
        # Compute the partial derivatives for the current column
        B[:,j] = (f_forward - f_backward)/(2*pmag_hue)

        # Restore the aircraft model object to its exact trim state
        #aircraft_model_object.x = x0
        #aircraft_model_object.u = ue

        # ----- Step 5. Similarity transformation to reorder and decouple states -----
        # Reorders states from [u, v, w, p, q, r, phi, theta, psi]
        # to [u, w, q, theta, v, p, r, phi, psi]
        STM = np.array([[1,0,0,0,0,0,0,0,0], 
                        [0,0,0,0,1,0,0,0,0], 
                        [0,1,0,0,0,0,0,0,0],
                        [0,0,0,0,0,1,0,0,0], 
                        [0,0,1,0,0,0,0,0,0], 
                        [0,0,0,0,0,0,1,0,0],
                        [0,0,0,0,0,0,0,1,0], 
                        [0,0,0,1,0,0,0,0,0], 
                        [0,0,0,0,0,0,0,0,1]])

        A_transformed = np.transpose(STM)@A@STM
        B_transformed = np.transpose(STM)@B

        # ----- Step 6. Report linearization results -----
    print("\n")
    print("="*45)
    print(Style.BRIGHT + "*** LINEALIZATION AROUND THE TRIM CONDITION ***" )
    print(f"    delta_x_dot = A*delta_x + B*delta_u")
    print("="*45)
    print("")


    # ----- Display decoupled state matrix A ------
    print(Style.BRIGHT + "- DECOUPLED STATE MATRIX [A]:")

    state_derivatives = ["u_dot", "w_dot", "q_dot", "theta_dot", "v_dot", "p_dot", "r_dot", "phi_dot", "psi_dot"]

    A_table = []
    for i in range(n_states):

        row = []
        row.append(Style.BRIGHT + state_derivatives[i] + Style.NORMAL)
    
        for j in range(n_states):

            if i <= 3 and j <= 3:

                if np.abs(A_transformed[i,j]) <= 1e-4:
                    row.append("0.0")
                else:
                    row.append(f"{A_transformed[i,j]:.4f}")
            
            elif (i >=4 and i < 9) and (j >=4 and j < 9):

                if np.abs(A_transformed[i,j]) <= 1e-4:
                    row.append("0.0")
                else:
                    row.append(f"{A_transformed[i,j]:.4f}")

            else:
                row.append(f".")
            
        A_table.append(row)

    header = [" ","u", "w", "q", "theta", "v", "p", "r", "phi", "psi"]
    
    print(tabulate(A_table,
                   headers=header, 
                   tablefmt="fancy_grid", 
                   stralign="center",
                   disable_numparse=True))
    

# ----- Display decoupled Control Input matrix B ------
    print("\n")
    print(Style.BRIGHT + "- DECOUPLED CONTROL INPUT MATRIX [B]:" + Style.NORMAL)

    B_table = []
    for i in range(n_states):

        row = []
        row.append(Style.BRIGHT + state_derivatives[i] + Style.NORMAL)
        
        for j in range(m_inputs):

            if np.abs(B_transformed[i,j]) <= 1e-4:
                row.append("0.0")
            else:
                row.append(f"{B_transformed[i,j]:.4f}")
            
        B_table.append(row)

    header = [Style.BRIGHT +" ","delta_a", "delta_e", "delta_r", "delta_th1", "delta_th2" + Style.NORMAL]
    
    print(tabulate(B_table,
                   headers=header, 
                   tablefmt="fancy_grid", 
                   stralign="center",
                   disable_numparse=True))

    return A, B, A_transformed, B_transformed
