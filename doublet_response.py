import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def response_doublet_input(z0, aircraft_params_object, aircraft_model_object, actuator, h=500, t_f=100, inc=0.01):
    """
    This function simulates and plots the response of the nonlinear time domain response of the RCAM to a doublet input 
    applied to a specific control surface or engine throttle.

    This function compares the aircraft response with ideal (instantaneous command) control inputs against modeled
    first order actuator and propulsion dynamics (including rate and position saturation limits). 
    It identifies and annotates the predominant dynamic modes on each plot.
    
    Inputs:
    =========================
        - z0 = [x0, ue]: Trim vector containing steady states values and equilibrium control input
        - aircraft_model_object: Instance of the Aircraft_model class
        - aircraft_param_object: Instance of the Aircraft_parameters dataclass
        - actuator: Index of the control input to excite:
            0 : Aileron (Excite Roll Subsidence & Spiral modes)
            1 : Elevator (Excite Short Period & Phugoid modes)
            2 : Rudder  (Excite Dutch Roll mode)
            3 : Throttle (Excite Short Period & Phugoid modes)
        - h: Initial simulation altitude [m], (default: 500.0)
        - t_f: Total simulation time [s], (default: 100.0)
        - inc: time step for solver [s], (default: 0.01)
    
    Returns:
    =========================
        None. This function plots the following results:
        - Actuator = 0: RCAM response to aileron doublet input (Roll subsidence and spiral modes)
        - Actuator = 1: RCAM response to elevator doublet input (Short Period and phugoid modes)
        - Actuator = 2: RCAM response to rudder doublet input (Dutch Roll (Coupled Lateral-Directional))
        - Actuator = 3: RCAM response to engines doublet input (Short Period and phugoid modes)
    """



    # Step 1. Define time vector for simulation
    t = np.arange(0.0, t_f+inc, inc) if (0 < inc <= t_f) else np.arange(0.0, 300.0 + 0.01, 0.01)

    # Step 2. Define 12 initial states (x0) and equilibrium control input vector (u_eq)
    # Note: In the NED frame, altitude h corresponds to PosD = -h
    x0 = np.concatenate([z0[:9], [0,0,-h]])
    u_eq = z0[9:]

    # Step 3. Define external doublet perturbation signal during flight simulation.
    def u_ext(t_current, actuator_idx, t_ini_step = 10.0, step_duration = 5.0, step_amplitude = 3.0 * np.pi / 180.0):

        if actuator_idx < 0 or actuator_idx >= len(u_eq):
            raise ValueError(f"Invalid actuator index: {actuator_idx}. Must be between 0 and {len(u_eq)-1}.")
        
        u_input = np.zeros(len(u_eq)) # Initialize control input vector with zeros

        if t_current < t_ini_step:

            return np.zeros(len(u_eq))  # No input 
        
        elif t_ini_step <= t_current < t_ini_step + step_duration:

            if actuator_idx == 3: # Simultaneous symmetric doublet input on both engines (throttles 1 and 2)
                u_input[3:] = [step_amplitude, step_amplitude]

            else:
                u_input[actuator_idx] = step_amplitude
                
            return u_input
        
        elif t_ini_step + step_duration <= t_current < t_ini_step + 2.0*step_duration:

            if actuator_idx == 3: # Simultaneous symmetric doublet input on both engines (throttles 1 and 2)
                u_input[actuator_idx:] = [-step_amplitude,-step_amplitude]
            else:
                u_input[actuator_idx] = -step_amplitude

            return u_input
        else:

            return np.zeros(len(u_eq))  # Return to trim command


    # Step 4. Define wrapper function combining 6 DoF equations of motion and navigation equations
    # (Ideal actuators: returns 12 state derivatives)
    def system_motion_nav(x, t_current, aircraft_model_object, u_eq, u_ext_fcn, actuator_idx):

        aircraft_model_object.x = x[:9]                      # Update state variables (9 states)
        aircraft_model_object.u = u_eq + u_ext_fcn(t_current,actuator_idx)   # Apply ideal instantaneous command

        # Compute derivatives of motion using equations of motion model for 9 states
        dxdt_motion = aircraft_model_object.equations_of_motion_model()  

        # Compute derivatives of position using navigation equation for 3 states
        dxdt_nav = aircraft_model_object.navigation_equation()

        # Combine derivatives into a single vector
        return np.concatenate((dxdt_motion, dxdt_nav))

    # Step 5. Define wrapper function combining equations of motion, navigation, and actuator dynamics
    # (Modeled actuators: returns 17 state derivatives)
    def system_motion_nav_actuators(x, t_current, aircraft_model_object, u_eq, u_ext_fcn, actuator_idx):

        # Unpack states, navigation states, and actual actuator states
        states_motion = x[:9]
        states_nav = x[9:12]
        u_response = x[12:]

        # # Compute total commanded control vector (ideal input)
        u_cmd = u_eq + u_ext_fcn(t_current,actuator_idx)

        # Set current states to the aircraft object
        aircraft_model_object.x = states_motion            

        # Set actual actuator positions to the aircraft object  
        aircraft_model_object.u = u_response

        # Compute derivatives for motion, navigation, and first order actuators
        dxdt_motion = aircraft_model_object.equations_of_motion_model()  
        dxdt_nav = aircraft_model_object.navigation_equation()
        dxdt_actuators = aircraft_model_object.actuator_dynamics(u_cmd, u_response)

        return np.concatenate((dxdt_motion, dxdt_nav, dxdt_actuators))

    # Step 6. Numerically integrate the nonlinear differential equations using odeint
    # Solve system without actuator dynamics (ideal response)
    x_ideal = odeint(system_motion_nav, x0, t, args=(aircraft_model_object, u_eq, u_ext,actuator))

    # Solve system with first order actuator dynamics (actual response)
    x_response = odeint(system_motion_nav_actuators, np.concatenate([x0,u_eq]), t, args=(aircraft_model_object, u_eq, u_ext,actuator))


    # Step 7. Plot simulation results
    match actuator:
        case 0: # Aileron

            # Rebuild command history over t (odeint only stores states) and convert to deg
            # u_ext(t_i,actuator)[actuator] extract the element given in the "actuator" position 
            aileron_cmd= [(u_eq[actuator] + u_ext(t_i,actuator)[actuator])*(180/np.pi) for t_i in t]
            
            # Actuator states are stored right after the 12 motion/nav states (len(x0))
            # Elevator response with actuator dynamics (real response)
            # x_response is 2D, we are extracting the column "len(x0)+actuator"
            aileron_response = x_response[:,len(x0)+actuator]*(180/np.pi) 

            # Define subplots settings
            fig, axs = plt.subplots(3)
            plt.figure(1)
            plt.suptitle("RCAM response to aileron doublet input", fontsize=16, fontweight="bold")

            # Plot aileron input
            # 1. Input: commanded vs. actual aileron deflection
            axs[0].set_title("Aileron deflection (δ_a)")
            axs[0].set_ylabel("δ_a [deg]")
            axs[0].plot(t, aileron_cmd, color="black", label="δ_a Command", linestyle="--", alpha=0.5, zorder=2)
            axs[0].plot(t, aileron_response, color="blue", label="δ_a Aileron deflection response")            
            axs[0].grid()
            axs[0].legend(loc="lower right", fontsize=9, framealpha=0.1)
            axs[0].text(0.2, 0.9, f"Actuator Limits: [{aircraft_params_object.aileron_saturation_min*180/np.pi:.1f}°, {aircraft_params_object.aileron_saturation_max*180/np.pi:.1f}°]", 
                        transform=axs[0].transAxes, fontsize=10, verticalalignment='top',bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


            # 2. Roll response: roll rate p (left) and roll angle phi (right)
            axs[1].set_title("Response: Roll rate (p) and Phi angle (φ)")
            axs[1].set_xlabel("Time [s]")
            axs[1].set_ylabel("p [deg/s]", color="dodgerblue")
            axs[1].plot(t, x_ideal[:,3]*180/np.pi, color="dodgerblue", label="p_ideal", linestyle="--", alpha=0.5, zorder=2)
            axs[1].plot(t, x_response[:,3]*180/np.pi, color="dodgerblue", label="p_actual")
            axs[1].tick_params(axis='y', labelcolor="dodgerblue")
            axs[1].grid()
            axs[1].legend(loc="lower right", fontsize=9, framealpha=0.1)

            # Doublet start/end markers
            axs[1].axvline(10, color="gray", linestyle="--", alpha = 0.5, linewidth=1, zorder=1)
            axs[1].axvline(20, color="gray", linestyle="--", alpha = 0.5, linewidth=1, zorder=1)
            # Textbox for meaning
            axs[1].text(0.6, 0.95, f"_ideal: Response without actuator dynamics,\n_actual: response with actuator modeled", 
                        transform=axs[1].transAxes, fontsize=8,  
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Textbox for dominant mode
            axs[1].text(0.75, 0.05, "Dominant Mode: (p) Roll Subsidence / (φ) Spiral", transform=axs[1].transAxes,        
                        fontsize=8, verticalalignment='bottom', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

            # Right y-axis for roll angle (phi)
            axs1 = axs[1].twinx()
            axs1.set_ylabel("φ [deg]", color="crimson")
            axs1.plot(t,x_ideal[:,6]*180/np.pi, color="crimson", label="φ_ideal", linestyle="--", alpha=0.5, zorder=2)
            axs1.plot(t,x_response[:,6]*180/np.pi, color="crimson", label="φ_actual")
            axs1.tick_params(axis='y', labelcolor="crimson")
            axs1.grid()
            axs1.legend(loc="upper right", fontsize=9, framealpha=0.1)


            # 3. Yaw coupling: yaw rate r (left) and yaw angle psi (right)
            axs[2].set_title("Response: Yaw rate (r) and Yaw angle (ψ)")
            axs[2].set_xlabel("Time [s]")
            axs[2].set_ylabel("r [deg/s]", color="dodgerblue")
            axs[2].plot(t, x_ideal[:,5]*180/np.pi, color="dodgerblue", label="r_ideal", linestyle="--", alpha=0.5, zorder=2)
            axs[2].plot(t, x_response[:,5]*180/np.pi, color="dodgerblue", label="r_actual")
            axs[2].tick_params(axis='y', labelcolor="dodgerblue")
            axs[2].grid()
            axs[2].legend(loc="lower right", fontsize=9, framealpha=0.1)

            # Doublet start/end markers
            axs[2].axvline(10, color="gray", linestyle="--", alpha = 0.5, linewidth=1, zorder=1)
            axs[2].axvline(20, color="gray", linestyle="--", alpha = 0.5, linewidth=1, zorder=1)
            
            # Textbox for meaning
            axs[2].text(0.6, 0.95, f"_ideal: Response without actuator dynamics,\n_actual: response with actuator modeled", 
                        transform=axs[2].transAxes, fontsize=8,  
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Right y-axis for yaw angle (psi)            
            axs1 = axs[2].twinx()
            axs1.set_ylabel("ψ [deg]", color="crimson")
            axs1.plot(t,x_ideal[:,8]*180/np.pi, color="crimson", label="ψ_ideal", linestyle="--", alpha=0.5, zorder=2)
            axs1.plot(t,x_response[:,8]*180/np.pi, color="crimson", label="ψ_actual")
            axs1.tick_params(axis='y', labelcolor="crimson")
            axs1.grid()
            axs1.legend(loc="upper right", fontsize=9, framealpha=0.1)

            fig.tight_layout()
            plt.show()


        case 1: # Elevator

            # Extract commanded and actual elevator deflection vectors [deg]
            elevator_cmd= [(u_eq[actuator] + u_ext(t_i,actuator)[actuator])*(180/np.pi) for t_i in t]
            elevator_response = x_response[:,len(x0)+actuator]*(180/np.pi) # Elevator response with actuator dynamics (real response)

            # Define subplots settings
            fig, axs = plt.subplots(3)
            plt.figure(1)
            plt.suptitle("RCAM response to elevator doublet input", fontsize=16, fontweight="bold")

            # Subplot 1: Elevator deflection
            axs[0].set_title("Elevator deflection (δ_e)")
            axs[0].set_ylabel("δ_e [deg]")
            axs[0].plot(t, elevator_cmd, color="black", label="δ_e Command", linestyle="--", alpha=0.5, zorder=2)
            axs[0].plot(t, elevator_response, color="blue", label="δ_e Elevator deflection response")            
            #axs[0].axvline(10, color="gray", linestyle="--", alpha = 0.5, linewidth=1, zorder=1)
            #axs[0].axvline(20, color="gray", linestyle="--", alpha = 0.5, linewidth=1, zorder=1)
            axs[0].grid()
            axs[0].legend(loc="lower right", fontsize=9, framealpha=0.1)
            axs[0].text(0.2, 0.9, f"Actuator Limits: [{aircraft_params_object.elevator_saturation_min*180/np.pi:.1f}°, {aircraft_params_object.elevator_saturation_max*180/np.pi:.1f}°]", 
                        transform=axs[0].transAxes, fontsize=10, verticalalignment='top',bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Subplot 2: Angle of Attack (alpha) and Pitch attitude (theta)
            alpha_ideal = np.arctan(x_ideal[:,2]/x_ideal[:,0])*(180/np.pi)
            alpha_response = np.arctan(x_response[:,2]/x_response[:,0])*(180/np.pi)
            axs[1].set_title("Response: Angle of Attack (α) and Pitch attitude (θ)")
            axs[1].set_ylabel("α & θ [deg]")
            axs[1].plot(t, alpha_ideal, color="blue", label="α_ideal", linestyle="--", alpha=0.5, zorder=2)
            axs[1].plot(t, alpha_response, color="blue", label="α_actual")
            axs[1].plot(t,x_ideal[:,7]*(180/np.pi), color="darkorange", label="θ_ideal", linestyle="--", alpha=0.5, zorder=2)
            axs[1].plot(t,x_response[:,7]*(180/np.pi), color="darkorange", label="θ_actual")
            axs[1].grid()
            axs[1].legend(loc="lower right", fontsize=9, framealpha=0.1)
            axs[1].axvline(10, color="gray", linestyle="--", alpha = 0.5, linewidth=1, zorder=1)
            axs[1].axvline(20, color="gray", linestyle="--", alpha = 0.5, linewidth=1, zorder=1)
            # Textbox for meaning
            axs[1].text(0.6, 0.95, f"_ideal: Response without actuator dynamics,\n_actual: response with actuator modeled", 
                        transform=axs[1].transAxes, fontsize=8,  
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            # Textbox for dominant mode
            axs[1].text(0.75, 0.05, "Dominant Mode: (α) Short Period / (θ) Short Period and Phugoid ", transform=axs[1].transAxes,        
                        fontsize=8, verticalalignment='bottom', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))


            # Subplot 3: Altitude (h) and Airspeed (V)
            axs[2].set_title("Response: Altitude (h) and Speed (V)")
            axs[2].set_xlabel("Time [s]")
            axs[2].set_ylabel("Altitude [m]", color="dodgerblue")
            axs[2].plot(t, -x_ideal[:,11], color="dodgerblue", label="h_ideal", linestyle="--", alpha=0.5, zorder=2)
            axs[2].plot(t, -x_response[:,11], color="dodgerblue", label="h_actual")
            axs[2].tick_params(axis='y', labelcolor="dodgerblue")
            axs[2].grid()
            axs[2].legend(loc="lower right", fontsize=9, framealpha=0.1)
            axs[2].axvline(10, color="gray", linestyle="--", alpha = 0.5, linewidth=1, zorder=1)
            axs[2].axvline(20, color="gray", linestyle="--", alpha = 0.5, linewidth=1, zorder=1)
            # Textbox for meaning
            axs[2].text(0.6, 0.95, f"_ideal: Response without actuator dynamics,\n_actual: response with actuator modeled", 
                        transform=axs[2].transAxes, fontsize=8,  
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            # Textbox for dominant mode
            axs[2].text(0.75, 0.05, "Dominant Mode: (h) Phugoid / (V) Phugoid", transform=axs[2].transAxes,        
                        fontsize=8, verticalalignment='bottom', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

            axs1 = axs[2].twinx()
            axs1.set_ylabel("Speed [m/s]", color="crimson")
            axs1.plot(t,np.sqrt(x_ideal[:,0]**2 + x_ideal[:,1]**2 + x_ideal[:,2]**2), color="crimson", label="V_ideal", linestyle="--", alpha=0.5, zorder=2)
            axs1.plot(t,np.sqrt(x_response[:,0]**2 + x_response[:,1]**2 + x_response[:,2]**2), color="crimson", label="V_actual")
            axs1.tick_params(axis='y', labelcolor="crimson")
            axs1.grid()
            axs1.legend(loc="upper right", fontsize=9, framealpha=0.1)

            fig.tight_layout()
            plt.show()



        case 2: # Rudder
            
            # Extract commanded and actual rudder deflection vectors [deg]
            rudder_cmd= [(u_eq[actuator] + u_ext(t_i,actuator)[actuator])*(180/np.pi) for t_i in t]
            rudder_response = x_response[:,len(x0)+actuator]*(180/np.pi) # Elevator response with actuator dynamics (real response)

            # Define subplots settings
            fig, axs = plt.subplots(3)
            plt.figure(1)
            plt.suptitle("RCAM response to rudder doublet input", fontsize=16, fontweight="bold")

            # Subplot 1: Rudder deflection
            axs[0].set_title("Rudder deflection (δ_r)")
            axs[0].set_ylabel("δ_r [deg]")
            axs[0].plot(t, rudder_cmd, color="black", label="δ_r Command", linestyle="--", alpha=0.5, zorder=2)
            axs[0].plot(t, rudder_response, color="blue", label="δ_r Rudder deflection response")            
            #axs[0].axvline(10, color="gray", linestyle="--", alpha = 0.5, linewidth=1, zorder=1)
            #axs[0].axvline(20, color="gray", linestyle="--", alpha = 0.5, linewidth=1, zorder=1)
            axs[0].grid()
            axs[0].legend(loc="lower right", fontsize=9, framealpha=0.1)
            axs[0].text(0.2, 0.9, f"Actuator Limits: [{aircraft_params_object.rudder_saturation_min*180/np.pi:.1f}°, {aircraft_params_object.rudder_saturation_max*180/np.pi:.1f}°]", 
                        transform=axs[0].transAxes, fontsize=10, verticalalignment='top',bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Subplot 2: Roll rate (p) and Yaw rate (r)
            axs[1].set_title("Response: Roll rate (p) and Yaw rate (r)")
            axs[1].set_xlabel("Time [s]")
            axs[1].set_ylabel("p [deg/s]", color="dodgerblue")
            axs[1].plot(t, x_ideal[:,3]*180/np.pi, color="dodgerblue", label="p_ideal", linestyle="--", alpha=0.5, zorder=2)
            axs[1].plot(t, x_response[:,3]*180/np.pi, color="dodgerblue", label="p_actual")
            axs[1].tick_params(axis='y', labelcolor="dodgerblue")
            axs[1].grid()
            axs[1].legend(loc="lower right", fontsize=9, framealpha=0.1)
            axs[1].axvline(10, color="gray", linestyle="--", alpha = 0.5, linewidth=1, zorder=1)
            axs[1].axvline(20, color="gray", linestyle="--", alpha = 0.5, linewidth=1, zorder=1)
            # Textbox for meaning
            axs[1].text(0.6, 0.95, f"_ideal: Response without actuator dynamics,\n_actual: response with actuator modeled", 
                        transform=axs[1].transAxes, fontsize=8,  
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Textbox for dominant mode
            axs[1].text(0.75, 0.05, "Dominant Mode: Dutch Roll (Coupled Lateral-Directional)", transform=axs[1].transAxes,        
                        fontsize=8, verticalalignment='bottom', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))


            axs1 = axs[1].twinx()
            axs1.set_ylabel("r [deg/s]", color="crimson")
            axs1.plot(t,x_ideal[:,5]*180/np.pi, color="crimson", label="r_ideal", linestyle="--", alpha=0.5, zorder=2)
            axs1.plot(t,x_response[:,5]*180/np.pi, color="crimson", label="r_actual")
            axs1.tick_params(axis='y', labelcolor="crimson")
            axs1.grid()
            axs1.legend(loc="upper right", fontsize=9, framealpha=0.1)


            # Subplot 3: Sideslip angle (beta) and Roll angle (phi)
            beta_ideal = x_ideal[:,1]/np.sqrt(x_ideal[:,0]**2 + x_ideal[:,1]**2 + x_ideal[:,2]**2)
            beta_response = x_response[:,1]/np.sqrt(x_response[:,0]**2 + x_response[:,1]**2 + x_response[:,2]**2)
            axs[2].set_title("Response: Beta (β) and Phi angle (φ)")
            axs[2].set_xlabel("Time [s]")
            axs[2].set_ylabel("β [deg]", color="dodgerblue")
            axs[2].plot(t, beta_ideal*180/np.pi, color="dodgerblue", label="β_ideal", linestyle="--", alpha=0.5, zorder=2)
            axs[2].plot(t, beta_response*180/np.pi, color="dodgerblue", label="β_actual")
            axs[2].tick_params(axis='y', labelcolor="dodgerblue")
            axs[2].grid()
            axs[2].legend(loc="lower right", fontsize=9, framealpha=0.1)
            axs[2].axvline(10, color="gray", linestyle="--", alpha = 0.5, linewidth=1, zorder=1)
            axs[2].axvline(20, color="gray", linestyle="--", alpha = 0.5, linewidth=1, zorder=1)
            
            # Textbox for meaning
            axs[2].text(0.6, 0.95, f"_ideal: Response without actuator dynamics,\n_actual: response with actuator modeled", 
                        transform=axs[2].transAxes, fontsize=8,  
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Textbox for dominant mode
            axs[2].text(0.75, 0.05, "Dominant Mode: Dutch Roll", transform=axs[2].transAxes,        
                        fontsize=8, verticalalignment='bottom', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

                        
            axs1 = axs[2].twinx()
            axs1.set_ylabel("φ [deg]", color="crimson")
            axs1.plot(t,x_ideal[:,6]*180/np.pi, color="crimson", label="φ_ideal", linestyle="--", alpha=0.5, zorder=2)
            axs1.plot(t,x_response[:,6]*180/np.pi, color="crimson", label="φ_actual")
            axs1.tick_params(axis='y', labelcolor="crimson")
            axs1.grid()
            axs1.legend(loc="upper right", fontsize=9, framealpha=0.1)

            fig.tight_layout()
            plt.show()


        case 3:     # Engine Throttle

            # Extract commanded and actual engine throttle vectors [deg]
            throttle_input_cmd = [(u_eq[actuator] + u_ext(t_i,actuator)[actuator])*(180/np.pi) for t_i in t]
            throttle_input_response = x_response[:,len(x0)+actuator]*(180/np.pi) # Elevator response with actuator dynamics (real response)

            # Define subplots settings
            fig, axs = plt.subplots(3)
            plt.figure(1)
            plt.suptitle("RCAM response to engine throttle doublet input", fontsize=16, fontweight="bold")

            # Subplot 1: Engine throttle
            axs[0].set_title("Engine throttle (δ_th)")
            axs[0].set_ylabel("δ_th [deg]")
            axs[0].plot(t, throttle_input_cmd, color="blue", label="δ_th Command", linestyle="--", alpha=0.5, zorder=2)
            axs[0].plot(t, throttle_input_response, color="blue", label="δ_th Response")      
            #axs[0].axvline(10, color="gray", linestyle="--", alpha = 0.5, linewidth=1, zorder=1)
            #axs[0].axvline(20, color="gray", linestyle="--", alpha = 0.5, linewidth=1, zorder=1)
            axs[0].grid()
            axs[0].legend(loc="lower right", fontsize=9, framealpha=0.1)
            axs[0].text(0.2, 0.9, f"Actuator Limits: [{aircraft_params_object.engine_saturation_min*180/np.pi:.1f}°, {aircraft_params_object.engine_saturation_max*180/np.pi:.1f}°]", 
                        transform=axs[0].transAxes, fontsize=8, verticalalignment='top',bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


            # Subplot 2: Angle of Attack (alpha) and Pitch attitude (theta)
            alpha_ideal = np.arctan(x_ideal[:,2]/x_ideal[:,0])*(180/np.pi)
            alpha_response = np.arctan(x_response[:,2]/x_response[:,0])*(180/np.pi)
            axs[1].set_title("Response: Angle of Attack (α) and Pitch attitude (θ)")
            axs[1].set_ylabel("α & θ [deg]")
            axs[1].plot(t, alpha_ideal, color="blue", label="α_ideal", linestyle="--", alpha=0.5, zorder=2)
            axs[1].plot(t, alpha_response, color="blue", label="α_actual")
            axs[1].plot(t,x_ideal[:,7]*(180/np.pi), color="darkorange", label="θ_ideal", linestyle="--", alpha=0.5, zorder=2)
            axs[1].plot(t,x_response[:,7]*(180/np.pi), color="darkorange", label="θ_actual")
            axs[1].grid()
            axs[1].legend(loc="lower right", fontsize=9, framealpha=0.1)
            axs[1].axvline(10, color="gray", linestyle="--", alpha = 0.5, linewidth=1, zorder=1)
            axs[1].axvline(20, color="gray", linestyle="--", alpha = 0.5, linewidth=1, zorder=1)
            # Textbox for meaning
            axs[1].text(0.6, 0.95, f"_ideal: Response without actuator dynamics,\n_actual: response with actuator modeled", 
                        transform=axs[1].transAxes, fontsize=8,  
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            # Textbox for dominant mode
            axs[1].text(0.75, 0.05, "Dominant Mode: (α) Short Period and Phugoid / (θ) Phugoid", transform=axs[1].transAxes,        
                        fontsize=8, verticalalignment='bottom', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))


            # Subplot 3: Altitude (h) and Airspeed (V)
            axs[2].set_title("Response: Altitude (h) and Speed (V)")
            axs[2].set_xlabel("Time [s]")
            axs[2].set_ylabel("Altitude [m]", color="dodgerblue")
            axs[2].plot(t, -x_ideal[:,11], color="dodgerblue", label="h_ideal", linestyle="--", alpha=0.5, zorder=2)
            axs[2].plot(t, -x_response[:,11], color="dodgerblue", label="h_actual")
            axs[2].tick_params(axis='y', labelcolor="dodgerblue")
            axs[2].grid()
            axs[2].legend(loc="lower right", fontsize=9, framealpha=0.5)
            axs[2].axvline(10, color="gray", linestyle="--", alpha = 0.5, linewidth=1, zorder=1)
            axs[2].axvline(20, color="gray", linestyle="--", alpha = 0.5, linewidth=1, zorder=1)
            # Textbox for meaning
            axs[2].text(0.6, 0.95, f"_ideal: Response without actuator dynamics,\n_actual: response with actuator modeled", 
                        transform=axs[2].transAxes, fontsize=8,  
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            # Textbox for dominant mode
            axs[2].text(0.75, 0.05, "Dominant Mode: (h) Phugoid / (V) Phugoid", transform=axs[2].transAxes,        
                        fontsize=8, verticalalignment='bottom', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

            axs1 = axs[2].twinx()
            axs1.set_ylabel("Speed [m/s]", color="crimson")
            axs1.plot(t,np.sqrt(x_ideal[:,0]**2 + x_ideal[:,1]**2 + x_ideal[:,2]**2), color="crimson", label="V_ideal", linestyle="--", alpha=0.5, zorder=2)
            axs1.plot(t,np.sqrt(x_response[:,0]**2 + x_response[:,1]**2 + x_response[:,2]**2), color="crimson", label="V_actual")
            axs1.tick_params(axis='y', labelcolor="crimson")
            axs1.grid()
            axs1.legend(loc="upper right", fontsize=9, framealpha=0.5)
            fig.tight_layout()

            plt.show()
