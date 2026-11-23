import numpy as np
from rcam_parameters import RCAM_parameters


class Aircraft_model:
    """
    This class implements the nonlinear 6 DoF equations of motion, navigation equations and
    RCAM dynamic models (aerodynamics, propulsion, gravity and actuators), providing the 
    equations and constraints used by external functions to:

        - Compute the steady state, straight and level flight trim condition
        - Linearize the RCAM nonlinear 6 DoF equations of motion around the trim condition
        - Find longitudinal and lateral/directional dynamic stability and flying qualities characteristics
        - Plot the comparison of the nonlinear response due to doublet inputs with idealized and dynamic actuators

        
    *** NOTE: Model assumptions and limitations

    1. Flat earth model: the equations of motion assume a non rotating, flat earth model as
    an inertial reference frame, it neglects the curvature of the earth as well as coriolis 
    and centrifugal effects. As a result, the navigation equations are valid for short range 
    trajectories rather than long range, which is sufficient to evaluate and test flight 
    control laws[3]

    2. Constant atmosphere and gravity at sea level: air density, pressure, temperature and 
    gravitational acceleration are assumed constant and don't vary with altitude
    
    3. Euler angles limitation: Attitude is represented using Euler Angles instead of quaternions 
    or poisson's kinematical equations.
    The disadvantage of this model representation is that it presents singularities (gimbal lock) at 
    a pitch angle of ±90°. However, the operational pitch range of a civil transport aircraft 
    remains below this limit, making this formulation valid. 
    
    4. Linear Lift coefficient region: The wing/body lift coefficient in this script is modeled 
    within the linear region, which is valid for alpha <= 14.5°. However, if it is required to 
    simulate high angles of attack or stall regimes, the cubic nonlinear lift approximation 
    provided in the Garteur reports [1],[2] must be implemented.

    This class is organized as:

    1. Special method: 
        __init__
        This method initializes the main attributes required to perform flight 
        dynamics operations. These initializations are:

        1. State variables (9 states: translational velocities, rotational velocities 
           and Euler angles)
        2. Control variables (5 inputs: surface actuators and engine throttles)
        3. Model parameters (imported from the RCAM_parameters dataclass)

    2. Intermediate variables:
        These variables depend on the main attributes and they are recomputed

    Methods:
    3. Compute the rotation matrix to transform vector components from the earth frame to the body frame
    4. Compute the rotation matrix to transform vector components from the stability frame to the body frame
    5. Compute the aerodynamic force coefficients given in stability axes (from GARTEUR handbook [1][2])
    6. Compute the aerodynamic forces projected on body axes
    7. Compute the aerodynamic moment coefficients about the Aerodynamic Centre (from GARTEUR handbook [1][2])
    8. Compute the aerodynamic moments about the Centre of Gravity of the aircraft model
    9. Compute the forces produced by engine thrust given in body axes (from GARTEUR handbook [1][2])
    10. Compute the moments produced by engine thrust about the Centre of Gravity of the aircraft model
    11. Compute the gravity force projected onto the aircraft body frame
    12. Compute the translational equations of motion in the body frame
    13. Compute the rotational equations of motion in the body frame
    14. Compute the Euler kinematic equations (attitude kinematics)
    15. Assemble the complete nonlinear 6 DoF equations of motion
    16. Compute the Navigation equations in the earth frame
    17. Compute the residual constraints for steady, straight and level flight
    18. Generate the boundaries for the optimization decision vector
    19. Compute the time derivatives of the actuator and engine responses

    """

    def __init__(self, params: RCAM_parameters, x= None,u= None)-> None:
        """
        This method initializes the main attributes required to perform flight 
        dynamics simulations. These initializations are:

        1. State variables (9 states: translational velocities, rotational velocities 
           and Euler angles)
        2. Control variables (5 inputs: actuators and throttles)
        3. Model parameters (imported from the RCAM_parameters dataclass)
        """

        self.states = 9
        self.actuators = 5
        self.params: RCAM_parameters = params

        # Initialize state and input vectors
        # Use temporary zero vectors when states and inputs are not provided (e.g. at instantiation)  
        # else, validate the dimensions
        if x is None:
            self.x = np.zeros(self.states)
        else:
            self.x = np.array(x, dtype = float)
             
            # Ensure the right number of states
            if len(self.x) != self.states:
                raise ValueError(f"The model requires {self.states} states") 

        if u is None:
            self.u = np.zeros(self.actuators)
        else:
            self.u = np.array(u, dtype = float)

            # Ensure the right number of inputs
            if len(self.u) != self.actuators:
                raise ValueError(f"The model requires {self.actuators} actuators")
            
    # ----- Intermediate variables -----
    # Required to be recomputed in each iteration when self.x is updated
    
    # Airspeed (m/s)
    @property 
    def vel_a(self):
        return np.sqrt(self.x[0]**2 + self.x[1]**2 + self.x[2]**2)
    
    # Angle of attack (rad)
    @property
    def alpha(self):
        return np.arctan2(self.x[2],self.x[0])
    
    # Angle of sideslip (rad)
    @property
    def beta(self):
        return np.arcsin(self.x[1]/self.vel_a)
    
    # Dynamic pressure (N/m^2), (Pa)
    @property
    def Q(self):
        return 0.5*self.params.rho*self.vel_a**2 
    
    # Translational velocity vector expressed in Fb (m/s)
    @property
    def V_b(self): 
        return np.array([self.x[0], self.x[1], self.x[2]]) 
    
    # Angular velocity vector of the body wrt earth, expressed in body frame Fb (rad/s)
    @property
    def wbe_b(self):
        return np.array([self.x[3], self.x[4], self.x[5]])
    
    # Euler angles describing the aircraft attitude (Fb wrt Fe) (rad)
    @property
    def Euler_angles(self):
        return np.array([self.x[6], self.x[7], self.x[8]])

    # ----- Methods -----
    def rotation_matrix_earth_body(self):
        """
        This method computes the rotation matrix that transforms the components 
        of a vector from the "earth frame" to the "body frame"

        Based on the 3-2-1 Euler angles sequence, where:
        x[6] = phi (roll)
        x[7] = theta (pitch)
        x[8] = psi (yaw)
        """
        return np.array([[np.cos(self.x[7])*np.cos(self.x[8]), np.cos(self.x[7])*np.sin(self.x[8]), -np.sin(self.x[7])],
                         [-np.cos(self.x[6])*np.sin(self.x[8]) + np.sin(self.x[6])*np.sin(self.x[7])*np.cos(self.x[8]), np.cos(self.x[6])*np.cos(self.x[8]) + np.sin(self.x[6])*np.sin(self.x[7])*np.sin(self.x[8]), np.sin(self.x[6])*np.cos(self.x[7])],
                         [np.sin(self.x[6])*np.sin(self.x[8]) + np.cos(self.x[6])*np.sin(self.x[7])*np.cos(self.x[8]), -np.sin(self.x[6])*np.cos(self.x[8]) + np.cos(self.x[6])*np.sin(self.x[7])*np.sin(self.x[8]), np.cos(self.x[6])*np.cos(self.x[7])]]) 

    def rotation_matrix_stability_body(self):
        """
        This method computes the rotation matrix, it transforms the components 
        of a vector from the "stability frame" to the "body frame"
        """
        return np.array([[np.cos(self.alpha), 0, -np.sin(self.alpha)],
                         [0, 1, 0],
                         [np.sin(self.alpha), 0, np.cos(self.alpha)]])       

    def aero_force_coeff(self):
        """
        This method computes the aerodynamic force coefficients.
        It returns the vector [CD, CY, CL] expressed in "stability axes".

        Expressions and parameters are taken from the GARTEUR report [1][2]. 
        """

        # Step 1. Compute the lift coefficient of the wing-body (linear region), 
        # it acts on the wing/body Aerodynamic Centre (AC)
        CL_wb = self.params.n*(self.alpha - self.params.alpha_L0)

        # Step 2. Compute the lift coefficient of the tailplane. 
        # It acts at the tailplane AC.
        # Define the downwash angle
        epsilon = self.params.depsda*(self.alpha - self.params.alpha_L0)
        
        # Define the angle of attack of the tailplane
        alpha_t = self.alpha - epsilon + self.u[1] + self.params.dpcf*self.x[4]*self.params.lt/self.vel_a

        # Compute the lift coefficient of the tailplane.  
        # (St/S) scales the tailplane lift contribution in terms of the "wing reference area"
        CL_t = self.params.cl_t*(self.params.St/self.params.S)*alpha_t

        # Step 3. Compute the total aerodynamic lift coefficient
        self.CL = CL_wb + CL_t

        # Step 4. Compute the aerodynamic drag coefficient. It acts on the wing/body AC
        self.CD = self.params.Cd0_wb + self.params.k*(self.params.n*self.alpha + self.params.cl_eff)**2

        # Step 5. Compute the aerodynamic sideforce coefficient. It acts on the wing/body AC
        self.CY = self.params.Cy_beta*self.beta + self.params.Cy_ur*self.u[2]
        return np.array([self.CD, self.CY,self.CL])

    def aero_force(self):
        """
        This method computes the aerodynamic forces projected onto the "body frame". 
        It returns the vector: F_b = [Fx_b,Fy_b,Fz_b].
        """
        
        # Step 1. Unpack coefficients from "aero_force_coeff()" method.
        CD,CY,CL = self.aero_force_coeff()

        # Step 2. Define aerodynamic forces in stability axes: F_s = [-D, Y, -L]
        # Note: Drag and Lift signs are negative because drag opposes the X stability axis
        # and Lift opposes the Z stability axis.
        FA_s = np.array([-CD*self.Q*self.params.S,
                         CY*self.Q*self.params.S,
                         -CL*self.Q*self.params.S])
        
        # Step 3. Transform aerodynamic forces from the "stability frame (Fs)" to the "body frame (Fb)"
        self.FA_b = self.rotation_matrix_stability_body()@FA_s
        return self.FA_b

    def aero_moment_coeff(self):
        """
        This method computes the aerodynamic moment coefficients (cl,cm,cn) 
        about the wing-body Aerodynamic Centre (AC).
        It returns the vector CMac_b = [cl,cm,cn]
        References: GARTEUR report [1][2]
        """

        # Step 1. Compute the contribution to the moment coefficients CMac_b
        # due to sideslip angle, angle of attack and aircraft model parameters (effectiveness)
        epsilon = self.params.depsda*(self.alpha - self.params.alpha_L0)
        eta11 = -1.4*self.beta
        eta21 = -0.59 - (3.1*(self.params.St*self.params.lt)/(self.params.S*self.params.cbar))*(self.alpha - epsilon)
        eta31 = (1 - self.alpha*(180/(15*np.pi)))*self.beta

        eta = np.array([eta11,
                        eta21,
                        eta31])

        # Step 2. Compute the contribution to the moment coefficients due to 
        # angular velocity (p,q,r) effects
        dCMdx = (self.params.cbar/self.vel_a)*np.array([[-11, 0, 5],
                                                        [0, (-4.03*(self.params.St*self.params.lt**2)/(self.params.S*self.params.cbar**2)), 0],
                                                        [1.7, 0, -11.5]])

        # Step 3. Compute the contribution to the moment coefficient due to 
        # control surface deflections (u_aileron, u_elevator, u_rudder)
        dCMdu = np.array([[-0.6, 0, 0.22],
                          [0, (-3.1*(self.params.St*self.params.lt)/(self.params.S*self.params.cbar)), 0],
                          [0, 0, -0.63]])

        # Step 4. Compute the total aerodynamic moment coefficient vector CMac_b = [Cl,Cm,Cn]
        # adding all contributions
        self.CMac_b = eta + dCMdx@self.wbe_b + dCMdu@np.array([self.u[0], self.u[1], self.u[2]])
        return self.CMac_b

    def aero_moment(self):
         
        """
        This method computes the aerodynamic moment about the Centre of Gravity (CG)
        It calls "aero_moment_coeff()" to obtain the aerodynamic moment coefficients about the AC
        This function returns the vector MA_cg = [M_l,M_m,M_n] [Nm]
        References: GARTEUR report[1][2]
        """

        # Step 1. Compute the aerodynamic moment MAac_b about the Aerodynamic Centre (AC)
        # Note: From RCAM documentation, Mean Aerodynamic Chord (cbar) is used as the reference length
        MAac_b = self.aero_moment_coeff()*self.Q*self.params.S*self.params.cbar

        # Step 2. Define the position vector of the CG wrt the aircraft fixed reference frame Fm
        rcg_b = np.array([self.params.Xcg, self.params.Ycg, self.params.Zcg])

        # Step 3. Define the position vector of the AC wrt the aircraft fixed reference frame Fm
        rac_b = np.array([self.params.Xac, self.params.Yac, self.params.Zac])

        # Step 4. Compute the moment transfer contribution about CG due to the aerodynamic forces acting at AC
        # M_transfer = FA_b x r. Where r = (rcg_b - rac_b) is the lever arm vector
        M_transfer = np.cross(self.aero_force(), rcg_b - rac_b)

        # Step 5. Compute the total aerodynamic moment MAcg_b about CG in the body frame Fb
        self.MAcg_b = MAac_b + M_transfer
        return self.MAcg_b

    def engine_propulsion(self):
        
        """
        This method computes the total thrust force produced by two engines 
        It returns the vector FTE_b = [FTEx, FTEy, FTEz] in Fb
        u[3] and u[4] represent the throttle settings 
        Reference: GARTEUR report [1][2]
        """

        # Step 1. Compute the thrust magnitude of each engine
        Teng_1 = self.params.m*self.params.g*self.u[3]
        Teng_2 = self.params.m*self.params.g*self.u[4]

        # Step 2. Define the thrust vector for each engine in Fb
        # It is assumed that engine thrust vectors are aligned with the aircraft 
        # longitudinal axis (x axis in Fb)
        # These vectors will be used to compute the moment produced by engines
        self.FE1_b = np.array([Teng_1, 0, 0])
        self.FE2_b = np.array([Teng_2, 0, 0])

        # Step 3. Compute the total propulsion force vector FTE_b
        self.FTE_b = self.FE1_b + self.FE2_b
        return self.FTE_b

    def engine_moment(self):

        """
        This method computes the moment produced by engine thrust 
        about the Centre of Gravity CG. 
        It returns the total moment vector MEcg_b = [MEcgx_b,MEcgy_b,MEcgz_b] [Nm]
        Reference: GARTEUR report[1][2]
        """

        # Compute the engine thrust vectors (self.FE1_b, self.FE2_b)
        self.engine_propulsion()

        # Step 1. Define the lever arm from the Aircraft CG to each engine's thrust 
        # application point expressed in Fb (coordinates are given in Fm) 
        # Note: Xm and Zm axes point opposite to Xb and Zb, 
        # while Ym keeps the same direction as Yb

        # Lever arm for engine 1
        eng1_apt_cg = np.array([self.params.Xcg - self.params.Xapt1,
                                self.params.Yapt1 - self.params.Ycg,
                                self.params.Zcg - self.params.Zapt1])
        
        # Lever arm for engine 2
        eng2_apt_cg = np.array([self.params.Xcg - self.params.Xapt2,
                                self.params.Yapt2 - self.params.Ycg,
                                self.params.Zcg - self.params.Zapt2])

        # Step 2. Compute the moment contribution of each engine wrt CG
        ME1cg_b = np.cross(eng1_apt_cg, self.FE1_b)
        ME2cg_b = np.cross(eng2_apt_cg, self.FE2_b)

        # Step 3. Compute the total engine moment acting on the aircraft
        self.MEcg_b = ME1cg_b + ME2cg_b
        return self.MEcg_b

    def gravity_force(self):

        """
        This method computes the gravity force vector projected onto the body frame Fb
        it returns the vector Fg_b = [Fgx_b, Fgy_b, Fgz_b ]. 
        In this model, gravity is not considered a function of altitude, 
        it is assumed to be constant.
        """

        # Step 1. Define the Gravity force vector given in the earth frame Fe.
        # In the NED convenction, gravity acts positive along Z axis (pointing to the earth center)
        Fg_e = self.params.m*np.array([0,
                                        0,
                                        self.params.g])

        # Step 2. Transform the gravity force components from the earth frame Fe to the body frame Fb
        # Required because the equations of motion are solved in the body frame
        # Apply the rotation matrix
        self.Fg_b = self.rotation_matrix_earth_body()@Fg_e
        return self.Fg_b

    def translational_equation(self):
        """
        This method computes the translational equation of motion of the Centre of Mass 
        of the RCAM in the body frame (Fb).
        
        It returns the time derivatives of the translational velocity state vector.
        self.x0tox2_dot = [u_dot, v_dot, w_dot] = [xdot[0], xdot[1], xdot[2]]

        The mathematical expression of the translational equation was derived 
        from Newton's second law, it is expressed in the rotating body frame as: 
        V_b_dot = (1/m)*F_b - wbe_b x V_b

        Where:
            V_b_dot = -> Rate of change of the translational velocity of the centre of mass
                       expressed in the body frame Fb

            F_b -> Sum of external forces expressed in the body frame Fb

            wbe_b -> Angular velocity vector of Fb wrt earth frame Fe

            V_b -> Translational velocity of the centre of mass expressed in body frame Fb
        """
        
        # Step 1. Compute the total external forces F_b acting on the aircraft in Fb
        self.F_b = self.gravity_force() + self.engine_propulsion() + self.aero_force()

        # Step 2. Compute the translational state derivatives
        self.x0tox2_dot = (1/self.params.m)*self.F_b - np.cross(self.wbe_b, self.V_b)
        return self.x0tox2_dot

    def rotational_equation(self):
        """
        This method computes the rotational equations of motion of the 
        RCAM vehicle in body frame (Fb).
        
        It returns the time derivatives of the rotational motion state vector
        x3tox5_dot = [p_dot, q_dot, r_dot] = [xdot[3], xdot[4], xdot[5]]

        The mathematical expression of the rotational equations for a rigid body is derived 
        from Newton's second law for rotational dynamics.

        The equation is expressed as: 
        wbe_b_dot = (Ib^-1)*(M_b - wbe_b x Ib*wbe_b)

        Where:
            wbe_b_dot -> Rate of change of the angular velocity vector (angular acceleration)  
            of the body frame Fb wrt the earth frame Fe

            M_b -> Sum of external moments acting about the centre of gravity CG expressed in Fb

            Ib -> Inertia tensor expressed in Fb
        """

        # Step 1. Compute the total moment acting about the Centre of Gravity (CG)
        # of the aircraft model in the body frame Fb
        self.Mcg_b = self.engine_moment() + self.aero_moment() 

        # Step 2. Compute the rotational state derivatives
        x3tox5_dot = self.params.invIb@(self.Mcg_b - np.cross(self.wbe_b, self.params.Ib@self.wbe_b))
        return x3tox5_dot

    def Euler_kinematics_equation(self):

        """
        This method computes the Euler Kinematic equations for Attitude kinematics
        
        It returns the time derivatives of the Euler angles
        x6tox8_dot = [dot_phi, dot_theta, dot_psi] = [dot_x[6], dot_x[7],dot_x[8]]

        The mathematical equation relates the angular velocity vector with 
        the rate of change of the Euler angles.

        The equation is expressed as:
        Euler_angles_dot = H_phi*wbe_b
            Where:
            H_phi -> Matrix relating angular rates (p,q,r) to Euler rates (dot_phi, dot_theta, dot_psi)            
        """
        # Step 1. Compute the kinematic matrix
        H_phi = np.array([[1, np.sin(self.x[6])*np.tan(self.x[7]), np.cos(self.x[6])*np.tan(self.x[7])], 
                        [0, np.cos(self.x[6]), -np.sin(self.x[6])], 
                        [0, np.sin(self.x[6])/np.cos(self.x[7]), np.cos(self.x[6])/np.cos(self.x[7])]])

        # Step 2. Compute the time derivatives of the Euler angles
        self.x6tox8_dot = H_phi@self.wbe_b
        return self.x6tox8_dot

    def equations_of_motion_model(self):
        """
        This method assembles the complete nonlinear 6 DoF
        equations of motion of the RCAM.

        It returns the time derivatives of the full state vector
        x_dot = [u_dot, v_dot, w_dot, p_dot, q_dot, r_dot, phi_dot, theta_dot, psi_dot]
              = [x[0], x[1], x[2], ..., x[8]]

        This model is known in flight dynamic literature as: 
        The flat earth equations of motion

        In this model, some assumptions are given:

            1. The earth frame is an inertial reference frame
            2. Position is measured in a tangent plane coordinate system
            3. The gravity vector is normal to the tangent plane and constant in magnitude
               (Earth curvature is ignored)
            4. Non rotating earth (Coriolis effects due to Earth rotation are ignored).
            5. The aircraft is modeled as a rigid body

            As consequences:
            6. This model is not suitable for precise determination of position over large
               distances on earth (valid only for short ranges)
            
            This model is used:
            7. To simulate aircraft performance and dynamic behaviour
            8. To derive linear state-space models for flight control system design
            """
        
        return np.concatenate([self.translational_equation(), self.rotational_equation(), self.Euler_kinematics_equation()])

    def navigation_equation(self):
    
        """
        This method computes the navigation equations of the RCAM in the earth frame (Fe).
        
        It returns the time derivatives of the position vector
        x9tox11_dot = [PosN_dot, PosE_dot, PosD_dot] = [xdot[9], xdot[10], xdot[11]]

        The navigation equations describe the trajectory and velocity of the 
        aircraft model in earth axes (NED). The expression is obtain by transforming the 
        translational velocity given in the body frame Fb to the earth frame Fe.
        
        the equation is expressed as: 
        V_e = Ce_b*V_b
        Where:

            V_b -> Translational velocity vector given in body frame Fb
        
            V_e -> Translational velocity vector given in the earth frame Fe

            Ce_b -> Rotation matrix from the body frame Fb to the earth frame Fe
            (Transpose of earth to body rotation matrix)
        """

        # Step 1. Transform the translational velocity vector in body axis "V_b" to the earth axis 
        # using the transpose of the "rotation_matrix_earth_body" method
        self.x9tox11_dot = np.linalg.matrix_transpose(self.rotation_matrix_earth_body())@self.V_b
        return self.x9tox11_dot

    def straight_level_flight_constraints(self, V_desired):
        """
        This method computes the residual constraints required for steady state, straight level flight
        condition for the RCAM

        This method requires from the user the desired airspeed "Vb_des" [m/s] at which the RCAM
        will be trimmed. 
        
        It returns the constraint residual vector evaluated at the current states and inputs
        The following equality constraints are required:

        1. Steady state constraints: From the equations of motion (x_dot = f(x0,ue) = 0)
        2. Flight condition constraints: For straight and level flight
        3. Physical constraints: For symmetry engine thrust
        
        """
        # Step 1: Steady state constraints
        # Required to ensure zero translational, rotational and attitude accelerations
        steady_state_const = self.equations_of_motion_model()

        # Step 2: Flight condition constraints 
        # Required to maintain constant airspeed, altitude, heading and wings level attitude

        # 1. Airspeed constraint: |Vb| = V_desired 
        # Required to maintain the desired/target cruise airspeed
        v_actual =  np.sqrt(self.x[0]**2 + self.x[1]**2 + self.x[2]**2)
        const_airspeed = v_actual - V_desired
        
        # 2. Flight path angle constraint: gamma = theta - alpha = 0
        # Required to maintain constant altitude (no climb/descent)
        const_gamma = self.x[7] - np.atan2(self.x[2], self.x[0])
        
        # 3. Lateral velocity constraint: v = 0
        # Required to ensure no sideslip (beta=0)
        const_beta = self.x[1]

        # 4. Bank angle constraint: phi = 0
        # Required to maintain wings level
        const_phi = self.x[6]

        # 5. Heading angle constraint: psi = 0
        # Required to point the aircraft with the north direction
        const_psi = self.x[8]
        
        # Flight condition constraint vector
        additional_const = np.array([const_airspeed, const_gamma, const_beta, const_phi, const_psi])


        # Step 3:Physical model constraints:
        # Since the aircraft is symmetric, engines must produce equal thrust to prevent yawing moments 
        const_engines = self.u[3] - self.u[4]
        physics_model_const = np.array([const_engines])
    
        # Total constraint residueals
        return np.concatenate([steady_state_const, additional_const, physics_model_const])

    def aircraft_model_bounds(self):
        """
        This method generates the boundaries for the decision vector "z" which is composed of:

        - n State variables (x): Unbounded 
        - m Actuators or Control variables (u): Bounded by the actuator and throotle saturation limts

        This list of tuples is passed as the "bound" parameter to the optimization solver
        "None" is used to specify no bound.
        """
        # State bounds
        state_bounds = [(None, None), # u
                        (None, None), # v
                        (None, None), # w
                        (None, None), # p
                        (None, None), # q
                        (None, None), # r
                        (None, None), # phi
                        (None, None), # theta
                        (None, None)] # psi
        
        # Control bounds: (saturation limits)
        control_bounds = [(self.params.aileron_saturation_min, self.params.aileron_saturation_max),     # Aileron saturation
                          (self.params.elevator_saturation_min, self.params.elevator_saturation_max),   # Elevator saturation
                          (self.params.rudder_saturation_min, self.params.rudder_saturation_max),       # Rudder saturation
                          (self.params.engine_saturation_min, self.params.engine_saturation_max),       # Engine 1 saturation
                          (self.params.engine_saturation_min, self.params.engine_saturation_max)]       # Engine 2 saturation

        z_bounds = state_bounds + control_bounds
        return z_bounds

    def actuator_dynamics(self,u_cmd, u_response):
        """
        This method computes the time derivatives of the control surface and engines 
        (u_dot_response) to simulate actuator and propulsion dynamics.

        Actuators (aileron, elevator, rudder) and engines (throotles 1 and 2) are modeled as
        first order systems (LPF) subjected to position saturation and rate limits[1][2].

        The differential equation for each command input is expressed as:
        u_dot_response = clip( (u_cmd_sat - u_response) / tau, rate_min, rate_max )
        
        Where:
            u_cmd:          Commanded control vector requested by the pilot or controller (5x1)
            u_cmd_sat:      Commanded control vector bounded by position saturation limits (5x1)
            u_response:     Current actual position of the actuators and engine throttles (5x1)
            tau:            First-order time constant for each actuator/engine [s] (5x1)
            u_dot_response: Time derivative (deflection/throttle rate) of the actual 
                            actuator and engine states [rad/s] (5x1)
        """

        # Step 1. Pack time constants, rate limits, and position saturation limits from RCAM_parameters
        time_constants = np.array([self.params.t_aileron, self.params.t_elevator, self.params.t_rudder, self.params.t_engine, self.params.t_engine])
        input_rates_min = np.array([self.params.aileron_rate_min, self.params.elevator_rate_min, self.params.rudder_rate_min, self.params.engine_rate_min, self.params.engine_rate_min])
        input_rates_max = np.array([self.params.aileron_rate_max, self.params.elevator_rate_max, self.params.rudder_rate_max, self.params.engine_rate_max, self.params.engine_rate_max])
        sat_min = np.array([self.params.aileron_saturation_min, self.params.elevator_saturation_min, self.params.rudder_saturation_min, self.params.engine_saturation_min, self.params.engine_saturation_min])
        sat_max = np.array([self.params.aileron_saturation_max, self.params.elevator_saturation_max, self.params.rudder_saturation_max, self.params.engine_saturation_max, self.params.engine_saturation_max])

        # Step 2. Apply position saturation limits to the commanded control vector
        u_cmd_sat = np.clip(u_cmd, sat_min, sat_max)
        
        # Step 3. Compute the first-order state derivatives and enforce actuator rate limits
        u_dot_response = np.clip((u_cmd_sat - u_response)/time_constants, input_rates_min, input_rates_max)

        return u_dot_response
