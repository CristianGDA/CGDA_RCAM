from dataclasses import dataclass, field
import numpy as np
import sys

@dataclass
class RCAM_parameters:

    """
    * This dataclass contains the parameters and properties for the 
      twin engine Research Civil Aircraft Model also known as RCAM. 

    * Parameters are taken from GARTEUR reports[1][2] 
    
    * Parameters are divided in the following sections:
    1. Aircraft inertial properties: 
        - Mass, (kg)
        - Inertia tensor, (kg*m^2)
        - Inverse of the inertia tensor, (1/((kg*m^2)))

    2. Aircraft geometric and aerodynamic properties:
        - Zero-lift angle of attack for the wing/body, (rad)
        - Wing/body lift curve slope in the linear region (dCL/dalpha), (1/rad)
        - Rate of change of downwash angle (depsilon/dalpha), (rad/rad)
        - Tailplane lift curve slope (dCLt/dalpha_t), (1/rad)
        - Zero-lift drag coefficient of the wing/body, (-)
        - Lift adjustment factor for the drag polar, (-)
        - Pitch rate correction factor, (-)
        - Induced drag coefficient of the wing/body, (-)
        - Sideforce coefficient slope due to the sideslip angle (dCy/dbeta), (1/rad)
        - Sideforce coefficient slope due to rudder deflection (dCy/dr), (1/rad)
        - Mean Aerodynamic Chord (MAC), (m)
        - Distance between the wing/body Aerodynamic Center (AC) and the tailplane Aerodynamic Center (AC), (m)
        - Wing planform reference area, (m^2)
        - Tailplane planform reference area, (m^2)
        - Position of the Centre of Gravity (CG) of the RCAM 
          defined in the fixed aircraft reference frame "Fm", (m)
        - Position of the Aerodynamic Centre (AC) of the wing/body 
          defined in the fixed aircraft reference frame "Fm", (m)

    3. Engine parameters:
        - Position vector of the thrust application point of engine 1 in "Fm", (m)
        - Position vector of the thrust application point of engine 2 in "Fm", (m)

    4. Dynamic models: Surface deflection (Ailerons, elevator, rudder) 
       and engine thrust response.
        - First order time constants for surface actuators and engine thrust, (s)
        - Rate limits for surface actuators and engine throttle, (rad/s)
        - Saturation limits for surface actuators and engine throttle, (rad)

    5. Atmosphere parameters:
        - Air density at sea level, (kg/m^3)
        - Static air pressure at sea level, (N/m^2)
        - Standard temperature at sea level, (K)

    6. Gravity model parameters:
        - Gravitational acceleration, (m/s^2)    
    """

    # ---------- Aircraft inertial properties ----------
    # RCAM Aircraft mass (kg)
    # Operational bounds given in the GARTEUR report: (100000 <= m <= 150000)
    m: float = 120000.00      

    # Inertia tensor (kg*m^2)
    # Expressed in the body frame "Fb"
    # The aircraft is modeled as an XZ symmetric rigid body with fixed shape 
    # and uniform mass distribution.
    # Inertia tensor given in "Fb" and defined as: Ib = m*I_geom
    # I_geom is the geometric inertia tensor of the RCAM, (m^2) 
    # DEPENDENT PARAMETER: Computed in __post_init__
    Ib: np.ndarray = field(init=False)
    
    # Inverse of the inertia tensor (1/((kg*m^2)))
    # DEPENDENT PARAMETER: Computed in __post_init__
    invIb: np.ndarray = field(init=False)
    

    # ---------- Aircraft geometric and aerodynamic properties ----------
    # Zero-lift angle of attack for the wing/body, (rad)
    alpha_L0: float = -11.5*np.pi/180
    
    # Wing/body lift curve slope in the linear region (dCL/dalpha), (1/rad)
    n: float = 5.5     

    # Rate of change of downwash angle (depsilon/dalpha), (rad/rad)
    depsda: float = 0.25                      

    # Tailplane lift curve slope (dCLt/dalpha_t), (1/rad)
    cl_t: float = 3.1

    # Zero-lift drag coefficient of the wing/body
    Cd0_wb: float = 0.13

    # Lift adjustment factor for drag polar
    cl_eff: float = 0.654

    # Pitch rate correction factor
    dpcf: float = 1.3

    # Induced drag coefficient of the wing/body
    k:float = 0.07

    # Sideforce coefficient slope due to the sideslip angle (dCy/dbeta), (1/rad)
    Cy_beta: float = -1.6

    # Sideforce coefficient slope due to the rudder deflection (dCy/dr), (1/rad)
    Cy_ur: float = 0.24

    # Mean Aerodynamic Chord (MAC), (m)
    cbar: float = 6.6 

    # Distance between the wing-body AC and the tail AC, (m)
    lt: float = 24.8

    # Wing planform reference area, (m^2)
    S: float = 260.0

    # Tailplane planform reference area, (m^2)
    St: float = 64.0


    # RCAM Centre of Gravity position (CG), (m)
    # defined in the fixed reference frame "Fm"
    # DEPENDENT PARAMETERS: Computed in __post_init__
    # field will be used to avoid external inputs
    Xcg: float = field(init=False)   # X position wrt the Fm frame 
    Ycg: float = field(init=False)   # Y position wrt the Fm frame
    Zcg: float = field(init=False)   # Z position wrt the Fm frame


    # Wing/body aerodynamic Centre position (AC), (m)
    # defined in the fixed reference frame "Fm"
    # DEPENDENT PARAMETERS: Computed exclusively in __post_init__
    # field will be used to avoid external inputs
    Xac: float = field(init=False)   # X position wrt the Fm frame
    Yac: float = field(init=False)   # Y position wrt the Fm frame
    Zac: float = field(init=False)   # Z position wrt the Fm frame

    # ---------- Engine parameters ----------
    # Position vector of the thrust application point of engine 1 in the "Fm" frame, (m)
    Xapt1: float = 0.0             # x position of thrust application point of engine 1 wrt Fm (m)
    Yapt1: float = -7.94           # y position of thrust application point of engine 1 wrt Fm (m)
    Zapt1: float = -1.9            # z position of thrust application point of engine 1 wrt Fm (m)

    # Position vector of the thrust application point of engine 2 in the "Fm" frame, (m)
    Xapt2: float = 0.0             # x position of thrust application point of engine 2 wrt Fm (m)
    Yapt2: float = 7.94            # y position of thrust application point of engine 2 wrt Fm (m)
    Zapt2: float = -1.9            # z position of thrust application point of engine 2 wrt Fm (m)

    
    # ---------- Actuator models and engine dynamics -------------
    # Actuators and engines are assumed to have first order system dynamics 
    # with rate limits and saturations[1],[2].

    # Time constant of first order system dynamics of the aileron actuator [s]:
    t_aileron: float = 0.15

    # Rate limits for aileron deflection [rad/s]:
    aileron_rate_min: float = -25*np.pi/180
    aileron_rate_max: float = 25*np.pi/180
    
    # Saturation limits for aileron deflection [rad]:
    aileron_saturation_min: float = -25*np.pi/180
    aileron_saturation_max: float = 25*np.pi/180
    
    # Time constant of first order system dynamics for the tailplane (elevator) actuator [s]:
    t_elevator: float = 0.15

    # Rate limits for tailplane (elevator) deflection [rad/s]:
    elevator_rate_min: float = -15*np.pi/180
    elevator_rate_max: float = 15*np.pi/180
    
    # Saturation limits of tailplane (elevator) deflection [rad]:
    elevator_saturation_min: float = -25*np.pi/180
    elevator_saturation_max: float = 10*np.pi/180

    # Time constant of first order system dynamics for the rudder actuator [s]:
    t_rudder: float = 0.3

    # Rate limits for rudder deflection [rad/s]:
    rudder_rate_min: float = -25*np.pi/180
    rudder_rate_max: float = 25*np.pi/180
    
    # Saturation limits of rudder deflection [rad]:
    rudder_saturation_min: float = -30*np.pi/180
    rudder_saturation_max: float = 30*np.pi/180
    
    # Time constant of first order system dynamics for the engine model [s]:
    t_engine: float = 1.5

    # Rate limits for Engine throttle [rad/s]:
    engine_rate_min: float = -1.6*np.pi/180
    engine_rate_max: float = 1.6*np.pi/180
    
    # Saturation limits of Engine throttle [rad]:
    engine_saturation_min: float = 0.5*np.pi/180
    engine_saturation_max: float = 10*np.pi/180


    # ---------- Atmosphere parameters ----------
    # Air density at sea level (kg/m^3)
    rho: float = 1.225

    # Static air pressure at sea level (N/m^2)
    press: float = 101325.0 

    # Standard temperature at sea level (K)
    Temp: float = 288.15


    # ---------- Gravity model parameters ----------
    # Gravitational acceleration (m/s^2)             
    g: float = 9.807               
    
    def __post_init__(self):
            """
            This special method computes dependent parameters after initialization.
            """

            # Compute inertia tensor (depends on aircraft mass)
            # Geometric inertia tensor (m^2)
            self.I_geom: np.ndarray = np.array([[40.07,0,-2.0923],
                                                [0,64,0],
                                                [-2.0923,0,99.92]])
            
            self.Ib = self.m*self.I_geom
    
            # Compute the inverse of the inertia tensor
            try:
                self.invIb = np.linalg.inv(self.Ib)
            except np.linalg.LinAlgError:
                 print("Singular matrix")
                 sys.exit()


            # Compute position of the Centre of Gravity (CG) of the RCAM 
            # defined in the fixed aircraft reference frame "Fm" 
            self.Xcg: float = 0.23*self.cbar         # X position wrt the Fm frame.  Range given by RCAM Report (0.15*cbar <= Xcg <= 0.31*cbar)
            self.Ycg: float = 0.0*self.cbar          # Y position wrt the Fm frame.  Range given by RCAM Report (-0.03*cbar <= Ycg <= 0.03*cbar)
            self.Zcg: float = 0.10*self.cbar         # Z position wrt the Fm frame.  Range given by RCAM Report (0.00*cbar <= Zcg <= 0.21*cbar)

            # Position of the Aerodynamic Centre (AC) of the wing/body 
            # defined in the fixed aircraft reference frame "Fm" 
            self.Xac: float = 0.12*self.cbar         # X position wrt the Fm frame
            self.Yac: float = 0.0*self.cbar          # Y position wrt the Fm frame
            self.Zac: float = 0.0*self.cbar          # Z position wrt the Fm frame
