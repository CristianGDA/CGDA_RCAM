import numpy as np
from tabulate import tabulate
from colorama import Fore, Style


def stability_analysis(A_transformed):

    """
    This function computes and displays the dynamic stability characteristics and flying qualities 
    compliance (MIL-HDBK-1797) for a linearized state-space representation of the aircraft 
    model around the trim condition (steady state, straight and level flight).

    Background
    =========================
    MIL-HDBK-1797 (previously, MIL-STD-1797A) provides guidance for flying qualities requirements 
    and compliance across different aircraft classes and flight categories.
    
    Inputs:
    =========================
        - A_transformed: Linearized state matrix "A" reordered to decouple longitudinal and lateral/directional 
        dynamics, provided by the "linearization" function.
        
        Assume state ordering:
            - Longitudinal states: [u, w, q, theta]
            - Lateral/directional states: [v, p, r, phi, psi]

    Returns:
    =========================
        None (prints the following tables):
        - TABLE 1: Longitudinal dynamic stability characteristics
        - TABLE 2: Longitudinal flying qualities compliance based on MIL-HDBK-1797 requirements
        - TABLE 3: Lateral/directional dynamic stability characteristics
        - TABLE 4: Lateral/directional flying qualities compliance based on MIL-HDBK-1797 requirements
    """

    print("\n")
    print("="*100)
    print(Style.BRIGHT + "*** LONGITUDINAL AND LATERAL/DIRECTIONAL DYNAMIC STABILITY AND FLYING QUALITIES ANALYSIS ***" )
    print("="*100)
    print("")


    # ----- STEP 1: LONGITUDINAL DYNAMIC STABILITY ANALYSIS ------
    # Compute longitudinal eigenvalues from the decoupled state matrix
    A_long = A_transformed[0:4,0:4]
    eig_long = np.linalg.eigvals(A_long)

    # Store only eigenvalues with positive imaginary part
    im_eigen_long = [i for i in eig_long if i.imag > 0]
    
    # Identify Short Period and Phugoid modes based on natural frequency magnitude  
    if np.abs(im_eigen_long[0])>np.abs(im_eigen_long[1]):
        short_period, phugoid = im_eigen_long
    else:
        phugoid, short_period = im_eigen_long

    # ----- TABLE 1: LONGITUDINAL DYNAMIC STABILITY -----
    long_list = []
    for i in [short_period, phugoid]:
                
        wn = np.sqrt(i.real**2 + i.imag**2)                     # Natural Frequency [rad/s] ("N/A" None)
        zeta = -i.real / wn if wn != 0 else None                # Damping Ratio [-]
        per = 2*np.pi / i.imag if i.imag != 0 else None         # Period [s],    i.imaginary = Damped natural frequency
        t_st = -np.log(0.02)/(wn*zeta) if i.real < 0 else None  # Settling time [s], Time to reach 2% of initial amplitude

        # Store modal properties in a list of dictionary
        long_list.append({    "Mode": "Short Period" if i==short_period else "Phugoid", 
                              "Eigenvalue": f"{i.real:.4f} ± {i.imag:.4f}j", 
                              "Natural Frequency": f"{wn:.4f}", 
                              "Damping Ratio": f"{zeta:.4f}",
                              "Period": f"{per:.4f}",
                              "Settling Time": f"{t_st:.4f}",
                              })

    # Header
    headers_map1 = {
        "Mode": Style.BRIGHT +"Mode" + Style.RESET_ALL,
        "Eigenvalue": Style.BRIGHT +"Eigenvalue" + Style.RESET_ALL,
        "Natural Frequency": Style.BRIGHT + "Natural Frequency\nω_n [rad/s]" + Style.RESET_ALL,
        "Damping Ratio": Style.BRIGHT + "Damping Ratio\n ζ [-]" + Style.RESET_ALL,
        "Period": Style.BRIGHT + "Period\n[s]" + Style.RESET_ALL,
        "Settling Time": Style.BRIGHT + "Settling Time\n[s]" + Style.RESET_ALL
    }

    # Display table
    print(Style.BRIGHT + "- LONGITUDINAL DYNAMIC STABILITY CHARACTERISTICS:" + Style.NORMAL)
    print(tabulate(
                    long_list,
                    headers=headers_map1,
                    tablefmt="fancy_grid", 
                    stralign="center",
                    disable_numparse=True))


    # ----- TABLE 2: LONGITUDINAL FLYING QUALITIES COMPLIANCE (MIL-HDBK-1797A) -----
    # Short Period damping ratio requirement (Level 1, Class III, Category B)
    
    wn_sp = np.abs(short_period)                                                # Short Period Natural frequency
    zeta_sp = -short_period.real / np.abs(short_period)                         # Short Period Damping Ratio
    per_sp = 2*np.pi / short_period.imag if short_period.imag != 0 else None    # Period,   short_period.imag: damped natural frequency
    req_zetamin_sp = 0.30                                                       # Minimum value of Damping Ratio requirement 
    req_zetamax_sp = 2.00                                                       # Maximum value of Damping Ratio requirement    
    settling_sp = np.log(0.02) / short_period.real                              # Settling time. Time to reach 2% of initial amplitude
    

    # Store Short Period Flying Qualities
    damp_ratio_sp = {"Mode": Style.BRIGHT +"Short Period"+ Style.RESET_ALL,
                    "Parameter": "Damping Ratio\n ζ [-]",
                    "Design": f"{zeta_sp:.4f}",
                    "Requirement": f" {req_zetamin_sp:.4f} ≤ ζ ≤ {req_zetamax_sp:.4f}",
                    "Margin": "Inside Range" if req_zetamin_sp <= zeta_sp <= req_zetamax_sp  else " Outside Range",
                    "Handling Qualities": Fore.GREEN +"Level 1"+ Style.RESET_ALL if req_zetamin_sp <= zeta_sp <= req_zetamax_sp else Fore.RED + "Unsatisfactory" + Style.RESET_ALL,
                    "Reference": "4.2.1.2"                
                    }

    # Phugoid damping ratio requirement (Level 1, Class III, Category B)
    wn_ph = np.abs(phugoid)                                                     # Phugoid Natural frequency
    zeta_ph = -phugoid.real / np.abs(phugoid)                                   # Phugoid Damping Ratio
    per_ph = 2*np.pi / phugoid.imag if phugoid.imag != 0 else None              # Period,     phugoid.imag: damped natural frequency
    req_zetamin_ph = 0.04                                                       # Minimum requirement value of Damping Ratio  
    settling_ph = np.log(0.02) / phugoid.real                                   # Phugoid Settling time. Time to reach 2% of initial amplitude

    # Store Phugoid Flying Qualities
    damp_ratio_ph = {"Mode": Style.BRIGHT +"Phugoid"+ Style.RESET_ALL,
                    "Parameter": "Damping Ratio\n ζ [-]",
                    "Design": f"{zeta_ph:.4f}",
                    "Requirement": f" ζ ≥ {req_zetamin_ph:.4f}",
                    "Margin": ("+" if zeta_ph > req_zetamin_ph else "") + f"{100*(zeta_ph - req_zetamin_ph)/req_zetamin_ph:.2f} %",
                    "Handling Qualities": Fore.GREEN +"Level 1"+ Style.RESET_ALL if zeta_ph > req_zetamin_ph else Fore.RED + "Unsatisfactory" + Style.RESET_ALL,
                    "Reference": "4.2.1.1"                
                    }

    # Header
    headers_map2 = {
    "Mode": Style.BRIGHT +"Mode" + Style.RESET_ALL,
    "Parameter": Style.BRIGHT +"Parameter\nfor requirement" + Style.RESET_ALL,
    "Design": Style.BRIGHT +"Design Value" + Style.RESET_ALL,
    "Requirement": Style.BRIGHT +"Requirement Value\nLevel 1, CATEGORY B - CRUISE" + Style.RESET_ALL,
    "Margin": Style.BRIGHT +"Margin" + Style.RESET_ALL,
    "Handling Qualities": Style.BRIGHT +"Flying\nQuality" + Style.RESET_ALL,
    "Reference": Style.BRIGHT +"Reference\n(MIL-HDBK-1797A)" + Style.RESET_ALL}

    print("\n")
    print(Style.BRIGHT + "- LONGITUDINAL FLYING QUALITIES COMPLIANCE (MIL-HDBK-1797):" + Style.NORMAL)

    # Display table
    print(tabulate([damp_ratio_sp, damp_ratio_ph],
            headers=headers_map2,
            tablefmt="fancy_grid", 
            stralign="center",
            disable_numparse=True))


    # ----- STEP 2: LATERAL/DIRECTIONAL DYNAMIC STABILITY ANALYSIS ------
    print("\n")

    # Compute lateral/directional eigenvalues from the decoupled matrix
    A_lat_dir = A_transformed[4:8,4:8]
    eig_lat_dir = np.linalg.eigvals(A_lat_dir)

    # Extract dutch roll eigenvalue (positive imaginary part)
    im_eigen_latd = [i for i in eig_lat_dir if i.imag > 0]
    dutch_roll = im_eigen_latd[0] if im_eigen_latd else None # Dutch Roll eigenvalue

    # Identify spiral (slower, closer to origin) and roll subsidence (faster, more negative) modes
    re_eigen_latd = [i.real for i in eig_lat_dir if i.imag == 0]
    spiral, roll_subs = sorted(re_eigen_latd, reverse=True) 

    # ----- TABLE 3: LATERAL/DIRECTIONAL DYNAMIC STABILITY -----
    lat_dir_list = []
    for i in [dutch_roll, spiral, roll_subs]:

        wn = np.abs(i) if i.imag != 0 else None                                  # Natural Frequency
        zeta = -i.real/wn if i.imag != 0 else None                               # Damping Ratio
        per = 2*np.pi / i.imag if i.imag != 0 else None                          # Period     i.imaginary = Damped natural frequency
        t_st = -np.log(0.02)/(wn*zeta) if (i.imag != 0 and zeta > 0) else None   # Settling time. Time to reach 2% of initial amplitude
        t_const = np.log(1/np.e)/i.real if i.imag == 0 and i.real < 0 else None  # Time constant for non-oscillatory modes. 
        
        # Store properties in dictionary (list of dicts)
        lat_dir_list.append({"Mode":                 "Dutch Roll" if i==dutch_roll else ("Spiral" if i==spiral else "Roll\nSubsidence"), 
                                "Eigenvalue":        f"{i.real:.4f} ± {i.imag:.4f}j" if i.imag != 0 else f"{i.real:.4f}", 
                                "Natural Frequency": "-" if wn == None else f"{wn:.4f}", 
                                "Damping Ratio":     "-" if wn == None else f"{zeta:.4f}",
                                "Period":            "-" if per == None else f"{per:.4f}",
                                "Settling Time":     "-" if t_st == None else f"{t_st:.4f}",
                                "Time Constant":     "-" if t_const == None else f"{t_const:.4f}",                            
                                })

    # Header
    headers_map = {
        "Mode": Style.BRIGHT +"Mode" + Style.RESET_ALL,
        "Eigenvalue": Style.BRIGHT +"Eigenvalue" + Style.RESET_ALL,
        "Natural Frequency": Style.BRIGHT + "Natural Frequency\nω_n [rad/s]" + Style.RESET_ALL,
        "Damping Ratio": Style.BRIGHT + "Damping Ratio\n ζ [-]" + Style.RESET_ALL,
        "Period": Style.BRIGHT + "Period\n[s]" + Style.RESET_ALL,
        "Settling Time": Style.BRIGHT + "Settling Time\n[s]" + Style.RESET_ALL,
        "Time Constant": Style.BRIGHT + "Time Constant\n𝜏 [s]" + Style.RESET_ALL
    }

    # Display table
    print(Style.BRIGHT + "- LATERAL/DIRECTIONAL DYNAMIC STABILITY CHARACTERISTICS:" + Style.NORMAL)
    print(tabulate(lat_dir_list,
                    headers=headers_map,
                    tablefmt="fancy_grid", 
                    stralign="center",
                    disable_numparse=True))


    # ----- TABLE 4: LATERAL/DIRECTIONAL FLYING QUALITIES COMPLIANCE (MIL-HDBK-1797A) -----
    # Dutch Roll requirements (Level 1, Class III, Category B)
    wn_dr = np.abs(dutch_roll) # Natural Frequency
    zeta_dr = -dutch_roll.real / wn_dr

    req_zeta_dr = 0.08 # Damping Ratio Requirement for Level 1 handling qualities
    req_wn_dr = 0.4    # rad/s.  Natural Frequency Requirement for Level 1 handling qualities
    req_zw_dr = 0.15   # rad/s.  Damping-Natural Frequency product Requirement for Level 1 handling qualities

    damp_ratio = {"Mode": Style.BRIGHT +"Dutch Roll"+ Style.RESET_ALL,
                "Parameter": " Damping Ratio\n ζ [-]",
                "Design": f"{zeta_dr:.4f}",
                "Requirement": f" ≥ {req_zeta_dr}",
                "Margin": ("+" if zeta_dr > req_zeta_dr else "") + f"{100*(zeta_dr-req_zeta_dr)/req_zeta_dr:.2f} %",
                "Handling Qualities": Fore.GREEN +"Level 1"+ Style.RESET_ALL if zeta_dr >= req_zeta_dr else Fore.RED + "Unsatisfactory" + Style.RESET_ALL,
                "Reference": "4.6.1.1"            
                }

    nat_freq = {"Mode": " - ",
                "Parameter": " Natural Frequency\nω_n [rad/s]",
                "Design": f"{wn_dr:.4f}",
                "Requirement": f" ≥ {req_wn_dr}",
                "Margin": ("+" if wn_dr > req_wn_dr else "") + f"{100*(wn_dr - req_wn_dr)/req_wn_dr:.2f} %",
                "Handling Qualities": Fore.GREEN +"Level 1"+ Style.RESET_ALL if wn_dr >= req_wn_dr else Fore.RED + "Unsatisfactory" + Style.RESET_ALL,
                "Reference": "4.6.1.1",              
                }

    product = {"Mode": " - ",
                "Parameter": " Product\n ζ*ω_n [rad/s]",
                "Design": f"{zeta_dr*wn_dr:.4f}",
                "Requirement": f" ≥ {req_zw_dr}",
                "Margin": ("+" if zeta_dr*wn_dr > req_zw_dr else "") + f"{100*(zeta_dr*wn_dr - req_zw_dr)/req_zw_dr:.2f} %",
                "Handling Qualities": Fore.GREEN +"Level 1"+ Style.RESET_ALL if zeta_dr*wn_dr >= req_zw_dr else Fore.RED + "Unsatisfactory" + Style.RESET_ALL,
                "Reference": "4.6.1.1",
                }

    # Spiral mode requirement (Level 1, Class III, Category B)
    # Note: MIL-HDBK-1797A requires T_2 >= 20 s if unstable; any stable mode (lambda < 0) satisfies Level 1.
    t_halve_spiral = np.log(1/2) / spiral.real  # Time to halve amplitude
    req_spiral = 20 # Spiral mode requirement for Level 1 handling qualities (time to halve amplitude) in seconds

    spiral = {"Mode": Style.BRIGHT +"Spiral"+ Style.RESET_ALL,
                "Parameter": "Time to halve\namplitude [s]",
                "Design": f"{t_halve_spiral:.4f}", # Time to halve amplitude. It is the time when e^(lambda*t) = 0.5. # time to reduce to half amplitude: 
                "Requirement": f" Stable ",
                "Margin": " - " if spiral.real < 0 else Fore.RED + "Unstable" + Style.RESET_ALL,
                "Handling Qualities": Fore.GREEN +"Level 1"+ Style.RESET_ALL if spiral.real < 0 else Fore.RED + "Unsatisfactory" + Style.RESET_ALL,
                "Reference": "4.5.1.2",
                }

    # Roll Subsidence mode requirement (Level 1, Class III, Category B)
    t_const_rolls = 1.0 / np.abs(roll_subs)  # Time to 1/e amplitude
    req_rolls = 1.4                          # Maximum allowable roll time constant [s]

    roll_sub = {"Mode": "Roll\nSubsidence",
                "Parameter": "Roll time\nconstant 𝜏 [s]",
                "Design": f"{t_const_rolls:.4f}",
                "Requirement": f" ≤ {req_rolls}",
                "Margin": ("+" if t_const_rolls < req_rolls else "") + f"{100*(req_rolls-t_const_rolls)/req_rolls:.2f} %",
                "Handling Qualities": Fore.GREEN +"Level 1"+ Style.RESET_ALL if t_const_rolls <= req_rolls else Fore.RED + "Unsatisfactory" + Style.RESET_ALL,
                "Reference": "4.5.1.1"
                }

    mode_param = [damp_ratio, nat_freq, product, spiral, roll_sub]
    # Header for tabulate
    headers_map2 = {
        "Mode": Style.BRIGHT +"Mode" + Style.RESET_ALL,
        "Parameter": Style.BRIGHT +"Parameter\nfor requirement" + Style.RESET_ALL,
        "Design": Style.BRIGHT +"Design Value" + Style.RESET_ALL,
        "Requirement": Style.BRIGHT +"Requirement Value\nLevel 1, CATEGORY B - CRUISE" + Style.RESET_ALL,
        "Margin": Style.BRIGHT +"Margin" + Style.RESET_ALL,
        "Handling Qualities": Style.BRIGHT +"Flying\nQuality" + Style.RESET_ALL,
        "Reference": Style.BRIGHT +"Reference\n(MIL-HDBK-1797A)" + Style.RESET_ALL}

    print("\n")
    print(Style.BRIGHT + "- LATERAL/DIRECTIONAL FLYING QUALITIES COMPLIANCE (MIL-HDBK-1797):" + Style.NORMAL)

    print(tabulate(mode_param,
                    headers=headers_map2,
                    tablefmt="fancy_grid", 
                    stralign="center",
                    disable_numparse=True))
