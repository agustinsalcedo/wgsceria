# %%
import numpy as np
from scipy.integrate import solve_ivp

# %%
# =========================
# Parameters
# =========================

# Physical constants and conversion factors
J2eV = 6.24150974e18           # eV/J
Na   = 6.0221415e23            # mol-1
h    = 6.62607015e-34 * J2eV   # eV*s
kb   = 1.38064852e-23 * J2eV   # eV/K

# Surface unit cell area per active site (m^2)
area = (118.059257638912 / 9.0) * 1e-20

# Molecular weights (kg)
mCO  = 28.01 / 1000 / Na
mH2O = 18.01528 / 1000 / Na
mCO2 = 44.01 / 1000 / Na
mH2  = 2.01588 / 1000 / Na

# %%
# =========================
# Kinetic model
# =========================

# Reaction energies (eV)
dE = np.array([
    -0.14, -1.30, 0.27, -0.72, -1.18,
     2.26, -0.04, -0.45, -0.07, -2.59,
    -0.58, -0.78, 1.49, 0.37, -0.40
])
  
# Activation energies (eV)
# Default: barrierless elementary steps
Ea = np.zeros_like(dE)
# Activated steps
Ea[[1,5,9,12,13,14]] = [0.40, 3.16, 0.42, 1.90, 1.84, 0.19]

# %%
# =========================
# Get kinetic constants
# =========================

def get_k(T, P):
    # Extract pressures
    PCO  = P[0]
    PH2O = P[1]
    PCO2 = P[2]
    PH2  = P[3]

    kbT = kb * T  # eV

    K = np.exp(-dE / kbT)

    Ea_corr = np.maximum.reduce([np.zeros_like(Ea), dE, Ea])
    kf = kbT/h * np.exp(-Ea_corr / kbT)
    kr = kf / K # Thermodynamic consistency

    # Fix adsorption steps
    K[0] = np.exp(-dE[0]/kbT - 1) * 2.8 / T * h / np.sqrt(area * 2 * np.pi * mCO * kbT * J2eV)
    K[3] = np.exp(-dE[3]/kbT - 1) * 2 * 21.5 / T * h / np.sqrt(area * 2 * np.pi * mH2O * kbT * J2eV)
    K[2] = 1 / (np.exp(dE[2]/kbT - 1) * 2 * 0.561 / T * h / np.sqrt(area * 2 * np.pi * mCO2 * kbT * J2eV))
    K[6] = 1 / (np.exp(dE[6]/kbT - 1) * 2 * 87.61 / T * h / np.sqrt(area * 2 * np.pi * mH2 * kbT * J2eV))
    K[7] = np.exp(-dE[7]/kbT - 1) * 2 * 21.5 / T * h / np.sqrt(area * 2 * np.pi * mH2O * kbT * J2eV)
    K[10] = np.exp(-dE[10]/kbT - 1) * 2 * 0.561 / T * h / np.sqrt(area * 2 * np.pi * mCO2 * kbT * J2eV)

    kf[0] = (area * PCO * 100000) / np.sqrt(2 * np.pi * kbT / J2eV * mCO)
    kf[3] = (area * PH2O * 100000) / np.sqrt(2 * np.pi * kbT / J2eV * mH2O)
    kr[2] = (area * PCO2 * 100000) / np.sqrt(2 * np.pi * kbT / J2eV * mCO2)
    kr[6] = (area * PH2 * 100000) / np.sqrt(2 * np.pi * kbT / J2eV * mH2)
    kf7_ads = kf[3]
    kf10_ads = kr[2]
    kf[7]  = kf7_ads
    kf[10] = kf10_ads

    kr[0] = kf[0] / K[0]
    kr[3] = kf[3] / K[3]
    kf[2] = kr[2] * K[2]
    kf[6] = kr[6] * K[6]
    kr[7]  = kf[7] / K[7]
    kr[10] = kf[10] / K[10]

    return K, kf, kr

# %%
# =========================
# Get reaction rates
# =========================
def get_rates(theta, kf, kr):

    (
        tCO,
        tH2,
        tvO,
        tCO2,
        tOH,
        tH2O,
        tH2Oads,
        tOHads,
        tCO2ads,
        tcarboxyl,
        tformate,
        tCe,
        tO,
    ) = theta

    rate = np.empty(15)

    rate[0]  = kf[0]  * tCe - kr[0] * tCO
    rate[1]  = kf[1]  * tCO * tO - kr[1] * tCO2 * tCe
    rate[2]  = kf[2]  * tCO2 - kr[2] * tvO
    rate[3]  = kf[3]  * tCe * tvO - kr[3] * tH2O
    rate[4]  = kf[4]  * tH2O * tO - kr[4] * tOH**2 * tCe
    rate[5]  = kf[5]  * tOH**2 * tCe - kr[5] * tH2 * tO**2
    rate[6]  = kf[6]  * tH2 - kr[6] * tCe
    rate[7]  = kf[7]  * tCe - kr[7] * tH2Oads
    rate[8]  = kf[8]  * tH2Oads * tO - kr[8] * tOHads * tOH
    rate[9]  = kf[9]  * tCO * tOHads * tO - kr[9] * tcarboxyl * tCe
    rate[10] = kf[10] * tCe**2 * tO - kr[10] * tCO2ads
    rate[11] = kf[11] * tCO2ads * tH2O - kr[11] * tcarboxyl * tCe**2 * tOH
    rate[12] = kf[12] * tcarboxyl * tCe - kr[12] * tformate * tO
    rate[13] = kf[13] * tformate * tOH * tCe - kr[13] * tCO2ads * tH2
    rate[14] = kf[14] * tcarboxyl * tCe * tO - kr[14] * tCO2ads * tOH

    return rate

# %%
# =========================
# Get system of ODEs
# =========================
def get_odes(rate):
    """
    Returns d(theta)/dt from elementary reaction rates.
    """

    dt = np.empty(13)

    dt[0]  = rate[0] - rate[1] - rate[9]                                     # d(CO)/dt
    dt[1]  = rate[5] + rate[13] - rate[6]                                    # d(H2)/dt
    dt[2]  = rate[2] - rate[3]                                               # d(vO)/dt
    dt[3]  = rate[1] - rate[2]                                               # d(CO2)/dt
    dt[4]  = (2 * rate[4] + rate[8] + rate[11]
              + rate[14] -2 * rate[5] - rate[13])                            # d(OH)/dt
    dt[5]  = rate[3] - rate[4] - rate[11]                                    # d(H2O)/dt
    dt[6]  = rate[7] - rate[8]                                               # d(H2Oads)/dt
    dt[7]  = rate[8] - rate[9]                                               # d(OHads)/dt
    dt[8]  = rate[10] + rate[13] + rate[14] - rate[11]                       # d(CO2ads)/dt
    dt[9]  = rate[9] + rate[11] - rate[12] - rate[14]                        # d(carboxyl)/dt
    dt[10] = rate[12] - rate[13]                                             # d(formate)/dt
    dt[11] = (
        rate[1] + rate[4] + rate[6] + rate[9] + 2 * rate[11]
        - rate[0] - rate[3] - rate[5] - rate[7]
        - 2 * rate[10] - rate[12] - rate[13] - rate[14]
    )                                                                        # d(Ce)/dt
    dt[12] = (
        2 * rate[5] + rate[12]
        - rate[1] - rate[4] - rate[8] - rate[9]
        - rate[10] - rate[14]
    )                                                                        # d(O)/dt

    return dt

# %%
# =========================
# Solve microkinetic ODEs
# =========================
def int_conv(kf, kr, theta0, tol=3e-9, tmax=1e12):
    """
    Integrate microkinetic ODEs until steady state.

    Parameters
    ----------
    kf, kr : Forward and reverse rate constants
    theta0 : Initial coverage guess (length 13)
    tol : Convergence criterion |dtheta/dt|
    tmax : Maximum integration time

    Returns
    -------
    theta_ss : Final coverage
    """

    theta0 = np.asarray(theta0, dtype=float)

    def rhs(t, theta):
        rate = get_rates(theta, kf, kr)
        return get_odes(rate)

#    def steady_event(t, theta):
#        rate = get_rates(theta, kf, kr)
#        dt = get_odes(rate)
#        return np.max(np.abs(dt)) - tol

#    steady_event.terminal = True
#    steady_event.direction = -1

    sol = solve_ivp(
        rhs,
        (0.0, tmax),
        theta0,
        method="BDF",
        atol=1e-12,
        rtol=1e-8,
#        events=steady_event,
    )

    theta_ss = sol.y[:, -1]

    # explicit convergence check
    rates = get_rates(theta_ss, kf, kr)
    dt = get_odes(rates)

    #if np.max(np.abs(dt)) > tol:
    #    print("WARNING: steady state not fully reached")
    #    print("max residual:", np.max(np.abs(dt)))

    return theta_ss
# %%
# =========================
# Get degree of rate control (DRC)
# =========================
def get_drc(K, kf, theta0, step_idx, eps=0.02):
           
    kf_up = kf.copy()
    kf_down = kf.copy()

    # Perturb kf of the step of interest
    kf_up[step_idx] *= (1 + eps)
    kf_down[step_idx] *= (1 - eps)

    # Recalculate kr to maintain thermodynamic consistency
    kr_up = kf_up / K
    kr_down = kf_down / K

    theta_up = int_conv(kf_up, kr_up, theta0)
    theta_down = int_conv(kf_down, kr_down, theta0)

    rate_up = get_rates(theta_up, kf_up, kr_up)
    rate_down = get_rates(theta_down, kf_down, kr_down)

    # Calculate DRC
    if rate_up[6] <= 1e-30 or rate_down[6] <= 1e-30:
        drc = np.nan
    else:
        drc = np.log(rate_up[6] / rate_down[6]) / np.log((1 + eps) / (1 - eps))

    return drc