"""Provides functions used to calculate the average spin state of FeO in Ferropericlase.

This file is associated with the article:
"Constraints on the composition and temperature of LLSVPs from seismic properties of
lower mantle minerals" by K. Vilella, T. Bodin, C.-E. Boukare, F. Deschamps, J. Badro,
M. D. Ballmer, and Y. Li

This file provides all the functions used to calculate the average spin state of FeO
in Ferropericlase (Fp) depending on temperature, pressure, and mineral composition.
The model is based on the work of Sturhahn et al. (2005) and this specific version is
described in Vilella et al. (2015).

The Mie-Gruneisen-Debye equation of state is used to calculate the properties of Fp.
These functions should not be used outside the class MineralProperties.

Typical usage example:

  from _spin_configuration import _calc_spin_configuration
  spin_config, P_table = _calc_spin_configuration(self)

Copyright, 2023,  Vilella Kenny.
"""
import numpy as np
import scipy.optimize
from _eos_implementation import _EOS_fp
#======================================================================================#
#                                                                                      #
#    Starting implementation of functions used to calculate the spin configuration     #
#                                                                                      #
#======================================================================================#
def _calc_spin_configuration(self):
    """Calculates the spin configuration of FeO in Ferropericlase.

    The theory behind this calculation is described in Vilella et al. (2015) and is
    based on a model developed in Sturhahn et al. (2005).

    As a brief summary, the spin state of FeO in the Ferropericlase (Fp) mineral
    changed with varying pressure and temperature. In particular, at ambient
    conditions, FeO is in a high spin state, while in the lowermost part of the
    Earth's mantle, FeO is in a low spin state.
    This transition from high spin state to low spin state is commonly referred to
    as the spin state transition and is associated with an increase in density.

    In this function, the average spin state of Fp (eta_ls) is determined by
    searching for the value of eta_ls minimizing the Helmholtz free energy (F), that
    is the most stable configuration. In practice, this is done by searching for
    the values of eta_ls leading to a derivative of F with respect to eta_ls equals
    to zero. However, this is challenging as several local minima may exist. As a
    result, the calculation is done using three different initial conditions, and
    the actual solution is the one associated with the lowest value of F.

    Returns:
        np.ndarray: Average spin state of FeO in Fp for a given value for the
                    temperature, volume of Fp, and FeO content in Fp.
        np.ndarray: Pressure for a given value for the temperature, volume of Fp, and
                    FeO content in Fp.
    """
    # Loading utility class for mineral properties calculation
    fp_eos = _EOS_fp()

    # Initializing range for spin configuration calculation
    x_min = 0.0
    x_max = 1.0
    delta_x = 0.01
    self.x_vec = np.arange(x_min, x_max + delta_x, delta_x)
    self.T_vec = self.T_am + np.arange(
        0.0, self.dT_max + self.delta_dT, self.delta_dT
    )

    # Calculating range for the volume of Fp using extreme cases
    solution = scipy.optimize.fsolve(
        lambda x: fp_eos._MGD(
            self, self.T_am + self.dT_max, self.P_am, x, 1.0, x_max
        ), 10.
    )
    v_min = solution[0] / 0.15055 - 2.0
    solution = scipy.optimize.fsolve(
        lambda x: fp_eos._MGD(self, self.T_am, self.P_am, x, 0.0, x_max), 10.
    )
    v_max = solution[0] / 0.15055

    n_T = len(self.T_vec)
    n_v = round((v_max - v_min) / self.delta_v) + 1
    n_x = len(self.x_vec)

    # Initializing
    spin_config = np.zeros((n_T, n_v, n_x))
    P_table = np.zeros((n_T, n_v, n_x))
    k_b = 8.617 * 10**(-5) # Boltzmann constant
    v_fp_0 = self.v_feo_hs_0 / 0.15055

    # The energy degeneracy of the electronic configuration for the low/high
    # spin state
    g_ls = 1.;
    g_hs = 15.

    for ii in range(n_x):
        x_fp = self.x_vec[ii]
        for jj in range(n_T):
            T = self.T_vec[jj]
            for kk in range(n_v):
                # Volume of Fp at P, T condition
                v_fp = kk * self.delta_v + v_min

                # Energy associated with low and high spin state
                E_ls = _energy_equation(self, 1, v_fp_0, v_fp)
                E_hs = _energy_equation(self, 3, v_fp_0, v_fp)

                # Coupling energy low spin state - low spin state
                wc = _splitting_energy(self, x_fp, v_fp_0, v_fp)

                # Equation parameters
                beta = 1 / (k_b * T)
                c = np.exp(beta * E_ls) / g_ls * g_hs * np.exp(-beta * E_hs)

                # Calculating solution for an initial condition equal to 0.0
                eta_ls_1 = scipy.optimize.fsolve(
                    lambda x: x * (1 + c * np.exp(-2 * beta * wc * x)) - 1, 0.
                )
                eta_hs_1 = 1 - eta_ls_1

                # Calculating the entropy to avoid issue with log
                s_1 = 0.0
                if (eta_ls_1 > 0.01): s_1 += eta_ls_1 * np.log(eta_ls_1 / g_ls)
                if (eta_hs_1 > 0.01): s_1 += eta_hs_1 * np.log(eta_hs_1 / g_hs)

                # Calculating the Helmholtz free energy
                F_1 = (
                    -wc * eta_ls_1 * eta_ls_1 + E_ls * eta_ls_1 + E_hs * eta_hs_1 +
                    (s_1 / beta)
                )

                # Calculating solution for an initial condition equal to 0.5
                eta_ls_2 = scipy.optimize.fsolve(
                    lambda x: x * (1 + c * np.exp(-2 * beta * wc * x)) - 1, 0.5
                )
                eta_hs_2 = 1 - eta_ls_2

                # Calculating the entropy to avoid issue with log
                s_2 = 0.0
                if (eta_ls_2 > 0.01): s_2 += eta_ls_2 * np.log(eta_ls_2 / g_ls)
                if (eta_hs_2 > 0.01): s_2 += eta_hs_2 * np.log(eta_hs_2 / g_hs)

                # Calculating the Helmholtz free energy
                F_2 = (
                    -wc * eta_ls_2 * eta_ls_2 + E_ls * eta_ls_2 + E_hs * eta_hs_2 +
                    (s_2 / beta)
                )

                # Calculating solution for an initial condition equal to 1.0
                eta_ls_3 = scipy.optimize.fsolve(
                    lambda x: x * (1 + c * np.exp(-2 * beta * wc * x)) - 1, 1.0
                )
                eta_hs_3 = 1 - eta_ls_3

                # Calculating the entropy to avoid issue with log
                s_3 = 0.0
                if (eta_ls_3 > 0.01): s_3 += eta_ls_3 * np.log(eta_ls_3 / g_ls)
                if (eta_hs_3 > 0.01): s_3 += eta_hs_3 * np.log(eta_hs_3 / g_hs)

                # Calculating the Helmholtz free energy
                F_3 = (
                    -wc * eta_ls_3 * eta_ls_3 + E_ls * eta_ls_3 + E_hs * eta_hs_3 +
                    (s_3 / beta)
                )

                # Determining the actual solution
                eta_ls_vect = [eta_ls_1, eta_ls_2, eta_ls_3]
                F_vect = [F_1, F_2, F_3]
                eta_ls = eta_ls_vect[np.argmin(F_vect)]

                # Storing information
                spin_config[jj, kk, ii] = eta_ls
                P_table[jj, kk, ii] = -fp_eos._MGD(
                    self, T, 0.0, v_fp*0.15055, eta_ls, x_fp
                )

    return spin_config, P_table


def _energy_equation(self, spin_state, v_0, v):
    """Calculates the energy level of Fe2+ associated with a given spin state.

    This function calculates the energy level associated with a given spin state (
    low spin state, mixed spin state, or high spin state), as shown in the Fig. 1
    of Sturhahn et al. (2005).

    The energy level has two main contributions:
    - The energy difference between the two electronic levels (delta_energy).
    - The energy required to pair two electrons (pairing_energy).

    Args:
        spin_state: Indicates the spin state considered.
                    1 for low spin state, 2 for mixed spin state, and 3 for high
                    spin state.
        v_0: Volume of Ferropericlase at ambient conditions. [A^3]
        v: Volume of Ferropericlase at considered conditions. [A^3]

    Returns:
        Float64: Energy of the spin state. [eV]
    """
    # Calculating energy required to pair electrons
    pairing_energy = self.delta_0 * (v_0 / self.v_trans)**self.xi
            
    # Calculating energy difference between the two energy levels
    delta_energy = self.delta_0 * (v_0 / v)**self.xi

    # Calculating energy associated with each spin state
    if (spin_state == 1):
        # Low spin state
        return 3 * pairing_energy - 2.4 * delta_energy
    elif (spin_state == 2):
        # Mixed spin state
        return 2 * pairing_energy - 1.4 * delta_energy
    elif (spin_state == 3):
        # High spin state
        return pairing_energy - 0.4 * delta_energy


def _splitting_energy(self, x_fp, v_0, v):
    """Calculates the splitting energy.

    This function calculates the coupling energy between low spin state iron atoms
    in Ferropericlase.

    Args:
        x_fp: FeO content in ferropericlase.
        v_0: Volume of Ferropericlase at ambient conditions. [A^3]
        v: Volume of Ferropericlase at considered conditions. [A^3]

    Returns:
        Float64: The splitting energy. [eV]
    """
    # Calculating energy difference between the two energy levels
    delta_energy = self.delta_0 * (v_0 / v)**self.xi

    return x_fp**self.xi * delta_energy