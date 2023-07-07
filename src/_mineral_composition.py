"""Calculates mineral composition of rock assemblages.

This file is associated with the article:
"Constraints on the composition and temperature of LLSVPs from seismic properties of
lower mantle minerals" by K. Vilella, T. Bodin, C.-E. Boukare, F. Deschamps, J. Badro,
M. D. Ballmer, and Y. Li

The purpose of the functions in this file is to calculate the mineral composition of a
rock assemblage for a wide range of input parameters. The six input parameters are the
temperature contrast against the ambient mantle, the proportion of Calcio Perovskite
(CaPv), the proportion of Bridgmanite (Bm), the FeO content, the alumina content, and
the oxidation state of iron in Bm. Using these input parameters, the simulator
calculates the molar concentration of FeO in Bm, the molar concentration of FeO in
Ferropericlase (Fp), the molar concentration of AlO2 in Bm, the density of Bm, and the
density of Fp.

The results are written into separate files named after the considered temperature
contrast. The simulation can be resumed if stopped, allowing for incremental
calculations. The obtained results can then be used to calculate the seismic properties
of all the rock assemblages.

Solving the equations governing this problem is challenging due to their highly
non-linear nature. If a solution is not found, a value of 0.0 is returned for each
property. To improve convergence, adjustments to the starting conditions or the
non-linear solver are required, both of which can be quite challenging.

Note that these functions are intended for use within the MineralProperties class and
should not be used outside of it.

Typical usage example:

  from _spin_configuration import _calc_spin_configuration
  from _mineral_composition import _calc_mineral_composition
  spin_config, P_table = _calc_spin_configuration(self)
  _calc_mineral_composition(self, spin_config, P_table)

Copyright, 2023,  Vilella Kenny.
"""
import numpy as np
from math import isclose
import os
import random
import scipy.optimize
from ._eos_implementation import _EOS_fp, _EOS_bm, _EOS_capv


# ==================================================================================== #
#                                                                                      #
#    Starting implementation of functions used to calculate the mineral composition    #
#                                                                                      #
# ==================================================================================== #
def _calc_mineral_composition(self, spin_config: np.ndarray, P_table: np.ndarray):
    """Calculates the mineral composition of a wide range of rock assemblages.

    This function calculates the properties of a large range of mineral composition and
    write the results into files.
    The calculated properties are the molar concentration of FeO in Bm, the molar
    concentration of FeO in Fp, the molar concentration of AlO2 in Bm, the density of
    Bm, and the density of Fp.

    The range of mineral compositions investigated are provided to the class
    MineralProperties as an optional input.

    The directory where the results are saved is given by the `path` attribute of the
    class MineralProperties. The name of the file corresponds to the value of the
    temperature contrast against the ambient mantle.
    The simulation can be resumed if existing files are already present. As a result, it
    is necessary to remove old files from the results folder before launching a new set
    of simulations.

    An attempt is made to use a previous solution as starting conditions for the solver,
    but if no previous solution with close enough composition is available, default
    starting conditions are used.

    Args:
        spin_config: Average spin state of FeO in Fp for a given value for the
                     temperature, volume of Fp, and FeO content in Fp.
        P_table: Pressure for a given value for the temperature, volume of Fp, and
                 FeO content in Fp.
    """
    # Loading utility class for mineral properties calculation
    capv_eos = _EOS_capv()

    # Calculating number of input for each parameter
    n_capv = round((self.p_capv_max - self.p_capv_min) / self.delta_capv) + 1
    n_dT = round((self.dT_max - self.dT_min) / self.delta_dT) + 1
    n_feo = round(
        (self.iron_content_max - self.iron_content_min) / self.delta_iron_content) + 1
    n_al = round((self.al_content_max - self.al_content_min) / self.delta_al) + 1
    n_ratio_fe_bm = round(
        (self.ratio_fe_bm_max - self.ratio_fe_bm_min) / self.delta_ratio_fe) + 1
    n_bm = round((1.0 - self.p_bm_min) / self.delta_bm) + 1

    print("Starting the loop...")
    for ii in range(n_dT):
        # Initializing the matrices
        p_capv_ii = np.zeros((n_capv, n_bm, n_feo, n_al, n_ratio_fe_bm))
        p_bm_ii = np.zeros((n_capv, n_bm, n_feo, n_al, n_ratio_fe_bm))
        feo_ii = np.zeros((n_capv, n_bm, n_feo, n_al, n_ratio_fe_bm))
        al_ii = np.zeros((n_capv, n_bm, n_feo, n_al, n_ratio_fe_bm))
        ratio_fe_ii = np.zeros((n_capv, n_bm, n_feo, n_al, n_ratio_fe_bm))
        x_feo_bm_ii = np.zeros((n_capv, n_bm, n_feo, n_al, n_ratio_fe_bm))
        x_feo_fp_ii = np.zeros((n_capv, n_bm, n_feo, n_al, n_ratio_fe_bm))
        x_alo2_bm_ii = np.zeros((n_capv, n_bm, n_feo, n_al, n_ratio_fe_bm))
        rho_bm_ii = np.zeros((n_capv, n_bm, n_feo, n_al, n_ratio_fe_bm))
        rho_fp_ii = np.zeros((n_capv, n_bm, n_feo, n_al, n_ratio_fe_bm))
        rho_capv_ii = np.zeros((n_capv, n_bm, n_feo, n_al, n_ratio_fe_bm))

        # Setting temperature contrast
        dT = self.dT_min + ii * self.delta_dT

        # Calculating density of CaPv at P, T conditions
        solution = scipy.optimize.fsolve(
            lambda x: capv_eos._MGD(self, self.P_am, self.T_am + dT, x), 30.)
        rho_capv = self.rho_capv_0 * self.v_casio3_0 / solution[0]

        # Initializing starting condition
        x_init = [0.3, 0.1, 0.1, 5000.0, 5000.0]

        filename = os.path.join(self.path, str(int(dT)) + ".csv")
        if os.path.isfile(filename):
            ### A file already exists ###
            print("File detected")
            print("Resuming the simulation")
            with open(filename, newline='') as f:
                data = f.readlines()
                for row in data:
                    # Calculating the corresponding indices
                    processed_row = [float(s) for s in row.split()]
                    jj = round((processed_row[1] - self.p_capv_min) / self.delta_capv)
                    kk = round((processed_row[2] - self.p_bm_min) / self.delta_bm)
                    ll = round(
                        (processed_row[3] - self.iron_content_min) /
                        self.delta_iron_content)
                    mm = round((processed_row[4] - self.al_content_min) / self.delta_al)
                    nn = round(
                        (processed_row[5] - self.ratio_fe_bm_min) / self.delta_ratio_fe)

                    # Loading the data
                    x_feo_bm_ii[jj, kk, ll, mm, nn] = processed_row[6]
                    x_feo_fp_ii[jj, kk, ll, mm, nn] = processed_row[7]
                    x_alo2_bm_ii[jj, kk, ll, mm, nn] = processed_row[8]
                    rho_bm_ii[jj, kk, ll, mm, nn] = processed_row[9]
                    rho_fp_ii[jj, kk, ll, mm, nn] = processed_row[10]

            # Initializing the loop
            jj_start = jj
            kk_start = kk
            ll_start = ll
            mm_start = mm
            nn_start = nn
        else:
            ### No file exist ###
            # Initializing the loop
            jj_start = 0
            kk_start = 0
            ll_start = 0
            mm_start = 0
            nn_start = 0

        # Starting the calculation
        for jj in np.arange(jj_start, n_capv, 1):
            p_capv = self.p_capv_min + jj * self.delta_capv
            n_bm_max = round((1 - p_capv - self.p_bm_min) / self.delta_bm) + 1
            for kk in np.arange(kk_start, n_bm_max, 1):
                p_bm = self.p_bm_min + kk * self.delta_bm
                p_fp = 1 - p_bm - p_capv
                for ll in np.arange(ll_start, n_feo, 1):
                    feo = self.iron_content_min + ll * self.delta_iron_content
                    for mm in np.arange(mm_start, n_al, 1):
                        al = self.al_content_min + mm * self.delta_al
                        for nn in np.arange(nn_start, n_ratio_fe_bm, 1):
                            ratio_fe = (self.ratio_fe_bm_min + nn * self.delta_ratio_fe)

                            if (mm != 0):
                                ### Using previous starting conditions ###
                                x_init = [
                                    x_feo_bm_ii[jj, kk, ll, mm - 1, nn],
                                    x_feo_fp_ii[jj, kk, ll, mm - 1, nn],
                                    x_alo2_bm_ii[jj, kk, ll, mm - 1, nn],
                                    rho_bm_ii[jj, kk, ll, mm - 1, nn],
                                    rho_fp_ii[jj, kk, ll, mm - 1, nn],
                                ]
                            if (x_init[0] < 0.0):
                                ### Starting condition is incorrect ###
                                ll_iter = ll
                                while ((x_init[0] < 0.0) and (ll_iter > 0)):
                                    ll_iter -= 1
                                x_init = [
                                    x_feo_bm_ii[jj, kk, ll_iter, mm, nn],
                                    x_feo_fp_ii[jj, kk, ll_iter, mm, nn],
                                    x_alo2_bm_ii[jj, kk, ll_iter, mm, nn],
                                    rho_bm_ii[jj, kk, ll_iter, mm, nn],
                                    rho_fp_ii[jj, kk, ll_iter, mm, nn],
                                ]

                            # Calculating the composition of the rock assemblage
                            x_result = _solve_mineral_composition(
                                self, x_init, dT, p_capv, p_bm, p_fp, feo, al, ratio_fe,
                                spin_config, P_table, rho_capv)

                            # Loading results
                            x_feo_bm_ii[jj, kk, ll, mm, nn] = x_result[0]
                            x_feo_fp_ii[jj, kk, ll, mm, nn] = x_result[1]
                            x_alo2_bm_ii[jj, kk, ll, mm, nn] = x_result[2]
                            rho_bm_ii[jj, kk, ll, mm, nn] = x_result[3]
                            rho_fp_ii[jj, kk, ll, mm, nn] = x_result[4]

                            # Writting the data
                            with open(filename, 'a', newline='') as f:
                                f.write(
                                    f"{dT:.0f}\t{p_capv:.2f}\t{p_bm:.2f}\t{feo:.3f}\t" +
                                    f"{al:.3f}\t{ratio_fe:.1f}\t{x_result[0]:.3f}\t" +
                                    f"{x_result[1]:.3f}\t{x_result[2]:.3f}\t" +
                                    f"{x_result[3]:.0f}\t{x_result[4]:.0f}\n")
                            p_capv_ii[jj, kk, ll, mm, nn] = p_capv
                            p_bm_ii[jj, kk, ll, mm, nn] = p_bm
                            feo_ii[jj, kk, ll, mm, nn] = feo
                            al_ii[jj, kk, ll, mm, nn] = al
                            ratio_fe_ii[jj, kk, ll, mm, nn] = ratio_fe
                            rho_capv_ii[jj, kk, ll, mm, nn] = rho_capv
                        # End of Ratio Fe loop

                        # Updating starting conditions
                        x_init = [
                            x_feo_bm_ii[jj, kk, ll, mm, 0],
                            x_feo_fp_ii[jj, kk, ll, mm, 0],
                            x_alo2_bm_ii[jj, kk, ll, mm, 0],
                            rho_bm_ii[jj, kk, ll, mm, 0],
                            rho_fp_ii[jj, kk, ll, mm, 0],
                        ]
                        nn_start = 0  # In case the calculation has been resumed
                    # End of Al content loop

                    # Updating starting conditions
                    x_init = [
                        x_feo_bm_ii[jj, kk, ll, 0, 0],
                        x_feo_fp_ii[jj, kk, ll, 0, 0],
                        x_alo2_bm_ii[jj, kk, ll, 0, 0],
                        rho_bm_ii[jj, kk, ll, 0, 0],
                        rho_fp_ii[jj, kk, ll, 0, 0],
                    ]
                    mm_start = 0  # In case the calculation has been resumed
                # End of FeO content loop

                # Updating starting conditions
                x_init = [
                    x_feo_bm_ii[jj, kk, 0, 0, 0],
                    x_feo_fp_ii[jj, kk, 0, 0, 0],
                    x_alo2_bm_ii[jj, kk, 0, 0, 0],
                    rho_bm_ii[jj, kk, 0, 0, 0],
                    rho_fp_ii[jj, kk, 0, 0, 0],
                ]
                ll_start = 0  # In case the calculation has been resumed
            # End of Bm proportion loop

            # Updating starting conditions
            x_init = [
                x_feo_bm_ii[jj, 0, 0, 0, 0],
                x_feo_fp_ii[jj, 0, 0, 0, 0],
                x_alo2_bm_ii[jj, 0, 0, 0, 0],
                rho_bm_ii[jj, 0, 0, 0, 0],
                rho_fp_ii[jj, 0, 0, 0, 0],
            ]
            kk_start = 0  # In case the calculation has been resumed
        # End of CaPv proportion loop

        # Updating starting conditions
        x_init = [
            x_feo_bm_ii[0, 0, 0, 0, 0],
            x_feo_fp_ii[0, 0, 0, 0, 0],
            x_alo2_bm_ii[0, 0, 0, 0, 0],
            rho_bm_ii[0, 0, 0, 0, 0],
            rho_fp_ii[0, 0, 0, 0, 0],
        ]
        jj_start = 0  # In case the calculation has been resumed
    # End of temperature loop


def _solve_mineral_composition(
        self, x_init: list, dT: float, p_capv: float, p_bm: float, p_fp: float,
        feo: float, al: float, ratio_fe: float, spin_config: np.ndarray,
        P_table: np.ndarray, rho_capv: float) -> list:
    """Calculates the mineral composition of the provided rock assemblage.

    This function calculates five properties that fully characterize the composition of
    the rock assemblage: molar concentration of FeO in Bridgmanite (Bm), molar
    concentration of FeO in Ferropericlase (Fp), molar concentration of AlO2 in Bm,
    density of Bm, and density of Fp. These properties are determined based on the
    values of six input parameters: temperature contrast against the ambient mantle,
    proportion of Calcio Perovskite (CaPv), proportion of Bm, FeO content,
    alumina content, and oxidation state in Bm.

    Because of the complexity of the calculation, several techniques are used to make
    the physics problem easier to solve. Firstly, all densities are divided by 5500 to
    prevent the generation of large numbers during the solving step. Secondly, an
    initial empirical guess is made regarding the composition of Bm, whether it contains
    Fe2O3 or Al2O3. The consistency of the initial solution is then checked, and the
    guess is reversed if inconsistencies arise. Lastly, the physics problem is divided
    into two cases: with and without the presence of Fp. This separation is necessary
    because the system of equations to solve differs between these two cases.

    Note that, if no solution can be found, a value of 0.0 is returned for each
    property.

    Args:
        x_init: Starting conditions for the physics problem, that is, the molar
                concentration of FeO in Bm, the molar concentration of FeO in Fp, the
                molar concentration of AlO2 in Bm, the rescaled density of Bm, and the
                rescaled density of Fp.
        dT: Temperature contrast against the ambient mantle. [K]
        p_capv: Proportion of CaPv. [vol%]
        p_bm: Proportion of Bm. [vol%]
        p_fp: Proportion of Fp. [vol%]
        feo: FeO content in the rock assemblage. [wt%]
        al: Al2O3 content in the rock assemblage. [wt%]
        ratio_fe: Oxidation state in Bm.
        spin_config: Average spin state of FeO in Fp for a given value for the
                     temperature, volume of Fp, and FeO content in Fp.
        P_table: Pressure for a given value for the temperature, volume of Fp, and
                 FeO content in Fp.
        rho_capv: Density of CaPv for the considered conditions. Its value has been
                  rescaled by 5500. [kg/m^3]

    Returns:
        A list containing the molar concentration of FeO in Bm, FeO in Fp, AlO2 in Bm,
        the density of Bm and Fp
    """
    # First guess on whether Al or Fe is in excess in Bm
    if (al < 0.75 * ratio_fe * feo):
        al_excess = True
    else:
        al_excess = False

    # Converting inputs
    rho_capv = rho_capv / 5500

    if (p_fp > 0.0):
        ### Ferropericlase is present in the rock assemblage ###
        # Solving the system of equation
        x_init = [x_init[1], x_init[3] / 5500, x_init[4] / 5500]
        x_feo_fp, rho_bm, rho_fp = _solve_with_fp(
            self, x_init, dT, p_capv, p_bm, p_fp, feo, al, ratio_fe, spin_config,
            P_table, rho_capv, al_excess)

        if (x_feo_fp != 0.0):
            ### A solution has been found ###
            # Getting extra outputs
            x_feo_bm, x_alo2_bm = _oxides_content_in_bm(
                self, p_capv, p_bm, p_fp, feo, al, ratio_fe, x_feo_fp, rho_capv, rho_bm,
                rho_fp)
        else:
            ### No solution has been found ###
            x_feo_bm, x_alo2_bm = 0.0, 0.0

        if ((not al_excess) and (x_feo_fp == 0.0 or ratio_fe * x_feo_bm < x_alo2_bm)):
            ### First guess for al_excess is incorrect ###
            # Trying to solve the system of equation with al_excess
            al_excess = True
            x_init = [x_feo_fp, rho_bm, rho_fp]
            x_feo_fp, rho_bm, rho_fp = _solve_with_fp(
                self, x_init, dT, p_capv, p_bm, p_fp, feo, al, ratio_fe, spin_config,
                P_table, rho_capv, al_excess)

            if (x_feo_fp != 0.0):
                ### A solution has been found ###
                # Getting extra outputs
                x_feo_bm, x_alo2_bm = _oxides_content_in_bm(
                    self, p_capv, p_bm, p_fp, feo, al, ratio_fe, x_feo_fp, rho_capv,
                    rho_bm, rho_fp)

            # Checking that solution is indeed correct
            if (x_feo_fp != 0.0 and ratio_fe * x_feo_bm > x_alo2_bm):
                ### Guess for al_excess is again incorrect ###
                # Skipping this calculation
                x_feo_bm, x_feo_fp, x_alo2_bm, rho_bm, rho_fp = (0., 0., 0., 0., 0.)
                print("Problem with Al_excess")
                print(
                    "p_bm = ", p_bm, " p_capv = ", p_capv, " feo = ", feo, " al = ", al,
                    " ratio_fe = ", ratio_fe, " dT = ", dT)
                print("Skip this condition")
        elif ((al_excess) and (x_feo_fp == 0.0 or ratio_fe * x_feo_bm > x_alo2_bm)):
            ### First guess for al_excess is incorrect ###
            # Trying to solve the system of equation without al_excess
            al_excess = False
            x_init = [x_feo_fp, rho_bm, rho_fp]
            x_feo_fp, rho_bm, rho_fp = _solve_with_fp(
                self, x_init, dT, p_capv, p_bm, p_fp, feo, al, ratio_fe, spin_config,
                P_table, rho_capv, al_excess)

            if (x_feo_fp != 0.0):
                ### A solution has been found ###
                # Getting extra outputs
                x_feo_bm, x_alo2_bm = _oxides_content_in_bm(
                    self, p_capv, p_bm, p_fp, feo, al, ratio_fe, x_feo_fp, rho_capv,
                    rho_bm, rho_fp)

            # Checking that solution is indeed correct
            if (x_feo_fp != 0.0 and ratio_fe * x_feo_bm < x_alo2_bm):
                ### Guess for al_excess is again incorrect ###
                # Skipping this calculation
                x_feo_bm, x_feo_fp, x_alo2_bm, rho_bm, rho_fp = (0., 0., 0., 0., 0.)
                print("Problem with Al_excess")
                print(
                    "p_bm = ", p_bm, " p_capv = ", p_capv, " feo = ", feo, " al = ", al,
                    " ratio_fe = ", ratio_fe, " dT = ", dT)
                print("Skip this condition")

        # Verifying that the solution is indeed consistent
        _set_eqs_with_fp(
            self, [x_feo_fp, rho_bm, rho_fp], dT, p_capv, p_bm, p_fp, feo, al, ratio_fe,
            spin_config, P_table, rho_capv, al_excess, True)
    else:
        ### Ferropericlase is absent from the rock assemblage ###
        # Solving the system of equation
        x_init = [x_init[0], x_init[2], x_init[3] / 5500]
        x_feo_bm, x_alo2_bm, rho_bm = _solve_without_fp(
            self, x_init, dT, p_capv, p_bm, feo, al, ratio_fe, rho_capv, al_excess)
        x_feo_fp = 0.0
        rho_fp = 0.0

        if ((not al_excess) and (ratio_fe * x_feo_bm < x_alo2_bm)):
            ### First guess for al_excess is incorrect ###
            # Trying to solve the system of equation with al_excess
            al_excess = True
            x_init = [x_feo_bm, x_alo2_bm, rho_bm]
            x_feo_bm, x_alo2_bm, rho_bm = _solve_without_fp(
                self, x_init, dT, p_capv, p_bm, feo, al, ratio_fe, rho_capv, al_excess)

            # Checking that solution is indeed correct
            if (ratio_fe * x_feo_bm > x_alo2_bm):
                ### Guess for al_excess is again incorrect ###
                # Skipping this calculation
                x_feo_bm, x_alo2_bm, rho_bm = (0., 0., 0.)
                print("Problem with Al_excess")
                print(
                    "p_bm = ", p_bm, " p_capv = ", p_capv, " feo = ", feo, " al = ", al,
                    " ratio_fe = ", ratio_fe, " dT = ", dT)
                print("Skip this condition")
        elif ((al_excess) and (ratio_fe * x_feo_bm > x_alo2_bm)):
            ### First guess for al_excess is incorrect ###
            # Trying to solve the system of equation without al_excess
            al_excess = False
            x_init = [x_feo_bm, x_alo2_bm, rho_bm]
            x_feo_bm, x_alo2_bm, rho_bm = _solve_without_fp(
                self, x_init, dT, p_capv, p_bm, feo, al, ratio_fe, rho_capv, al_excess)

            # Checking that solution is indeed correct
            if (ratio_fe * x_feo_bm < x_alo2_bm):
                ### Guess for al_excess is again incorrect ###
                # Skipping this calculation
                x_feo_bm, x_alo2_bm, rho_bm = (0., 0., 0.)
                print("Problem with Al_excess")
                print(
                    "p_bm = ", p_bm, " p_capv = ", p_capv, " feo = ", feo, " al = ", al,
                    " ratio_fe = ", ratio_fe, " dT = ", dT)
                print("Skip this condition")

    return [x_feo_bm, x_feo_fp, x_alo2_bm, 5500 * rho_bm, 5500 * rho_fp]


def _solve_with_fp(
        self, x_init: list, dT: float, p_capv: float, p_bm: float, p_fp: float,
        feo: float, al: float, ratio_fe: float, spin_config: np.ndarray,
        P_table: np.ndarray, rho_capv: float, al_excess: bool) -> list:
    """Solves the physics problem with Ferropericlase.

    This function implements a solver for the system of equations governing the physics
    problem when Ferropericlase (Fp) is present. Solving this highly non-linear problem
    is challenging, especially when a good initial guess is not available, such as at
    the beginning of the simulation loop.

    To address this challenge, the function first attempts to find a solution using the
    provided starting conditions. If a solution cannot be found, the starting conditions
    are randomly sampled within predetermined ranges until a solution is obtained or the
    number of attempts exceeds 10. If a solution cannot be found under these
    conditions, a value of 0.0 is returned for each property.

    If finding a solution remains difficult, the user may manually increase the maximum
    number of attempts or the maximum number of iterations (maxiter) in the solver call.

    It is important to note that various solvers from the scipy.optimize library have
    been tested (e.g., minimize, fsolve, least_squares), but they did not yield
    satisfactory results.

    Args:
        x_init: Starting conditions for the physics problem, that is, the molar
                concentration of FeO in Fp, the rescaled density of Bm, and the
                rescaled density of Fp.
        dT: Temperature contrast against the ambient mantle. [K]
        p_capv: Proportion of CaPv. [vol%]
        p_bm: Proportion of Bm. [vol%]
        p_fp: Proportion of Fp. [vol%]
        feo: FeO content in the rock assemblage. [wt%]
        al: Al2O3 content in the rock assemblage. [wt%]
        ratio_fe: Oxidation state in Bm.
        spin_config: Average spin state of FeO in Fp for a given value for the
                     temperature, volume of Fp, and FeO content in Fp.
        P_table: Pressure for a given value for the temperature, volume of Fp, and
                 FeO content in Fp.
        rho_capv: Density of CaPv for the considered conditions. Its value has been
                  rescaled by 5500. [kg/m^3]
        al_excess: Flag indicating whether alumina is assumed to be in excess in Bm.

    Returns:
        A list containing the molar concentration of FeO in Fp, the rescaled density
        of Bm and Fp.
    """
    # Initialization
    n_iter = 0

    while (n_iter < 10):
        ### Solution has not yet been found ###
        try:
            ### Solution has been found ###
            # Solving the system of equation
            solution = scipy.optimize.newton_krylov(
                lambda x: _set_eqs_with_fp(
                    self, x, dT, p_capv, p_bm, p_fp, feo, al, ratio_fe, spin_config,
                    P_table, rho_capv, al_excess),
                x_init,
                maxiter=1000)

            return solution
        except Exception:
            ### Solution has not been found ###
            # Setting random starting conditions
            x_init[0] = random.uniform(0.0, 1.0)
            x_init[1] = random.uniform(0.8, 1.8)
            x_init[2] = random.uniform(0.8, 1.8)
            n_iter += 1

    return [0.0, 0.0, 0.0]


def _set_eqs_with_fp(
        self,
        var_in: list,
        dT: float,
        p_capv: float,
        p_bm: float,
        p_fp: float,
        feo: float,
        al: float,
        ratio_fe: float,
        spin_config: np.ndarray,
        P_table: np.ndarray,
        rho_capv: float,
        al_excess: bool,
        testing: bool = False) -> list:
    """Implements the equations for the physics problem with Ferropericlase.

    This function calculates the residue for the system of equations governing the
    physics problem when Ferropericlase (Fp) is present.

    The first and second equation come from the equation of state for Fp and
    Bridgmanite (Bm), respectively. More specifically, the Mie-Gruneisen-Debye equation
    of state, which corresponds to the eq. (33) of Jackson and Rigden (1996).
    The third equation comes from the alumina content in Bm. It corresponds to the
    eq. (13) in the supplementary material of Vilella et al. (2021).

    Args:
        var_in: Vector composed of the equation unknowns, that is, the molar
                concentration of FeO in Fp, the rescaled density of Bm, and the
                rescaled density of Fp.
        dT: Temperature contrast against the ambient mantle. [K]
        p_capv: Proportion of CaPv. [vol%]
        p_bm: Proportion of Bm. [vol%]
        p_fp: Proportion of Fp. [vol%]
        feo: FeO content in the rock assemblage. [wt%]
        al: Al2O3 content in the rock assemblage. [wt%]
        ratio_fe: Oxidation state in Bm.
        spin_config: Average spin state of FeO in Fp for a given value for the
                     temperature, volume of Fp, and FeO content in Fp.
        P_table: Pressure for a given value for the temperature, volume of Fp, and
                 FeO content in Fp.
        rho_capv: Density of CaPv for the considered conditions. Its value has been
                  rescaled by 5500. [kg/m^3]
        al_excess: Flag indicating whether alumina is assumed to be in excess in Bm.
        testing: Whether tests are conducted to make sure that the provided values of
                 var_in lead to a consistent composition.

    Returns:
        A list containing the residue for the equation of state for Ferropericlase, the
        equation of state for Bridgmanite, and the equation for the alumina content in
        Bridgmanite.
    """
    # Loading utility class for mineral properties calculation
    fp_eos = _EOS_fp()
    bm_eos = _EOS_bm()

    # Unpacking input variables
    x_feo_fp, rho_bm, rho_fp = var_in

    # Checking that input conditions makes sense
    if ((rho_fp < 0.1) or (rho_bm < 0.1) or (x_feo_fp < 0.0) or (x_feo_fp > 1.0)):
        return None

    # Average spin state of FeO
    index_x = np.argmin(np.abs(self.x_feo_fp_vec - x_feo_fp))
    index_T = np.argmin(np.abs(self.T_vec - (self.T_am + dT)))
    index_P = np.argmin(np.abs(P_table[index_T, :, index_x] - self.P_am))
    eta_ls = spin_config[index_T, index_P, index_x]

    # Volume of Fp
    m_fp = self.m_mgo * (1 - x_feo_fp) + self.m_feo * x_feo_fp
    v_fp = 1000 * m_fp / (5500 * rho_fp)

    # Mineral composition of Bm
    x_feo_bm, x_alo2_bm = _oxides_content_in_bm(
        self, p_capv, p_bm, p_fp, feo, al, ratio_fe, x_feo_fp, rho_capv, rho_bm, rho_fp)

    # Checking that the oxides content makes sense
    if ((x_alo2_bm < 0.0) or (x_alo2_bm > 1.0) or (x_feo_bm < 0.0) or (x_feo_bm > 1.0)):
        return None

    if (al_excess):
        ### Al is asssumed to be in excess ###
        # Calculating molar proportion of the different components of Bm
        x_mgsio3_bm = 1 - x_alo2_bm - (2 - ratio_fe) * x_feo_bm
        x_fesio3_bm = 2 * (1 - ratio_fe) * x_feo_bm
        x_fealo3_bm = 2 * ratio_fe * x_feo_bm
        x_fe2o3_bm = 0.0
        x_al2o3_bm = x_alo2_bm - ratio_fe * x_feo_bm
    else:
        ### Fe is assumed to be in excess ###
        # Calculating molar proportion of the different components of Bm
        x_mgsio3_bm = 1 - x_alo2_bm - (2 - ratio_fe) * x_feo_bm
        x_fesio3_bm = 2 * (1 - ratio_fe) * x_feo_bm
        x_fealo3_bm = 2 * x_alo2_bm
        x_fe2o3_bm = ratio_fe * x_feo_bm - x_alo2_bm
        x_al2o3_bm = 0.0

    # Mass proportion of Bm
    x_m_bm = 1 / (
        1 + p_capv * rho_capv / (p_bm * rho_bm) + p_fp * rho_fp / (p_bm * rho_bm))
    m_bm = (
        self.m_mgsio3 * x_mgsio3_bm + self.m_fealo3 * x_fealo3_bm +
        self.m_al2o3 * x_al2o3_bm + self.m_fe2o3 * x_fe2o3_bm +
        self.m_fesio3 * x_fesio3_bm)
    v_bm = 1000 * m_bm / (5500 * rho_bm)

    # Equation from the EOS for Fp
    eq_MGD_fp = fp_eos._MGD(self, self.P_am, self.T_am + dT, v_fp, eta_ls, x_feo_fp)

    # Equation from the EOS for Bm
    eq_MGD_bm = bm_eos._MGD(
        self, self.P_am, self.T_am + dT, v_bm, x_mgsio3_bm, x_fesio3_bm, x_fealo3_bm,
        x_fe2o3_bm, x_al2o3_bm)

    # Equation from the alumina content
    eq_alo2 = -x_alo2_bm * self.m_al2o3 * x_m_bm + m_bm * al

    if (testing):
        # Verifying that error on the spin configuration is reasonable
        delta_x_feo_fp = self.x_feo_fp_vec[1] - self.x_feo_fp_vec[0]
        if (abs(P_table[index_T, index_P, index_x] - self.P_am) > self.delta_P):
            print("Error on P for the spin transition is large")
            print(
                "Pressure for the spin configuration: `", P_table[index_T, index_P,
                                                                  index_x],
                "` actual pressure: `", self.P_am, "`")
        elif ((self.x_feo_fp_vec[index_x] - x_feo_fp) > delta_x_feo_fp):
            print("Error on x for the spin transition is large")
            print(
                "FeO content for the spin configuration: `", self.x_feo_fp_vec[index_x],
                "` calculated FeO content: `", x_feo_fp, "`")

        # Verifying that the sum of all chemical elements is equal to 1
        x_m_capv = (p_capv * rho_capv) / (
            (p_capv * rho_capv) + p_fp * rho_fp + p_bm * rho_bm)
        x_m_fp = 1 / (
            1 + p_capv * rho_capv / (p_fp * rho_fp) + p_bm * rho_bm / (p_fp * rho_fp))
        m_ca, m_al, m_mg, m_fe, m_o = (40.078, 26.982, 24.305, 55.845, 15.999)
        m_si = 28.086
        sum_elements = (
            m_ca / self.m_capv * x_m_capv + m_al / m_bm * 2 * x_m_bm * x_alo2_bm +
            m_mg / m_bm * x_m_bm * x_mgsio3_bm + m_mg / m_fp * x_m_fp * (1 - x_feo_fp) +
            m_fe / m_bm * 2 * x_m_bm * x_feo_bm + m_fe / m_fp * x_m_fp * x_feo_fp +
            m_si / m_bm * x_m_bm * (x_mgsio3_bm + x_fesio3_bm) +
            m_si / self.m_capv * x_m_capv + m_o / m_bm * 3 * x_m_bm +
            m_o / m_fp * x_m_fp + m_o / self.m_capv * 3 * x_m_capv)
        if (not isclose(sum_elements, 1.0, abs_tol=1e-3)):
            print("Problem with the sum of elements")
            print("Sum of elements is equal to `", sum_elements, "`")

        # Verifying that the FeO content is consistent
        sum_feo = self.m_feo * (2 * x_m_bm * x_feo_bm / m_bm + x_m_fp * x_feo_fp / m_fp)
        if (not isclose(sum_feo, feo, abs_tol=1e-3)):
            print("Problem with the FeO content")
            print(
                "Calculated FeO content is `", sum_feo,
                "` , while the actual value is `", feo, "`")

    return [eq_MGD_fp, eq_MGD_bm, eq_alo2]


def _oxides_content_in_bm(
        self, p_capv: float, p_bm: float, p_fp: float, feo: float, al: float,
        ratio_fe: float, x_feo_fp: float, rho_capv: float, rho_bm: float,
        rho_fp: float) -> list:
    """Calculates the molar concentration of FeO and AlO2 in Bridgmanite.

    This function calculates the molar concentration of FeO and AlO2 in Bridgmanite.
    The calculation is a combination of eqs. (1), (2), (4) and (13) in the
    supplementary material of Vilella et al. (2021).

    Args:
        p_capv: Proportion of CaPv. [vol%]
        p_bm: Proportion of Bm. [vol%]
        p_fp: Proportion of Fp. [vol%]
        feo: FeO content in the rock assemblage. [wt%]
        al: Al2O3 content in the rock assemblage. [wt%]
        ratio_fe: Oxidation state in Bm.
        x_feo_fp: The molar concentration of FeO in Fp.
        rho_capv: Density of CaPv for the considered conditions. Its value has been
                  rescaled by 5500. [kg/m^3]
        rho_bm: Density of Bm for the considered conditions. Its value has been
                rescaled by 5500. [kg/m^3]
        rho_fp: Density of Fp for the considered conditions. Its value has been
                rescaled by 5500. [kg/m^3]

    Returns:
        A list containing the molar concentration of AlO2 in Bm and FeO in Bm.
    """
    # Molar mass of Fp
    m_fp = self.m_mgo * (1 - x_feo_fp) + self.m_feo * x_feo_fp

    # Mass proportion of Fp
    x_m_fp = 1 / (
        1 + p_capv * rho_capv / (p_fp * rho_fp) + p_bm * rho_bm / (p_fp * rho_fp))

    # Equations giving the FeO and AlO2 content in Bm
    c_1 = 0.5 * self.m_al2o3 / al * (feo / self.m_feo - x_feo_fp * x_m_fp / m_fp)
    x_alo2_bm = self.kd_ref_am * x_feo_fp / (
        2 * c_1 *
        (1 - x_feo_fp) + self.kd_ref_am * x_feo_fp + self.kd_ref_am * x_feo_fp *
        (2 - ratio_fe) * c_1)
    x_feo_bm = c_1 * x_alo2_bm

    return [x_feo_bm, x_alo2_bm]


def _solve_without_fp(
        self, x_init: list, dT: float, p_capv: float, p_bm: float, feo: float,
        al: float, ratio_fe: float, rho_capv: float, al_excess: bool) -> list:
    """Solves the physics problem without Ferropericlase.

    This function implements a solver for the system of equations governing the physics
    problem when no Ferropericlase (Fp) is present. Solving this highly non-linear
    problem is challenging, especially when a good initial guess is not available,
    such as at the beginning of the simulation loop.

    To address this challenge, the function first attempts to find a solution using the
    provided starting conditions. If a solution cannot be found, the starting conditions
    are randomly sampled within predetermined ranges until a solution is obtained or the
    number of attempts exceeds 10. If a solution cannot be found under these
    conditions, a value of 0.0 is returned for each property.

    If finding a solution remains difficult, the user may manually increase the maximum
    number of attempts or the maximum number of iterations (maxiter) in the solver call.

    It is important to note that various solvers from the scipy.optimize library have
    been tested (e.g., minimize, fsolve, least_squares), but they did not yield
    satisfactory results.

    Args:
        x_init: Starting conditions for the physics problem, that is, the molar
                concentration of FeO in Bm, the molar concentration of AlO2 in Bm, and
                the rescaled density of Bm.
        dT: Temperature contrast against the ambient mantle. [K]
        p_capv: Proportion of CaPv. [vol%]
        p_bm: Proportion of Bm. [vol%]
        feo: FeO content in the rock assemblage. [wt%]
        al: Al2O3 content in the rock assemblage. [wt%]
        ratio_fe: Oxidation state in Bm.
        rho_capv: Density of CaPv for the considered conditions. Its value has been
                  rescaled by 5500. [kg/m^3]
        al_excess: Flag indicating whether alumina is assumed to be in excess in Bm.

    Returns:
        A list containing the molar concentration of FeO in Bm, AlO2 in Bm, and the
        rescaled density of Bm.
    """
    # Initialization
    n_iter = 0

    while (n_iter < 10):
        ### Solution has not yet been found ###
        try:
            ### Solution has been found ###
            # Solving the system of equation
            solution = scipy.optimize.newton_krylov(
                lambda x: _set_eqs_without_fp(
                    self, x, dT, p_capv, p_bm, feo, al, ratio_fe, rho_capv, al_excess),
                x_init,
                maxiter=1000)

            return solution
        except Exception:
            ### Solution has not been found ###
            # Setting random starting conditions
            x_init[0] = random.uniform(0.0, 1.0)
            x_init[1] = random.uniform(0.0, 1.0)
            x_init[2] = random.uniform(0.8, 1.8)
            n_iter += 1

    return [0.0, 0.0, 0.0]


def _set_eqs_without_fp(
        self, var_in: list, dT: float, p_capv: float, p_bm: float, feo: float,
        al: float, ratio_fe: float, rho_capv: float, al_excess: bool) -> list:
    """Implements the equations for the physics problem without Ferropericlase.

    This function calculates the residue for the system of equations governing the
    physics problem when no Ferropericlase (Fp) is present.

    The first equation comes from the equation of state for Bridgmanite (Bm). More
    specifically, the Mie-Gruneisen-Debye equation of state, which corresponds to the
    eq. (33) of Jackson and Rigden (1996).
    The second equation comes from the ratio of FeO and alumina content in Bm. It
    corresponds to the eq. (16) in the supplementary material of Vilella et al. (2021).
    The third equation comes from the alumina content in Bm. It corresponds to the
    eq. (13) in the supplementary material of Vilella et al. (2021).

    Args:
        var_in: Vector composed of the equation unknowns, that is, the molar
                concentration of FeO in Bm, the molar concentration of AlO2 in Bm, and
                the rescaled density of Bm.
        dT: Temperature contrast against the ambient mantle. [K]
        p_capv: Proportion of CaPv. [vol%]
        p_bm: Proportion of Bm. [vol%]
        feo: FeO content in the rock assemblage. [wt%]
        al: Al2O3 content in the rock assemblage. [wt%]
        ratio_fe: Oxidation state in Bm.
        rho_capv: Density of CaPv for the considered conditions. Its value has been
                  rescaled by 5500. [kg/m^3]
        al_excess: Flag indicating whether alumina is assumed to be in excess in Bm.

    Returns:
        A list containing the residue for the equation of state for Bridgmanite, the
        equation for the ratio of FeO and alumina content in Bridgmanite, and the
        equation for the alumina content in Bridgmanite.
    """
    # Loading utility class for mineral properties calculation
    bm_eos = _EOS_bm()

    # Unpacking input variables
    x_feo_bm, x_alo2_bm, rho_bm = var_in

    # Checking that input conditions makes sense
    if ((rho_bm < 0.1) or (x_feo_bm < 0.0) or (x_feo_bm > 1.0) or (x_alo2_bm < 0.0) or
            (x_alo2_bm > 1.0)):
        return None

    if (al_excess):
        ### Al is asssumed to be in excess ###
        # Calculating molar proportion of the different components of Bm
        x_mgsio3_bm = 1 - x_alo2_bm - (2 - ratio_fe) * x_feo_bm
        x_fesio3_bm = 2 * (1 - ratio_fe) * x_feo_bm
        x_fealo3_bm = 2 * ratio_fe * x_feo_bm
        x_fe2o3_bm = 0.0
        x_al2o3_bm = x_alo2_bm - ratio_fe * x_feo_bm
    else:
        ### Fe is assumed to be in excess ###
        # Calculating molar proportion of the different components of Bm
        x_mgsio3_bm = 1 - x_alo2_bm - (2 - ratio_fe) * x_feo_bm
        x_fesio3_bm = 2 * (1 - ratio_fe) * x_feo_bm
        x_fealo3_bm = 2 * x_alo2_bm
        x_fe2o3_bm = ratio_fe * x_feo_bm - x_alo2_bm
        x_al2o3_bm = 0.0

    # Mass proportion of Bm
    x_m_bm = 1 / (1 + p_capv * rho_capv / (p_bm * rho_bm))
    m_bm = (
        self.m_mgsio3 * x_mgsio3_bm + self.m_fealo3 * x_fealo3_bm +
        self.m_al2o3 * x_al2o3_bm + self.m_fe2o3 * x_fe2o3_bm +
        self.m_fesio3 * x_fesio3_bm)
    v_bm = 1000 * m_bm / (5500 * rho_bm)

    # Equation from the EOS for Bm
    eq_MGD_bm = bm_eos._MGD(
        self, self.P_am, self.T_am + dT, v_bm, x_mgsio3_bm, x_fesio3_bm, x_fealo3_bm,
        x_fe2o3_bm, x_al2o3_bm)

    # Equation from alumina and FeO content
    eq_feo_al = -feo / al + (2 * self.m_feo * x_feo_bm) / (self.m_al2o3 * x_alo2_bm)

    # Equation from the alumina content
    eq_alo2 = -x_alo2_bm + m_bm * al / (self.m_al2o3 * x_m_bm)

    return [eq_MGD_bm, eq_feo_al, eq_alo2]
