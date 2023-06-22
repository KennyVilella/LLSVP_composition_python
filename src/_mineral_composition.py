"""Provides functions used to calculate the mineral composition of the assemblage.

This file is associated with the article:
"Constraints on the composition and temperature of LLSVPs from seismic properties of
lower mantle minerals" by K. Vilella, T. Bodin, C.-E. Boukare, F. Deschamps, J. Badro,
M. D. Ballmer, and Y. Li

This file provides all the functions used to calculate for a large range of input
parameters the composition of the corresponding rock assemblage.
The six input parameters are the temperature contrast against the ambient mantle, the 
proportion of Calcio Perovskite, the proportion of Bridgmanite, the FeO content, the
alumina content, and the oxidation state of iron in Bridgmanite.
Given these input parameters, the simulator returns the molar concentration of FeO in
Bridgmanite, the molar concentration of FeO in Ferropericlase, the molar
concentration of AlO2 in Bridgmanite, the density of Bridgmanite, and the density of
Ferropericlase.

The results are written into files separated by the considered temperature contrast,
which also gives the name of the file. The simulation can be resumed if stopped.
The results can then be used to calculate the seismic properties of all the rock
assemblages.

Solving the equations governing this problem is quite difficult as they are highly
non-linear. If a solution is not found, a value of 0.0 is returned for each property.
To improve convergence, one has to either tweak the starting conditions or the
non-linear solver, both being somewhat challenging.

These functions should not be used outside the class MineralProperties.

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
import csv
import random
import scipy.optimize
from ._eos_implementation import _EOS_fp, _EOS_bm, _EOS_capv
#======================================================================================#
#                                                                                      #
#    Starting implementation of functions used to calculate the mineral composition    #
#                                                                                      #
#======================================================================================#
def _calc_mineral_composition(self, spin_config, P_table):
    """Calculates the mineral composition of a large range of rock assemblages.

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
    The simulation can be resumed if exsiting files are already present. As a result, it
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
        (self.iron_content_max - self.iron_content_min) / self.delta_iron_content
    ) + 1
    n_al = round((self.al_content_max - self.al_content_min) / self.delta_al) + 1
    n_ratio_fe_bm = round(
        (self.ratio_fe_bm_max - self.ratio_fe_bm_min) / self.delta_ratio_fe
    ) + 1
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
            lambda x: capv_eos._MGD(self, self.T_am + dT, self.P_am, x), 30.
        )
        rho_capv = self.rho_capv_0 * self.v_casio3_0 / solution[0]

        # Initializing starting condition
        x_init = [0.3, 0.1, 0.1, 5000.0, 5000.0]

        filename = os.path.join(self.path, str(int(dT)) + ".csv")
        if os.path.isfile(filename):
            ### A file already exists ###
            print("File detected")
            print("Resuming the simulation")
            with open(filename, newline='') as csvfile:
                data = csv.reader(csvfile, delimiter=' ')
                for row in data:
                    # Calculating the corresponding indices
                    jj = round((float(row[1]) - self.p_capv_min) / self.delta_capv)
                    kk = round((float(row[2]) - self.p_bm_min) / self.delta_bm)
                    ll = round(
                        (float(row[3]) - self.iron_content_min) /
                        self.delta_iron_content
                    )
                    mm = round(
                        (float(row[4]) - self.al_content_min) / self.delta_al
                    )
                    nn = round(
                        (float(row[5]) - self.ratio_fe_bm_min) / self.delta_ratio_fe
                    )

                    # Loading the data
                    x_feo_bm_ii[jj, kk, ll, mm, nn] = row[6]
                    x_feo_fp_ii[jj, kk, ll, mm, nn] = row[7]
                    x_alo2_bm_ii[jj, kk, ll, mm, nn] = row[8]
                    rho_bm_ii[jj, kk, ll, mm, nn] = row[9]
                    rho_fp_ii[jj, kk, ll, mm, nn] = row[10]

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
                            ratio_fe = (
                                self.ratio_fe_bm_min + nn * self.delta_ratio_fe
                            )

                            if (mm != 0):
                                ### Using previous starting conditions ###
                                x_init = [
                                    x_feo_bm_ii[jj, kk, ll, mm-1, nn],
                                    x_feo_fp_ii[jj, kk, ll, mm-1, nn],
                                    x_alo2_bm_ii[jj, kk, ll, mm-1, nn],
                                    rho_bm_ii[jj, kk, ll, mm-1, nn],
                                    rho_fp_ii[jj, kk, ll, mm-1, nn]
                                ]
                            if (x_init[0] < 0.0):
                                ### Starting condition is incorrect ###
                                ll_iter = ll
                                while (x_init[0] < 0.0):
                                    ll_iter -= 1
                                x_init = [
                                    x_feo_bm_ii[jj, kk, ll_iter, mm, nn],
                                    x_feo_fp_ii[jj, kk, ll_iter, mm, nn],
                                    x_alo2_bm_ii[jj, kk, ll_iter, mm, nn],
                                    rho_bm_ii[jj, kk, ll_iter, mm, nn],
                                    rho_fp_ii[jj, kk, ll_iter, mm, nn]
                                ]

                            # Calculating the composition of the rock assemblage
                            x_result = _solve_mineral_composition(
                                self, dT, p_capv, p_bm, feo, al, ratio_fe,
                                spin_config, P_table, rho_capv, p_fp, x_init
                            )

                            # Loading results
                            x_feo_bm_ii[jj, kk, ll, mm, nn] = x_result[0]
                            x_feo_fp_ii[jj, kk, ll, mm, nn] = x_result[1]
                            x_alo2_bm_ii[jj, kk, ll, mm, nn] = x_result[2]
                            rho_bm_ii[jj, kk, ll, mm, nn] = x_result[3]
                            rho_fp_ii[jj, kk, ll, mm, nn] = x_result[4]

                            # Writting the data
                            with open(filename, 'a', newline='') as csvfile:
                                data_writer = csv.writer(csvfile, delimiter=' ')
                                data_writer.writerow([
                                    dT, p_capv, p_bm, feo, al, ratio_fe, x_result[0],
                                    x_result[1], x_result[2], x_result[3], x_result[4]
                                ])
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
                            rho_fp_ii[jj, kk, ll, mm, 0]
                        ]
                        nn_start = 0 # In case the calculation has been resumed
                    # End of Al content loop

                    # Updating starting conditions
                    x_init = [
                        x_feo_bm_ii[jj, kk, ll, 0, 0],
                        x_feo_fp_ii[jj, kk, ll, 0, 0],
                        x_alo2_bm_ii[jj, kk, ll, 0, 0],
                        rho_bm_ii[jj, kk, ll, 0, 0],
                        rho_fp_ii[jj, kk, ll, 0, 0]
                    ]
                    mm_start = 0 # In case the calculation has been resumed
                # End of FeO content loop

                # Updating starting conditions
                x_init = [
                    x_feo_bm_ii[jj, kk, 0, 0, 0], x_feo_fp_ii[jj, kk, 0, 0, 0],
                    x_alo2_bm_ii[jj, kk, 0, 0, 0], rho_bm_ii[jj, kk, 0, 0, 0],
                    rho_fp_ii[jj, kk, 0, 0, 0]
                ]
                ll_start = 0 # In case the calculation has been resumed
            # End of Bm proportion loop

            # Updating starting conditions
            x_init = [
                x_feo_bm_ii[jj, 0, 0, 0, 0], x_feo_fp_ii[jj, 0, 0, 0, 0],
                x_alo2_bm_ii[jj, 0, 0, 0, 0], rho_bm_ii[jj, 0, 0, 0, 0],
                rho_fp_ii[jj, 0, 0, 0, 0]
            ]
            kk_start = 0 # In case the calculation has been resumed
        # End of CaPv proportion loop

        # Updating starting conditions
        x_init = [
            x_feo_bm_ii[0, 0, 0, 0, 0], x_feo_fp_ii[0, 0, 0, 0, 0],
            x_alo2_bm_ii[0, 0, 0, 0, 0], rho_bm_ii[0, 0, 0, 0, 0],
            rho_fp_ii[0, 0, 0, 0, 0]
        ]
        jj_start = 0 # In case the calculation has been resumed
    # End of temperature loop


def _solve_mineral_composition(
    self, dT, p_capv, p_bm, feo, al, ratio_fe,
    spin_config, P_table, rho_capv, p_fp, x_init
):
    """Calculates the mineral composition of the provided rock assemblage.

    This function calculates five properties fully characterizing the composition of
    the rock assemblage (molar concentration of FeO in Bm, molar concentration of FeO
    in Fp, molar concentration of AlO2 in Bm, density of Bm, and density of Fp) from the
    value of six input parameters (temperature contrast against the ambient mantle,
    proportion of CaPv, proportion of Bm, FeO content, alumina content, and oxidation
    state in Bm).

    The calculation is quite complex so that several tricks are used to make the physics
    problem easier to solve.
    First, all densities are divided by 5500 to avoid the generation of large numbers
    during the solving step.
    Second, a first-order empirical guess is made concerning the composition of Bm, that
    is, whether Fe2O3 or Al2O3 is formed. The consistency of the first obtained solution
    is then checked, and the inital guess is reversed in case of inconsistency.
    Third, the physics problem is separated in two cases, with and without the presence
    of Fp. This is because the system of equations to solve is different in these two
    cases.

    Note that a value of 0.0 is returned for each property, if no solution can be found.

    Args:
        dT: Temperature contrast against the ambient mantle. [K]
        p_capv: Proportion of CaPv. [vol%]
        p_bm: Proportion of Bm. [vol%]
        feo: FeO content in the rock assemblage. [wt%]
        al: Al2O3 content in the rock assemblage. [wt%]
        ratio_fe: Oxidation state in Bm.
        spin_config: Average spin state of FeO in Fp for a given value for the
                     temperature, volume of Fp, and FeO content in Fp.
        P_table: Pressure for a given value for the temperature, volume of Fp, and
                 FeO content in Fp.
        rho_capv: Density of CaPv for the considered conditions. Its value has been
                  rescaled by 5500. [kg/m^3]
        p_fp: Proportion of Fp. [vol%]
        x_init: Starting conditions for the physics problem, that is, the molar
                concentration of FeO in Bm, the molar concentration of FeO in Fp, the
                molar concentration of AlO2 in Bm, the rescaled density of Bm, and the
                rescaled density of Fp.

    Returns:
        Float64: The molar concentration of FeO in Bm.
        Float64: The molar concentration of FeO in Fp.
        Float64: The molar concentration of AlO2 in Bm.
        Float64: The rescaled density of Bm.
        Float64: The rescaled density of Fp.
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
            self, x_init, dT, p_capv, p_bm, feo, al, ratio_fe, spin_config, P_table,
            rho_capv, p_fp, al_excess
        )

        # Getting extra outputs
        x_alo2_bm, x_feo_bm = _oxides_content_in_bm(
            self, x_feo_fp, rho_bm, rho_fp, p_capv, p_bm, feo, al, ratio_fe, rho_capv,
            p_fp
        )

        if ((not al_excess) and (ratio_fe * x_feo_bm < x_alo2_bm)):
            ### First guess for al_excess is incorrect ###
            # Trying to solve the system of equation with al_excess
            al_excess = True
            x_init = [x_feo_fp, rho_bm, rho_fp]
            x_feo_fp, rho_bm, rho_fp = _solve_with_fp(
                self, x_init, dT, p_capv, p_bm, feo, al, ratio_fe, spin_config,
                P_table, rho_capv, p_fp, al_excess
            )

            # Getting extra outputs
            x_alo2_bm, x_feo_bm = _oxides_content_in_bm(
                self, x_feo_fp, rho_bm, rho_fp, p_capv, p_bm, feo, al, ratio_fe,
                rho_capv, p_fp
            )

            # Checking that solution is indeed correct
            if (ratio_fe * x_feo_bm > x_alo2_bm):
                ### Guess for al_excess is again incorrect ###
                # Skipping this calculation
                x_feo_bm, x_feo_fp, x_alo2_bm, rho_bm, rho_fp = (0., 0., 0., 0., 0.)
                print("Problem with Al_excess")
                print("p_bm = ", p_bm, " p_capv = ", p_capv," feo = ", feo,
                    " al = ", al, " ratio_fe = ", ratio_fe, " dT = ", dT)
                print("Skip this condition")
        elif ((al_excess) and (ratio_fe * x_feo_bm > x_alo2_bm)):
            ### First guess for al_excess is incorrect ###
            # Trying to solve the system of equation without al_excess
            al_excess = False
            x_init = [x_feo_fp, rho_bm, rho_fp]
            x_feo_fp, rho_bm, rho_fp = _solve_with_fp(
                self, x_init, dT, p_capv, p_bm, feo, al, ratio_fe, spin_config,
                P_table, rho_capv, p_fp, al_excess
            )

            # Getting extra outputs
            x_alo2_bm, x_feo_bm = _oxides_content_in_bm(
                self, x_feo_fp, rho_bm, rho_fp, p_capv, p_bm, feo, al, ratio_fe,
                rho_capv, p_fp
            )

            # Checking that solution is indeed correct
            if (ratio_fe * x_feo_bm < x_alo2_bm):
                ### Guess for al_excess is again incorrect ###
                # Skipping this calculation
                x_feo_bm, x_feo_fp, x_alo2_bm, rho_bm, rho_fp = (0., 0., 0., 0., 0.)
                print("Problem with Al_excess")
                print("p_bm = ", p_bm, " p_capv = ", p_capv," feo = ", feo,
                    " al = ", al, " ratio_fe = ", ratio_fe, " dT = ", dT)
                print("Skip this condition")

        # Verifying that the solution is indeed consistent
        _set_eqs_with_fp(
            self, [x_feo_fp, rho_bm, rho_fp], dT, p_capv, p_bm, feo, al, ratio_fe,
            spin_config, P_table, rho_capv, p_fp, al_excess, True
        )
    else:
        ### Ferropericlase is absent from the rock assemblage ###
        # Solving the system of equation
        x_init = [x_init[0], x_init[2], x_init[3] / 5500]
        x_feo_bm, x_alo2_bm, rho_bm = _solve_without_fp(
            self, x_init, dT, p_capv, p_bm, feo, al, ratio_fe, rho_capv, al_excess
        )
        x_feo_fp = 0.0
        rho_fp = 0.0

        if ((not al_excess) and (ratio_fe * x_feo_bm < x_alo2_bm)):
            ### First guess for al_excess is incorrect ###
            # Trying to solve the system of equation with al_excess
            al_excess = True
            x_init = [x_feo_bm, x_alo2_bm, rho_bm]
            x_feo_bm, x_alo2_bm, rho_bm = _solve_without_fp(
                self, x_init, dT, p_capv, p_bm, feo, al, ratio_fe, rho_capv, al_excess
            )

            # Checking that solution is indeed correct
            if (ratio_fe * x_feo_bm > x_alo2_bm):
                ### Guess for al_excess is again incorrect ###
                # Skipping this calculation
                x_feo_bm, x_alo2_bm, rho_bm = (0., 0., 0.)
                print("Problem with Al_excess")
                print("p_bm = ", p_bm, " p_capv = ", p_capv," feo = ", feo,
                    " al = ", al, " ratio_fe = ", ratio_fe, " dT = ", dT)
                print("Skip this condition")
        elif ((al_excess) and (ratio_fe * x_feo_bm > x_alo2_bm)):
            ### First guess for al_excess is incorrect ###
            # Trying to solve the system of equation without al_excess
            al_excess = False
            x_init = [x_feo_bm, x_alo2_bm, rho_bm]
            x_feo_bm, x_alo2_bm, rho_bm = _solve_without_fp(
                self, x_init, dT, p_capv, p_bm, feo, al, ratio_fe, rho_capv, al_excess
            )

            # Checking that solution is indeed correct
            if (ratio_fe * x_feo_bm < x_alo2_bm):
                ### Guess for al_excess is again incorrect ###
                # Skipping this calculation
                x_feo_bm, x_alo2_bm, rho_bm = (0., 0., 0.)
                print("Problem with Al_excess")
                print("p_bm = ", p_bm, " p_capv = ", p_capv," feo = ", feo,
                    " al = ", al, " ratio_fe = ", ratio_fe, " dT = ", dT)
                print("Skip this condition")

    return [x_feo_bm, x_feo_fp, x_alo2_bm, 5500 * rho_bm, 5500 * rho_fp]


def _solve_with_fp(
    self, x_init, dT, p_capv, p_bm, feo, al, ratio_fe, spin_config, P_table,
    rho_capv, p_fp, al_excess
):
    """Implements the solver for the physics problem with Ferropericlase.

    This function implements a solver for the system of equations governing the physics
    problem when Ferropericlase is present.
    This physics problem is highly non-linear such that finding a proper solution is
    challenging. It is particularly challenging when the starting conditions are not
    well known, for instance at the start of the simulation loop.

    To circumvent this issue, this function first try to find a solution using the
    suggested starting conditions. If a solution cannot be found, the starting
    conditions are randomly sampled within predetermined ranges until a solution is
    found or that the number of attempts exceeds 1000.
    If a solution cannot be found within these conditions, a value of 0.0 is returned
    for each property.

    If it remains too difficult to find a solution, the user can manually increase the
    maximum number of attempts or the maximum number of iterations (maxiter) in the
    solver call.

    Note that several solvers from scipy.optimize have been tried (minimize, fsolve,
    least_squares) and they have produced unsatisfactory results.

    Args:
        x_init: Starting conditions for the physics problem, that is, the molar
                concentration of FeO in Fp, the rescaled density of Bm, and the
                rescaled density of Fp.
        dT: Temperature contrast against the ambient mantle. [K]
        p_capv: Proportion of CaPv. [vol%]
        p_bm: Proportion of Bm. [vol%]
        feo: FeO content in the rock assemblage. [wt%]
        al: Al2O3 content in the rock assemblage. [wt%]
        ratio_fe: Oxidation state in Bm.
        spin_config: Average spin state of FeO in Fp for a given value for the
                     temperature, volume of Fp, and FeO content in Fp.
        P_table: Pressure for a given value for the temperature, volume of Fp, and
                 FeO content in Fp.
        rho_capv: Density of CaPv for the considered conditions. Its value has been
                  rescaled by 5500. [kg/m^3]
        p_fp: Proportion of Fp. [vol%]
        al_excess: Flag indicating whether alumina is assumed to be in excess in Bm.

    Returns:
        Float64: The molar concentration of FeO in Fp.
        Float64: The rescaled density of Bm.
        Float64: The rescaled density of Fp.
    """
    # Initialization
    n_iter = 0

    while (n_iter < 1000):
        ### Solution has not yet been found ###
        try:
            ### Solution has been found ###
            # Solving the system of equation
            solution = scipy.optimize.newton_krylov(
                lambda x: _set_eqs_with_fp(
                    self, x, dT, p_capv, p_bm, feo, al, ratio_fe, spin_config,
                    P_table, rho_capv, p_fp, al_excess
                ),
                x_init,
                maxiter=1000
            )

            return solution
        except:
            ### Solution has not been found ###
            # Setting random starting conditions
            x_init[0] = random.uniform(0.0, 1.0)
            x_init[1] = random.uniform(0.8, 1.8)
            x_init[2] = random.uniform(0.8, 1.8)
            n_iter += 1

    return 0.0, 0.0, 0.0


def _set_eqs_with_fp(
    self, var_in, dT, p_capv, p_bm, feo, al, ratio_fe, spin_config, P_table,
    rho_capv, p_fp, al_excess, testing=False
):
    """Calculates the equations for the physics problem with Ferropericlase.

    This function calculates the residue for the system of equations governing the
    physics problem when Ferropericlase is present.

    The first and second equation come from the equation of state for Ferropericlase
    and Bridgmanite, respectively. More specifically, the Mie-Gruneisen-Debye equation
    of state, which corresponds to the eq. (33) of Jackson and Rigden (1996).
    The third equation comes from the alumina content in Bridgmanite. It corresponds
    to the eq. (13) in the supplementary material of Vilella et al. (2021).

    Args:
        var_in: Vector composed of the equation unknowns, that is, the molar
                concentration of FeO in Fp, the rescaled density of Bm, and the
                rescaled density of Fp.
        dT: Temperature contrast against the ambient mantle. [K]
        p_capv: Proportion of CaPv. [vol%]
        p_bm: Proportion of Bm. [vol%]
        feo: FeO content in the rock assemblage. [wt%]
        al: Al2O3 content in the rock assemblage. [wt%]
        ratio_fe: Oxidation state in Bm.
        spin_config: Average spin state of FeO in Fp for a given value for the
                     temperature, volume of Fp, and FeO content in Fp.
        P_table: Pressure for a given value for the temperature, volume of Fp, and
                 FeO content in Fp.
        rho_capv: Density of CaPv for the considered conditions. Its value has been
                  rescaled by 5500. [kg/m^3]
        p_fp: Proportion of Fp. [vol%]
        al_excess: Flag indicating whether alumina is assumed to be in excess in Bm.
        testing: Whether tests are conducted to make sure that the provided values of
                 var_in lead to a consistent composition.

    Returns:
        Float64: Residue of the equation of state for Bridgmanite.
        Float64: Residue of the equation for the ratio of FeO and alumina content in
                 Bridgmanite.
        Float64: Residue of the equation for the alunina content in Bridgmanite.
    """
    # Loading utility class for mineral properties calculation
    fp_eos = _EOS_fp()
    bm_eos = _EOS_bm()

    # Unpacking input variables
    x_feo_fp, rho_bm, rho_fp = var_in

    # Average spin state of FeO
    index_x = np.argmin(np.abs(self.x_vec - x_feo_fp))
    index_T = np.argmin(np.abs(self.T_vec - (self.T_am + dT)))
    index_P = np.argmin(np.abs(P_table[index_T, :, index_x] - self.P_am))
    eta_ls = spin_config[index_T, index_P, index_x]

    # Volume of Fp
    m_fp = self.m_mgo * (1 - x_feo_fp) + self.m_feo * x_feo_fp
    v_fp_0 = fp_eos._v_fp_0(self, eta_ls, x_feo_fp)
    v_fp = 1000 * m_fp / (5500 * rho_fp)

    # Bulk modulus of Fp
    k_fp_0 = fp_eos._k_fp_0_VRH_average(self, eta_ls, x_feo_fp)

    # Mineral composition of Bm
    x_alo2_bm, x_feo_bm = _oxides_content_in_bm(
        self, x_feo_fp, rho_bm, rho_fp, p_capv, p_bm, feo, al, ratio_fe, rho_capv, p_fp
    )

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
        1 + p_capv * rho_capv / (p_bm * rho_bm) + p_fp * rho_fp / (p_bm * rho_bm)
    )
    m_bm = (
        self.m_mgsio3 * x_mgsio3_bm + self.m_fealo3 * x_fealo3_bm +
        self.m_al2o3 * x_al2o3_bm + self.m_fe2o3 * x_fe2o3_bm +
        self.m_fesio3 * x_fesio3_bm
    )
    v_bm = 1000 * m_bm / (5500 * rho_bm)

    # Equation from the EOS for Fp
    eq_MGD_fp = fp_eos._MGD(
        self, self.T_am + dT, self.P_am, v_fp, eta_ls, x_feo_fp
    ) 

    # Equation from the EOS for Bm
    eq_MGD_bm = bm_eos._MGD(
        self, self.T_am + dT, self.P_am, v_bm, x_mgsio3_bm, x_fesio3_bm, x_fealo3_bm,
        x_fe2o3_bm, x_al2o3_bm
    )

    # Equation from the alumina content
    eq_alo2 = -x_alo2_bm * self.m_al2o3 * x_m_bm + m_bm * al

    if (testing):
        # Verifying that error on the spin configuration is reasonable
        if (abs(P_table[index_T, index_P, index_x] - self.P_am) > self.delta_P):
            print("Error on P for the spin transition is large")
            print("Pressure for the spin configuration: `",
                P_table[index_T, index_P, index_x], "` actual pressure: `", self.P_am, "`")
        elif ((self.x_vec[index_x] - x_feo_fp) > self.x_vec[1] - self.x_vec[0]):
            print("Error on x for the spin transition is large")
            print("FeO content for the spin configuration: `",
                self.x_vec[index_x], "` calculated FeO content: `", x_feo_fp, "`")


        # Verifying that the sum of all chemical elements is equal to 1
        x_m_capv = (p_capv * rho_capv) / (
            (p_capv * rho_capv) + p_fp * rho_fp + p_bm * rho_bm
        )
        x_m_fp = 1 / (
            1 + p_capv * rho_capv / (p_fp * rho_fp) + p_bm * rho_bm / (p_fp * rho_fp)
        )
        m_ca, m_al, m_mg, m_fe, m_o = (40.078, 26.982, 24.305, 55.845, 15.999)
        m_si = 28.086
        sum_elements = (
            m_ca / self.m_capv * x_m_capv + m_al / m_bm * 2 * x_m_bm * x_alo2_bm +
            m_mg / m_bm * x_m_bm * x_mgsio3_bm + m_mg / m_fp * x_m_fp * (1 - x_feo_fp) +
            m_fe / m_bm * 2 * x_m_bm * x_feo_bm + m_fe / m_fp * x_m_fp * x_feo_fp +
            m_si / m_bm * x_m_bm * (x_mgsio3_bm + x_fesio3_bm) +
            m_si / self.m_capv * x_m_capv + m_o / m_bm * 3 * x_m_bm +
            m_o / m_fp * x_m_fp + m_o / self.m_capv * 3 * x_m_capv
        )
        if (not isclose(sum_elements, 1.0, abs_tol=1e-3)):
            print("Problem with the sum of elements")
            print("Sum of elements is equal to `", sum_elements, "`")

        # Verifying that the FeO content is consistent
        sum_feo = self.m_feo * (
            2 * x_m_bm * x_feo_bm / m_bm + x_m_fp * x_feo_fp / m_fp
        )
        if (not isclose(sum_feo, feo, abs_tol=1e-3)):
            print("Problem with the FeO content")
            print("Calculated FeO content is `", sum_feo,
                "` , while the actual value is `", feo, "`")

    return eq_MGD_fp, eq_MGD_bm, eq_alo2


def _oxides_content_in_bm(
    self, x_feo_fp, rho_bm, rho_fp, p_capv, p_bm, feo, al, ratio_fe, rho_capv, p_fp
):
    """Calculates the molar concentration of FeO and AlO2 in Bridgmanite.

    This function calculates the molar concentration of FeO and AlO2 in Bridgmanite.
    The calculation here is a combination of eqs. (1), (2), (4) and (13) in the
    supplementary material of Vilella et al. (2021).

    Args:
        x_feo_fp: The molar concentration of FeO in Fp.
        rho_bm: Density of Bm for the considered conditions. Its value has been
                rescaled by 5500. [kg/m^3]
        rho_fp: Density of Fp for the considered conditions. Its value has been
                rescaled by 5500. [kg/m^3]
        p_capv: Proportion of CaPv. [vol%]
        p_bm: Proportion of Bm. [vol%]
        feo: FeO content in the rock assemblage. [wt%]
        al: Al2O3 content in the rock assemblage. [wt%]
        ratio_fe: Oxidation state in Bm.
        rho_capv: Density of CaPv for the considered conditions. Its value has been
                  rescaled by 5500. [kg/m^3]
        p_fp: Proportion of Fp. [vol%]

    Returns:
        Float64: The molar concentration of AlO2 in Bm.
        Float64: The molar concentration of FeO in Bm.
    """
    # Molar mass of Fp
    m_fp = self.m_mgo * (1 - x_feo_fp) + self.m_feo * x_feo_fp

    # Mass proportion of Fp
    x_m_fp = 1 / (
        1 + p_capv * rho_capv / (p_fp * rho_fp) + p_bm * rho_bm / (p_fp * rho_fp)
    )

    # Equations giving the FeO and AlO2 content in Bm
    c_1 = 0.5 * self.m_al2o3 / al * (feo / self.m_feo - x_feo_fp * x_m_fp / m_fp)
    x_alo2_bm = self.kd_ref_am * x_feo_fp / (
        2 * c_1 * (1 - x_feo_fp) + self.kd_ref_am * x_feo_fp +
        self.kd_ref_am * x_feo_fp * (2 - ratio_fe) * c_1
    )
    x_feo_bm  = c_1 * x_alo2_bm

    return x_alo2_bm, x_feo_bm


def _solve_without_fp(
    self, x_init, dT, p_capv, p_bm, feo, al, ratio_fe, rho_capv, al_excess
):
    """Implements the solver for the physics problem without Ferropericlase.

    This function implement a solver for the system of equations governing the physics
    problem when no Ferropericlase is present.
    This physics problem is highly non-linear such that finding a proper solution is
    challenging. It is particularly challenging when the starting conditions are not
    well known, for instance at the start of the simulation loop.

    To circumvent this issue, this function first try to find a solution using the
    suggested starting conditions. If a solution cannot be found, the starting
    conditions are randomly sampled within predetermined ranges until a solution is
    found or that the number of attempts exceeds 1000.
    If a solution cannot be found within these conditions, a value of 0.0 is returned
    for each property.

    If it remains too difficult to find a solution, the user can manually increase the
    maximum number of attempts or the maximum number of iterations (maxiter) in the
    solver call.

    Note that several solvers from scipy.optimize have been tried (minimize, fsolve,
    least_squares) and they have produced unsatisfactory results.

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
        Float64: The molar concentration of FeO in Bm.
        Float64: The molar concentration of AlO2 in Bm.
        Float64: The rescaled density of Bm.
    """
    # Initialization
    n_iter = 0

    while (n_iter < 1000):
        ### Solution has not yet been found ###
        try:
            ### Solution has been found ###
            # Solving the system of equation
            solution = scipy.optimize.newton_krylov(
                lambda x: _set_eqs_without_fp(
                    self, x, dT, p_capv, p_bm, feo, al, ratio_fe, rho_capv,
                    al_excess
                ),
                x_init,
                maxiter=1000
            )

            return solution
        except:
            ### Solution has not been found ###
            # Setting random starting conditions
            x_init[0] = random.uniform(0.0, 1.0)
            x_init[1] = random.uniform(0.0, 1.0)
            x_init[2] = random.uniform(0.8, 1.8)
            n_iter += 1

    return 0.0, 0.0, 0.0


def _set_eqs_without_fp(
    self, var_in, dT, p_capv, p_bm, feo, al, ratio_fe, rho_capv, al_excess
):
    """Calculates the equations for the physics problem without Ferropericlase.

    This function calculates the residue for the system of equations governing the
    physics problem when no Ferropericlase is present.

    The first equation comes from the equation of state for Bridgmanite. More
    specifically, the Mie-Gruneisen-Debye equation of state, which corresponds to the
    eq. (33) of Jackson and Rigden (1996).
    The second equation comes from the ratio of FeO and alumina content in Bridgmanite.
    It corresponds to the eq. (16) in the supplementary material of Vilella et al.
    (2021).
    The third equation comes from the alumina content in Bridgmanite. It corresponds
    to the eq. (13) in the supplementary material of Vilella et al. (2021).

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
        Float64: Residue of the equation of state for Bridgmanite.
        Float64: Residue of the equation for the ratio of FeO and alumina content in
                 Bridgmanite.
        Float64: Residue of the equation for the alunina content in Bridgmanite.
    """
    # Loading utility class for mineral properties calculation
    bm_eos = _EOS_bm()

    # Unpacking input variables
    x_feo_bm, x_alo2_bm, rho_bm = var_in

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
        self.m_fesio3 * x_fesio3_bm
    )
    v_bm = 1000 * m_bm / (5500 * rho_bm)

    # Equation from the EOS for Bm
    eq_MGD_bm = bm_eos._MGD(
        self, self.T_am + dT, self.P_am, v_bm, x_mgsio3_bm, x_fesio3_bm, x_fealo3_bm,
        x_fe2o3_bm, x_al2o3_bm
    )

    # Equation from alumina and FeO content
    eq_feo_al = -feo / al + (2 * self.m_feo * x_feo_bm) / (self.m_al2o3 * x_alo2_bm)

    # Equation from the alumina content
    eq_alo2 = -x_alo2_bm + m_bm * al / (self.m_al2o3 * x_m_bm)

    return eq_MGD_bm, eq_feo_al, eq_alo2
