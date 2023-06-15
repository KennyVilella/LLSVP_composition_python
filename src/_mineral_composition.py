"""

This file is associated with the article:
"Constraints on the composition and temperature of LLSVPs from seismic properties of
lower mantle minerals" by K. Vilella, T. Bodin, C.-E. Boukare, F. Deschamps, J. Badro,
M. D. Ballmer, and Y. Li

Typical usage example:


Copyright, 2023,  Vilella Kenny.
"""
import numpy as np
from math import isclose
import os
import csv
import scipy.optimize
from _eos_implementation import _EOS_fp, _EOS_bm, _EOS_capv
#======================================================================================#
#                                                                                      #
#    Starting implementation of functions used to calculate the mineral composition    #
#                                                                                      #
#======================================================================================#
def _calc_mineral_composition(self, spin_config, P_table):
    """Work in progress.
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
        rho_capv = self.rho_capv_0 * solution[0]

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
                                self, dT, p_capv, p_bm, feo, al, ratio_fe, ii,
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
                                    dT, p_capv, p_bm, feo, al, ratio_fe, x_init[0],
                                    x_init[1], x_init[2], x_init[3], x_init[4]
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
    self, dT, p_capv, p_bm, feo, al, ratio_fe, ii,
    spin_config, P_table, rho_capv, p_fp, x_init
):
    """Work in progress.
    """
    # First guess on whether Al or Fe is in excess in Bm
    if (al < 0.75 * ratio_fe * feo):
        al_excess = True
    else:
        al_excess = False

    if (p_fp > 0.0):
        ### Ferropericlase is present in the rock assemblage ###
        # Setting lower bound and upper bound for the solution
        x_feo_fp_bound = (0.001, 1.0)
        rho_bm_bound = (0.5, 1.8)
        rho_fp_bound = (0.5, 1.8)

        # Solving the system of equation
        solution = scipy.optimize.minimize(
            lambda x: _set_eqs_with_fp(
                self, x, dT, p_capv, p_bm, feo, al, ratio_fe, ii, spin_config, P_table,
                rho_capv, p_fp, al_excess
            ),
            [x_init[1], x_init[3] / 5500, x_init[4] / 5500],
            bounds=(x_feo_fp_bound, rho_bm_bound, rho_fp_bound)
        )

        # Unpacking the results
        x_feo_fp, rho_bm, rho_fp = solution.x
        x_alo2_bm, x_feo_bm = _oxides_content_in_bm(
            self, x_feo_fp, rho_bm, rho_fp, p_capv, p_bm, feo, al, ratio_fe, rho_capv,
            p_fp
        )

        if ((not al_excess) and (ratio_fe * x_feo_bm < x_alo2_bm)):
            ### First guess for al_excess is incorrect ###
            # Trying to solve the system of equation with al_excess
            al_excess = True
            solution = scipy.optimize.minimize(
                lambda x: _set_eqs_with_fp(
                    self, x, dT, p_capv, p_bm, feo, al, ratio_fe, ii, spin_config,
                    P_table, rho_capv, p_fp, al_excess
                ),
                [x_init[1], x_init[3] / 5500, x_init[4] / 5500],
                bounds=(x_feo_fp_bound, rho_bm_bound, rho_fp_bound)
            )

            # Unpacking the results
            x_feo_fp, rho_bm, rho_fp = solution.x
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
            solution = scipy.optimize.minimize(
                lambda x: _set_eqs_with_fp(
                    self, x, dT, p_capv, p_bm, feo, al, ratio_fe, ii, spin_config,
                    P_table, rho_capv, p_fp, al_excess
                ),
                [x_init[1], x_init[3] / 5500, x_init[4] / 5500],
                bounds=(x_feo_fp_bound, rho_bm_bound, rho_fp_bound)
            )

            # Unpacking the results
            x_feo_fp, rho_bm, rho_fp = solution.x
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
            self, solution.x, dT, p_capv, p_bm, feo, al, ratio_fe, ii, spin_config,
            P_table, rho_capv, p_fp, al_excess, True
        )
    else:
        ### Ferropericlase is absent from the rock assemblage ###
        # Setting lower bound and upper bound for the solution
        x_feo_bm_bound = (0.0, 1.0)
        x_alo2_bm_bound = (0.0, 1.0)
        rho_bm_bound = (0.5, 1.8)

        # Solving the system of equation
        solution = scipy.optimize.minimize(
            lambda x: _set_eqs_without_fp(
                self, x, dT, p_capv, p_bm, feo, al, ratio_fe, rho_capv, al_excess
            ),
            [x_init[0], x_init[2], x_init[3] / 5500],
            bounds=(x_feo_bm_bound, x_alo2_bm_bound, rho_bm_bound)
        )

        # Unpacking the results
        x_feo_bm, x_alo2_bm, rho_bm = solution.x
        x_feo_fp = 0.0
        rho_fp = 0.0

        if ((not al_excess) and (ratio_fe * x_feo_bm < x_alo2_bm)):
            ### First guess for al_excess is incorrect ###
            # Trying to solve the system of equation with al_excess
            al_excess = True
            solution = scipy.optimize.minimize(
                lambda x: _set_eqs_without_fp(
                    self, x, dT, p_capv, p_bm, feo, al, ratio_fe, rho_capv, al_excess
                ),
                [x_init[0], x_init[2], x_init[3] / 5500],
                bounds=(x_feo_bm_bound, x_alo2_bm_bound, rho_bm_bound)
            )

            # Unpacking the results
            x_feo_bm, x_alo2_bm, rho_bm = solution.x

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
            solution = scipy.optimize.minimize(
                lambda x: _set_eqs_without_fp(
                    self, x, dT, p_capv, p_bm, feo, al, ratio_fe, rho_capv, al_excess
                ),
                [x_init[0], x_init[2], x_init[3] / 5500],
                bounds=(x_feo_bm_bound, x_alo2_bm_bound, rho_bm_bound)
            )

            # Unpacking the results
            x_feo_bm, x_alo2_bm, rho_bm = solution.x

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


def _set_eqs_with_fp(
    self, var_in, dT, p_capv, p_bm, feo, al, ratio_fe, ii, spin_config, P_table,
    rho_capv, p_fp, al_excess, testing=False
):
    """Work in progress.
    """
    # Loading utility class for mineral properties calculation
    fp_eos = _EOS_fp()
    bm_eos = _EOS_bm()

    # Unpacking input variables
    x_feo_fp, rho_bm, rho_fp = var_in

    # Average spin state of FeO
    index_x = np.argmin(np.abs(self.x_vec - x_feo_fp))
    index_P = np.argmin(np.abs(P_table[ii, :, index_x] - self.P_am))
    eta_ls = spin_config[ii, index_P, index_x]

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
        if (abs(P_table[ii, index_P, index_x] - self.P_am) > self.delta_P):
            print("Error on P for the spin transition is large")
            print("Pressure for the spin configuration: `",
                P_table[ii, index_P, index_x], "` actual pressure: `", self.P_am, "`")
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

    return abs(eq_MGD_fp) + abs(eq_MGD_bm) + abs(eq_alo2)


def _oxides_content_in_bm(
    self, x_feo_fp, rho_bm, rho_fp, p_capv, p_bm, feo, al, ratio_fe, rho_capv, p_fp
):
    """Work in progress.
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

    return (x_alo2_bm, x_feo_bm)


def _set_eqs_without_fp(
    self, var_in, dT, p_capv, p_bm, feo, al, ratio_fe, rho_capv, al_excess
):
    """Work in progress.
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

    return abs(eq_MGD_bm) + abs(eq_feo_al) + abs(eq_alo2)
