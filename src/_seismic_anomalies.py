"""

This file is associated with the article:
"Constraints on the composition and temperature of LLSVPs from seismic properties of
lower mantle minerals" by K. Vilella, T. Bodin, C.-E. Boukare, F. Deschamps, J. Badro,
M. D. Ballmer, and Y. Li

These functions should not be used outside the class MineralProperties.

Typical usage example:


Copyright, 2023,  Vilella Kenny.
"""
import numpy as np
from math import isclose
import os
import csv
import scipy.optimize
from _eos_implementation import _EOS_fp, _EOS_bm, _EOS_capv
from _mineral_composition import _solve_mineral_composition
#======================================================================================#
#                                                                                      #
#     Starting implementation of functions used to calculate the seismic anomalies     #
#                                                                                      #
#======================================================================================#
def _calc_seismic_anomalies(self, spin_config, P_table):
    """
    """
    # Loading utility class for mineral properties calculation
    capv_eos = _EOS_capv()

    # Calculating the density of CaPv in the ambient mantle
    solution = scipy.optimize.fsolve(
        lambda x: capv_eos._MGD(self, self.T_am, self.P_am, x), 20.
    )
    rho_capv_am = self.rho_capv_0 * solution[0]

    # Calculating the composition of the ambient mantle
    p_fp_am = 1 - self.p_capv_am - self.p_bm_am
    x_init = [0.1, 0.3, 0.1, 5500.0, 6500.0]
    solution = _solve_mineral_composition(
        self, 0., self.p_capv_am, self.p_bm_am, self.iron_content_am,
        self.al_content_am, self.ratio_fe_bm_am, spin_config, P_table, rho_capv_am,
        p_fp_am, x_init
    )
    x_feo_bm_am, x_feo_fp_am, x_alo2_bm_am, rho_bm_am, rho_fp_am = solution

    # Calculating the seismic properties of the ambient mantle
    rho_ref, v_phi_ref, v_s_ref, v_p_ref = _calc_seismic_properties(
        self, spin_config, P_table, 0., self.p_capv_am, self.p_bm_am, p_fp_am,
        self.iron_content_am, self.al_content_am, self.ratio_fe_bm_am, x_feo_bm_am,
        x_feo_fp_am, x_alo2_bm_am, rho_bm_am, rho_fp_am
    )

    for file in os.listdir(self.path):
        if (not file.endswith(".csv")):
            continue
        print("Starting to process file: `", file, "`")

        # Determining name for processed file
        processed_filename = os.path.join(
            self.path, os.path.splitext(file)[0] + "_processed.csv"
        )
        if os.path.isfile(processed_filename):
            ### A file already exists ###
            print("File detected")
            print("Resuming the simulation is not available")
            print("Deleting old file")
            os.remove(processed_filename)

        with open(os.path.join(self.path, file), newline='') as csvfile:
            data = csv.reader(csvfile, delimiter=' ')
            for row in data:
                # Loading the data
                dT = float(row[0])
                p_capv = float(row[1])
                p_bm = float(row[2])
                feo = float(row[3])
                al = float(row[4])
                ratio_fe = float(row[5])
                x_feo_bm = float(row[6])
                x_feo_fp = float(row[7])
                x_alo2_bm = float(row[8])
                rho_bm = float(row[9])
                rho_fp = float(row[10])
                p_fp = 1 - p_bm - p_capv

                if isclose(p_fp, 0.0, abs_tol=1e-4):
                    ### No Ferropericlase, changing density to avoid errors ###
                    rho_fp = 1.0

                if (rho_bm == 0.0):
                    ### No results for this condition ###
                    continue

                # Calculating the seismic properties of the rock assemblage
                rho, v_phi, v_s, v_p = _calc_seismic_properties(
                    self, spin_config, P_table, dT, p_capv, p_bm, p_fp, feo, al,
                    ratio_fe, x_feo_bm, x_feo_fp, x_alo2_bm, rho_bm, rho_fp
                )

                # Calculating the seismic anomalies of the rock assemblage
                delta_rho = 100 * (rho - rho_ref) / rho
                delta_v_phi = 100 * (v_phi - v_phi_ref) / v_phi
                delta_v_s = 100 * (v_s - v_s_ref) / v_s
                delta_v_p = 100 * (v_p - v_p_ref) / v_p

                # Writting the data
                with open(processed_filename, 'a', newline='') as csvfile:
                    data_writer = csv.writer(csvfile, delimiter=' ')
                    data_writer.writerow([
                        dT, p_capv, p_bm, feo, al, ratio_fe, delta_rho, delta_v_phi,
                        delta_v_s, delta_v_p
                    ])

def _calc_seismic_properties(
    self, spin_config, P_table, dT, p_capv, p_bm, p_fp, feo, al, ratio_fe, x_feo_bm,
    x_feo_fp, x_alo2_bm, rho_bm, rho_fp
):
    """
    """
    # Loading utility class for mineral properties calculation
    fp_eos = _EOS_fp()
    bm_eos = _EOS_bm()
    capv_eos = _EOS_capv()
   
    # Calculating the density of CaPv
    solution = scipy.optimize.fsolve(
        lambda x: capv_eos._MGD(self, self.T_am + dT, self.P_am, x), 20.
    )
    rho_capv = self.rho_capv_0 * solution[0]

    # Calculating the spin configuration
    index_x = np.argmin(np.abs(self.x_vec - x_feo_fp))
    index_T = np.argmin(np.abs(self.T_vec - (self.T_am + dT)))
    index_P = np.argmin(np.abs(P_table[index_T, :, index_x] - self.P_am))
    eta_ls = spin_config[index_T, index_P, index_x]

    # Calculating composition of Bridgmanite in term of components
    al_excess = (feo * x_feo_bm < x_alo2_bm)
    x_mgsio3 = 1 - x_alo2_bm - (2 - feo) * x_feo_bm
    x_fesio3 = 2 * (1 - feo) * x_feo_bm
    x_fealo3 = 2 * (1.0 - al_excess) * x_alo2_bm + 2 * al_excess * feo * x_feo_bm
    x_fe2o3 = (1.0 - al_excess) * (feo * x_feo_bm - x_alo2_bm)
    x_al2o3 = al_excess * (x_alo2_bm - feo * x_feo_bm)

    # Calculating molar mass of Fp and Bm
    m_fp = self.m_mgo * (1 - x_feo_fp) + self.m_feo * x_feo_fp
    m_bm = (
        self.m_mgsio3 * x_mgsio3 + self.m_fealo3 * x_fealo3 + self.m_al2o3 * x_al2o3 +
        self.m_fe2o3 * x_fe2o3 + self.m_fesio3 * x_fesio3
    )

    # Calculating volume of the three minerals
    v_fp = 1000 * m_fp / rho_fp
    v_bm = 1000 * m_bm / rho_bm
    v_capv = 1000 * self.m_capv / rho_capv

    # Calculating the shear modulus of the three minerals
    g_fp = fp_eos._g(self, self.T_am + dT, v_fp, eta_ls, x_feo_fp)
    g_bm = bm_eos._g(
        self, self.T_am + dT, v_bm, x_mgsio3, x_fesio3, x_fealo3, x_fe2o3, x_al2o3
    )
    g_capv = capv_eos._g(self, self.T_am + dT, v_capv)

    # Calculating the isentropic bulk modulus of the three minerals
    k_s_fp = fp_eos._k_s(self, self.T_am + dT, v_fp, eta_ls, x_feo_fp)
    k_s_bm = bm_eos._k_s(
        self, self.T_am + dT, v_bm, x_mgsio3, x_fesio3, x_fealo3, x_fe2o3, x_al2o3
    )
    k_s_capv = capv_eos._k_s(self, self.T_am + dT, v_capv)

    # Calculating the shear modulus of the rock assemblage
    g_tot_r = 1 / (p_bm / g_bm + p_capv / g_capv + p_fp / g_fp)
    g_tot_v = p_bm * g_bm + p_capv * g_capv + p_fp * g_fp
    g_tot = 0.5 * (g_tot_r + g_tot_v)

    # Calculating the isentropic bulk modulus of the rock assemblage
    k_s_tot_r = 1 / (p_bm / k_s_bm + p_capv / k_s_capv + p_fp / k_s_fp)
    k_s_tot_v = p_bm * k_s_bm + p_capv * k_s_capv + p_fp * k_s_fp
    k_s_tot = 0.5 * (k_s_tot_r + k_s_tot_v)

    # Calculating the density of the rock assemblage
    rho = p_fp * rho_fp + p_bm * rho_bm + p_capv * rho_capv

    # Calculating the seismic velocities of the rock assemblage
    v_phi = ((k_s_tot * 10**9) / rho)**(1/2) / 1000
    v_s = ((g_tot * 10**9) / rho)**(1/2) / 1000
    v_p = np.sqrt(v_phi * v_phi + (4/3) * v_s * v_s)

    return rho, v_phi, v_s, v_p