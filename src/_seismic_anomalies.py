""""Calculates the seismic anomalies of the rock assemblage.

This file is associated with the article:
"Constraints on the composition and temperature of LLSVPs from seismic properties of
lower mantle minerals" by K. Vilella, T. Bodin, C.-E. Boukare, F. Deschamps, J. Badro,
M. D. Ballmer, and Y. Li

The purpose of the functions in this module is to calculate the seismic anomalies of a rock assemblage relative to the ambient mantle, considering a wide range of mineral compositions. The main function reads the file generated by _calc_mineral_composition and calculates the density and seismic wave speed anomalies for each composition. The seismic anomalies are expressed as relative differences in percent.

The purpose of the functions in this file is to calculate the seismic anomalies of a
rock assemblage relative to the ambient mantle, considering a wide range of mineral
compositions. The main function reads the file generated by _calc_mineral_composition
and calculates for each composition the density and seismic wave speed anomalies. The
seismic anomalies are expressed as relative differences compared to the ambient mantle
in percent.

It is important to note that the calculation of S-wave and P-wave seismic velocities is
considered less reliable compared to the calculation of density and bulk sound speed.
Further discussion on this topic is available in the associated article.

The results are written into new files with the name of the input file appended with
"_processed". The simulation cannot be resumed, as it is expected to be a relatively
quick calculation.

Note that these functions are intended for use within the MineralProperties class and
should not be used outside of it.

Typical usage example:

  from _spin_configuration import _calc_spin_configuration
  from _seismic_anomalies import _calc_seismic_anomalies
  spin_config, P_table = _calc_spin_configuration(self)
  _calc_seismic_anomalies(self, spin_config, P_table)

Copyright, 2023,  Vilella Kenny.
"""
import numpy as np
from math import isclose
import os
import scipy.optimize
from ._eos_implementation import _EOS_fp, _EOS_bm, _EOS_capv
from ._mineral_composition import _solve_mineral_composition
#======================================================================================#
#                                                                                      #
#     Starting implementation of functions used to calculate the seismic anomalies     #
#                                                                                      #
#======================================================================================#
def _calc_seismic_anomalies(self, spin_config: np.ndarray, P_table: np.ndarray):
    """"Calculates the seismic anomalies of a wide range of rock assemblages.

    This function calculates the seismic anomalies of a wide range of mineral
    compositions and writes the results into files. The calculated seismic anomalies are
    the relative differences in percent for density, bulk sound speed, S-wave velocity
    and P-wave velocity compared to the ambient mantle.

    The investigated mineral compositions are read from the files generated by
    _calc_mineral_composition in the directory given by the `path` attribute of the
    class MineralProperties. The results are then written in new files in the same
    directory with the same name except that "_processed" is added.

    The function reads the mineral compositions from files generated by
    _calc_mineral_composition in the directory specified by the `path` attribute of the
    MineralProperties class. The results are then written to new files in the same
    directory, with the addition of "_processed" to the original file names.

    It is important to note that the simulation cannot be resumed, and any existing old
    files in the directory will be automatically deleted.

    Args:
        spin_config: Average spin state of FeO in Fp for a given value for the
                     temperature, volume of Fp, and FeO content in Fp.
        P_table: Pressure for a given value for the temperature, volume of Fp, and
                 FeO content in Fp.
    """
    # Loading utility class for mineral properties calculation
    capv_eos = _EOS_capv()

    # Calculating the density of CaPv in the ambient mantle
    solution = scipy.optimize.fsolve(
        lambda x: capv_eos._MGD(self, self.P_am, self.T_am, x), 20.
    )
    rho_capv_am = self.rho_capv_0 * self.v_casio3_0 / solution[0]

    # Calculating the composition of the ambient mantle
    p_fp_am = 1 - self.p_capv_am - self.p_bm_am
    x_init = [0.1, 0.3, 0.1, 5500.0, 6500.0]
    solution = _solve_mineral_composition(
        self, x_init, 0., self.p_capv_am, self.p_bm_am, p_fp_am, self.iron_content_am,
        self.al_content_am, self.ratio_fe_bm_am, spin_config, P_table, rho_capv_am
    )
    x_feo_bm_am, x_feo_fp_am, x_alo2_bm_am, rho_bm_am, rho_fp_am = solution

    # Calculating the seismic properties of the ambient mantle
    rho_ref, v_phi_ref, v_s_ref, v_p_ref = _calc_seismic_properties(
        self, 0., self.p_capv_am, self.p_bm_am, p_fp_am, self.iron_content_am,
        self.al_content_am, self.ratio_fe_bm_am, spin_config, P_table, x_feo_bm_am,
        x_feo_fp_am, x_alo2_bm_am, rho_bm_am, rho_fp_am
    )

    for file in os.listdir(self.path):
        if (not file.endswith(".csv") or file.endswith("_processed.csv")):
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

        with open(os.path.join(self.path, file), newline='') as f:
            data = f.readlines()
            for row in data:
                # Loading the data
                processed_row = [float(s) for s in row.split()]
                dT, p_capv, p_bm, feo, al, ratio_fe = processed_row[:6]
                x_feo_bm, x_feo_fp, x_alo2_bm, rho_bm, rho_fp = processed_row[6:]
                p_fp = 1 - p_bm - p_capv

                if isclose(p_fp, 0.0, abs_tol=1e-4):
                    ### No Ferropericlase, changing density to avoid errors ###
                    rho_fp = 5500.0

                if (rho_bm == 0.0):
                    ### No results for this condition ###
                    continue

                # Calculating the seismic properties of the rock assemblage
                rho, v_phi, v_s, v_p = _calc_seismic_properties(
                    self, dT, p_capv, p_bm, p_fp, feo, al, ratio_fe, spin_config,
                    P_table, x_feo_bm, x_feo_fp, x_alo2_bm, rho_bm, rho_fp
                )

                # Calculating the seismic anomalies of the rock assemblage
                delta_rho = 100 * (rho - rho_ref) / rho
                delta_v_phi = 100 * (v_phi - v_phi_ref) / v_phi
                delta_v_s = 100 * (v_s - v_s_ref) / v_s
                delta_v_p = 100 * (v_p - v_p_ref) / v_p

                # Writting the data
                with open(processed_filename, 'a', newline='') as f:
                    f.write(
                        f"{dT:.0f}\t{p_capv:.2f}\t{p_bm:.2f}\t{feo:.3f}\t{al:.3f}\t" +
                        f"{ratio_fe:.1f}\t{delta_rho:.3f}\t{delta_v_phi:.3f}\t" +
                        f"{delta_v_s:.3f}\t{delta_v_p:.0f}\n"
                    )

def _calc_seismic_properties(
    self, dT: float, p_capv: float, p_bm: float, p_fp: float, feo: float, al: float,
    ratio_fe: float, spin_config: np.ndarray, P_table: np.ndarray, x_feo_bm: float,
    x_feo_fp: float, x_alo2_bm: float, rho_bm: float, rho_fp: float
) -> list:
    """Calculates the seismic properties of a rock assemblage.

    This function calculates the density and seismic wave speeds of a given mineral
    composition.

    In addition to the calculation described in _mineral_composition, this function
    requires the calculation of the shear modulus and isentropic bulk modulus. The
    shear modulus is calculated using a combination of eqs. (21) and (38) of Bina and
    Helffrich (1992). The isentropic bulk modulus is calculated using multiple
    equations, primarily described in (B5) and (B6) of Jackson and Rigden (1996).

    The value of the shear modulus and the isentropic bulk modulus of the rock
    assemblage is calculated from the value for its individual components using the
    Voigt-Reuss-Hill average.

    The shear modulus and isentropic bulk modulus of the rock assemblage are calculated
    by taking the Voigt-Reuss-Hill average of the individual components values.

    Note that the calculation of S-wave and P-wave seismic velocities is considered to
    be less reliable compared to the calculation of density and bulk sound speed. This
    is mainly due to the assumption made in the shear modulus calculation, particularly
    the assumption of a constant temperature dependence of the shear modulus is
    throughout the considered parameter space.

    Args:
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
        x_feo_bm: Molar concentration of FeO in Bm.
        x_feo_fp: Molar concentration of FeO in Fp.
        x_alo2_bm: Molar concentration of AlO2 in Bm.
        rho_bm: Density of Bm. [kg/m^3]
        rho_fp: Density of Fp. [kg/m^3]

    Returns:
        A list containing the density of the rock assemblage [kg/m^3], its bulk sound
        speed [km/s], its S-wave velocity [km/s], and its P-wave velocity [km/s].
    """
    # Loading utility class for mineral properties calculation
    fp_eos = _EOS_fp()
    bm_eos = _EOS_bm()
    capv_eos = _EOS_capv()
   
    # Calculating the density of CaPv
    solution = scipy.optimize.fsolve(
        lambda x: capv_eos._MGD(self, self.P_am, self.T_am + dT, x), 20.
    )
    rho_capv = self.rho_capv_0 * self.v_casio3_0 / solution[0]

    # Calculating the spin configuration
    index_x = np.argmin(np.abs(self.x_vec - x_feo_fp))
    index_T = np.argmin(np.abs(self.T_vec - (self.T_am + dT)))
    index_P = np.argmin(np.abs(P_table[index_T, :, index_x] - self.P_am))
    eta_ls = spin_config[index_T, index_P, index_x]

    # Calculating composition of Bridgmanite in term of components
    al_excess = (ratio_fe * x_feo_bm < x_alo2_bm)
    x_mgsio3_bm = 1 - x_alo2_bm - (2 - ratio_fe) * x_feo_bm
    x_fesio3_bm = 2 * (1 - ratio_fe) * x_feo_bm
    x_fealo3_bm = (
        2 * (1.0 - al_excess) * x_alo2_bm + 2 * al_excess * ratio_fe * x_feo_bm
    )
    x_fe2o3_bm = (1.0 - al_excess) * (ratio_fe * x_feo_bm - x_alo2_bm)
    x_al2o3_bm = al_excess * (x_alo2_bm - ratio_fe * x_feo_bm)

    # Calculating molar mass of Fp and Bm
    m_fp = self.m_mgo * (1 - x_feo_fp) + self.m_feo * x_feo_fp
    m_bm = (
        self.m_mgsio3 * x_mgsio3_bm + self.m_fealo3 * x_fealo3_bm +
        self.m_al2o3 * x_al2o3_bm + self.m_fe2o3 * x_fe2o3_bm +
        self.m_fesio3 * x_fesio3_bm
    )

    # Calculating volume of the three minerals
    v_fp = 1000 * m_fp / rho_fp
    v_bm = 1000 * m_bm / rho_bm
    v_capv = 1000 * self.m_capv / rho_capv

    # Calculating the shear modulus of the three minerals
    g_fp = fp_eos._g(self, self.T_am + dT, v_fp, eta_ls, x_feo_fp)
    g_bm = bm_eos._g(
        self, self.T_am + dT, v_bm, x_mgsio3_bm, x_fesio3_bm, x_fealo3_bm, x_fe2o3_bm,
        x_al2o3_bm
    )
    g_capv = capv_eos._g(self, self.T_am + dT, v_capv)

    # Calculating the isentropic bulk modulus of the three minerals
    k_s_fp = fp_eos._k_s(self, self.T_am + dT, v_fp, eta_ls, x_feo_fp)
    k_s_bm = bm_eos._k_s(
        self, self.T_am + dT, v_bm, x_mgsio3_bm, x_fesio3_bm, x_fealo3_bm, x_fe2o3_bm,
        x_al2o3_bm
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

    return [rho, v_phi, v_s, v_p]
