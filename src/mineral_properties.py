"""Calculates the mineral properties of potential lower mantle compositions.

This file is associated with the article:
"Constraints on the composition and temperature of LLSVPs from seismic properties of
lower mantle minerals" by K. Vilella, T. Bodin, C.-E. Boukare, F. Deschamps, J. Badro,
M. D. Ballmer, and Y. Li

This class calculates the properties of a large range of mineral composition at
pressure and temperature conditions appropriate for the lowermost mantle. These
compositions are assumed to be close to a typical pyrolitic composition.
The ultimate goal of this program is to calculate the density and seismic wave
velocities of various rock assemblages in order to identify potential compositions for
the LLSVPs. As a result, the outputs of this class are the density of ferropericlase
(Fp) and bridgmanite (Bm) as well as the molar concentration of FeO in Fp, FeO in Bm,
and AlO2 in Bm. These five properties fully characterize the rock assemblage and allow
for easy calculation of their seismic wave velocities.

The next step is to calculate the associated seismic wave velocities, and optionally
analyze the obtained results.

The formalism for the Mie-Gruneisen-Debye equation of state (EOS) used in this class can
be found in Jackson and Rigden (1996). The calculation of the iron spin transition is
described in Sturhahn et al. (2005). Additionnal information is provided in
Vilella et al. (2015) and Vilella et al. (2021).

Typical usage example:

  LLSVP_compositions_simulator = MineralProperties()
  LLSVP_compositions_simulator.calc_mineral_properties()

Copyright, 2023,  Vilella Kenny.
"""
import numpy as np
import os
import csv
import scipy.optimize
import scipy.integrate
from spin_configuration import SpinConfiguration
#======================================================================================#
#                                                                                      #
#           Starting implementation of class calculating mineral properties            #
#                                                                                      #
#======================================================================================#
class MineralProperties:
    """Calculates mineral properties of lower mantle compositions.

    This class provides the functionality required to calculate mineral properties for
    a wide range of mineral compositions under pressure and temperature conditions
    suitable for the lowermost mantle. It is assumed that the desired compositions are
    close to a typical pyrolitic composition.

    During class initialization, a default value is assigned to each mineral property.
    References are provided for each default value. Note that some values are derived
    from the provided references. Alternatively, the user can override the default
    values by providing a dictionary.

    The class method `calc_mineral_properties` calculates the mineral properties for
    a large range of compositions. A dictionary can be provided to specify the desired
    range for each input property (CaPv proportion, Bm proportion, FeO content,
    Al2O3 content, oxidation state, temperature contrast).

    Attributes:
        R: Gas constant. [cm^3 GPa K^−1 mol^−1]
        xi: Scaling of the volume dependence of the crystal-field splitting used for the
            calculation of the spin transition. Default to 1.5.
        delta_0: Crystal-field splitting at ambient condition used for the calculation
                 of the spin transition. Default to 1.35. [eV]
        v_trans: Volume of FeO at which the spin transition occurs. Default to 59. [A^3]
        n_capv_am: Proportion of CaPv in the ambient lower mantle.
                   Default to 0.07. [vol%]
        iron_content_am: FeO content in the ambient mantle. Default to 0.08. [wt%]
        n_bm_am: Proportion of Bm in the ambient lower mantle. Default to 0.75. [vol%]
        al_content_am: Al2O3 content in the ambient mantle. Default to 0.036. [mol%]
        kd_ref_am: Fe partioning coefficient in the ambient mantle. Default to 0.5.
        ratio_fe_bm_am: Oxidation state in Bm. Default to 0.5.
        k0t_prime_bm: Pressure derivative of the bulk modulus for Bm.
                      Default to 3.7.
        theta_bm_0: Debye temperature for Bm at ambient conditions. Default to 1100. [K]
        gamma_bm_0: Gruneisen parameter for Bm at ambient conditions. Default to 1.4.
        q_bm: Exponent of the Gruneisen parameter for Bm. Default to 1.4.
        k0t_prime_fp: Pressure derivative of the bulk modulus for Fp.
                      Default to 4.0.
        theta_fp_0: Debye temperature for Fp at ambient conditions. Default to 673. [K]
        gamma_fp_0: Gruneisen parameter for Fp at ambient conditions. Default to 1.41.
        q_fp: Exponent of the Gruneisen parameter for Fp. Default to 1.3.
        k0t_prime_capv: Pressure derivative of the bulk modulus for CaPv.
                        Default to 3.9.
        theta_capv_0: Debye temperature for CaPv at ambient conditions.
                      Default to 1000. [K]
        gamma_capv_0: Gruneisen parameter for CaPv at ambient conditions.
                      Default to 1.92.
        q_capv: Exponent of the Gruneisen parameter for CaPv. Default to 0.6.
        k_mgo_0: Bulk modulus of MgO at ambient conditions. Default to 160. [GPa]
        v_mgo_0: Volume of MgO at ambient conditions. Default to 11.25. [cm^3/mol]
        k_feo_ls_0: Bulk modulus of FeO in the low spin state at ambient conditions.
                    Default to 150. [GPa] 
        v_feo_ls_0: Volume of FeO in the low spin state at ambient conditions.
                    Default to 10.82. [cm^3/mol]
        k_feo_hs_0: Bulk modulus of FeO in the high spin state at ambient conditions.
                    Default to 150. [GPa]
        v_feo_hs_0: Volume of FeO in the high spin state at ambient conditions.
                    Default to 12.18. [cm^3/mol]
        k_mgsio3_0: Bulk modulus of MgSiO3 at ambient conditions. Default to 261. [GPa]
        v_mgsio3_0: Volume of MgSiO3 at ambient conditions. Default to 24.43. [cm^3/mol]
        k_fesio3_0: Bulk modulus of FeSiO3 at ambient conditions. Default to 248. [GPa]
        v_fesio3_0: Volume of FeSiO3 at ambient conditions. Default to 25.44. [cm^3/mol]
        k_fe2o3_0: Bulk modulus of Fe2O3 at ambient conditions. Default to 95. [GPa]
        v_fe2o3_0: Volume of Fe2O3 at ambient conditions. Default to 30.6. [cm^3/mol]
        k_fealo3_0: Bulk modulus of FeAlO3 at ambient conditions. Default to 271. [GPa]
        v_fealo3_0: Volume of FeAlO3 at ambient conditions. Default to 28.21. [cm^3/mol]
        k_al2o3_0: Bulk modulus of Al2O3 at ambient conditions. Default to 137. [GPa]
        v_al2o3_0: Volume of Al2O3 at ambient conditions. Default to 26.74. [cm^3/mol]
        k_casio3_0: Bulk modulus of CaSiO3 at ambient conditions. Default to 236. [GPa]
        v_casio3_0: Volume of CaSiO3 at ambient conditions. Default to 27.45. [cm^3/mol]
        m_capv: Molar mass of CaPv. [g/mol]
        rho_capv_0: Density of CaPv at ambient conditions. [kg/m^3]
        P_am: Pressure of the ambient mantle. Default to 130. [GPa]
        T_am: Temperature of the ambient mantle. Default to 3000. [K]
        p_capv_min: Minimum proportion of CaPv assumed for the considered compositions.
                    Default to 0. [vol%]
        p_capv_max: Maximum proportion of CaPv assumed for the considered compositions.
                    Default to 0.40. [vol%]
        delta_capv: Step value for the proportion of CaPv. Default to 0.01. [vol%]
        p_bm_min: Minimum proportion of Bm assumed for the considered compositions.
                  Default to 0.60. [vol%]
        delta_bm: Step value for the proportion of Bm. Default to 0.01. [vol%]
        dT_min: Minimum temperature contrast against the ambient mantle assumed for the
                considered compositions. Default to 0. [K]
        dT_max: Maximum temperature contrast against the ambient mantle assumed for the
                considered compositions. Default to 1000. [K]
        delta_dT: Step value for the temperature contrast against the ambient mantle.
                  Default to 100. [K] 
        iron_content_min: Minimum FeO content assumed for the considered compositions.
                          Default to 0.08. [wt%]
        iron_content_max: Maximum FeO content assumed for the considered compositions.
                          Default to 0.28. [wt%]
        delta_iron_content: Step value for the FeO content. Default to 0.01. [wt%]
        al_content_min: Minimum Al2O3 content assumed for the considered compositions.
                        Default to 0.01. [wt%]
        al_content_max: Maximum Al2O3 content assumed for the considered compositions.
                        Default to 0.19. [wt%]
        delta_al: Step value for the Al2O3 content. Default to 0.005. [wt%]
        ratio_fe_bm_min: Minimum oxidation state in Bm assumed for the considered
                         compositions. Default to 0.
        ratio_fe_bm_max: Maximum oxidation state in Bm assumed for the considered
                         compositions. Default to 1.
        delta_ratio_fe: Step value for the oxidation state in Bm. Default to 0.1.
        delta_v: Step value for the volume in the spin transition calculation.
                 Default to 0.05. [A^3]
        delta_P: Step value for the pressure in the spin transition calculation.
                 Default to 0.5. [GPa]
    """
    def __init__(self, prop={}):
        """Initializes all the mineral properties.

        All units are given in the class description.
        Note volumes are converted from the unit cell volume given in the
        reference to the cm3 unit used in this simulator.
        The conversion is done as follows

            V     =   V  *    10^-24   *  6.02*10^23   /      4        =   V*0.15055
         cm^3/mol    A^3    A^3->cm^3       ->mol        number atoms

        The number 4 corresponds to the number of atoms in the primitive cell of Fp
        and Bm.

        More information is provided in Vilella et al. (2015) and Vilella et al. (2021).

        Args:
            prop: A dictionary providing the value of mineral properties.
                  Default to empty dictionary.
        """
        # Directory to write the results
        filepath = os.path.abspath(os.path.dirname(__file__))
        result_path = os.path.join(filepath, "..", "results")
        self.path = prop.get("path", result_path)

        # Gas constant
        self.R = 8.31446*10**(-3)

        # Parameters related to the spin transition
        self.xi = prop.get("xi", 1.5) # Sturhahn 2005
        self.delta_0 = prop.get("delta_0", 1.35) # Sturhahn 2005
        self.v_trans = prop.get("v_trans", 59.) # Fei 2007

        # Parameters of the considered ambient mantle
        self.n_capv_am = prop.get("n_capv_am", 0.07) # Irifune 1994, 2010
        self.iron_content_am = prop.get("iron_content_am", 0.08) # Irifune 2010
        self.n_bm_am = prop.get("n_bm_am", 0.75) # Irifune 2010
        self.al_content_am = prop.get("al_content_am", 0.036) # Irifune 2010
        self.kd_ref_am = prop.get("kd_ref_am", 0.5) # Piet 2016
        self.ratio_fe_bm_am = prop.get("ratio_fe_bm_am", 0.5) # Piet 2016

        # EOS parameters for Bridgmanite
        self.k0t_prime_bm = prop.get("k0t_prime_bm", 3.7) # Fiquet 2000
        self.theta_bm_0 = prop.get("theta_bm_0", 1100.) # Fiquet 2000
        self.gamma_bm_0 = prop.get("gamma_bm_0", 1.4) # Fiquet 2000
        self.q_bm = prop.get("q_bm", 1.4) # Fiquet 2000

        # EOS parameters for Ferropericlase
        self.k0t_prime_fp = prop.get("k0t_prime_fp", 4.0) # Jackson 1982
        self.theta_fp_0 = prop.get("theta_fp_0", 673) # Jackson 1982
        self.gamma_fp_0 = prop.get("gamma_fp_0", 1.41) # Jackson 1982
        self.q_fp = prop.get("q_fp", 1.3) # Jackson 1982

        # EOS parameters for Calcio Perovskite
        self.k0t_prime_capv = prop.get("k0t_prime_capv", 3.9) # Shim 2000
        self.theta_capv_0 = prop.get("theta_capv_0", 1000.) # Shim 2000
        self.gamma_capv_0 = prop.get("gamma_capv_0", 1.92) # Shim 2000
        self.q_capv = prop.get("q_capv", 0.6) # Shim 2000

        # EOS parameters for various components
        self.k_mgo_0 = prop.get("k_mgo_0", 160.) # Speziale 2001
        self.v_mgo_0 = prop.get("v_mgo_0", 11.25) # Speziale 2001
        self.k_feo_ls_0 = prop.get("k_feo_ls_0", 150.) # Fei 2007
        self.v_feo_ls_0 = prop.get("v_feo_ls_0", 10.82) # Fei 2007
        self.k_feo_hs_0 = prop.get("k_feo_hs_0", 150.) # Fei 2007
        self.v_feo_hs_0 = prop.get("v_feo_hs_0", 12.18) # Fei 2007
        self.k_mgsio3_0 = prop.get("k_mgsio3_0", 261.) # Lundin 2008
        self.v_mgsio3_0 = prop.get("v_mgsio3_0", 24.43) # Lundin 2008
        self.k_fesio3_0 = prop.get("k_fesio3_0", 248.) # Lundin 2008
        self.v_fesio3_0 = prop.get("v_fesio3_0", 25.44) # Lundin 2008
        self.k_fe2o3_0 = prop.get("k_fe2o3_0", 95.) # Catalli 2010
        self.v_fe2o3_0 = prop.get("v_fe2o3_0", 30.6) # Catalli 2010
        self.k_fealo3_0 = prop.get("k_fealo3_0", 271.) # Catalli 2011
        self.v_fealo3_0 = prop.get("v_fealo3_0", 28.21) # Catalli 2011
        self.k_al2o3_0 = prop.get("k_al2o3_0", 137.) # Catalli 2011
        self.v_al2o3_0 = prop.get("v_al2o3_0", 26.74) # Catalli 2011
        self.k_casio3_0 = prop.get("k_casio3_0", 236.) # Shin 2000
        self.v_casio3_0 = prop.get("v_casio3_0", 27.45) # Shin 2000

        # Parameters related to Calcio Perovskite
        self.m_capv = 116.161
        self.rho_capv_0 = 1000. * self.m_capv / self.v_casio3_0


    def calc_mineral_properties(self, conditions={}):
        """Work in progress.
        """
        # Loading simulation conditions
        self._load_conditions(conditions)

        # Calculating the spin configuration
        spin_config, P_table = self._calc_spin_configuration()

        # Calculating the mineral compositions
        self._find_mineral_composition(spin_config, P_table)


    def _load_conditions(self, conditions={}):
        """Initializes the input conditions of the simulator.

        Default values are chosen in order to reproduce the results presented in
        Vilella et al. (2021).

        Args:
            conditions: A dictionary providing the values for input conditions.
                        Default to empty dictionary.
        """
        # P, T conditions of the ambient mantle
        self.P_am = conditions.get("P_am", 130.)
        self.T_am = conditions.get("T_an", 3000.)

        # Ranges considered for all input parameters
        self.p_capv_min = conditions.get("p_capv_min", 0.00)
        self.p_capv_max = conditions.get("p_capv_max", 0.40)
        self.delta_capv = conditions.get("delta_capv", 0.01)

        self.p_bm_min = conditions.get("p_bm_min", 0.60)
        self.delta_bm = conditions.get("delta_bm", 0.01)

        self.dT_min = conditions.get("dT_min", 0.)
        self.dT_max = conditions.get("dT_max", 1000.)
        self.delta_dT = conditions.get("delta_dT", 100.)

        self.iron_content_min = conditions.get("iron_content_min", 0.08)
        self.iron_content_max = conditions.get("iron_content_max", 0.28)
        self.delta_iron_content = conditions.get("delta_iron_content", 0.01)

        self.al_content_min = conditions.get("al_content_min", 0.01)
        self.al_content_max = conditions.get("al_content_max", 0.19)
        self.delta_al = conditions.get("delta_al", 0.005)

        self.ratio_fe_bm_min = conditions.get("ratio_fe_bm_min", 0.)
        self.ratio_fe_bm_max = conditions.get("ratio_fe_bm_max", 1.0)
        self.delta_ratio_fe = conditions.get("delta_ratio_fe", 0.1)

        # Stepping used for the spin transition calculation
        self.delta_v = conditions.get("delta_v", 0.05)
        self.delta_P = conditions.get("delta_P", 0.5)


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
        is the msot stable configuration. In practice, this is done by searching for
        the values of eta_ls leading to a derivative of F with respect to eta_ls equals
        to zero. However, this is challenging as several local minima may exist. As a
        result, the calculation is done using three different initial conditions, and
        the actual solution is the one associated with the lowest value of F.

        Returns:
            np.ndarray: The average spin state of FeO in Ferropericalse for a given
                        value for the temperature, volume of Ferropericlase, and FeO
                        content in Ferropericalse.
            np.ndarray: The corresponding pressure for a given value for the
                        temperature, volume of Ferropericlase, and FeO content in
                        Ferropericalse.
        """
        # Initializing SpinConfiguration class with utility functions
        spin = SpinConfiguration(self)

        # Initializing range for spin configuration calculation
        x_min = 0.0
        x_max = 1.0
        delta_x = 0.01
        self.x_vec = np.arange(x_min, x_max + delta_x, delta_x)
        self.T_vec = self.T_am + np.arange(
            self.dT_min, self.dT_max + self.delta_dT, self.delta_dT
        )

        # Calculating range for the volume of Fp using extreme cases
        v_min = spin._calc_volume_MGD(self.P_am, self.T_am + self.dT_max, 1.0, x_max)
        v_max = spin._calc_volume_MGD(self.P_am, self.T_am, 0.0, x_max)
        v_min -= 2.0

        n_T = len(self.T_vec)
        n_v = round((v_max - v_min) / self.delta_v)
        n_x = len(self.x_vec)

        # Initializing
        spin_config = np.zeros((n_T, n_v, n_x))
        P_table = np.zeros((n_T, n_v, n_x))
        k_b = 8.617**(-5) # Boltzmann constant
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
                    E_ls = spin._energy_equation(1, v_fp_0, v_fp)
                    E_hs = spin._energy_equation(3, v_fp_0, v_fp)

                    # Coupling energy low spin state - low spin state
                    wc = spin._splitting_energy(x_fp, v_fp_0, v_fp)

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
                    P_table[jj, kk, ii] = spin._calc_pressure_MGD(
                        T, 0.15055*v_fp, eta_ls, x_fp
                    )

        return spin_config, P_table


    def _find_mineral_composition(self, spin_config, P_table):
        """Work in progress.
        """
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
                lambda x: self._MGD_capv(
                    x, self.T_am + dT, self.P_am, self.k_casio3_0, self.v_casio3_0
                ), 1.5
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
                                x_result = self._calc_mineral_composition(
                                    dT, p_capv, p_bm, feo, al, ratio_fe, ii,
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


    def _calc_mineral_composition(
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
                self._set_eqs_with_fp,
                [x_init[1], x_init[3] / 5500, x_init[4] / 5500],
                args=(
                    dT, p_capv, p_bm, feo, al, ratio_fe, ii,
                    spin_config, P_table, rho_capv, p_fp, al_excess
                ),
                bounds=(x_feo_fp_bound, rho_bm_bound, rho_fp_bound)
            )

            # Unpacking the results
            x_feo_fp, rho_bm, rho_fp = solution.x
            x_feo_bm, x_alo2_bm, x_fe2o3_bm, x_al2o3_bm = self._set_eqs_with_fp(
                solution.x, dT, p_capv, p_bm, feo, al, ratio_fe, ii,
                spin_config, P_table, rho_capv, p_fp, al_excess, True
            )

            if (
                (not al_excess) and
                ((ratio_fe * x_feo_bm < x_alo2_bm) or (x_fe2o3_bm < 0))
            ):
                ### First guess for al_excess is incorrect ###
                # Trying to solve the system of equation with al_excess
                al_excess = True
                solution = scipy.optimize.minimize(
                    self._set_eqs_with_fp,
                    [x_init[1], x_init[3] / 5500, x_init[4] / 5500],
                    args=(
                        dT, p_capv, p_bm, feo, al, ratio_fe, ii,
                        spin_config, P_table, rho_capv, p_fp, al_excess
                    ),
                    bounds=(x_feo_fp_bound, rho_bm_bound, rho_fp_bound)
                )

                # Unpacking the results
                x_feo_fp, rho_bm, rho_fp = solution.x
                x_feo_bm, x_alo2_bm, x_fe2o3_bm, x_al2o3_bm = self._set_eqs_with_fp(
                    solution.x, dT, p_capv, p_bm, feo, al, ratio_fe, ii,
                    spin_config, P_table, rho_capv, p_fp, al_excess, True
                )

                # Checking that solution is indeed correct
                if ((ratio_fe * x_feo_bm > x_alo2_bm) or (x_al2o3_bm < 0)):
                    ### Guess for al_excess is again incorrect ###
                    print("Problem with Al_excess")
                    print("p_bm = ", p_bm, " p_capv = ", p_capv," feo = ", feo,
                        " al = ", al, " ratio_fe = ", ratio_fe, " dT = ", dT)
                    print("Skip this condition")
            elif (
                (al_excess) and
                ((ratio_fe * x_feo_bm > x_alo2_bm) or (x_al2o3_bm < 0))
            ):
                ### First guess for al_excess is incorrect ###
                # Trying to solve the system of equation without al_excess
                al_excess = False
                solution = scipy.optimize.minimize(
                    self._set_eqs_with_fp,
                    [x_init[1], x_init[3] / 5500, x_init[4] / 5500],
                    args=(
                        dT, p_capv, p_bm, feo, al, ratio_fe, ii,
                        spin_config, P_table, rho_capv, p_fp, al_excess
                    ),
                    bounds=(x_feo_fp_bound, rho_bm_bound, rho_fp_bound)
                )

                # Unpacking the results
                x_feo_fp, rho_bm, rho_fp = solution.x
                x_feo_bm, x_alo2_bm, x_fe2o3_bm, x_al2o3_bm = self._set_eqs_with_fp(
                    solution.x, dT, p_capv, p_bm, feo, al, ratio_fe, ii,
                    spin_config, P_table, rho_capv, p_fp, al_excess, True
                )

                # Checking that solution is indeed correct
                if ((ratio_fe * x_feo_bm < x_alo2_bm) or (x_fe2o3_bm < 0)):
                    ### Guess for al_excess is again incorrect ###
                    print("Problem with Al_excess")
                    print("p_bm = ", p_bm, " p_capv = ", p_capv," feo = ", feo,
                        " al = ", al, " ratio_fe = ", ratio_fe, " dT = ", dT)
                    print("Skip this condition")

            # Verifying that the solution is indeed consistent


        else:
            ### Ferropericlase is absent from the rock assemblage ###
            # Setting lower bound and upper bound for the solution
            x_feo_bm_bound = (0.0, 1.0)
            x_alo2_bm_bound = (0.0, 1.0)
            rho_bm_bound = (0.5, 1.8)

            x_feo_bm = 0.1
            x_feo_fp = 0.0
            x_alo2_bm = 0.1
            rho_bm = 1.0
            rho_fp = 0.0

        return [x_feo_bm, x_feo_fp, x_alo2_bm, 5500 * rho_bm, 5500 * rho_fp]

    def _set_eqs_with_fp(
        self, xy, dT, p_capv, p_bm, feo, al, ratio_fe, ii,
        spin_config, P_table, rho_capv, p_fp, al_excess, extra=False
    ):
        """Work in progress.
        """
        # Setting molar masses
        m_mgo = 40.3040
        m_feo = 71.8440
        m_alo2 = 58.9800
        m_sio2 = 60.0840
        m_mgsio3 = m_mgo + m_sio2
        m_fesio3 = m_feo + m_sio2
        m_fealo3 = m_feo + m_alo2
        m_fe2o3 = 2 * m_feo + 15.999
        m_al2o3 = 101.961

        index_x = lambda w: np.argmin(np.abs(self.x_vec - w))
        index_P = lambda w: np.argmin(np.abs(P_table[ii, :, index_x(w)] - self.P_am))
        eta_ls = lambda w: spin_config[ii, index_P(w), index_x(w)]

        v_feo_0 = lambda w: (
            eta_ls(w) * self.v_feo_ls_0 + (1 - eta_ls(w)) * self.v_feo_hs_0
        )
        m_fp = lambda w: m_mgo * (1 - w) + m_feo * w
        v_fp_0 = lambda w: (1 - w) * self.v_mgo_0 + w * v_feo_0(w)
        v_fp = lambda w,v: 1000 * m_fp(w) / (5500 * v)

        # Mass proportions of the minerals
        x_m_bm = lambda u,v: 1 / (
            1 + p_capv * rho_capv / (p_bm * u) + p_fp * v / (p_bm * u)
        )
        x_m_fp = lambda u,v: 1 / (
            1 + p_capv * rho_capv / (p_fp * v) + p_bm * u / (p_fp * v)
        )
        x_m_capv = lambda u,v: 1 / (
            1 + p_fp * v / (p_capv * rho_capv) + p_bm * u / (p_capv * rho_capv)
        )

        # Equations setting the mineral compositions
        c_1 = lambda w,u,v: 0.5 * m_al2o3 / (
            al * (feo / m_feo - w * x_m_fp(u, v) / m_fp(w))
        )
        x_alo2_bm = lambda w,u,v: self.kd_ref_am * w / (
            2 * c_1(w, u, v) * (1 - w) + self.kd_ref_am * w +
            self.kd_ref_am * w * (2 - ratio_fe) * c_1(w, u, v)
        )
        x_feo_bm  = lambda w,u,v: c_1(w, u, v) * x_alo2_bm(w, u, v)

        # Calculating the bulk modulus of FeO using the VRH average
        k_feo_0_r = lambda w: v_feo_0(w) / (
            (1 - eta_ls(w)) * self.v_feo_hs_0 / self.k_feo_hs_0 +
            eta_ls(w) * self.v_feo_ls_0 / self.k_feo_ls_0
        )
        k_feo_0_v = lambda w: (
            (1 - eta_ls(w)) * self.v_feo_hs_0 / v_feo_0(w) * self.k_feo_hs_0 +
            eta_ls(w) * self.v_feo_ls_0 / v_feo_0(w) * self.k_feo_ls_0
        )
        k_feo_0 = lambda w: 0.5 * (k_feo_0_r(w) + k_feo_0_v(w))

        # Calculating the bulk modulus of Fp using the VRH average
        k_fp_0_r = lambda w: v_fp_0(w) / (
            (1 - w) * self.v_mgo_0 / self.k_mgo_0 + w * v_feo_0(w) / k_feo_0(w)
        )
        k_fp_0_v = lambda w: (
            (1 - w) * self.v_mgo_0 / v_fp_0(w) * self.k_mgo_0 +
            w * v_feo_0(w) / v_fp_0(w) * k_feo_0(w)
        )
        k_fp_0 = lambda w: 0.5 * (k_fp_0_r(w) + k_fp_0_v(w))

        if (al_excess):
            ### Al is asssumed to be in excess ###
            # Calculating molar proportion of the different components of Bm
            x_mgsio3_bm = lambda w,u,v: (
                1 - x_alo2_bm(w, u, v) - (2 - ratio_fe) * x_feo_bm(w, u, v)
            )
            x_fesio3_bm = lambda w,u,v: 2 * (1 - ratio_fe) * x_feo_bm(w, u, v)
            x_fealo3_bm = lambda w,u,v: 2 * ratio_fe * x_feo_bm(w, u, v)
            x_fe2o3_bm = lambda w,u,v: 0.0
            x_al2o3_bm = lambda w,u,v: x_alo2_bm(w, u, v) - ratio_fe * x_feo_bm(w, u, v)
        else:
            ### Fe is assumed to be in excess ###
            # Calculating molar proportion of the different components of Bm
            x_mgsio3_bm = lambda w,u,v: (
                1 - x_alo2_bm(w, u, v) - (2 - ratio_fe) * x_feo_bm(w, u, v)
            )
            x_fesio3_bm = lambda w,u,v: 2 * (1 - ratio_fe) * x_feo_bm(w, u, v)
            x_fealo3_bm = lambda w,u,v: 2 * x_alo2_bm(w, u, v)
            x_fe2o3_bm = lambda w,u,v: ratio_fe * x_feo_bm(w, u, v) - x_alo2_bm(w, u, v)
            x_al2o3_bm = lambda w,u,v: 0.0

        v_bm_0 = lambda w,u,v: (
            x_mgsio3_bm(w, u, v) * self.v_mgsio3_0 +
            x_fesio3_bm(w, u, v) * self.v_fesio3_0 +
            x_fealo3_bm(w, u, v) * self.v_fealo3_0 +
            x_al2o3_bm(w, u, v) * self.v_al2o3_0 +
            x_fe2o3_bm(w, u, v) * self.v_fe2o3_0
        )

        # Calculating the bulk modulus of Bm using the VRH average
        k_bm_0_r = lambda w,u,v:  v_bm_0(w, u, v) / (
            x_mgsio3_bm(w, u, v) * self.v_mgsio3_0 / self.k_mgsio3_0 +
            x_fesio3_bm(w, u, v) * self.v_fesio3_0 / self.k_fesio3_0 +
            x_fealo3_bm(w, u, v) * self.v_fealo3_0 / self.k_fealo3_0 +
            x_fe2o3_bm(w, u, v) * self.v_fe2o3_0 / self.k_fe2o3_0 +
            x_al2o3_bm(w, u, v) * self.v_al2o3_0 / self.k_al2o3_0
        )
        k_bm_0_v = lambda w,u,v:  (
            x_mgsio3_bm(w, u, v) * self.v_mgsio3_0 * self.k_mgsio3_0 +
            x_fesio3_bm(w, u, v) * self.v_fesio3_0 * self.k_fesio3_0 +
            x_fealo3_bm(w, u, v) * self.v_fealo3_0 * self.k_fealo3_0 +
            x_fe2o3_bm(w, u, v) * self.v_fe2o3_0 * self.k_fe2o3_0 +
            x_al2o3_bm(w, u, v) * self.v_al2o3_0 * self.k_al2o3_0
        ) / v_bm_0(w, u, v)
        k_bm_0 = lambda w,u,v: 0.5 * (k_bm_0_r(w, u, v) + k_bm_0_v(w, u, v))

        m_bm = lambda w,u,v: (
            m_mgsio3 * x_mgsio3_bm(w,u,v) + m_fealo3 * x_fealo3_bm(w,u,v) +
            m_al2o3 * x_al2o3_bm(w,u,v) + m_fe2o3 * x_fe2o3_bm(w,u,v) +
            m_fesio3 * x_fesio3_bm(w,u,v)
        )
        v_bm = lambda w,u,v: 1000 * m_bm(w, u, v) / (5500 * u)
        x_mgo_bm = lambda w,u,v: x_feo_bm(w, u, v) * (1 - w) / (self.kd_ref_am * w)

        # Equation from the EOS for Fp
        gamma_fp = lambda x: self.gamma_fp_0 * x**(-self.q_fp)
        theta_fp = lambda x: self.theta_fp_0 * np.exp(
            (self.gamma_fp_0 - gamma_fp(x)) / self.q_fp
        )
        BM3_fp = lambda x,w: -self.P_am + (
            1.5 * k_fp_0(w) * (x**(7/3) - x**(5/3)) *
            (1 + 3/4 * (self.k0t_prime_fp - 4) * (x**(2/3) - 1))
        )
        E_th_fp = lambda T,x: 9. * 2. * self.R * T**4 / theta_fp(x)**3 * (
            self._integral_vibrational_energy(theta_fp(x) / T)
        )

        eq_MGD_fp = lambda w,u,v: (
            BM3_fp(v_fp_0(w) / v_fp(w, v), w) +
            gamma_fp(v_fp_0(w) / v_fp(w, v)) / v_fp(w, v) *
            (
                E_th_fp(self.T_am + dT, v_fp_0(w) / v_fp(w, v)) -
                E_th_fp(300, v_fp_0(w) / v_fp(w, v))
            )
        )

        # Equation from the EOS for Bm
        gamma_bm = lambda x: self.gamma_bm_0 * x**(-self.q_bm)
        theta_bm = lambda x: self.theta_bm_0 * np.exp(
            (self.gamma_bm_0 - gamma_bm(x)) / self.q_bm
        )
        BM3_bm = lambda x,w,u,v: -self.P_am + (
            1.5 * k_bm_0(w, u, v) * (x**(7/3) - x**(5/3)) *
            (1 + 3/4 * (self.k0t_prime_bm - 4) * (x**(2/3) - 1))
        )
        E_th_bm = lambda T,x:9. * 2. * self.R * T**4 / theta_bm(x)**3 * (
            self._integral_vibrational_energy(theta_bm(x) / T)
        )

        eq_MGD_bm = lambda w,u,v: (
            BM3_bm(v_bm_0(w, u, v) / v_bm(w, u, v), w, u, v) +
            gamma_bm(v_bm_0(w, u, v) / v_bm(w, u, v)) / v_bm(w, u, v) *
            (
                E_th_bm(self.T_am + dT, v_bm_0(w, u, v) / v_bm(w, u, v)) -
                E_th_bm(300, v_bm_0(w, u, v) / v_bm(w, u, v))
            )
        )

        # Equation from the alumina content
        eq_alo2 = lambda w,u,v: (
            -x_alo2_bm(w, u, v) * m_al2o3 * x_m_bm(u, v) + m_bm(w, u, v) * al
        )

        if extra:
            return (
                x_feo_bm(xy[0], xy[1], xy[2]), x_alo2_bm(xy[0], xy[1], xy[2]),
                x_fe2o3_bm(xy[0], xy[1], xy[2]), x_al2o3_bm(xy[0], xy[1], xy[2])
            )
        else:
            return (
                abs(eq_MGD_fp(xy[0], xy[1], xy[2])) +
                abs(eq_MGD_bm(xy[0], xy[1], xy[2])) +
                abs(eq_alo2(xy[0], xy[1], xy[2]))
            )


    def _MGD_fp(self, x, T_i, P_i, k_fp_0, v_fp_0):
        """Implements the Mie-Gruneisen-Debye EOS for Ferropericlase.

        This function calculates the residue of the Mie-Gruneisen-Debye equation of
        state applied to Ferropericlase (Fp) and can be used to obtain the volume of Fp
        given the pressure and temperature conditions.
        It corresponds to the eq. (6) of Vilella et al. (2015).

        This function can also be used to obtained the pressure by providing a zero
        pressure as the residue would simply be the pressure.

        The formalism of the Mie-Gruneisen-Debye equation of state can be found in
        Jackson and Rigden (1996).

        Args:
            x: Volume ratio V0 / V, where V0 is the volume at ambient conditions and V
               the volume at the considered conditions.
            T_i: Considered temperature. [K]
            P_i: Considered pressure. [GPa]
            k_fp_0: Bulk modulus of Ferropericlase at ambient conditions. [GPa]
            v_fp_0: Volume of Ferropericlase at ambient conditions. [cm^3/mol]

        Returns:
            Float64: The residue of the Mie-Gruneisen-Debye EOS. [GPa]
        """
        # Gruneisen parameter at P, T conditions
        gamma_fp = self.gamma_fp_0 * x**(-self.q_fp)

        # Debye temperature at P, T conditions
        theta_fp = self.theta_fp_0 * np.exp(
            (self.gamma_fp_0 - gamma_fp) / self.q_fp
        )

        # The third-order Birch–Murnaghan isothermal equation of state
        BM3 = P_i - (
            1.5 * k_fp_0 * (x**(7/3) - x**(5/3)) *
            (1 + 3/4 * (self.k0t_prime_fp - 4) * (x**(2/3) - 1))
        )

        E_th = 9. * 2. * self.R * T_i**4 / theta_fp**3 * (
            self._integral_vibrational_energy(theta_fp / T_i)
        )
        E_th_0 = 9. * 2. * self.R * 300.**4 / theta_fp**3 * (
            self._integral_vibrational_energy(theta_fp / 300.)
        )

        # The Mie-Gruneisen-Debye equation of state
        MGD =  BM3  - gamma_fp * x / v_fp_0 * (E_th - E_th_0)

        return MGD


    def _MGD_capv(self, x, T_i, P_i, k_capv_0, v_capv_0):
        """Implements the Mie-Gruneisen-Debye EOS for Calcio Perovskite.

        This function calculates the residue of the Mie-Gruneisen-Debye equation of
        state applied to Calcio Perovskite (CaPv) and can be used to obtain the volume
        of CaPv given the pressure and temperature conditions.
        It corresponds to the eq. (6) of Vilella et al. (2015).

        The formalism of the Mie-Gruneisen-Debye equation of state can be found in
        Jackson and Rigden (1996).

        Args:
            x: Volume ratio V0 / V, where V0 is the volume at ambient conditions and V
               the volume at the considered conditions.
            T_i: Considered temperature. [K]
            P_i: Considered pressure. [GPa]
            k_capv_0: Bulk modulus of CaPv at ambient conditions. [GPa]
            v_capv_0: Volume of CaPv at ambient conditions. [cm^3/mol]

        Returns:
            Float64: The residue of the Mie-Gruneisen-Debye EOS. [GPa]
        """
        # Gruneisen parameter at P, T conditions
        gamma_capv = self.gamma_capv_0 * x**(-self.q_capv)

        # Debye temperature at P, T conditions
        theta_capv = self.theta_capv_0 * np.exp(
            (self.gamma_capv_0 - gamma_capv) / self.q_capv
        )

        # The third-order Birch–Murnaghan isothermal equation of state
        BM3 = P_i - (
            1.5 * k_capv_0 * (x**(7/3) - x**(5/3)) *
            (1 + 3/4 * (self.k0t_prime_capv - 4) * (x**(2/3) - 1))
        )

        E_th = 9. * 2. * self.R * T_i**4 / theta_capv**3 * (
            self._integral_vibrational_energy(theta_capv / T_i)
        )
        E_th_0 = 9. * 2. * self.R * 300.**4 / theta_capv**3 * (
            self._integral_vibrational_energy(theta_capv / 300.)
        )

        # The Mie-Gruneisen-Debye equation of state
        MGD =  BM3  - gamma_capv * x / v_capv_0 * (E_th - E_th_0)

        return MGD


    def _integral_vibrational_energy(self, x_max):
        """Calculates the integral part of the vibrational energy.

        This function calculates the integral part of the vibrational energy and returns
        its value. It corresponds to the eq. (9) of Vilella et al. (2015).

        This function is necessary because the method scipy.integrate.quad returns both         the value and the error, while only the value is needed.

        Args:
            x_max: Value for the upper bound of the integral.

        Returns:
            Float64: Value of the integral.
        """
        value, err = scipy.integrate.quad(
            lambda x: x**3 / (np.exp(x) - 1), 0, x_max
        )
        return value


if __name__ == "__main__":
    # Building simple simulator
    LLSVP_compositions_simulator = MineralProperties()

    # Setting up dummy conditions
    conditions = {
         "dT_min" : 100, "dT_max": 200,
         "p_capv_min": 0.10, "p_capv_max": 0.20,
         "iron_content_min": 0.12, "iron_content_max": 0.14,
         "al_content_min": 0.05, "al_content_max": 0.06,
         "ratio_fe_bm_min": 0.5, "ratio_fe_bm_max":0.6
    }

    # Calculating mineral proeprties
    LLSVP_compositions_simulator.calc_mineral_properties(conditions)
