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
        n_capv_min: Minimum proportion of CaPv assumed for the considered compositions.
                    Default to 0. [vol%]
        n_capv_max: Maximum proportion of CaPv assumed for the considered compositions.
                    Default to 0.40. [vol%]
        delta_capv: Step value for the proportion of CaPv. Default to 0.01. [vol%]
        n_bm_min: Minimum proportion of Bm assumed for the considered compositions.
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
        # Gas constant
        self.R = 8.31446*10^(-3)

        # Parameters related to the spin transition
        self.xi = prop.get(xi, 1.5) # Sturhahn 2005
        self.delta_0 = prop.get(delta_0, 1.35) # Sturhahn 2005
        self.v_trans = prop.get(v_trans, 59.) # Fei 2007

        # Parameters of the considered ambient mantle
        self.n_capv_am = prop.get(n_capv_am, 0.07) # Irifune 1994, 2010
        self.iron_content_am = prop.get(iron_content_am, 0.08) # Irifune 2010
        self.n_bm_am = prop.get(n_bm_am, 0.75) # Irifune 2010
        self.al_content_am = prop.get(al_content_am, 0.036) # Irifune 2010
        self.kd_ref_am = prop.get(kd_ref_am, 0.5) # Piet 2016
        self.ratio_fe_bm_am = prop.get(ratio_fe_bm_am, 0.5) # Piet 2016

        # EOS parameters for Bridgmanite
        self.k0t_prime_bm = prop.get(k0t_prime_bm, 3.7) # Fiquet 2000
        self.theta_bm_0 = prop.get(theta_bm_0, 1100.) # Fiquet 2000
        self.gamma_bm_0 = prop.get(gamma_bm_0, 1.4) # Fiquet 2000
        self.q_bm = prop.get(q_bm, 1.4) # Fiquet 2000

        # EOS parameters for Ferropericlase
        self.k0t_prime_fp = prop.get(k0t_prime_fp, 4.0) # Jackson 1982
        self.theta_fp_0 = prop.get(theta_fp_0, 673) # Jackson 1982
        self.gamma_fp_0 = prop.get(gamma_fp_0, 1.41) # Jackson 1982
        self.q_fp = prop.get(q_fp, 1.3) # Jackson 1982

        # EOS parameters for Calcio Perovskite
        self.k0t_prime_capv = prop.get(k0t_prime_capv, 3.9) # Shim 2000
        self.theta_capv_0 = prop.get(theta_capv_0, 1000.) # Shim 2000
        self.gamma_capv_0 = prop.get(gamma_capv_0, 1.92) # Shim 2000
        self.q_capv = prop.get(q_capv, 0.6) # Shim 2000

        # EOS parameters for various components
        self.k_mgo_0 = prop.get(k_mgo_0, 160.) # Speziale 2001
        self.v_mgo_0 = prop.get(v_mgo_0, 11.25) # Speziale 2001
        self.k_feo_ls_0 = prop.get(k_feo_ls_0, 150.) # Fei 2007
        self.v_feo_ls_0 = prop.get(v_feo_ls_0, 10.82) # Fei 2007
        self.k_feo_hs_0 = prop.get(k_feo_hs_0, 150.) # Fei 2007
        self.v_feo_hs_0 = prop.get(v_feo_hs_0, 12.18) # Fei 2007
        self.k_mgsio3_0 = prop.get(k_mgsio3_0, 261.) # Lundin 2008
        self.v_mgsio3_0 = prop.get(v_mgsio3_0, 24.43.) # Lundin 2008
        self.k_fesio3_0 = prop.get(k_fesio3_0, 248.) # Lundin 2008
        self.v_fesio3_0 = prop.get(v_fesio3_0, 25.44) # Lundin 2008
        self.k_fe2o3_0 = prop.get(k_fe2o3_0, 95.) # Catalli 2010
        self.v_fe2o3_0 = prop.get(v_fe2o3_0, 30.6) # Catalli 2010
        self.k_fealo3_0 = prop.get(k_fealo3_0, 271.) # Catalli 2011
        self.v_fealo3_0 = prop.get(v_fealo3_0, 28.21) # Catalli 2011
        self.k_al2o3_0 = prop.get(k_al2o3_0, 137.) # Catalli 2011
        self.v_al2o3_0 = prop.get(v_al2o3_0, 26.74) # Catalli 2011
        self.k_casio3_0 = prop.get(k_casio3_0, 236.) # Shin 2000
        self.v_casio3_0 = prop.get(v_casio3_0, 27.45) # Shin 2000

        # Parameters related to Calcio Perovskite
        self.m_capv = 116.161
        self.rho_capv_0 = 1000. * m_capv/v_casio3_0

    def calc_mineral_properties(self, conditions={}):
        """Work in progress.
        """
        # Load simulation conditions
        _load_conditions(conditions)

    def _load_conditions(self, conditions={}):
        """Initializes the input conditions of the simulator.

        Default values are chosen in order to reproduce the results presented in
        Vilella et al. (2021).

        Args:
            conditions: A dictionary providing the values for input conditions.
                        Default to empty dictionary.
        """
        # P, T conditions of the ambient mantle
        self.P_am = conditions.get(P_am, 130.)
        self.T_am = conditions.get(T_an, 3000.)

        # Ranges considered for all input parameters
        self.n_capv_min = conditions.get(n_capv_min, 0.00)
        self.n_capv_max = conditions.get(n_capv_max, 0.40)
        self.delta_capv = conditions.get(delta_capv, 0.01)

        self.n_bm_min = conditions.get(n_bm_min, 0.60)
        self.delta_bm = conditions.get(delta_bm, 0.01)

        self.dT_min = conditions.get(dT_min, 0.)
        self.dT_max = conditions.get(dT_max, 1000.)
        self.delta_dT = conditions.get(delta_dT, 100.)

        self.iron_content_min = conditions.get(iron_content_min, 0.08)
        self.iron_content_max = conditions.get(iron_content_max, 0.28)
        self.delta_iron_content = conditions.get(delta_iron_content, 0.01)

        self.al_content_min = conditions.get(al_content_min, 0.01)
        self.al_content_max = conditions.get(al_content_max, 0.19)
        self.delta_al = conditions.get(delta_al, 0.005)

        self.ratio_fe_bm_min = conditions.get(ratio_fe_bm_min, 0.)
        self.ratio_fe_bm_max = conditions.get(ratio_fe_bm_max, 1.0)
        self.delta_ratio_fe = conditions.get(delta_ratio_fe, 0.1)

        # Stepping used for the spin transition calculation
        self.delta_v = conditions.get(delta_v, 0.05)
        self.delta_P = conditions.get(delta_P, 0.5)