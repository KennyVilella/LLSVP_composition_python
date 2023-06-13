"""Provides functions used to calculate the average spin state of FeO in Ferropericlase.

This file is associated with the article:
"Constraints on the composition and temperature of LLSVPs from seismic properties of
lower mantle minerals" by K. Vilella, T. Bodin, C.-E. Boukare, F. Deschamps, J. Badro,
M. D. Ballmer, and Y. Li

This class provides all the functions used to calculate the average spin state of FeO
in Ferropericlase (Fp) depending on temperature, pressure, and mineral composition.
The model is based on the work of Sturhahn et al. (2005) and this specific version is
described in Vilella et al. (2015).

The Mie-Gruneisen-Debye equation of state is used to calculate the properties of Fp.
This class should not be used indeoendently of the main class MineralProperties.

Typical usage example:

  LLSVP_compositions_simulator = MineralProperties()
  spin = SpinConfiguration(LLSVP_compositions_simulator)

Copyright, 2023,  Vilella Kenny.
"""
import numpy as np
import scipy.optimize
#======================================================================================#
#                                                                                      #
#         Starting implementation of class calculating the spin configuration          #
#                                                                                      #
#======================================================================================#
class SpinConfiguration:
    """Provides utilities to model the spin state transition in Ferropericlase

    This class provides the functionality required to calculate the average spin state
    of FeO in Ferropericlase (Fp) as a function of temperature, pressure and composition
    of Fp.

    The MineralProperties class is taken as input to pass the mineral properties to this
    class.

    Attributes:
        MP: The MineralProperties class.
    """
    def __init__(self, MineralProperties):
        """Initializes all the mineral properties.

        Attributes of the MineralProperties class are passed to this class and loaded.
        This enables to use all the mineral properties without providing them as
        arguments.

        Raises:
            ValiueError: The MineralProperties has not been provided as an argument.
        """
        if MineralProperties is None:
            raise ValueError("MineralProperties must be provided")

        self.MP = MineralProperties


    def _calc_volume_MGD(self, P_i, T_i, eta_ls, x_fp):
        """Calculates the volume of Fp using the Mie-Gruneisen-Debye EOS.

        This function calculates the volume of Ferropericlase (Fp) for the given
        conditions using the Mie-Gruneisen-Debye equation of state.
        It corresponds to the eq. (6) of Vilella et al. (2015).
        To do so, the fsolve method from the scipy.optimize package is used.

        The bulk modulus of the mineral is calculated from the bulk modulus of its
        individual components suing the Voigt-Reuss-Hill (VRH) average.

        Args:
            P_i: Considered pressure. [GPa]
            T_i: Considered temperature. [K]
            eta_ls: Average proportion of FeO in the low spin state.
            x_fp: FeO content in ferropericlase.

        Returns:
            Float64: Volume of Ferropericlase. [A^3]
        """
        # Calculating proportion of high spin state
        eta_hs = 1 - eta_ls

        # Calculating density of components at ambient conditions
        rho_mgo_0 = (24.305 + 15.999) / self.MP.v_mgo_0
        rho_feo_hs_0  = (55.845 + 15.999) / self.MP.v_feo_hs_0
        rho_feo_ls_0  = (55.845 + 15.999) / self.MP.v_feo_ls_0
        rho_feo_0 = eta_hs * rho_feo_hs_0 + eta_ls * rho_feo_ls_0
        rho_fp_0 = (1 - x_fp) * rho_mgo_0 + x_fp * rho_feo_0

        # Calculating the bulk modulus of FeO at ambient conditions using the
        # VRH average
        k_feo_0_r = 1 / (
            eta_hs * rho_feo_hs_0 / (self.MP.k_feo_hs_0 * rho_feo_0) +
            eta_ls * rho_feo_ls_0 / (self.MP.k_feo_ls_0 * rho_feo_0)
        )
        k_feo_0_v = (
            eta_hs * rho_feo_hs_0 * self.MP.k_feo_hs_0 / rho_feo_0 +
            eta_ls * rho_feo_ls_0 * self.MP.k_feo_ls_0 / rho_feo_0
        )
        k_feo_0   = 0.5 * (k_feo_0_r + k_feo_0_r)

        # Calculating the bulk modulus of Fp at ambient conditions using the VRH average
        k_fp_0_r = 1 / (
            (1 - x_fp) * rho_mgo_0 / (self.MP.k_mgo_0 * rho_fp_0) +
            x_fp * rho_feo_0 / (self.MP.k_feo_hs_0 * rho_fp_0)
        )
        k_fp_0_v = (
            (1 - x_fp) * rho_mgo_0 * self.MP.k_mgo_0 / rho_fp_0 +
            x_fp * rho_feo_0 * self.MP.k_feo_hs_0 / rho_fp_0
        )
        k_fp_0 = 0.5 * (k_fp_0_r + k_fp_0_v)

        # Calculating volume at ambient conditions
        v_feo_0 = eta_ls * self.MP.v_feo_ls_0 + eta_hs * self.MP.v_feo_hs_0
        v_fp_0 = x_fp * v_feo_0 + (1 - x_fp) * self.MP.v_mgo_0

        solution = scipy.optimize.fsolve(
            lambda x: self.MP._MGD_fp(x, T_i, P_i, k_fp_0, v_fp_0), 1.5
        )
        v_fp = v_fp_0 / (0.15055 * solution)

        return v_fp[0]


    def _calc_pressure_MGD(self, T_i, v_fp, eta_ls, x_fp):
        """Calculates the pressure of Fp using the Mie-Gruneisen-Debye EOS.

        This function calculates the pressure of Ferropericlase (Fp) for the given
        conditions using the Mie-Gruneisen-Debye equation of state.
        It corresponds to the eq. (6) of Vilella et al. (2015).
        To do so, the function _MGD_fp is used with a zero pressure.

        The bulk modulus of the mineral is calculated from the bulk modulus of its
        individual components suing the Voigt-Reuss-Hill (VRH) average.

        Args:
            T_i: Considered temperature. [K]
            v_fp: Volume of Ferropericlase. [cm^3/mol]
            eta_ls: Average proportion of FeO in the low spin state.
            x_fp: FeO content in ferropericlase.

        Returns:
            Float64: Pressure of the mineral. [GPa]
        """
        # Calculating proportion of high spin state
        eta_hs = 1 - eta_ls

        # Calculating density of components at ambient conditions
        rho_mgo_0 = (24.305 + 15.999) / self.MP.v_mgo_0
        rho_feo_hs_0  = (55.845 + 15.999) / self.MP.v_feo_hs_0
        rho_feo_ls_0  = (55.845 + 15.999) / self.MP.v_feo_ls_0
        rho_feo_0 = eta_hs * rho_feo_hs_0 + eta_ls * rho_feo_ls_0
        rho_fp_0 = (1 - x_fp) * rho_mgo_0 + x_fp * rho_feo_0

        # Calculating the bulk modulus of FeO at ambient conditions using the
        # VRH average
        k_feo_0_r = 1 / (
            eta_hs * rho_feo_hs_0 / (self.MP.k_feo_hs_0 * rho_feo_0) +
            eta_ls * rho_feo_ls_0 / (self.MP.k_feo_ls_0 * rho_feo_0)
        )
        k_feo_0_v = (
            eta_hs * rho_feo_hs_0 * self.MP.k_feo_hs_0 / rho_feo_0 +
            eta_ls * rho_feo_ls_0 * self.MP.k_feo_ls_0 / rho_feo_0
        )
        k_feo_0   = 0.5 * (k_feo_0_r + k_feo_0_r)

        # Calculating the bulk modulus of Fp at ambient conditions using the VRH average
        k_fp_0_r = 1 / (
            (1 - x_fp) * rho_mgo_0 / (self.MP.k_mgo_0 * rho_fp_0) +
            x_fp * rho_feo_0 / (self.MP.k_feo_hs_0 * rho_fp_0)
        )
        k_fp_0_v = (
            (1 - x_fp) * rho_mgo_0 * self.MP.k_mgo_0 / rho_fp_0 +
            x_fp * rho_feo_0 * self.MP.k_feo_hs_0 / rho_fp_0
        )
        k_fp_0 = 0.5 * (k_fp_0_r + k_fp_0_v)

        # Calculating volume at ambient conditions
        v_feo_0 = eta_ls * self.MP.v_feo_ls_0 + eta_hs * self.MP.v_feo_hs_0
        v_fp_0 = x_fp * v_feo_0 + (1 - x_fp) * self.MP.v_mgo_0

        return -self.MP._MGD_fp(v_fp_0 / v_fp, T_i, 0.0, k_fp_0, v_fp_0)


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
        pairing_energy = self.MP.delta_0 * (v_0 / self.MP.v_trans)**self.MP.xi
            
        # Calculating energy difference between the two energy levels
        delta_energy = self.MP.delta_0 * (v_0 / v)**self.MP.xi

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
        delta_energy = self.MP.delta_0 * (v_0 / v)**self.MP.xi

        return x_fp**self.MP.xi * delta_energy
