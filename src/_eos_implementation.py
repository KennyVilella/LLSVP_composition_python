"""Provides utility functions to calculate the properties of the considered minerals.

This file is associated with the article:
"Constraints on the composition and temperature of LLSVPs from seismic properties of
lower mantle minerals" by K. Vilella, T. Bodin, C.-E. Boukare, F. Deschamps, J. Badro,
M. D. Ballmer, and Y. Li

This file provides a class for each considered minerals (Ferropericlase, Bridgmanite,
Calcio Perovskite) containing all the utility functions required to calculate their
properties.

These functions are mainly derived from the Mie-Gruneisen-Debye equation of state (EOS).
A thorough description of this EOS can be found in Jackson and Rigden (1996) and in the
supplementary material of Vilella et al. (2021).

These functions should not be used outside the class MineralProperties.

Typical usage example:

  from _eos_implementation import _EOS_fp
  fp_eos = _EOS_fp()
  eq_MGD_fp = fp_eos._MGD(self, 3000, 120, 11.0, 0.5, 0.2)

Copyright, 2023,  Vilella Kenny.
"""
import numpy as np
import scipy.integrate
from abc import ABC, abstractmethod
#======================================================================================#
#                                                                                      #
#   Starting implementation of the abstract class for mineral properties calculation   #
#                                                                                      #
#======================================================================================#
class _EOS(ABC):
    """Provides base utilities to implement the Mie-Gruneisen-Debye equation of state.

    This class provides the basic functionality required to implement the
    Mie-Gruneisen-Debye equation of state.

    The abstract methods are then applied to each mineral in their corresponding derived
    class.

    The formalism for the Mie-Gruneisen-Debye equation of state used in this class can
    be found in Jackson and Rigden (1996).
    """
    @abstractmethod
    def _gamma(self, gamma_0, v_ratio, q):
        """Calculates the Gruneisen parameter.

        It corresponds to the eq. (31) of Jackson and Rigden (1996).

        Args:
            gamma_0: Gruneisen parameter at ambient conditions.
            v_ratio: Volume ratio V0 / V, where V0 is the volume at ambient conditions
                     and V the volume at the considered conditions.
            q: Exponent of the Gruneisen parameter.

        Returns:
            Float64: Gruneisen parameter at the considered conditions.
        """
        return gamma_0 * v_ratio**(-q)

    @abstractmethod
    def _theta(self, theta_0, gamma_0, gamma, q):
        """Calculates the Debye temperature.

        It corresponds to the eq. (32) of Jackson and Rigden (1996).

        Args:
            theta_0: Debye temperature at ambient conditions. [K]
            gamma_0: Gruneisen parameter at ambient conditions.
            gamma: Gruneisen parameter at the considered conditions.
            q: Exponent of the Gruneisen parameter.

        Returns:
            Float64: Debye temperature at the considered conditions. [K]
        """
        return theta_0 * np.exp((gamma_0 - gamma) / q)

    @abstractmethod
    def _BM3(self, P, k_0, v_ratio, k0t_prime):
        """Calculates the third-order Birch–Murnaghan isothermal equation of state.

        It corresponds to the eq. (B1) of Jackson and Rigden (1996).

        Args:
            P: Considered pressure. [GPa]
            k_0: Bulk modulus at ambient conditions. [GPa]
            v_ratio: Volume ratio V0 / V, where V0 is the volume at ambient conditions
                     and V the volume at the considered conditions.
            k0t_prime: Pressure derivative of the bulk modulus at ambient conditions.

        Returns:
            Float64: Residue of the third-order Birch–Murnaghan isothermal
                     equation of state. [GPa]
        """
        return (P -
            1.5 * k_0 * (v_ratio**(7/3) - v_ratio**(5/3)) *
            (1 + 3/4 * (k0t_prime - 4) * (v_ratio**(2/3) - 1))
        )

    @abstractmethod
    def _E_th(self, n, R, T, theta, int_part):
        """Calculates the vibrational energy.

        The integral part of the expression is calculated separately because the method 
        scipy.integrate.quad returns both the value and the error, while only the value
        is needed.

        It corresponds to the eq. (30) of Jackson and Rigden (1996).

        Args:
            n: Number of atoms per formula unit.
            R: Gas constant. [cm^3 GPa K^−1 mol^−1]
            T: Considered temperature. [K]
            theta: Debye temperature at the considered conditions. [K]
            int_part: Integral part of the vibrational energy.

        Returns:
            Float64: Vibrational energy at the considered conditions. [cm^3 GPa mol^−1]
        """
        return 9. * n * R * T**4 / theta**3 * int_part

    @abstractmethod
    def _MGD(self, BM3, gamma, v, E_th, E_th_0):
        """Calculates the Mie-Gruneisen-Debye equation of state.

       It corresponds to the eq. (33) of Jackson and Rigden (1996).

        Args:
            BM3: Residue of the third-order Birch–Murnaghan isothermal
                 equation of state. [GPa]
            gamma: Gruneisen parameter at the considered conditions.
            v: Volume at considered conditions. [cm^3/mol]
            E_th: Vibrational energy at the considered conditions. [cm^3 GPa mol^−1]
            E_th_0: Vibrational energy at ambient conditions. [cm^3 GPa mol^−1]

        Returns:
            Float64: Residue of the Mie-Gruneisen-Debye equation of state. [GPa]
        """
        return BM3  - gamma / v * (E_th - E_th_0)

    def _integral_vibrational_energy(self, x_max):
        """Calculates the integral part of the vibrational energy.

        It corresponds to the eq. (30) of Jackson and Rigden (1996).

        Args:
            x_max: Value for the upper bound of the integral.

        Returns:
            Float64: Value of the integral.
        """
        value, err = scipy.integrate.quad(
            lambda x: x**3 / (np.exp(x) - 1), 0, x_max
        )
        return value


class _EOS_fp(_EOS):
    """
    """
    def _v_feo_0(self, data, eta_ls):
        return eta_ls * data.v_feo_ls_0 + (1 - eta_ls) * data.v_feo_hs_0

    def _v_fp_0(self, data, eta_ls, x_fp):
        v_feo_0 = self. _v_feo_0(data, eta_ls)
        return x_fp * v_feo_0 + (1 - x_fp) * data.v_mgo_0

    def _k_VRH_average(self, x_1, v_1, k_1, v_2, k_2):
        """Calculates the Voigt-Reuss-Hill average of the bulk modulus."""
        x_2 = 1 - x_1
        v_tot = x_1 * v_1 + x_2 * v_2
        k_v = x_1 * v_1 / v_tot * k_1 + x_2 * v_2 / v_tot * k_2
        k_r = v_tot / (x_1 * v_1 / k_1 + x_2 * v_2 / k_2)
        return 0.5 * (k_v + k_r)

    def _k_feo_0_VRH_average(self, data, eta_ls):
        return self._k_VRH_average(
            eta_ls, data.v_feo_ls_0, data.k_feo_ls_0, data.v_feo_hs_0, data.k_feo_hs_0
        )

    def _k_fp_0_VRH_average(self, data, eta_ls, x_fp):
        # Volume of FeO at ambient conditions
        v_feo_0 = self._v_feo_0(data, eta_ls)
        # Bulk modulus of FeO at ambient conditions
        k_feo_0 = self._k_feo_0_VRH_average(data, eta_ls)
        return self._k_VRH_average(
            x_fp, v_feo_0, k_feo_0, data.v_mgo_0, data.k_mgo_0
        )

    def _gamma(self, data, v_ratio):
        return super()._gamma(data.gamma_fp_0, v_ratio, data.q_fp)

    def _theta(self, data, v_ratio):
        # Gruneisen parameter
        gamma_fp = self._gamma(data, v_ratio)
        return super()._theta(data.theta_fp_0, data.gamma_fp_0, gamma_fp, data.q_fp)

    def _BM3(self, data, P, k_fp_0, v_ratio):
        return super()._BM3(P, k_fp_0, v_ratio, data.k0t_prime_fp)

    def _E_th(self, data, T, v_ratio):
        # Debye temperature
        theta_fp = self._theta(data, v_ratio)
        # Integral part of the vibrational energy
        int_part_fp = super()._integral_vibrational_energy(theta_fp / T)
        return super()._E_th(2, data.R, T, theta_fp, int_part_fp)

    def _MGD(self, data, T, P, v_fp, eta_ls, x_fp):
        """Implements the Mie-Gruneisen-Debye EOS for Ferropericlase.

        This function calculates the residue of the Mie-Gruneisen-Debye equation of
        state applied to Ferroepriclase (Fp). It can be used to obtain one of the
        conditions among the pressure, temperature, volume, knowing the remaining two.

        It corresponds to the eq. (33) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            T: Considered temperature. [K]
            P: Considered pressure. [GPa]
            v_fp: Volume of Fp at considered conditions. [cm^3/mol]
            eta_ls: Average proportion of FeO in the low spin state.
            x_fp: Molar concentration of FeO in Fp.

        Returns:
            Float64: The residue of the Mie-Gruneisen-Debye EOS. [GPa]
        """
        # Bulk modulus of Fp at ambient conditions
        k_fp_0 = self._k_fp_0_VRH_average(data, eta_ls, x_fp)
        # Calculate volume of Fp at ambient conditions
        v_fp_0 = self._v_fp_0(data, eta_ls, x_fp)
        # Calculate volume ratio
        v_ratio = v_fp_0 / v_fp
        # Gruneisen parameter
        gamma_fp = self._gamma(data, v_ratio)
        # Third-order Birch–Murnaghan isothermal equation of state
        BM3_fp = self._BM3(data, P, k_fp_0, v_ratio)
        # Vibrational energy at T
        E_th_fp = self._E_th(data, T, v_ratio)
        # Vibrational energy at ambient conditions
        E_th_fp_0 = self._E_th(data, 300, v_ratio)
        return super()._MGD(BM3_fp, gamma_fp, v_fp, E_th_fp, E_th_fp_0)


class _EOS_bm(_EOS):
    """
    """
    def _v_bm_0(self, data, x_mgsio3, x_fesio3, x_fealo3, x_fe2o3, x_al2o3):
        v_bm_0 = (
            x_mgsio3 * data.v_mgsio3_0 + x_fesio3 * data.v_fesio3_0 +
            x_fealo3 * data.v_fealo3_0 + x_al2o3 * data.v_al2o3_0 +
            x_fe2o3 * data.v_fe2o3_0
        )
        return v_bm_0

    def _k_bm_0_VRH_average(
        self, data, x_mgsio3, x_fesio3, x_fealo3, x_fe2o3, x_al2o3, v_tot
    ):
        """Calculates the Voigt-Reuss-Hill average of the bulk modulus."""
        k_v = (
            x_mgsio3 * data.v_mgsio3_0 * data.k_mgsio3_0 +
            x_fesio3 * data.v_fesio3_0 * data.k_fesio3_0 +
            x_fealo3 * data.v_fealo3_0 * data.k_fealo3_0 +
            x_al2o3 * data.v_al2o3_0 * data.k_al2o3_0 +
            x_fe2o3 * data.v_fe2o3_0 * data.k_fe2o3_0
        ) / v_tot
        k_r = v_tot / (
            x_mgsio3 * data.v_mgsio3_0 / data.k_mgsio3_0 + 
            x_fesio3 * data.v_fesio3_0 / data.k_fesio3_0 +
            x_fealo3 * data.v_fealo3_0 / data.k_fealo3_0 +
            x_al2o3 * data.v_al2o3_0 / data.k_al2o3_0 +   
            x_fe2o3 * data.v_fe2o3_0 / data.k_fe2o3_0 
        )
        return 0.5 * (k_v + k_r)

    def _gamma(self, data, v_ratio):
        return super()._gamma(data.gamma_bm_0, v_ratio, data.q_bm)

    def _theta(self, data, v_ratio):
        # Gruneisen parameter
        gamma_bm = self._gamma(data, v_ratio)
        return super()._theta(
            data.theta_bm_0, data.gamma_bm_0, gamma_bm, data.q_bm
        )

    def _BM3(self, data, P, k_bm_0, v_ratio):
        return super()._BM3(P, k_bm_0, v_ratio, data.k0t_prime_bm)

    def _E_th(self, data, T, v_ratio):
        # Debye temperature
        theta_bm = self._theta(data, v_ratio)
        # Integral part of the vibrational energy
        int_part_bm = super()._integral_vibrational_energy(theta_bm / T)
        return super()._E_th(5, data.R, T, theta_bm, int_part_bm)

    def _MGD(self, data, T, P, v_bm, x_mgsio3, x_fesio3, x_fealo3, x_fe2o3, x_al2o3):
       """Implements the Mie-Gruneisen-Debye EOS for Bridgmanite.

        This function calculates the residue of the Mie-Gruneisen-Debye equation of
        state applied to Bridgmanite (Bm). It can be used to obtain one of the
        conditions among the pressure, temperature, volume, knowing the remaining two.

        It corresponds to the eq. (33) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            T: Considered temperature. [K]
            P: Considered pressure. [GPa]
            v_bm: Volume of Bm at considered conditions. [cm^3/mol]
            x_mgsio3: Molar concentration of MgSiO3 in Bm.
            x_fesio3: Molar concentration of FeSiO3 in Bm.
            x_fealo3: Molar concentration of FeAlO3 in Bm.
            x_fe2o3: Molar concentration of Fe2O3 in Bm.
            x_al2o3: Molar concentration of Al2O3 in Bm.

        Returns:
            Float64: The residue of the Mie-Gruneisen-Debye EOS. [GPa]
        """
        # Calculate volume of Bm at ambient conditions
        v_bm_0 = self._v_bm_0(data, x_mgsio3, x_fesio3, x_fealo3, x_fe2o3, x_al2o3)
        # Calculate volume ratio
        v_ratio = v_bm_0 / v_bm
        # Bulk modulus of Bm at ambient conditions
        k_bm_0 = self._k_bm_0_VRH_average(
            data, x_mgsio3, x_fesio3, x_fealo3, x_fe2o3, x_al2o3, v_bm_0
        )
        # Gruneisen parameter
        gamma_bm = self._gamma(data, v_ratio)
        # Third-order Birch–Murnaghan isothermal equation of state
        BM3_bm = self._BM3(data, P, k_bm_0, v_ratio)
        # Vibrational energy at T
        E_th_bm = self._E_th(data, T, v_ratio)
        # Vibrational energy at ambient conditions
        E_th_bm_0 = self._E_th(data, 300, v_ratio)
        return super()._MGD(BM3_bm, gamma_bm, v_bm, E_th_bm, E_th_bm_0)


class _EOS_capv(_EOS):
    """
    """
    def _gamma(self, data, v_ratio):
        return super()._gamma(data.gamma_capv_0, v_ratio, data.q_capv)

    def _theta(self, data, v_ratio):
        # Gruneisen parameter
        gamma_capv = self._gamma(data, v_ratio)
        return super()._theta(
            data.theta_capv_0, data.gamma_capv_0, gamma_capv, data.q_capv
        )

    def _BM3(self, data, P, v_ratio):
        return super()._BM3(P, data.k_casio3_0, v_ratio, data.k0t_prime_capv)

    def _E_th(self, data, T, v_ratio):
        # Debye temperature
        theta_capv = self._theta(data, v_ratio)
        # Integral part of the vibrational energy
        int_part_capv = super()._integral_vibrational_energy(theta_capv / T)
        return super()._E_th(5, data.R, T, theta_capv, int_part_capv)

    def _MGD(self, data, T, P, v_capv):
        """Implements the Mie-Gruneisen-Debye EOS for Calcio Perovskite.

        This function calculates the residue of the Mie-Gruneisen-Debye equation of
        state applied to Calcio Perovskite (CaPv). It can be used to obtain one of the
        conditions among the pressure, temperature, volume, knowing the remaining two.

        It corresponds to the eq. (33) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            T: Considered temperature. [K]
            P: Considered pressure. [GPa]
            v_capv: Volume of CaPv at considered conditions. [cm^3/mol]

        Returns:
            Float64: The residue of the Mie-Gruneisen-Debye EOS. [GPa]
        """
        # Calculate volume ratio
        v_ratio = data.v_casio3_0 / v_capv
        # Gruneisen parameter
        gamma_capv = self._gamma(data, v_ratio)
        # Third-order Birch–Murnaghan isothermal equation of state
        BM3_capv = self._BM3(data, P, v_ratio)
        # Vibrational energy at T
        E_th_capv = self._E_th(data, T, v_ratio)
        # Vibrational energy at ambient conditions
        E_th_capv_0 = self._E_th(data, 300, v_ratio)
        return super()._MGD(BM3_capv, gamma_capv, v_capv, E_th_capv, E_th_capv_0)
