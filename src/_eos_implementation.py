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
supplementary material of Vilella et al. (2021). The shear modulus calculation follows
the model presented in Bina and Helffrich (1992).

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
    Mie-Gruneisen-Debye equation of state and to calculate the shear modulus.

    The abstract methods are then applied to each mineral in their corresponding derived
    class.

    The formalism for the Mie-Gruneisen-Debye equation of state used in this class can
    be found in Jackson and Rigden (1996), while the shear modulus calculation can be
    found in Bina and Helffrich (1992).
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
    def _E_th_dv(self, n, R, T, theta, gamma, v, E_th, E_th_0):
        """
        """
        d_int_part = 9 * n * R * (
            1 / (np.exp(theta / T) - 1) - 1 / (np.exp(theta / 300) - 1)
        )
        return (gamma * theta / v) * (3 * (E_th - E_th_0) / theta - d_int_part)

    @abstractmethod
    def _E_th_dT(self, n, R, T, theta, E_th):
        """
        """
        return 4 * E_th / T - 9 * n * R * (theta / T) / (np.exp(theta / T) - 1)

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

    @abstractmethod
    def _alpha(self, k_0, gamma, q, v, E_th, E_th_0, E_th_dv, E_th_dT):
        """
        """
        return gamma / v * E_th_dT / (
            k_0 - (q - 1) * gamma * (E_th - E_th_0) / v - gamma * E_th_dv
        )

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

    @abstractmethod
    def _g_t0(self, g_0, g_prime, k_0, v_ratio):
        """
        """
        return v_ratio**(5/3) * (
            g_0 + (1 - v_ratio**(2/3)) * 0.5 * (5 * g_0 - 3 * k_0 * g_prime)
        )

    @abstractmethod
    def _k_s(
        self, T, k_0, k0t_prime, gamma, q, alpha, v, v_ratio, E_th, E_th_0, E_th_dv
    ):
        """
        """
        # Bulk modulus at ambient temperature and considered pressure/volume
        k_v = k_0 * (
            (v_ratio**(7/3) - v_ratio**(5/3)) * 3/4 * (k0t_prime - 4) * v_ratio**(2/3) +
            0.5 * (7 * v_ratio**(7/3) - 5 * v_ratio**(5/3)) *
            (1 + 3/4 * (k0t_prime - 4) * (v_ratio**(2/3) - 1))
        )
        # Bulk modulus at considered temperature/pressure/volume
        k_t = k_v - (q - 1) * gamma / v * (E_th - E_th_0) - gamma * E_th_dv
        return k_t * (1 + alpha * gamma * T)

#======================================================================================#
#                                                                                      #
#   Starting implementation of the class for calculation of Ferropericlase properties  #
#                                                                                      #
#======================================================================================#
class _EOS_fp(_EOS):
    """Implement the Mie-Gruneisen-Debye equation of state for Ferropericlase.

    This class is derived from the abstract class _EOS, which implements the basic
    functionality required to implement the Mie-Gruneisen-Debye equation of state.

    The formalism for the Mie-Gruneisen-Debye equation of state used in this class can
    be found in Jackson and Rigden (1996), while the shear modulus calculation can be
    found in Bina and Helffrich (1992).

    This class should not be used outside the class MineralProperties.
    """
    def _v_feo_0(self, data, eta_ls):
        """Calculates the volume of FeO at ambient conditions.

        Args:
            data: Data holder for the MineralProperties class.
            eta_ls: Average proportion of FeO in the low spin state.

        Returns:
            Float64: Volume of FeO at ambient conditions. [cm^3/mol]
        """
        return eta_ls * data.v_feo_ls_0 + (1 - eta_ls) * data.v_feo_hs_0

    def _v_fp_0(self, data, eta_ls, x_fp):
        """Calculates the volume of Ferropericlase at ambient conditions.

        Args:
            data: Data holder for the MineralProperties class.
            eta_ls: Average proportion of FeO in the low spin state.
            x_fp: Molar concentration of FeO in Fp.

        Returns:
            Float64: Volume of Fp at ambient conditions. [cm^3/mol]
        """
        v_feo_0 = self. _v_feo_0(data, eta_ls)
        return x_fp * v_feo_0 + (1 - x_fp) * data.v_mgo_0

    def _VRH_average(self, x_1, v_1, c_1, v_2, c_2):
        """Calculates the Voigt-Reuss-Hill average.

        Implement the Voigt-Reuss-Hill average for a mixture of two components. This is
        traditionally used to calculate the elasticity of polycrystal from the
        elasticity of its individual components. Here, it is used to calculate the bulk
        modulus or shear modulus of Ferropericlase.

        Note that the volume can be given in any unit as long as the two volumes are
        given in the same unit.

        Args:
            x_1: Molar concentraion of the first component.
            v_1: Volume of the first component.
            c_1: Bulk or shear modulus of the first component. [GPa]
            v_2: Volume of the second component.
            c_2: Bulk or shear modulus of the second component. [GPa]

        Returns:
            Float64: Voigt-Reuss-Hill average of the bulk or shear modulus. [GPa]
        """
        x_2 = 1 - x_1
        v_tot = x_1 * v_1 + x_2 * v_2
        c_v = x_1 * v_1 / v_tot * c_1 + x_2 * v_2 / v_tot * c_2
        c_r = v_tot / (x_1 * v_1 / c_1 + x_2 * v_2 / c_2)
        return 0.5 * (c_v + c_r)

    def _k_feo_0_VRH_average(self, data, eta_ls):
        """Calculates the bulk modulus of FeO in Ferropericlase at ambient conditions.

        This function calculates the bulk modulus of FeO at ambient conditions using
        the Voigt-Reuss-Hill average. It is assumed that the FeO in Ferropericlase is a
        mixture of FeO in a low spin state and FeO in a high spin state.

        Args:
            data: Data holder for the MineralProperties class.
            eta_ls: Average proportion of FeO in the low spin state.

        Returns:
            Float64: Bulk modulus of FeO in Fp at ambient conditions. [GPa]
        """
        return self._VRH_average(
            eta_ls, data.v_feo_ls_0, data.k_feo_ls_0, data.v_feo_hs_0, data.k_feo_hs_0
        )

    def _g_feo_0_VRH_average(self, data, eta_ls):
        """Calculates the shear modulus of FeO in Ferropericlase at ambient conditions.

        This function calculates the shear modulus of FeO at ambient conditions using
        the Voigt-Reuss-Hill average. It is assumed that the FeO in Ferropericlase is a
        mixture of FeO in a low spin state and FeO in a high spin state.

        Args:
            data: Data holder for the MineralProperties class.
            eta_ls: Average proportion of FeO in the low spin state.

        Returns:
            Float64: Shear modulus of FeO in Fp at ambient conditions. [GPa]
        """
        return self._VRH_average(
            eta_ls, data.v_feo_ls_0, data.g_feo_ls_0, data.v_feo_hs_0, data.g_feo_hs_0
        )

    def _g_prime_feo_VRH_average(self, data, eta_ls):
        """Calculates the pressure derivative of the shear modulus for FeO in Fp.

        This function calculates the pressure derivative of the shear modulus for FeO
        using the Voigt-Reuss-Hill average. It is assumed that the FeO in Ferropericlase
        is a mixture of FeO in a low spin state and FeO in a high spin state.

        Args:
            data: Data holder for the MineralProperties class.
            eta_ls: Average proportion of FeO in the low spin state.

        Returns:
            Float64: Pressure derivative of the shear modulus for FeO in Fp.
        """
        return self._VRH_average(
            eta_ls, data.v_feo_ls_0, data.g_prime_feo_ls, data.v_feo_hs_0,
            data.g_prime_feo_hs
        )

    def _k_fp_0_VRH_average(self, data, eta_ls, x_fp):
        """Calculates the bulk modulus of Ferropericlase at ambient conditions.

        This function calculates the bulk modulus of Fp at ambient conditions using
        the Voigt-Reuss-Hill average. It is assumed that Fp is a mixture of FeO and MgO.

        Args:
            data: Data holder for the MineralProperties class.
            eta_ls: Average proportion of FeO in the low spin state.
            x_fp: Molar concentration of FeO in Fp.

        Returns:
            Float64: Bulk modulus of Fp at ambient conditions. [GPa]
        """
        # Volume of FeO at ambient conditions
        v_feo_0 = self._v_feo_0(data, eta_ls)
        # Bulk modulus of FeO at ambient conditions
        k_feo_0 = self._k_feo_0_VRH_average(data, eta_ls)
        return self._VRH_average(
            x_fp, v_feo_0, k_feo_0, data.v_mgo_0, data.k_mgo_0
        )

    def _g_fp_0_VRH_average(self, data, eta_ls, x_fp):
        """Calculates the shear modulus of Ferropericlase at ambient conditions.

        This function calculates the shear modulus of Fp at ambient conditions using
        the Voigt-Reuss-Hill average. It is assumed that Fp is a mixture of FeO and MgO.

        Args:
            data: Data holder for the MineralProperties class.
            eta_ls: Average proportion of FeO in the low spin state.
            x_fp: Molar concentration of FeO in Fp.

        Returns:
            Float64: Shear modulus of Fp at ambient conditions. [GPa]
        """
        # Volume of FeO at ambient conditions
        v_feo_0 = self._v_feo_0(data, eta_ls)
        # Shear modulus of FeO at ambient conditions
        g_feo_0 = self._g_feo_0_VRH_average(data, eta_ls)
        return self._VRH_average(
            x_fp, v_feo_0, g_feo_0, data.v_mgo_0, data.g_mgo_0
        )

    def _g_prime_fp_VRH_average(self, data, eta_ls, x_fp):
        """Calculates the pressure derivative of the shear modulus for Ferropericlase.

        This function calculates the pressure derivative of the shear modulus for Fp
        using the Voigt-Reuss-Hill average. It is assumed that Fp is a mixture of FeO
        and MgO.

        Args:
            data: Data holder for the MineralProperties class.
            eta_ls: Average proportion of FeO in the low spin state.
            x_fp: Molar concentration of FeO in Fp.

        Returns:
            Float64: Pressure derivative of the shear modulus for Fp.
        """
        # Volume of FeO at ambient conditions
        v_feo_0 = self._v_feo_0(data, eta_ls)
        # Pressure derivative of the shear modulus for FeO
        g_prime_feo = self._g_prime_feo_VRH_average(data, eta_ls)
        return self._VRH_average(
            x_fp, v_feo_0, g_prime_feo, data.v_mgo_0, data.g_prime_mgo
        )

    def _gamma(self, data, v_ratio):
        """Calculates the Gruneisen parameter of Ferropericlase.

        It corresponds to the eq. (31) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            v_ratio: Volume ratio V0 / V of Fp, where V0 is the volume of Fp at ambient
                     conditions and V the volume of Fp at the considered conditions.

        Returns:
            Float64: Gruneisen parameter of Fp at the considered conditions.
        """
        return super()._gamma(data.gamma_fp_0, v_ratio, data.q_fp)

    def _theta(self, data, v_ratio):
        """Calculates the Debye temperature of Ferropericlase.

        It corresponds to the eq. (32) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            v_ratio: Volume ratio V0 / V of Fp, where V0 is the volume of Fp at ambient
                     conditions and V the volume of Fp at the considered conditions.

        Returns:
            Float64: Debye temperature of Fp at the considered conditions. [K]
        """
        # Gruneisen parameter
        gamma_fp = self._gamma(data, v_ratio)
        return super()._theta(data.theta_fp_0, data.gamma_fp_0, gamma_fp, data.q_fp)

    def _BM3(self, data, P, k_fp_0, v_ratio):
        """Implements the third-order Birch–Murnaghan isothermal EOS for Ferropericlase.

        This function calculates the residue of the third-order Birch–Murnaghan
        isothermal equation of state applied to Ferroepriclase (Fp).

        It corresponds to the eq. (B1) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            P: Considered pressure. [GPa]
            k_fp_0: Bulk modulus of Fp at ambient conditions. [GPa]
            v_ratio: Volume ratio V0 / V of Fp, where V0 is the volume of Fp at ambient
                     conditions and V the volume of Fp at the considered conditions.

        Returns:
            Float64: Residue of the third-order Birch–Murnaghan isothermal
                     equation of state for Fp. [GPa]
        """
        return super()._BM3(P, k_fp_0, v_ratio, data.k0t_prime_fp)

    def _E_th(self, data, T, v_ratio):
        """Calculates the vibrational energy of Ferropericlase.

        The integral part of the expression is calculated separately because the method 
        scipy.integrate.quad returns both the value and the error, while only the value
        is needed.

        It corresponds to the eq. (30) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            T: Considered temperature. [K]
            v_ratio: Volume ratio V0 / V of Fp, where V0 is the volume of Fp at ambient
                     conditions and V the volume of Fp at the considered conditions.

        Returns:
            Float64: Vibrational energy of Fp at the considered conditions.
                     [cm^3 GPa mol^−1]
        """
        # Debye temperature
        theta_fp = self._theta(data, v_ratio)
        # Integral part of the vibrational energy
        int_part_fp = super()._integral_vibrational_energy(theta_fp / T)
        return super()._E_th(2, data.R, T, theta_fp, int_part_fp)

    def _E_th_dv(self, data, T, theta_fp, gamma_fp, v_fp, E_th_fp, E_th_fp_0):
        """
        """
        return super()._E_th_dv(
            2, data.R, T, theta_fp, gamma_fp, v_fp, E_th_fp, E_th_fp_0
        )

    def _E_th_dT(self, data, T, theta_fp, E_th_fp):
        """
        """
        return super()._E_th_dT(2, data.R, T, theta_fp, E_th_fp)

    def _alpha(self, data, T, k_fp_0, theta_fp, gamma_fp, v_fp, E_th_fp, E_th_fp_0):
        """
        """
        # Partial derivative of the vibrational energy with respect to volume
        E_th_fp_dv = self._E_th_dv(
            data, T, theta_fp, gamma_fp, v_fp, E_th_fp, E_th_fp_0
        )
        # Partial derivative of the vibrational energy with respect to temperature
        E_th_fp_dT = self._E_th_dT(data, T, theta_fp, E_th_fp)
        return super()._alpha(
            k_fp_0, gamma_fp, data.q_fp, v_fp, E_th_fp, E_th_fp_0, E_th_fp_dv,
            E_th_fp_dT
        )

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
            Float64: Residue of the Mie-Gruneisen-Debye EOS for Fp. [GPa]
        """
        # Bulk modulus of Fp at ambient conditions
        k_fp_0 = self._k_fp_0_VRH_average(data, eta_ls, x_fp)
        # Volume of Fp at ambient conditions
        v_fp_0 = self._v_fp_0(data, eta_ls, x_fp)
        # Volume ratio
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

    def _g_t0(self, data, v_fp, eta_ls, x_fp):
        """
        """
        # Volume of Fp at ambient conditions
        v_fp_0 = self._v_fp_0(data, eta_ls, x_fp)
        # Volume ratio
        v_ratio = v_fp_0 / v_fp
        # Bulk modulus of Fp at ambient conditions
        k_fp_0 = self._k_fp_0_VRH_average(data, eta_ls, x_fp)
        # Shear modulus of Fp at ambient conditions
        g_fp_0 = self._g_fp_0_VRH_average(data, eta_ls, x_fp)
        # Pressure derivative of the shear modulus for Fp
        g_prime_fp = self._g_prime_fp_VRH_average(data, eta_ls, x_fp)
        return super()._g_t0(g_fp_0, g_prime_fp, k_fp_0, v_ratio)

    def _g(self, data, T, v_fp, eta_ls, x_fp):
        """
        """
        # Shear modulus at ambient temperature
        g_fp_t0 = self._g_t0(data, v_fp, eta_ls, x_fp)
        return g_fp_t0 + data.g_dot_fp * (T - 300)

    def _k_s(self, data, T, v_fp, eta_ls, x_fp):
        """
        """
        # Volume of Fp at ambient conditions
        v_fp_0 = self._v_fp_0(data, eta_ls, x_fp)
        # Volume ratio
        v_ratio = v_fp_0 / v_fp
        # Gruneisen parameter
        gamma_fp = self._gamma(data, v_ratio)
        # Debye temperature
        theta_fp = self._theta(data, v_ratio)
        # Bulk modulus of Fp at ambient conditions
        k_fp_0 = self._k_fp_0_VRH_average(data, eta_ls, x_fp)
        # Vibrational energy at T
        E_th_fp = self._E_th(data, T, v_ratio)
        # Vibrational energy at ambient conditions
        E_th_fp_0 = self._E_th(data, 300, v_ratio)
        # Partial derivative of the vibrational energy with respect to volume
        E_th_fp_dv = self._E_th_dv(
            data, T, theta_fp, gamma_fp, v_fp, E_th_fp, E_th_fp_0
        )
        # Thermal expansion
        alpha_fp = self._alpha(
            data, T, k_fp_0, theta_fp, gamma_fp, v_fp, E_th_fp, E_th_fp_0
        )
        return super()._k_s(
            T, k_fp_0, data.k0t_prime_fp, gamma_fp, data.q_fp, alpha_fp, v_fp, v_ratio,
            E_th_fp, E_th_fp_0, E_th_fp_dv
        )

#======================================================================================#
#                                                                                      #
#    Starting implementation of the class for calculation of Bridgmanite properties    #
#                                                                                      #
#======================================================================================#
class _EOS_bm(_EOS):
    """Implement the Mie-Gruneisen-Debye equation of state for Bridgmanite.

    This class is derived from the abstract class _EOS, which implements the basic
    functionality required to implement the Mie-Gruneisen-Debye equation of state.

    The formalism for the Mie-Gruneisen-Debye equation of state used in this class can
    be found in Jackson and Rigden (1996) while the shear modulus calculation can be
    found in Bina and Helffrich (1992).

    This class should not be used outside the class MineralProperties.
    """
    def _v_bm_0(self, data, x_mgsio3, x_fesio3, x_fealo3, x_fe2o3, x_al2o3):
        """Calculates the volume of Bridgmanite at ambient conditions.

        Args:
            data: Data holder for the MineralProperties class.
            x_mgsio3: Molar concentration of MgSiO3 in Bm.
            x_fesio3: Molar concentration of FeSiO3 in Bm.
            x_fealo3: Molar concentration of FeAlO3 in Bm.
            x_fe2o3: Molar concentration of Fe2O3 in Bm.
            x_al2o3: Molar concentration of Al2O3 in Bm.

        Returns:
            Float64: Volume of Bm at ambient conditions. [cm^3/mol]
        """
        v_bm_0 = (
            x_mgsio3 * data.v_mgsio3_0 + x_fesio3 * data.v_fesio3_0 +
            x_fealo3 * data.v_fealo3_0 + x_al2o3 * data.v_al2o3_0 +
            x_fe2o3 * data.v_fe2o3_0
        )
        return v_bm_0

    def _k_bm_0_VRH_average(
        self, data, x_mgsio3, x_fesio3, x_fealo3, x_fe2o3, x_al2o3, v_tot
    ):
        """Calculates the bulk modulus of Bridgmanite at ambient conditions.

        This function calculates the bulk modulus of Bm at ambient conditions using
        the Voigt-Reuss-Hill average. The Voigt-Reuss-Hill average is traditionally
        used to calculate the elasticity of polycrystal from the elasticity of its
        individual components. Here, it is assumed that Bm is a mixture of MgSiO3, 
        FeSiO3, FeAlO3, Fe2O3, and Al2O3.

        Args:
            data: Data holder for the MineralProperties class.
            x_mgsio3: Molar concentration of MgSiO3 in Bm.
            x_fesio3: Molar concentration of FeSiO3 in Bm.
            x_fealo3: Molar concentration of FeAlO3 in Bm.
            x_fe2o3: Molar concentration of Fe2O3 in Bm.
            x_al2o3: Molar concentration of Al2O3 in Bm.
            v_tot: Volume of Bm at ambient conditions. [cm^3/mol]

        Returns:
            Float64: Bulk modulus of Bm at ambient conditions. [GPa]
        """
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

    def _g_bm_0_VRH_average(
        self, data, x_mgsio3, x_fesio3, x_fealo3, x_fe2o3, x_al2o3, v_tot
    ):
        """Calculates the shear modulus of Bridgmanite at ambient conditions.

        This function calculates the shear modulus of Bm at ambient conditions using
        the Voigt-Reuss-Hill average. The Voigt-Reuss-Hill average is traditionally
        used to calculate the elasticity of polycrystal from the elasticity of its
        individual components. Here, it is assumed that Bm is a mixture of MgSiO3, 
        FeSiO3, FeAlO3, Fe2O3, and Al2O3.

        Args:
            data: Data holder for the MineralProperties class.
            x_mgsio3: Molar concentration of MgSiO3 in Bm.
            x_fesio3: Molar concentration of FeSiO3 in Bm.
            x_fealo3: Molar concentration of FeAlO3 in Bm.
            x_fe2o3: Molar concentration of Fe2O3 in Bm.
            x_al2o3: Molar concentration of Al2O3 in Bm.
            v_tot: Volume of Bm at ambient conditions. [cm^3/mol]

        Returns:
            Float64: Shear modulus of Bm at ambient conditions. [GPa]
        """
        g_v = (
            x_mgsio3 * data.v_mgsio3_0 * data.g_mgsio3_0 +
            x_fesio3 * data.v_fesio3_0 * data.g_fesio3_0 +
            x_fealo3 * data.v_fealo3_0 * data.g_fealo3_0 +
            x_al2o3 * data.v_al2o3_0 * data.g_al2o3_0 +
            x_fe2o3 * data.v_fe2o3_0 * data.g_fe2o3_0
        ) / v_tot
        g_r = v_tot / (
            x_mgsio3 * data.v_mgsio3_0 / data.g_mgsio3_0 +
            x_fesio3 * data.v_fesio3_0 / data.g_fesio3_0 +
            x_fealo3 * data.v_fealo3_0 / data.g_fealo3_0 +
            x_al2o3 * data.v_al2o3_0 / data.g_al2o3_0 +
            x_fe2o3 * data.v_fe2o3_0 / data.g_fe2o3_0
        )
        return 0.5 * (g_v + g_r)

    def _g_prime_bm_VRH_average(
        self, data, x_mgsio3, x_fesio3, x_fealo3, x_fe2o3, x_al2o3, v_tot
    ):
        """Calculates the pressure derivative of the shear modulus for Bridgmanite.

        This function calculates the pressure derivative of the shear modulus for Bm
        using the Voigt-Reuss-Hill average. The Voigt-Reuss-Hill average is
        traditionally used to calculate the elasticity of polycrystal from the
        elasticity of its individual components. Here, it is assumed that Bm is a
        mixture of MgSiO3, FeSiO3, FeAlO3, Fe2O3, and Al2O3.

        Args:
            data: Data holder for the MineralProperties class.
            x_mgsio3: Molar concentration of MgSiO3 in Bm.
            x_fesio3: Molar concentration of FeSiO3 in Bm.
            x_fealo3: Molar concentration of FeAlO3 in Bm.
            x_fe2o3: Molar concentration of Fe2O3 in Bm.
            x_al2o3: Molar concentration of Al2O3 in Bm.
            v_tot: Volume of Bm at ambient conditions. [cm^3/mol]

        Returns:
            Float64: Pressure derivative of the shear modulus for Bm.
        """
        g_prime_v = (
            x_mgsio3 * data.v_mgsio3_0 * data.g_prime_mgsio3 +
            x_fesio3 * data.v_fesio3_0 * data.g_prime_fesio3 +
            x_fealo3 * data.v_fealo3_0 * data.g_prime_fealo3 +
            x_al2o3 * data.v_al2o3_0 * data.g_prime_al2o3 +
            x_fe2o3 * data.v_fe2o3_0 * data.g_prime_fe2o3
        ) / v_tot
        g_prime_r = v_tot / (
            x_mgsio3 * data.v_mgsio3_0 / data.g_prime_mgsio3 +
            x_fesio3 * data.v_fesio3_0 / data.g_prime_fesio3 +
            x_fealo3 * data.v_fealo3_0 / data.g_prime_fealo3 +
            x_al2o3 * data.v_al2o3_0 / data.g_prime_al2o3 +
            x_fe2o3 * data.v_fe2o3_0 / data.g_prime_fe2o3
        )
        return 0.5 * (g_prime_v + g_prime_r)

    def _gamma(self, data, v_ratio):
        """Calculates the Gruneisen parameter of Bridgmanite.

        It corresponds to the eq. (31) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            v_ratio: Volume ratio V0 / V of Bm, where V0 is the volume of Bm at ambient
                     conditions and V the volume of Bm at the considered conditions.

        Returns:
            Float64: Gruneisen parameter of Bm at the considered conditions.
        """
        return super()._gamma(data.gamma_bm_0, v_ratio, data.q_bm)

    def _theta(self, data, v_ratio):
        """Calculates the Debye temperature of Bridgmanite.

        It corresponds to the eq. (32) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            v_ratio: Volume ratio V0 / V of Bm, where V0 is the volume of Bm at ambient
                     conditions and V the volume of Bm at the considered conditions.

        Returns:
            Float64: Debye temperature of Bm at the considered conditions. [K]
        """
        # Gruneisen parameter
        gamma_bm = self._gamma(data, v_ratio)
        return super()._theta(
            data.theta_bm_0, data.gamma_bm_0, gamma_bm, data.q_bm
        )

    def _BM3(self, data, P, k_bm_0, v_ratio):
        """Implements the third-order Birch–Murnaghan isothermal EOS for Bridgmanite.

        This function calculates the residue of the third-order Birch–Murnaghan
        isothermal equation of state applied to Bridgmanite (Bm).

        It corresponds to the eq. (B1) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            P: Considered pressure. [GPa]
            k_bm_0: Bulk modulus of Bm at ambient conditions. [GPa]
            v_ratio: Volume ratio V0 / V of Bm, where V0 is the volume of Bm at ambient
                     conditions and V the volume of Bm at the considered conditions.

        Returns:
            Float64: Residue of the third-order Birch–Murnaghan isothermal
                     equation of state for Bm. [GPa]
        """
        return super()._BM3(P, k_bm_0, v_ratio, data.k0t_prime_bm)

    def _E_th(self, data, T, v_ratio):
        """Calculates the vibrational energy of Bridgmanite.

        The integral part of the expression is calculated separately because the method 
        scipy.integrate.quad returns both the value and the error, while only the value
        is needed.

        It corresponds to the eq. (30) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            T: Considered temperature. [K]
            v_ratio: Volume ratio V0 / V of Bm, where V0 is the volume of Bm at ambient
                     conditions and V the volume of Bm at the considered conditions.

        Returns:
            Float64: Vibrational energy of Bm at the considered conditions.
                     [cm^3 GPa mol^−1]
        """
        # Debye temperature
        theta_bm = self._theta(data, v_ratio)
        # Integral part of the vibrational energy
        int_part_bm = super()._integral_vibrational_energy(theta_bm / T)
        return super()._E_th(5, data.R, T, theta_bm, int_part_bm)

    def _E_th_dv(self, data, T, theta_bm, gamma_bm, v_bm, E_th_bm, E_th_bm_0):
        """
        """
        return super()._E_th_dv(
            5, data.R, T, theta_bm, gamma_bm, v_bm, E_th_bm, E_th_bm_0
        )

    def _E_th_dT(self, data, T, theta_bm, E_th_bm):
        """
        """
        return super()._E_th_dT(2, data.R, T, theta_bm, E_th_bm)

    def _alpha(self, data, T, k_bm_0, theta_bm, gamma_bm, v_bm, E_th_bm, E_th_bm_0):
        """
        """
        # Partial derivative of the vibrational energy with respect to volume
        E_th_bm_dv = self._E_th_dv(
            data, T, theta_bm, gamma_bm, v_bm, E_th_bm, E_th_bm_0
        )
        # Partial derivative of the vibrational energy with respect to temperature
        E_th_bm_dT = self._E_th_dT(data, T, theta_bm, E_th_bm)
        return super()._alpha(
            k_bm_0, gamma_bm, data.q_bm, v_bm, E_th_bm, E_th_bm_0, E_th_bm_dv,
            E_th_bm_dT
        )

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
            Float64: Residue of the Mie-Gruneisen-Debye EOS for Bm. [GPa]
        """
        # Volume of Bm at ambient conditions
        v_bm_0 = self._v_bm_0(data, x_mgsio3, x_fesio3, x_fealo3, x_fe2o3, x_al2o3)
        # Volume ratio
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

    def _g_t0(self, data, v_bm, x_mgsio3, x_fesio3, x_fealo3, x_fe2o3, x_al2o3):
        """
        """
        # Volume of Bm at ambient conditions
        v_bm_0 = self._v_bm_0(data, x_mgsio3, x_fesio3, x_fealo3, x_fe2o3, x_al2o3)
        # Volume ratio
        v_ratio = v_bm_0 / v_bm
        # Bulk modulus of Bm at ambient conditions
        k_bm_0 = self._k_bm_0_VRH_average(
            data, x_mgsio3, x_fesio3, x_fealo3, x_fe2o3, x_al2o3, v_bm_0
        )
        # Shear modulus of Bm at ambient conditions
        g_bm_0 = self._g_bm_0_VRH_average(
            data, x_mgsio3, x_fesio3, x_fealo3, x_fe2o3, x_al2o3, v_bm_0
        )
        # Pressure derivative of the shear modulus for Bm
        g_prime_bm = self._g_prime_bm_VRH_average(
            data, x_mgsio3, x_fesio3, x_fealo3, x_fe2o3, x_al2o3, v_bm_0
        )
        return super()._g_t0(g_bm_0, g_prime_bm, k_bm_0, v_ratio)

    def _g(self, data, T, v_bm, x_mgsio3, x_fesio3, x_fealo3, x_fe2o3, x_al2o3):
        """
        """
        # Shear modulus at ambient temperature
        g_bm_t0 = self._g_t0(data, v_bm, x_mgsio3, x_fesio3, x_fealo3, x_fe2o3, x_al2o3)
        return g_bm_t0 + data.g_dot_bm * (T - 300)

    def _k_s(self, data, T, v_bm, x_mgsio3, x_fesio3, x_fealo3, x_fe2o3, x_al2o3):
        """
        """
        # Volume of Bm at ambient conditions
        v_bm_0 = self._v_bm_0(data, x_mgsio3, x_fesio3, x_fealo3, x_fe2o3, x_al2o3)
        # Volume ratio
        v_ratio = v_bm_0 / v_bm
        # Gruneisen parameter
        gamma_bm = self._gamma(data, v_ratio)
        # Debye temperature
        theta_bm = self._theta(data, v_ratio)
        # Bulk modulus of Bm at ambient conditions
        k_bm_0 = self._k_bm_0_VRH_average(
            data, x_mgsio3, x_fesio3, x_fealo3, x_fe2o3, x_al2o3, v_bm_0
        )
        # Vibrational energy at T
        E_th_bm = self._E_th(data, T, v_ratio)
        # Vibrational energy at ambient conditions
        E_th_bm_0 = self._E_th(data, 300, v_ratio)
        # Partial derivative of the vibrational energy with respect to volume
        E_th_bm_dv = self._E_th_dv(
            data, T, theta_bm, gamma_bm, v_bm, E_th_bm, E_th_bm_0
        )
        # Thermal expansion
        alpha_bm = self._alpha(
            data, T, k_bm_0, theta_bm, gamma_bm, v_bm, E_th_bm, E_th_bm_0
        )
        return super()._k_s(
            T, k_bm_0, data.k0t_prime_bm, gamma_bm, data.q_bm, alpha_bm, v_bm, v_ratio,
            E_th_bm, E_th_bm_0, E_th_bm_dv
        )

#======================================================================================#
#                                                                                      #
# Starting implementation of the class for calculation of Calcio Perovskite properties #
#                                                                                      #
#======================================================================================#
class _EOS_capv(_EOS):
    """Implement the Mie-Gruneisen-Debye equation of state for Calcio Perovskite.

    This class is derived from the abstract class _EOS, which implements the basic
    functionality required to implement the Mie-Gruneisen-Debye equation of state.

    The formalism for the Mie-Gruneisen-Debye equation of state used in this class can
    be found in Jackson and Rigden (1996) while the shear modulus calculation can be
    found in Bina and Helffrich (1992).

    This class should not be used outside the class MineralProperties.
    """
    def _gamma(self, data, v_ratio):
        """Calculates the Gruneisen parameter of Calcio Perovskite.

        It corresponds to the eq. (31) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            v_ratio: Volume ratio V0 / V of CaPv, where V0 is the volume of CaPv at
                     ambient conditions and V the volume of CaPv at the considered
                     conditions.

        Returns:
            Float64: Gruneisen parameter of CaPv at the considered conditions.
        """
        return super()._gamma(data.gamma_capv_0, v_ratio, data.q_capv)

    def _theta(self, data, v_ratio):
        """Calculates the Debye temperature of Calcio Perovskite.

        It corresponds to the eq. (32) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            v_ratio: Volume ratio V0 / V of CaPv, where V0 is the volume of CaPv at
                     ambient conditions and V the volume of CaPv at the considered
                     conditions.

        Returns:
            Float64: Debye temperature of CaPv at the considered conditions. [K]
        """
        # Gruneisen parameter
        gamma_capv = self._gamma(data, v_ratio)
        return super()._theta(
            data.theta_capv_0, data.gamma_capv_0, gamma_capv, data.q_capv
        )

    def _BM3(self, data, P, v_ratio):
        """Implements the third-order Birch–Murnaghan EOS for Calcio Perovskite.

        This function calculates the residue of the third-order Birch–Murnaghan
        isothermal equation of state applied to Calcio Perovskite (CaPv).

        It corresponds to the eq. (B1) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            P: Considered pressure. [GPa]
            v_ratio: Volume ratio V0 / V of CaPv, where V0 is the volume of CaPv at
                     ambient conditions and V the volume of CaPv at the considered
                     conditions.

        Returns:
            Float64: Residue of the third-order Birch–Murnaghan isothermal
                     equation of state for CaPv. [GPa]
        """
        return super()._BM3(P, data.k_casio3_0, v_ratio, data.k0t_prime_capv)

    def _E_th(self, data, T, v_ratio):
        """Calculates the vibrational energy of Calcio Perovskite.

        The integral part of the expression is calculated separately because the method 
        scipy.integrate.quad returns both the value and the error, while only the value
        is needed.

        It corresponds to the eq. (30) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            T: Considered temperature. [K]
            v_ratio: Volume ratio V0 / V of CaPv, where V0 is the volume of CaPv at
                     ambient conditions and V the volume of CaPv at the considered
                     conditions.

        Returns:
            Float64: Vibrational energy of CaPv at the considered conditions.
                     [cm^3 GPa mol^−1]
        """
        # Debye temperature
        theta_capv = self._theta(data, v_ratio)
        # Integral part of the vibrational energy
        int_part_capv = super()._integral_vibrational_energy(theta_capv / T)
        return super()._E_th(5, data.R, T, theta_capv, int_part_capv)

    def _E_th_dv(self, data, T, theta_capv, gamma_capv, v_capv, E_th_capv, E_th_capv_0):
        """
        """
        return super()._E_th_dv(
            5, data.R, T, theta_capv, gamma_capv, v_capv, E_th_capv, E_th_capv_0
        )

    def _E_th_dT(self, data, T, theta_capv, E_th_capv):
        """
        """
        return super()._E_th_dT(2, data.R, T, theta_capv, E_th_capv)

    def _alpha(
        self, data, T, theta_capv, gamma_capv, v_capv, E_th_capv, E_th_capv_0
    ):
        """
        """
        # Partial derivative of the vibrational energy with respect to volume
        E_th_capv_dv = self._E_th_dv(
            data, T, theta_capv, gamma_capv, v_capv, E_th_capv, E_th_capv_0
        )
        # Partial derivative of the vibrational energy with respect to temperature
        E_th_capv_dT = self._E_th_dT(data, T, theta_capv, E_th_capv)
        return super()._alpha(
            data.k_casio3_0, gamma_capv, data.q_capv, v_capv, E_th_capv, E_th_capv_0,
            E_th_capv_dv, E_th_capv_dT
        )

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
            Float64: Residue of the Mie-Gruneisen-Debye EOS for CaPv. [GPa]
        """
        # Volume ratio
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

    def _g_t0(self, data, v_ratio):
        """
        """
        return super()._g_t0(
            data.g_casio3_0, data.g_prime_casio3, data.k_casio3_0, v_ratio
        )

    def _g(self, data, T, v_capv):
        """
        """
        # Volume ratio
        v_ratio = data.v_casio3_0 / v_capv
        # Shear modulus at ambient temperature
        g_capv_t0 = self._g_t0(data, v_ratio)
        return g_capv_t0 + data.g_dot_capv * (T - 300)

    def _k_s(self, data, T, v_capv):
        """
        """
        # Volume ratio
        v_ratio = data.v_casio3_0 / v_capv
        # Gruneisen parameter
        gamma_capv = self._gamma(data, v_ratio)
        # Debye temperature
        theta_capv = self._theta(data, v_ratio)
        # Vibrational energy at T
        E_th_capv = self._E_th(data, T, v_ratio)
        # Vibrational energy at ambient conditions
        E_th_capv_0 = self._E_th(data, 300, v_ratio)
        # Partial derivative of the vibrational energy with respect to volume
        E_th_capv_dv = self._E_th_dv(
            data, T, theta_capv, gamma_capv, v_capv, E_th_capv, E_th_capv_0
        )
        # Thermal expansion
        alpha_capv = self._alpha(
            data, T, theta_capv, gamma_capv, v_capv, E_th_capv, E_th_capv_0
        )
        return super()._k_s(
            T, data.k_casio3_0, data.k0t_prime_capv, gamma_capv, data.q_capv,
            alpha_capv, v_capv, v_ratio, E_th_capv, E_th_capv_0, E_th_capv_dv
        )
