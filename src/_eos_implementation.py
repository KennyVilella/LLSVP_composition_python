"""Provides utility functions to calculate the properties of considered minerals.

This file is associated with the article:
"Constraints on the composition and temperature of LLSVPs from seismic properties of
lower mantle minerals" by K. Vilella, T. Bodin, C.-E. Boukare, F. Deschamps, J. Badro,
M. D. Ballmer, and Y. Li

This file includes a base class that implements the basic formulas for calculating the
mineral properties required for this simulator. These base utilities are then applied to
each considered mineral in separate derived classes.

The functions in this file are primarily based on the Mie-Gruneisen-Debye equation of
state (EOS), with additional considerations for shear modulus based on the model
presented in Bina and Helffrich (1992). A detailed description of the EOS can be found
in Jackson and Rigden (1996) and in the supplementary material of Vilella et al. (2021).

Note that these functions are intended for use within the MineralProperties class and
should not be used outside of it.

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

    The abstract methods defined in this class are intended to be implemented in the
    derived classes corresponding to each specific mineral.

    The formalism for the Mie-Gruneisen-Debye equation of state used in this class can
    be found in the work of Jackson and Rigden (1996). The calculation of the shear
    modulus follows the model presented by Bina and Helffrich (1992).
    """
    @abstractmethod
    def _gamma(self, v_ratio: float, gamma_0: float, q: float) -> float:
        """Calculates the Gruneisen parameter.

        This function calculates the Gruneisen parameter at the considered conditions.

        It corresponds to the eq. (31) of Jackson and Rigden (1996).

        Args:
            v_ratio: Volume ratio V0 / V, where V0 is the volume at ambient conditions
                     and V the volume at the considered conditions.
            gamma_0: Gruneisen parameter at ambient conditions.
            q: Exponent of the Gruneisen parameter.

        Returns:
            Gruneisen parameter at the considered conditions.
        """
        return gamma_0 * v_ratio**(-q)

    @abstractmethod
    def _theta(self, theta_0: float, gamma_0: float, gamma: float, q: float) -> float:
        """Calculates the Debye temperature.

        This function calculates the Debye temperature at the considered conditions.

        It corresponds to the eq. (32) of Jackson and Rigden (1996).

        Args:
            theta_0: Debye temperature at ambient conditions. [K]
            gamma_0: Gruneisen parameter at ambient conditions.
            gamma: Gruneisen parameter at the considered conditions.
            q: Exponent of the Gruneisen parameter.

        Returns:
            Debye temperature at the considered conditions. [K]
        """
        return theta_0 * np.exp((gamma_0 - gamma) / q)

    @abstractmethod
    def _BM3(self, P: float, v_ratio: float, k_0: float, k0t_prime: float) -> float:
        """Calculates the third-order Birch–Murnaghan isothermal equation of state.

        This function calculates the residue of the third-order Birch–Murnaghan
        isothermal equation of state at the consideredconditions.

        It corresponds to the eq. (B1) of Jackson and Rigden (1996).

        Args:
            P: Considered pressure. [GPa]
            v_ratio: Volume ratio V0 / V, where V0 is the volume at ambient conditions
                     and V the volume at the considered conditions.
            k_0: Isothermal bulk modulus at ambient conditions. [GPa]
            k0t_prime: Pressure derivative of the isothermal bulk modulus at ambient
                       conditions.

        Returns:
            Residue of the third-order Birch–Murnaghan isothermal equation of state.
            [GPa]
        """
        return (P -
            1.5 * k_0 * (v_ratio**(7/3) - v_ratio**(5/3)) *
            (1 + 3/4 * (k0t_prime - 4) * (v_ratio**(2/3) - 1))
        )

    @abstractmethod
    def _E_th(self, T: float, theta: float, int_part: float, n: int, R: float) -> float:
        """Calculates the vibrational energy.

        This function calculates the vibrational energy at the considered conditions.
        The integral part of the expression is calculated separately by the function
        _integral_vibrational_energy because the method scipy.integrate.quad returns
        both the value and the error, while only the value is needed.

        It corresponds to the eq. (30) of Jackson and Rigden (1996).

        Args:
            T: Considered temperature. [K]
            theta: Debye temperature at the considered conditions. [K]
            int_part: Integral part of the vibrational energy.
            n: Number of atoms per formula unit.
            R: Gas constant. [cm^3 GPa K^−1 mol^−1]

        Returns:
            Vibrational energy at the considered conditions. [cm^3 GPa mol^−1]
        """
        return 9. * n * R * T**4 / theta**3 * int_part

    @abstractmethod
    def _E_th_dv(
        self, T: float, v: float, theta: float, gamma: float, E_th_0: float,
        E_th: float, n: int, R: float
    ) -> float:
        """Calculates the derivative of the vibrational energy with respect to volume.

        This function calculates the derivative of the vibrational energy with respect
        to volume at the considered conditions.

        It corresponds to the fourth equation in (B6) of Jackson and Rigden (1996).
        Alternatively, it can be directly calculated from the expression of the
        vibrational energy.

        Args:
            T: Considered temperature. [K]
            v: Volume at considered conditions. [cm^3/mol]
            theta: Debye temperature at the considered conditions. [K]
            gamma: Gruneisen parameter at the considered conditions.
            E_th_0: Vibrational energy at ambient conditions. [cm^3 GPa mol^−1]
            E_th: Vibrational energy at the considered conditions. [cm^3 GPa mol^−1]
            n: Number of atoms per formula unit.
            R: Gas constant. [cm^3 GPa K^−1 mol^−1]

        Returns:
            Partial derivative of the vibrational energy with respect to temperature.
            [cm^3 GPa mol^−1 K^-1]
        """
        d_int_part = 9 * n * R * (
            1 / (np.exp(theta / T) - 1) - 1 / (np.exp(theta / 300) - 1)
        )
        return (gamma * theta / v) * (3 * (E_th - E_th_0) / theta - d_int_part)

    @abstractmethod
    def _E_th_dT(self, T: float, theta: float, E_th: float, n: int, R: float) -> float:
        """Calculates the derivative of vibrational energy with respect to temperature.

        This function calculates the derivative of the vibrational energy with respect
        to temperature at the considered conditions.

        It corresponds to the third equation in (B6) of Jackson and Rigden (1996).
        Alternatively, it can be directly calculated from the expression of the
        vibrational energy.

        Args:
            T: Considered temperature. [K]
            theta: Debye temperature at the considered conditions. [K]
            E_th: Vibrational energy at the considered conditions. [cm^3 GPa mol^−1]
            n: Number of atoms per formula unit.
            R: Gas constant. [cm^3 GPa K^−1 mol^−1]

        Returns:
            Partial derivative of the vibrational energy with respect to temperature.
            [cm^3 GPa mol^−1 K^-1]
        """
        return 4 * E_th / T - 9 * n * R * (theta / T) / (np.exp(theta / T) - 1)

    def _integral_vibrational_energy(self, x_max: float) -> float:
        """Calculates the integral part of the vibrational energy.

        This function calculates the integral part of the vibrational energy at the
        considered conditions.

        It corresponds to the eq. (30) of Jackson and Rigden (1996).

        Args:
            x_max: Value for the upper bound of the integral.

        Returns:
            Value of the integral.
        """
        value, err = scipy.integrate.quad(
            lambda x: x**3 / (np.exp(x) - 1), 0, x_max
        )
        return value

    @abstractmethod
    def _alpha(
        self, v: float, gamma: float, k_v: float, E_th_0: float, E_th: float,
        E_th_dv: float, E_th_dT: float, q: float
    ) -> float:
        """Calculates the thermal expansion coefficient.

        This function calculates the thermal expansion coefficient at the considered
        conditions.

        It corresponds to the sixth equation in (B5) of Jackson and Rigden (1996).

        Args:
            v: Volume at considered conditions. [cm^3/mol]
            gamma: Gruneisen parameter at the considered conditions.
            k_v: Isothermal bulk modulus at ambient temperature. [GPa]
            E_th_0: Vibrational energy at ambient conditions. [cm^3 GPa mol^−1]
            E_th: Vibrational energy at the considered conditions. [cm^3 GPa mol^−1]
            E_th_dv: Partial derivative of the vibrational energy with respect to
                     temperature. [cm^3 GPa mol^−1 K^-1]
            E_th_dT: Partial derivative of the vibrational energy with respect to
                     temperature. [cm^3 GPa mol^−1 K^-1]
            q: Exponent of the Gruneisen parameter.

        Returns:
            Thermal expansion coefficient. [K^-1]
        """
        return gamma / v * E_th_dT / (
            k_v - (q - 1) * gamma * (E_th - E_th_0) / v - gamma * E_th_dv
        )

    @abstractmethod
    def _MGD(
        self, v: float, gamma: float, E_th_0: float, E_th: float, BM3: float
    ) -> float:
        """Calculates the residue of the Mie-Gruneisen-Debye equation of state.

        This function calculates the residue of the Mie-Gruneisen-Debye equation of
        state applied at the considered conditions.

        It corresponds to the eq. (33) of Jackson and Rigden (1996).

        Args:
            v: Volume at considered conditions. [cm^3/mol]
            gamma: Gruneisen parameter at the considered conditions.
            E_th_0: Vibrational energy at ambient conditions. [cm^3 GPa mol^−1]
            E_th: Vibrational energy at the considered conditions. [cm^3 GPa mol^−1]
            BM3: Residue of the third-order Birch–Murnaghan isothermal
                 equation of state. [GPa]

        Returns:
            Residue of the Mie-Gruneisen-Debye equation of state. [GPa]
        """
        return BM3  - gamma / v * (E_th - E_th_0)

    @abstractmethod
    def _g_t0(self, v_ratio: float, k_0: float, g_0: float, g_prime: float) -> float:
        """Calculates the shear modulus at ambient temperature.

        This function calculates the shear modulus at ambient temperature.

        It corresponds to the eq. (21) of Bina and Helffrich (1992) when assuming that
        the second-order terms can be neglected.

        Args:
            v_ratio: Volume ratio V0 / V, where V0 is the volume at ambient conditions
                     and V the volume at the considered conditions.
            k_0: Isothermal bulk modulus at ambient conditions. [GPa]
            g_0: Shear modulus at ambient conditions. [GPa]
            g_prime: Pressure derivative of the shear modulus.

        Returns:
            Shear modulus at ambient temperature. [GPa]
        """
        return v_ratio**(5/3) * (
            g_0 + (1 - v_ratio**(2/3)) * 0.5 * (5 * g_0 - 3 * k_0 * g_prime)
        )

    def _k_v(self, v_ratio: float, k_0: float, k0t_prime: float) -> float:
        """Calculates the isothermal bulk modulus at ambient temperature.

        This function calculates the isothermal bulk modulus at ambient temperature.

        The expression for the isothermal bulk modulus at ambient temperature is not
        given in Jackson and Rigden (1996), but it can be calculated from the
        third-order Birch–Murnaghan isothermal equation of state (B1) and the definition
        of the isothermal bulk modulus (eq. 5).

        Args:
            v_ratio: Volume ratio V0 / V, where V0 is the volume at ambient conditions
                     and V the volume at the considered conditions.
            k_0: Isothermal bulk modulus at ambient conditions. [GPa]
            k0t_prime: Pressure derivative of the isothermal bulk modulus at ambient
                       conditions.

        Returns:
            Isothermal bulk modulus at ambient temperature. [GPa]
        """
        return k_0 * (
            (v_ratio**(7/3) - v_ratio**(5/3)) * 3/4 * (k0t_prime - 4) * v_ratio**(2/3) +
            0.5 * (7 * v_ratio**(7/3) - 5 * v_ratio**(5/3)) *
            (1 + 3/4 * (k0t_prime - 4) * (v_ratio**(2/3) - 1))
        )

    @abstractmethod
    def _k_t(
        self, v: float, gamma: float, k_v: float, E_th_0: float, E_th: float,
        E_th_dv: float, q: float
    ) -> float:
        """Calculates the isothermal bulk modulus.

        This function calculates the isothermal bulk modulus at the considered
        conditions.

        It corresponds to the fifth equation in (B5) of Jackson and Rigden (1996).

        Args:
            v: Volume at considered conditions. [cm^3/mol]
            gamma: Gruneisen parameter at the considered conditions.
            k_v: Isothermal bulk modulus at ambient temperature. [GPa]
            E_th_0: Vibrational energy at ambient conditions. [cm^3 GPa mol^−1]
            E_th: Vibrational energy at the considered conditions. [cm^3 GPa mol^−1]
            E_th_dv: Partial derivative of the vibrational energy with respect to
                     temperature. [cm^3 GPa mol^−1 K^-1]
            q: Exponent of the Gruneisen parameter.

        Returns:
            Isothermal bulk modulus. [GPa]
        """
        return k_v - (q - 1) * gamma / v * (E_th - E_th_0) - gamma * E_th_dv

    @abstractmethod
    def _k_s(self, T: float, alpha: float, gamma: float, k_t: float) -> float:
        """Calculates the isentropic bulk modulus.

        This function calculates the isentropic bulk modulus at the considered
        conditions.

        It corresponds to the twelfth equation in (B5) of Jackson and Rigden (1996).

        Args:
            T: Considered temperature. [K]
            alpha: Thermal expansion coefficient. [K^-1]
            gamma: Gruneisen parameter at the considered conditions.
            k_t: Isothermal bulk modulus at considered conditions. [GPa]

        Returns:
            Isentropic bulk modulus. [GPa]
        """
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

    Note that these functions are intended for use within the MineralProperties class
    and should not be used outside of it.
    """
    def _v_feo_0(self, data, eta_ls: float) -> float:
        """Calculates the volume of FeO at ambient conditions.

        This function calculates the volume of FeO in Ferropericlase (Fp) at ambient
        conditions. FeO in Fp is assumed to be a mixture of FeO in a low spin state and
        FeO in a high spin state.

        Args:
            data: Data holder for the MineralProperties class.
            eta_ls: Average proportion of FeO in the low spin state.

        Returns:
            Volume of FeO at ambient conditions. [cm^3/mol]
        """
        return eta_ls * data.v_feo_ls_0 + (1 - eta_ls) * data.v_feo_hs_0

    def _v_fp_0(self, data, eta_ls: float, x_fp: float) -> float:
        """Calculates the volume of Ferropericlase at ambient conditions.

        This function calculates the volume of Ferropericlase (Fp) at ambient
        conditions. Fp is assumed to be a mixture of FeO and MgO.

        Args:
            data: Data holder for the MineralProperties class.
            eta_ls: Average proportion of FeO in the low spin state.
            x_fp: Molar concentration of FeO in Fp.

        Returns:
            Volume of Fp at ambient conditions. [cm^3/mol]
        """
        v_feo_0 = self. _v_feo_0(data, eta_ls)
        return x_fp * v_feo_0 + (1 - x_fp) * data.v_mgo_0

    def _VRH_average(
        self, x_1: float, v_1: float, c_1: float, v_2: float, c_2: float
    ) -> float:
        """Calculates the Voigt-Reuss-Hill average.

        This function implements the Voigt-Reuss-Hill average for a mixture of two
        components. This is traditionally used to calculate the elasticity of
        polycrystal from the elasticity of its individual components. Here, it is used
        to calculate the bulk modulus or shear modulus of Ferropericlase (Fp).

        Note that the volume can be given in any unit as long as the two volumes are
        given in the same unit.

        Args:
            x_1: Molar concentraion of the first component.
            v_1: Volume of the first component.
            c_1: Bulk or shear modulus of the first component. [GPa]
            v_2: Volume of the second component.
            c_2: Bulk or shear modulus of the second component. [GPa]

        Returns:
            Voigt-Reuss-Hill average of the bulk or shear modulus. [GPa]
        """
        x_2 = 1 - x_1
        v_tot = x_1 * v_1 + x_2 * v_2
        c_v = x_1 * v_1 / v_tot * c_1 + x_2 * v_2 / v_tot * c_2
        c_r = v_tot / (x_1 * v_1 / c_1 + x_2 * v_2 / c_2)
        return 0.5 * (c_v + c_r)

    def _k_feo_0_VRH_average(self, data, eta_ls: float) -> float:
        """Calculates the bulk modulus of FeO in Ferropericlase at ambient conditions.

        This function calculates the isothermal bulk modulus of FeO at ambient
        conditions using the Voigt-Reuss-Hill average. FeO in Ferropericlase (Fp) is
        assumed to be a mixture of FeO in a low spin state and FeO in a high spin state.

        Args:
            data: Data holder for the MineralProperties class.
            eta_ls: Average proportion of FeO in the low spin state.

        Returns:
            Isothermal bulk modulus of FeO in Fp at ambient conditions. [GPa]
        """
        return self._VRH_average(
            eta_ls, data.v_feo_ls_0, data.k_feo_ls_0, data.v_feo_hs_0, data.k_feo_hs_0
        )

    def _g_feo_0_VRH_average(self, data, eta_ls: float) -> float:
        """Calculates the shear modulus of FeO in Ferropericlase at ambient conditions.

        This function calculates the shear modulus of FeO at ambient conditions using
        the Voigt-Reuss-Hill average. FeO in Ferropericlase (Fp) is assumed to be a
        mixture of FeO in a low spin state and FeO in a high spin state.

        Args:
            data: Data holder for the MineralProperties class.
            eta_ls: Average proportion of FeO in the low spin state.

        Returns:
            Shear modulus of FeO in Fp at ambient conditions. [GPa]
        """
        return self._VRH_average(
            eta_ls, data.v_feo_ls_0, data.g_feo_ls_0, data.v_feo_hs_0, data.g_feo_hs_0
        )

    def _g_prime_feo_VRH_average(self, data, eta_ls: float) -> float:
        """Calculates the pressure derivative of the shear modulus for FeO in Fp.

        This function calculates the pressure derivative of the shear modulus for FeO
        using the Voigt-Reuss-Hill average. FeO in Ferropericlase (Fp) is assumed to be
        a mixture of FeO in a low spin state and FeO in a high spin state.

        Args:
            data: Data holder for the MineralProperties class.
            eta_ls: Average proportion of FeO in the low spin state.

        Returns:
            Pressure derivative of the shear modulus for FeO in Fp.
        """
        return self._VRH_average(
            eta_ls, data.v_feo_ls_0, data.g_prime_feo_ls, data.v_feo_hs_0,
            data.g_prime_feo_hs
        )

    def _k_fp_0_VRH_average(self, data, eta_ls: float, x_fp: float) -> float:
        """Calculates the bulk modulus of Ferropericlase at ambient conditions.

        This function calculates the isothermal bulk modulus of Ferropericlase (Fp) at
        ambient conditions using the Voigt-Reuss-Hill average. Fp is assumed to be a
        mixture of FeO and MgO.

        Args:
            data: Data holder for the MineralProperties class.
            eta_ls: Average proportion of FeO in the low spin state.
            x_fp: Molar concentration of FeO in Fp.

        Returns:
            Isothermal bulk modulus of Fp at ambient conditions. [GPa]
        """
        # Volume of FeO at ambient conditions
        v_feo_0 = self._v_feo_0(data, eta_ls)
        # Isothermal bulk modulus of FeO at ambient conditions
        k_feo_0 = self._k_feo_0_VRH_average(data, eta_ls)
        return self._VRH_average(
            x_fp, v_feo_0, k_feo_0, data.v_mgo_0, data.k_mgo_0
        )

    def _g_fp_0_VRH_average(self, data, eta_ls: float, x_fp: float) -> float:
        """Calculates the shear modulus of Ferropericlase at ambient conditions.

        This function calculates the shear modulus of Ferropericlase (Fp) at ambient
        conditions using the Voigt-Reuss-Hill average. Fp is assumed to be a mixture of
        FeO and MgO.

        Args:
            data: Data holder for the MineralProperties class.
            eta_ls: Average proportion of FeO in the low spin state.
            x_fp: Molar concentration of FeO in Fp.

        Returns:
            Shear modulus of Fp at ambient conditions. [GPa]
        """
        # Volume of FeO at ambient conditions
        v_feo_0 = self._v_feo_0(data, eta_ls)
        # Shear modulus of FeO at ambient conditions
        g_feo_0 = self._g_feo_0_VRH_average(data, eta_ls)
        return self._VRH_average(
            x_fp, v_feo_0, g_feo_0, data.v_mgo_0, data.g_mgo_0
        )

    def _g_prime_fp_VRH_average(self, data, eta_ls: float, x_fp: float) -> float:
        """Calculates the pressure derivative of the shear modulus for Ferropericlase.

        This function calculates the pressure derivative of the shear modulus for
        Ferropericlase (Fp) using the Voigt-Reuss-Hill average. Fp is assumed to be a
        mixture of FeO and MgO.

        Args:
            data: Data holder for the MineralProperties class.
            eta_ls: Average proportion of FeO in the low spin state.
            x_fp: Molar concentration of FeO in Fp.

        Returns:
            Pressure derivative of the shear modulus for Fp.
        """
        # Volume of FeO at ambient conditions
        v_feo_0 = self._v_feo_0(data, eta_ls)
        # Pressure derivative of the shear modulus for FeO
        g_prime_feo = self._g_prime_feo_VRH_average(data, eta_ls)
        return self._VRH_average(
            x_fp, v_feo_0, g_prime_feo, data.v_mgo_0, data.g_prime_mgo
        )

    def _gamma(self, data, v_ratio: float) -> float:
        """Calculates the Gruneisen parameter of Ferropericlase.

        This function calculates the Gruneisen parameter of Ferropericlase (Fp) at the
        considered conditions.

        It corresponds to the eq. (31) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            v_ratio: Volume ratio V0 / V of Fp, where V0 is the volume of Fp at ambient
                     conditions and V the volume of Fp at the considered conditions.

        Returns:
            Gruneisen parameter of Fp at the considered conditions.
        """
        return super()._gamma(v_ratio, data.gamma_fp_0, data.q_fp)

    def _theta(self, data, v_ratio: float) -> float:
        """Calculates the Debye temperature of Ferropericlase.

        This function calculates the Debye temperature of Ferropericlase (Fp) at the
        considered conditions.

        It corresponds to the eq. (32) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            v_ratio: Volume ratio V0 / V of Fp, where V0 is the volume of Fp at ambient
                     conditions and V the volume of Fp at the considered conditions.

        Returns:
            Debye temperature of Fp at the considered conditions. [K]
        """
        # Gruneisen parameter
        gamma_fp = self._gamma(data, v_ratio)
        return super()._theta(data.theta_fp_0, data.gamma_fp_0, gamma_fp, data.q_fp)

    def _BM3(self, data, P: float, v_ratio: float, k_fp_0: float) -> float:
        """Implements the third-order Birch–Murnaghan isothermal EOS for Ferropericlase.

        This function calculates the residue of the third-order Birch–Murnaghan
        isothermal equation of state applied to Ferroepriclase (Fp) at the considered
        conditions.

        It corresponds to the eq. (B1) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            P: Considered pressure. [GPa]
            v_ratio: Volume ratio V0 / V of Fp, where V0 is the volume of Fp at ambient
                     conditions and V the volume of Fp at the considered conditions.
            k_fp_0: Isothermal bulk modulus of Fp at ambient conditions. [GPa]

        Returns:
            Residue of the third-order Birch–Murnaghan isothermal equation of state
            for Fp. [GPa]
        """
        return super()._BM3(P, v_ratio, k_fp_0, data.k0t_prime_fp)

    def _E_th(self, data, T: float, v_ratio: float) -> float:
        """Calculates the vibrational energy of Ferropericlase.

        This function calculates the vibrational energy of Ferropericlase (Fp) at the
        considered conditions. The integral part of the expression is calculated
        separately because the method scipy.integrate.quad returns both the value and
        the error, while only the value is needed.

        It corresponds to the eq. (30) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            T: Considered temperature. [K]
            v_ratio: Volume ratio V0 / V of Fp, where V0 is the volume of Fp at ambient
                     conditions and V the volume of Fp at the considered conditions.

        Returns:
            Vibrational energy of Fp at the considered conditions. [cm^3 GPa mol^−1]
        """
        # Debye temperature
        theta_fp = self._theta(data, v_ratio)
        # Integral part of the vibrational energy
        int_part_fp = super()._integral_vibrational_energy(theta_fp / T)
        return super()._E_th(T, theta_fp, int_part_fp, 2, data.R)

    def _E_th_dv(
        self, data, T: float, v_fp: float, theta_fp: float, gamma_fp: float,
        E_th_fp_0: float, E_th_fp: float
    ) -> float:
        """Calculates derivative of vibrational energy with respect to volume for Fp.

        This function calculates the derivative of the vibrational energy with respect
        to volume for Ferropericlase (Fp) at the considered conditions.

        It corresponds to the fourth equation in (B6) of Jackson and Rigden (1996).
        Alternatively, it can be directly calculated from the expression of the
        vibrational energy.

        Args:
            data: Data holder for the MineralProperties class.
            T: Considered temperature. [K]
            v_fp: Volume of Fp at considered conditions. [cm^3/mol]
            theta_fp: Debye temperature of Fp at the considered conditions. [K]
            gamma_fp: Gruneisen parameter of Fp at the considered conditions.
            E_th_fp_0: Vibrational energy of Fp at ambient conditions. [cm^3 GPa mol^−1]
            E_th_fp: Vibrational energy of Fp at the considered conditions.
                     [cm^3 GPa mol^−1]

        Returns:
            Partial derivative of the vibrational energy with respect to temperature
            for Fp. [cm^3 GPa mol^−1 K^-1]
        """
        return super()._E_th_dv(
            T, v_fp, theta_fp, gamma_fp, E_th_fp_0, E_th_fp, 2, data.R
        )

    def _E_th_dT(self, data, T: float, theta_fp: float, E_th_fp: float) -> float:
        """Calculates the derivative of the vibrational energy wrt temperature for Fp.

        This function calculates the derivative of the vibrational energy with respect
        to temperature for Ferropericlase (Fp) at the considered conditions.

        It corresponds to the third equation in (B6) of Jackson and Rigden (1996).
        Alternatively, it can be directly calculated from the expression of the
        vibrational energy.

        Args:
            data: Data holder for the MineralProperties class.
            T: Considered temperature. [K]
            theta_fp: Debye temperature of Fp at the considered conditions. [K]
            E_th_fp: Vibrational energy of Fp at the considered conditions.
                     [cm^3 GPa mol^−1]

        Returns:
            Partial derivative of the vibrational energy with respect to temperature
            for Fp. [cm^3 GPa mol^−1 K^-1]
        """
        return super()._E_th_dT(T, theta_fp, E_th_fp, 2, data.R)

    def _alpha(
        self, data, T: float, k_v_fp: float, theta_fp: float, gamma_fp: float,
        v_fp: float, E_th_fp: float, E_th_fp_0: float, E_th_fp_dv: float
    ) -> float:
        """Calculates the thermal expansion coefficient of Ferropericlase.

        This function calculates the thermal expansion coefficient of Ferropericlase
        (Fp) at the considered conditions.

        It corresponds to the sixth equation in (B5) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            T: Considered temperature. [K]
            k_v_fp: Isothermal bulk modulus of Fp at ambient temperature. [GPa]
            theta_fp: Debye temperature of Fp at the considered conditions. [K]
            gamma_fp: Gruneisen parameter of Fp at the considered conditions.
            v_fp: Volume of Fp at considered conditions. [cm^3/mol]
            E_th_fp: Vibrational energy of Fp at the considered conditions.
                     [cm^3 GPa mol^−1]
            E_th_fp_0: Vibrational energy of Fp at ambient conditions. [cm^3 GPa mol^−1]
            E_th_fp_dv: Partial derivative of the vibrational energy with respect to
                        temperature for Fp. [cm^3 GPa mol^−1 K^-1]

        Returns:
            Thermal expansion coefficient of Fp. [K^-1]
        """
        # Partial derivative of the vibrational energy with respect to temperature
        E_th_fp_dT = self._E_th_dT(data, T, theta_fp, E_th_fp)
        return super()._alpha(
            v_fp, gamma_fp, k_v_fp, E_th_fp_0, E_th_fp, E_th_fp_dv, E_th_fp_dT
            data.q_fp
        )

    def _MGD(
        self, data, T: float, P: float, v_fp: float, eta_ls: float, x_fp: float
    ) -> float:
        """Implements the Mie-Gruneisen-Debye EOS for Ferropericlase.

        This function calculates the residue of the Mie-Gruneisen-Debye equation of
        state applied to Ferroepriclase (Fp) at the considered conditions. It can be
        used to obtain one of the conditions among the pressure, temperature, volume,
        knowing the remaining two.

        It corresponds to the eq. (33) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            T: Considered temperature. [K]
            P: Considered pressure. [GPa]
            v_fp: Volume of Fp at considered conditions. [cm^3/mol]
            eta_ls: Average proportion of FeO in the low spin state.
            x_fp: Molar concentration of FeO in Fp.

        Returns:
            Residue of the Mie-Gruneisen-Debye EOS for Fp. [GPa]
        """
        # Isothermal bulk modulus of Fp at ambient conditions
        k_fp_0 = self._k_fp_0_VRH_average(data, eta_ls, x_fp)
        # Volume of Fp at ambient conditions
        v_fp_0 = self._v_fp_0(data, eta_ls, x_fp)
        # Volume ratio
        v_ratio = v_fp_0 / v_fp
        # Gruneisen parameter
        gamma_fp = self._gamma(data, v_ratio)
        # Third-order Birch–Murnaghan isothermal equation of state
        BM3_fp = self._BM3(data, P, v_ratio, k_fp_0)
        # Vibrational energy at T
        E_th_fp = self._E_th(data, T, v_ratio)
        # Vibrational energy at ambient conditions
        E_th_fp_0 = self._E_th(data, 300, v_ratio)
        return super()._MGD(v_fp, gamma_fp, E_th_fp_0, E_th_fp, BM3_fp)

    def _g_t0(self, data, v_fp: float, eta_ls: float, x_fp: float) -> float:
        """Calculates the shear modulus of Ferropericlase at ambient temperature.

        This function calculates the shear modulus of Ferropericlase (Fp) at ambient
        temperature.

        It corresponds to the eq. (21) of Bina and Helffrich (1992) when assuming that
        the second-order terms can be neglected.

        Args:
            data: Data holder for the MineralProperties class.
            v_fp: Volume of Fp at considered conditions. [cm^3/mol]
            eta_ls: Average proportion of FeO in the low spin state.
            x_fp: Molar concentration of FeO in Fp.

        Returns:
            Shear modulus of Fp at ambient temperature. [GPa]
        """
        # Volume of Fp at ambient conditions
        v_fp_0 = self._v_fp_0(data, eta_ls, x_fp)
        # Volume ratio
        v_ratio = v_fp_0 / v_fp
        # Isothermal bulk modulus of Fp at ambient conditions
        k_fp_0 = self._k_fp_0_VRH_average(data, eta_ls, x_fp)
        # Shear modulus of Fp at ambient conditions
        g_fp_0 = self._g_fp_0_VRH_average(data, eta_ls, x_fp)
        # Pressure derivative of the shear modulus for Fp
        g_prime_fp = self._g_prime_fp_VRH_average(data, eta_ls, x_fp)
        return super()._g_t0(v_ratio, k_fp_0, g_fp_0, g_prime_fp)

    def _g(self, data, T: float, v_fp: float, eta_ls: float, x_fp: float) -> float:
        """Calculates the shear modulus of Ferropericlase.

        This function calculates the shear modulus of Ferropericlase (Fp) at the
        considered conditions.

        Following eq. (38) of Bina and Helffrich (1992), the temperature dependence
        of the shear modulus is assumed to be constant, so that it can be simply
        calculated from its temperature derivative.
        It should however be noted that it is a rough estimation that is unlikely to be
        valid. 

        Args:
            data: Data holder for the MineralProperties class.
            T: Considered temperature. [K]
            v_fp: Volume of Fp at considered conditions. [cm^3/mol]
            eta_ls: Average proportion of FeO in the low spin state.
            x_fp: Molar concentration of FeO in Fp.

        Returns:
            Shear modulus of Fp. [GPa]
        """
        # Shear modulus at ambient temperature
        g_fp_t0 = self._g_t0(data, v_fp, eta_ls, x_fp)
        return g_fp_t0 + data.g_dot_fp * (T - 300)

    def _k_t(
        self, data, k_v_fp: float, gamma_fp: float, v_fp: float, E_th_fp: float,
        E_th_fp_0: float, E_th_fp_dv: float
    ) -> float:
        """Calculates the isothermal bulk modulus of Ferropericlase.

        This function calculates the isothermal bulk modulus of Ferropericlase (Fp) at
        the considered conditions.

        It corresponds to the fifth equation in (B5) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            k_v_fp: Isothermal bulk modulus of Fp at ambient temperature. [GPa]
            gamma_fp: Gruneisen parameter of Fp at the considered conditions.
            v_fp: Volume of Fp at considered conditions. [cm^3/mol]
            E_th_fp: Vibrational energy of Fp at the considered conditions.
                     [cm^3 GPa mol^−1]
            E_th_fp_0: Vibrational energy of Fp at ambient conditions. [cm^3 GPa mol^−1]
            E_th_fp_dv: Partial derivative of the vibrational energy with respect to
                        temperature for Fp. [cm^3 GPa mol^−1 K^-1]

        Returns:
            Isothermal bulk modulus Fp. [GPa]
        """
        return super()._k_t(
            v_fp, gamma_fp, k_v_fp, E_th_fp_0, E_th_fp, E_th_fp_dv, data.q_fp
        )

    def _k_s(self, data, T: float, v_fp: float, eta_ls: float, x_fp: float) -> float:
        """Calculates the isentropic bulk modulus of Ferropericlase.

        This function calculates the isentropic bulk modulus of Ferropericlase (Fp) at
        the considered conditions.

        It corresponds to the twelfth equation in (B5) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            T: Considered temperature. [K]
            v_fp: Volume of Fp at considered conditions. [cm^3/mol]
            eta_ls: Average proportion of FeO in the low spin state.
            x_fp: Molar concentration of FeO in Fp.

        Returns:
            Isentropic bulk modulus of Fp. [GPa]
        """
        # Volume of Fp at ambient conditions
        v_fp_0 = self._v_fp_0(data, eta_ls, x_fp)
        # Volume ratio
        v_ratio = v_fp_0 / v_fp
        # Gruneisen parameter
        gamma_fp = self._gamma(data, v_ratio)
        # Debye temperature
        theta_fp = self._theta(data, v_ratio)
        # Isothermal bulk modulus of Fp at ambient conditions
        k_fp_0 = self._k_fp_0_VRH_average(data, eta_ls, x_fp)
        # Vibrational energy at T
        E_th_fp = self._E_th(data, T, v_ratio)
        # Vibrational energy at ambient conditions
        E_th_fp_0 = self._E_th(data, 300, v_ratio)
        # Partial derivative of the vibrational energy with respect to volume
        E_th_fp_dv = self._E_th_dv(
            data, T, v_fp, theta_fp, gamma_fp, E_th_fp_0, E_th_fp
        )
        # Isothermal bulk modulus at ambient temperature
        k_v_fp = super()._k_v(v_ratio, k_fp_0, data.k0t_prime_fp)
        # Isothermal bulk modulus
        k_t_fp = self._k_t(
            data, k_v_fp, gamma_fp, v_fp, E_th_fp, E_th_fp_0, E_th_fp_dv
        )
        # Thermal expansion coefficient
        alpha_fp = self._alpha(
            data, T, k_v_fp, theta_fp, gamma_fp, v_fp, E_th_fp, E_th_fp_0, E_th_fp_dv
        )
        return super()._k_s(T, alpha_fp, gamma_fp, k_t_fp)

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

    Note that these functions are intended for use within the MineralProperties class
    and should not be used outside of it.
    """
    def _v_bm_0(
        self, data, x_mgsio3: float, x_fesio3: float, x_fealo3: float, x_fe2o3: float,
        x_al2o3: float
    ) -> float:
        """Calculates the volume of Bridgmanite at ambient conditions.

        This function calculates the volume of Bridgmanite (Bm) at ambient conditions.
        Bm is assumed to be a mixture of MgSiO3, FeSiO3, FeAlO3, Fe2O3, and Al2O3.

        Args:
            data: Data holder for the MineralProperties class.
            x_mgsio3: Molar concentration of MgSiO3 in Bm.
            x_fesio3: Molar concentration of FeSiO3 in Bm.
            x_fealo3: Molar concentration of FeAlO3 in Bm.
            x_fe2o3: Molar concentration of Fe2O3 in Bm.
            x_al2o3: Molar concentration of Al2O3 in Bm.

        Returns:
            Volume of Bm at ambient conditions. [cm^3/mol]
        """
        v_bm_0 = (
            x_mgsio3 * data.v_mgsio3_0 + x_fesio3 * data.v_fesio3_0 +
            x_fealo3 * data.v_fealo3_0 + x_al2o3 * data.v_al2o3_0 +
            x_fe2o3 * data.v_fe2o3_0
        )
        return v_bm_0

    def _k_bm_0_VRH_average(
        self, data, x_mgsio3: float, x_fesio3: float, x_fealo3: float, x_fe2o3: float,
        x_al2o3: float, v_tot: float
    ) -> float:
        """Calculates the isothermal bulk modulus of Bridgmanite at ambient conditions.

        This function calculates the isothermal bulk modulus of Bridgmanite (Bm) at
        ambient conditions using the Voigt-Reuss-Hill average. Bm is assumed to be a
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
            Isothermal bulk modulus of Bm at ambient conditions. [GPa]
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
        self, data, x_mgsio3: float, x_fesio3: float, x_fealo3: float, x_fe2o3: float,
        x_al2o3: float, v_tot: float
    ) -> float:
        """Calculates the shear modulus of Bridgmanite at ambient conditions.

        This function calculates the shear modulus of Bridgmanite (Bm) at ambient
        conditions using the Voigt-Reuss-Hill average. Bm is assumed to be a mixture of
        MgSiO3, FeSiO3, FeAlO3, Fe2O3, and Al2O3.

        Args:
            data: Data holder for the MineralProperties class.
            x_mgsio3: Molar concentration of MgSiO3 in Bm.
            x_fesio3: Molar concentration of FeSiO3 in Bm.
            x_fealo3: Molar concentration of FeAlO3 in Bm.
            x_fe2o3: Molar concentration of Fe2O3 in Bm.
            x_al2o3: Molar concentration of Al2O3 in Bm.
            v_tot: Volume of Bm at ambient conditions. [cm^3/mol]

        Returns:
            Shear modulus of Bm at ambient conditions. [GPa]
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
        self, data, x_mgsio3: float, x_fesio3: float, x_fealo3: float, x_fe2o3: float,
        x_al2o3: float, v_tot: float
    ) -> float:
        """Calculates the pressure derivative of the shear modulus for Bridgmanite.

        This function calculates the pressure derivative of the shear modulus for
        Bridgmanite (Bm) using the Voigt-Reuss-Hill average. Bm is assumed to be a
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
            Pressure derivative of the shear modulus for Bm.
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

    def _gamma(self, data, v_ratio: float) -> float:
        """Calculates the Gruneisen parameter of Bridgmanite.

        This function calculates the Gruneisen parameter of Bridgmanite (Bm) at the
        considered conditions.

        It corresponds to the eq. (31) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            v_ratio: Volume ratio V0 / V of Bm, where V0 is the volume of Bm at ambient
                     conditions and V the volume of Bm at the considered conditions.

        Returns:
            Gruneisen parameter of Bm at the considered conditions.
        """
        return super()._gamma(v_ratio, data.gamma_bm_0, data.q_bm)

    def _theta(self, data, v_ratio: float) -> float:
        """Calculates the Debye temperature of Bridgmanite.

        This function calculates the Debye temperature of Bridgmanite (Bm) at the
        considered conditions.

        It corresponds to the eq. (32) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            v_ratio: Volume ratio V0 / V of Bm, where V0 is the volume of Bm at ambient
                     conditions and V the volume of Bm at the considered conditions.

        Returns:
            Debye temperature of Bm at the considered conditions. [K]
        """
        # Gruneisen parameter
        gamma_bm = self._gamma(data, v_ratio)
        return super()._theta(
            data.theta_bm_0, data.gamma_bm_0, gamma_bm, data.q_bm
        )

    def _BM3(self, data, P: float, v_ratio: float, k_bm_0: float) -> float:
        """Implements the third-order Birch–Murnaghan isothermal EOS for Bridgmanite.

        This function calculates the residue of the third-order Birch–Murnaghan
        isothermal equation of state applied to Bridgmanite (Bm) at the considered
        conditions.

        It corresponds to the eq. (B1) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            P: Considered pressure. [GPa]
            v_ratio: Volume ratio V0 / V of Bm, where V0 is the volume of Bm at ambient
                     conditions and V the volume of Bm at the considered conditions.
            k_bm_0: Isothermal bulk modulus of Bm at ambient conditions. [GPa]

        Returns:
            Residue of the third-order Birch–Murnaghan isothermal equation of state
            for Bm. [GPa]
        """
        return super()._BM3(P, v_ratio, k_bm_0, data.k0t_prime_bm)

    def _E_th(self, data, T: float, v_ratio: float) -> float:
        """Calculates the vibrational energy of Bridgmanite.

        This function calculates the vibrational energy of Bridgmanite (Bm) at the
        considered conditions. The integral part of the expression is calculated
        separately because the method scipy.integrate.quad returns both the value and
        the error, while only the value is needed.

        It corresponds to the eq. (30) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            T: Considered temperature. [K]
            v_ratio: Volume ratio V0 / V of Bm, where V0 is the volume of Bm at ambient
                     conditions and V the volume of Bm at the considered conditions.

        Returns:
            Vibrational energy of Bm at the considered conditions. [cm^3 GPa mol^−1]
        """
        # Debye temperature
        theta_bm = self._theta(data, v_ratio)
        # Integral part of the vibrational energy
        int_part_bm = super()._integral_vibrational_energy(theta_bm / T)
        return super()._E_th(T, theta_bm, int_part_bm, 5, data.R)

    def _E_th_dv(
        self, data, T: float, v_bm: float, theta_bm: float, gamma_bm: float,
        E_th_bm_0: float, E_th_bm: float
    ) -> float:
        """Calculates derivative of vibrational energy with respect to volume for Bm.

        This function calculates the derivative of the vibrational energy with respect
        to volume for Bridgmanite (Bm) at the considered conditions.

        It corresponds to the fourth equation in (B6) of Jackson and Rigden (1996).
        Alternatively, it can be directly calculated from the expression of the
        vibrational energy.

        Args:
            data: Data holder for the MineralProperties class.
            T: Considered temperature. [K]
            v_bm: Volume of Bm at considered conditions. [cm^3/mol]
            theta_bm: Debye temperature of Bm at the considered conditions. [K]
            gamma_bm: Gruneisen parameter of Bm at the considered conditions.
            E_th_bm_0: Vibrational energy of Bm at ambient conditions. [cm^3 GPa mol^−1]
            E_th_bm: Vibrational energy of Bm at the considered conditions.
                     [cm^3 GPa mol^−1]

        Returns:
            Partial derivative of the vibrational energy with respect to temperature
            for Bm. [cm^3 GPa mol^−1 K^-1]
        """
        return super()._E_th_dv(
            T, v_bm, theta_bm, gamma_bm, E_th_bm_0, E_th_bm, 5, data.R
        )

    def _E_th_dT(self, data, T: float, theta_bm: float, E_th_bm: float) -> float:
        """Calculates the derivative of the vibrational energy wrt temperature for Bm.

        This function calculates the derivative of the vibrational energy with respect
        to temperature for Bridgmanite (Bm) at the considered conditions.

        It corresponds to the third equation in (B6) of Jackson and Rigden (1996).
        Alternatively, it can be directly calculated from the expression of the
        vibrational energy.

        Args:
            data: Data holder for the MineralProperties class.
            T: Considered temperature. [K]
            theta_bm: Debye temperature of Bm at the considered conditions. [K]
            E_th_bm: Vibrational energy of Bm at the considered conditions.
                     [cm^3 GPa mol^−1]

        Returns:
            Partial derivative of the vibrational energy with respect to temperature
            for Bm. [cm^3 GPa mol^−1 K^-1]
        """
        return super()._E_th_dT(T, theta_bm, E_th_bm, 5, data.R)

    def _alpha(
        self, data, T: float, k_v_bm: float, theta_bm: float, gamma_bm: float,
        v_bm: float, E_th_bm: float, E_th_bm_0: float, E_th_bm_dv: float
    ) -> float:
        """Calculates the thermal expansion coefficient of Bridgmanite.

        This function calculates the thermal expansion coefficient of Bridgmanite (Bm)
        at the considered conditions.

        It corresponds to the sixth equation in (B5) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            T: Considered temperature. [K]
            k_v_bm: Isothermal bulk modulus of Bm at ambient temperature. [GPa]
            theta_bm: Debye temperature of Bm at the considered conditions. [K]
            gamma_bm: Gruneisen parameter of Bm at the considered conditions.
            v_bm: Volume of Bm at considered conditions. [cm^3/mol]
            E_th_bm: Vibrational energy of Bm at the considered conditions.
                     [cm^3 GPa mol^−1]
            E_th_bm_0: Vibrational energy of Bm at ambient conditions. [cm^3 GPa mol^−1]
            E_th_bm_dv: Partial derivative of the vibrational energy with respect to
                        temperature for Bm. [cm^3 GPa mol^−1 K^-1]

        Returns:
            Thermal expansion coefficient of Bm. [K^-1]
        """
        # Partial derivative of the vibrational energy with respect to temperature
        E_th_bm_dT = self._E_th_dT(data, T, theta_bm, E_th_bm)
        return super()._alpha(
            v_bm, gamma_bm, k_v_bm, E_th_bm_0, E_th_bm, E_th_bm_dv, E_th_bm_dT,
            data.q_bm
        )

    def _MGD(
        self, data, T: float, P: float, v_bm: float, x_mgsio3: float, x_fesio3: float,
        x_fealo3: float, x_fe2o3: float, x_al2o3: float
    ) -> float:
        """Implements the Mie-Gruneisen-Debye EOS for Bridgmanite.

        This function calculates the residue of the Mie-Gruneisen-Debye equation of
        state applied to Bridgmanite (Bm) at the considered conditions. It can be used
        to obtain one of the conditions among the pressure, temperature, volume,
        knowing the remaining two.

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
            Residue of the Mie-Gruneisen-Debye EOS for Bm. [GPa]
        """
        # Volume of Bm at ambient conditions
        v_bm_0 = self._v_bm_0(data, x_mgsio3, x_fesio3, x_fealo3, x_fe2o3, x_al2o3)
        # Volume ratio
        v_ratio = v_bm_0 / v_bm
        # Isothermal bulk modulus of Bm at ambient conditions
        k_bm_0 = self._k_bm_0_VRH_average(
            data, x_mgsio3, x_fesio3, x_fealo3, x_fe2o3, x_al2o3, v_bm_0
        )
        # Gruneisen parameter
        gamma_bm = self._gamma(data, v_ratio)
        # Third-order Birch–Murnaghan isothermal equation of state
        BM3_bm = self._BM3(data, P, v_ratio, k_bm_0)
        # Vibrational energy at T
        E_th_bm = self._E_th(data, T, v_ratio)
        # Vibrational energy at ambient conditions
        E_th_bm_0 = self._E_th(data, 300, v_ratio)
        return super()._MGD(v_bm, gamma_bm, E_th_bm_0, E_th_bm, BM3_bm)

    def _g_t0(
        self, data, v_bm: float, x_mgsio3: float, x_fesio3: float, x_fealo3: float,
        x_fe2o3: float, x_al2o3: float
    ) -> float:
        """Calculates the shear modulus of Bridgmanite at ambient temperature.

        This function calculates the shear modulus of Bridgmanite (Bm) at ambient
        temperature.

        It corresponds to the eq. (21) of Bina and Helffrich (1992) when assuming that
        the second-order terms can be neglected.

        Args:
            data: Data holder for the MineralProperties class.
            v_bm: Volume of Bm at considered conditions. [cm^3/mol]
            x_mgsio3: Molar concentration of MgSiO3 in Bm.
            x_fesio3: Molar concentration of FeSiO3 in Bm.
            x_fealo3: Molar concentration of FeAlO3 in Bm.
            x_fe2o3: Molar concentration of Fe2O3 in Bm.
            x_al2o3: Molar concentration of Al2O3 in Bm.

        Returns:
            Shear modulus of Bm at ambient temperature. [GPa]
        """
        # Volume of Bm at ambient conditions
        v_bm_0 = self._v_bm_0(data, x_mgsio3, x_fesio3, x_fealo3, x_fe2o3, x_al2o3)
        # Volume ratio
        v_ratio = v_bm_0 / v_bm
        # Isothermal bulk modulus of Bm at ambient conditions
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
        return super()._g_t0(v_ratio, k_bm_0, g_bm_0, g_prime_bm)

    def _g(
        self, data, T: float, v_bm: float, x_mgsio3: float, x_fesio3: float,
        x_fealo3: float, x_fe2o3: float, x_al2o3: float
    ) -> float:
        """Calculates the shear modulus of Bridgmanite.

        This function calculates the shear modulus of Bridgmanite (Bm) at the
        considered conditions.

        Following eq. (38) of Bina and Helffrich (1992), the temperature dependence
        of the shear modulus is assumed to be constant, so that it can be simply
        calculated from its temperature derivative.
        It should however be noted that it is a rough estimation that is unlikely to be
        valid. 

        Args:
            data: Data holder for the MineralProperties class.
            T: Considered temperature. [K]
            v_bm: Volume of Bm at considered conditions. [cm^3/mol]
            x_mgsio3: Molar concentration of MgSiO3 in Bm.
            x_fesio3: Molar concentration of FeSiO3 in Bm.
            x_fealo3: Molar concentration of FeAlO3 in Bm.
            x_fe2o3: Molar concentration of Fe2O3 in Bm.
            x_al2o3: Molar concentration of Al2O3 in Bm.

        Returns:
            Shear modulus of Bm. [GPa]
        """
        # Shear modulus at ambient temperature
        g_bm_t0 = self._g_t0(data, v_bm, x_mgsio3, x_fesio3, x_fealo3, x_fe2o3, x_al2o3)
        return g_bm_t0 + data.g_dot_bm * (T - 300)

    def _k_t(
        self, data, k_v_bm: float, gamma_bm: float, v_bm: float, E_th_bm: float,
        E_th_bm_0: float, E_th_bm_dv: float
    ) -> float:
        """Calculates the isothermal bulk modulus of Bridgmanite.

        This function calculates the isothermal bulk modulus of Bridgmanite (Bm) at
        the considered conditions.

        It corresponds to the fifth equation in (B5) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            k_v_bm: Isothermal bulk modulus of Bm at ambient temperature. [GPa]
            gamma_bm: Gruneisen parameter of Bm at the considered conditions.
            v_bm: Volume of Bm at considered conditions. [cm^3/mol]
            E_th_bm: Vibrational energy of Bm at the considered conditions.
                     [cm^3 GPa mol^−1]
            E_th_bm_0: Vibrational energy of Bm at ambient conditions. [cm^3 GPa mol^−1]
            E_th_bm_dv: Partial derivative of the vibrational energy with respect to
                        temperature for Bm. [cm^3 GPa mol^−1 K^-1]

        Returns:
            Isothermal bulk modulus of Bm. [GPa]
        """
        return super()._k_t(
            v_bm, gamma_bm, k_v_bm, E_th_bm_0, E_th_bm, E_th_bm_dv, data.q_bm
        )

    def _k_s(
        self, data, T: float, v_bm: float, x_mgsio3: float, x_fesio3: float,
        x_fealo3: float, x_fe2o3: float, x_al2o3: float
    ) -> float:
        """Calculates the isentropic bulk modulus of Bridgmanite.

        This function calculates the isentropic bulk modulus of Bridgmanite (Bm) at
        the considered conditions.

        It corresponds to the twelfth equation in (B5) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            T: Considered temperature. [K]
            v_bm: Volume of Bm at considered conditions. [cm^3/mol]
            x_mgsio3: Molar concentration of MgSiO3 in Bm.
            x_fesio3: Molar concentration of FeSiO3 in Bm.
            x_fealo3: Molar concentration of FeAlO3 in Bm.
            x_fe2o3: Molar concentration of Fe2O3 in Bm.
            x_al2o3: Molar concentration of Al2O3 in Bm.

        Returns:
            Isentropic bulk modulus of Bm. [GPa]
        """
        # Volume of Bm at ambient conditions
        v_bm_0 = self._v_bm_0(data, x_mgsio3, x_fesio3, x_fealo3, x_fe2o3, x_al2o3)
        # Volume ratio
        v_ratio = v_bm_0 / v_bm
        # Gruneisen parameter
        gamma_bm = self._gamma(data, v_ratio)
        # Debye temperature
        theta_bm = self._theta(data, v_ratio)
        # Isothermal bulk modulus of Bm at ambient conditions
        k_bm_0 = self._k_bm_0_VRH_average(
            data, x_mgsio3, x_fesio3, x_fealo3, x_fe2o3, x_al2o3, v_bm_0
        )
        # Vibrational energy at T
        E_th_bm = self._E_th(data, T, v_ratio)
        # Vibrational energy at ambient conditions
        E_th_bm_0 = self._E_th(data, 300, v_ratio)
        # Partial derivative of the vibrational energy with respect to volume
        E_th_bm_dv = self._E_th_dv(
            data, T, v_bm, theta_bm, gamma_bm, E_th_bm_0, E_th_bm
        )
        # Isothermal bulk modulus at ambient temperature
        k_v_bm = super()._k_v(v_ratio, k_bm_0, data.k0t_prime_bm)
        # Isothermal bulk modulus
        k_t_bm = self._k_t(
            data, k_v_bm, gamma_bm, v_bm, E_th_bm, E_th_bm_0, E_th_bm_dv
        )
        # Thermal expansion coefficient
        alpha_bm = self._alpha(
            data, T, k_v_bm, theta_bm, gamma_bm, v_bm, E_th_bm, E_th_bm_0, E_th_bm_dv
        )
        return super()._k_s(T, alpha_bm, gamma_bm, k_t_bm)

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

    Note that these functions are intended for use within the MineralProperties class
    and should not be used outside of it.
    """
    def _gamma(self, data, v_ratio: float) -> float:
        """Calculates the Gruneisen parameter of Calcio Perovskite.

        This function calculates the Gruneisen parameter of Calcio Perovskite (CaPv) at
        the considered conditions.

        It corresponds to the eq. (31) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            v_ratio: Volume ratio V0 / V of CaPv, where V0 is the volume of CaPv at
                     ambient conditions and V the volume of CaPv at the considered
                     conditions.

        Returns:
            Gruneisen parameter of CaPv at the considered conditions.
        """
        return super()._gamma(v_ratio, data.gamma_capv_0, data.q_capv)

    def _theta(self, data, v_ratio: float) -> float:
        """Calculates the Debye temperature of Calcio Perovskite.

        This function calculates the Debye temperature of Calcio Perovskite (CaPv) at
        the considered conditions.

        It corresponds to the eq. (32) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            v_ratio: Volume ratio V0 / V of CaPv, where V0 is the volume of CaPv at
                     ambient conditions and V the volume of CaPv at the considered
                     conditions.

        Returns:
            Debye temperature of CaPv at the considered conditions. [K]
        """
        # Gruneisen parameter
        gamma_capv = self._gamma(data, v_ratio)
        return super()._theta(
            data.theta_capv_0, data.gamma_capv_0, gamma_capv, data.q_capv
        )

    def _BM3(self, data, P: float, v_ratio: float) -> float:
        """Implements the third-order Birch–Murnaghan EOS for Calcio Perovskite.

        This function calculates the residue of the third-order Birch–Murnaghan
        isothermal equation of state applied to Calcio Perovskite (CaPv) at the
        considered conditions.

        It corresponds to the eq. (B1) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            P: Considered pressure. [GPa]
            v_ratio: Volume ratio V0 / V of CaPv, where V0 is the volume of CaPv at
                     ambient conditions and V the volume of CaPv at the considered
                     conditions.

        Returns:
            Residue of the third-order Birch–Murnaghan isothermal equation of state
            for CaPv. [GPa]
        """
        return super()._BM3(P, v_ratio, data.k_casio3_0, data.k0t_prime_capv)

    def _E_th(self, data, T: float, v_ratio: float) -> float:
        """Calculates the vibrational energy of Calcio Perovskite.

        This function calculates the vibrational energy of Calcio Perovskite (CaPv) at
        the considered conditions. The integral part of the expression is calculated
        separately because the method scipy.integrate.quad returns both the value and
        the error, while only the value is needed.

        It corresponds to the eq. (30) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            T: Considered temperature. [K]
            v_ratio: Volume ratio V0 / V of CaPv, where V0 is the volume of CaPv at
                     ambient conditions and V the volume of CaPv at the considered
                     conditions.

        Returns:
            Vibrational energy of CaPv at the considered conditions. [cm^3 GPa mol^−1]
        """
        # Debye temperature
        theta_capv = self._theta(data, v_ratio)
        # Integral part of the vibrational energy
        int_part_capv = super()._integral_vibrational_energy(theta_capv / T)
        return super()._E_th(T, theta_capv, int_part_capv, 5, data.R)

    def _E_th_dv(
        self, data, T: float, v_capv: float, theta_capv: float, gamma_capv: float,
        E_th_capv_0: float, E_th_capv: float
    ) -> float:
        """Calculates derivative of vibrational energy with respect to volume for CaPv.

        This function calculates the derivative of the vibrational energy with respect
        to volume for Calcio Perovskite (CaPv) at the considered conditions.

        It corresponds to the fourth equation in (B6) of Jackson and Rigden (1996).
        Alternatively, it can be directly calculated from the expression of the
        vibrational energy.

        Args:
            data: Data holder for the MineralProperties class.
            T: Considered temperature. [K]
            v_capv: Volume of CaPv at considered conditions. [cm^3/mol]
            theta_capv: Debye temperature of CaPv at the considered conditions. [K]
            gamma_capv: Gruneisen parameter of CaPv at the considered conditions.
            E_th_capv_0: Vibrational energy of CaPv at ambient conditions.
                         [cm^3 GPa mol^−1]
            E_th_capv: Vibrational energy of CaPv at the considered conditions.
                       [cm^3 GPa mol^−1]

        Returns:
            Partial derivative of the vibrational energy with respect to temperature
            for CaPv. [cm^3 GPa mol^−1 K^-1]
        """
        return super()._E_th_dv(
            T, v_capv, theta_capv, gamma_capv, E_th_capv_0, E_th_capv, 5, data.R
        )

    def _E_th_dT(self, data, T: float, theta_capv: float, E_th_capv: float) -> float:
        """Calculates the derivative of the vibrational energy wrt temperature for CaPv.

        This function calculates the derivative of the vibrational energy with respect
        to temperature for Calcio Perovskite (CaPv) at the considered conditions.

        It corresponds to the third equation in (B6) of Jackson and Rigden (1996).
        Alternatively, it can be directly calculated from the expression of the
        vibrational energy.

        Args:
            data: Data holder for the MineralProperties class.
            T: Considered temperature. [K]
            theta_capv: Debye temperature of CaPv at the considered conditions. [K]
            E_th_capv_0: Vibrational energy of CaPv at ambient conditions.
                         [cm^3 GPa mol^−1]

        Returns:
            Partial derivative of the vibrational energy with respect to temperature
            for CaPv. [cm^3 GPa mol^−1 K^-1]
        """
        return super()._E_th_dT(T, theta_capv, E_th_capv, 5, data.R)

    def _alpha(
        self, data, T: float, k_v_capv: float, theta_capv: float, gamma_capv: float,
        v_capv: float, E_th_capv: float, E_th_capv_0: float, E_th_capv_dv: float
    ) -> float:
        """Calculates the thermal expansion coefficient of Calcio Perovskite.

        This function calculates the thermal expansion coefficient of Calcio Perovskite
        (CaPv) at the considered conditions.

        It corresponds to the sixth equation in (B5) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            T: Considered temperature. [K]
            k_v_capv: Isothermal bulk modulus of CaPv at ambient temperature. [GPa]
            theta_capv: Debye temperature of CaPv at the considered conditions. [K]
            gamma_capv: Gruneisen parameter of CaPv at the considered conditions.
            v_capv: Volume of CaPv at considered conditions. [cm^3/mol]
            E_th_capv: Vibrational energy of CaPv at the considered conditions.
                       [cm^3 GPa mol^−1]
            E_th_capv_0: Vibrational energy of CaPv at ambient conditions.
                         [cm^3 GPa mol^−1]
            E_th_capv_dv: Partial derivative of the vibrational energy with respect to
                          temperature for CaPv. [cm^3 GPa mol^−1 K^-1]

        Returns:
            Thermal expansion coefficient of CaPv. [K^-1]
        """
        # Partial derivative of the vibrational energy with respect to temperature
        E_th_capv_dT = self._E_th_dT(data, T, theta_capv, E_th_capv)
        return super()._alpha(
            v_capv, gamma_capv, k_v_capv, E_th_capv_0, E_th_capv, E_th_capv_dv,
            E_th_capv_dT, data.q_capv
        )

    def _MGD(self, data, T: float, P: float, v_capv: float) -> float:
        """Implements the Mie-Gruneisen-Debye EOS for Calcio Perovskite.

        This function calculates the residue of the Mie-Gruneisen-Debye equation of
        state applied to Calcio Perovskite (CaPv) at the considered conditions. It can
        be used to obtain one of the conditions among the pressure, temperature, volume,
        knowing the remaining two.

        It corresponds to the eq. (33) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            T: Considered temperature. [K]
            P: Considered pressure. [GPa]
            v_capv: Volume of CaPv at considered conditions. [cm^3/mol]

        Returns:
            Residue of the Mie-Gruneisen-Debye EOS for CaPv. [GPa]
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
        return super()._MGD(v_capv, gamma_capv, E_th_capv_0, E_th_capv, BM3_capv)

    def _g_t0(self, data, v_ratio: float) -> float:
        """Calculates the shear modulus of Calcio Perovskite at ambient temperature.

        This function calculates the shear modulus of Calcio Perovskite (CaPv) at
        ambient temperature.

        It corresponds to the eq. (21) of Bina and Helffrich (1992) when assuming that
        the second-order terms can be neglected.

        Args:
            data: Data holder for the MineralProperties class.
            v_ratio: Volume ratio V0 / V of CaPv, where V0 is the volume of CaPv at
                     ambient conditions and V the volume of CaPv at the considered
                     conditions.

        Returns:
            Shear modulus of Bm at ambient temperature. [GPa]
        """
        return super()._g_t0(
            v_ratio, data.k_casio3_0, data.g_casio3_0, data.g_prime_casio3
        )

    def _g(self, data, T: float, v_capv: float) -> float:
        """Calculates the shear modulus of Calcio Perovskite.

        This function calculates the shear modulus of Calcio Perovskite (CaPv) at the
        considered conditions.

        Following eq. (38) of Bina and Helffrich (1992), the temperature dependence
        of the shear modulus is assumed to be constant, so that it can be simply
        calculated from its temperature derivative.
        It should however be noted that it is a rough estimation that is unlikely to be
        valid. 

        Args:
            data: Data holder for the MineralProperties class.
            T: Considered temperature. [K]
            v_capv: Volume of CaPv at considered conditions. [cm^3/mol]

        Returns:
            Shear modulus of CaPv. [GPa]
        """
        # Volume ratio
        v_ratio = data.v_casio3_0 / v_capv
        # Shear modulus at ambient temperature
        g_capv_t0 = self._g_t0(data, v_ratio)
        return g_capv_t0 + data.g_dot_capv * (T - 300)

    def _k_t(
        self, data, k_v_capv: float, gamma_capv: float, v_capv: float,
        E_th_capv: float, E_th_capv_0: float, E_th_capv_dv: float
    ) -> float:
        """Calculates the isothermal bulk modulus of Calcio Perovskite.

        This function calculates the isothermal bulk modulus of Calcio Perovskite (CaPv)
        at the considered conditions.

        It corresponds to the fifth equation in (B5) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            k_v_capv: Isothermal bulk modulus of CaPv at ambient temperature. [GPa]
            gamma_capv: Gruneisen parameter of CaPv at the considered conditions.
            v_capv: Volume of CaPv at considered conditions. [cm^3/mol]
            E_th_capv: Vibrational energy of CaPv at the considered conditions.
                       [cm^3 GPa mol^−1]
            E_th_capv_0: Vibrational energy of CaPv at ambient conditions.
                         [cm^3 GPa mol^−1]
            E_th_capv_dv: Partial derivative of the vibrational energy with respect to
                          temperature for CaPv. [cm^3 GPa mol^−1 K^-1]

        Returns:
            Isothermal bulk modulus of CaPv. [GPa]
        """
        return super()._k_t(
            v_capv, gamma_capv, k_v_capv, E_th_capv_0, E_th_capv, E_th_capv_dv,
            data.q_capv
        )

    def _k_s(self, data, T: float, v_capv: float) -> float:
        """Calculates the isentropic bulk modulus of Calcio Perovskite.

        This function calculates the isentropic bulk modulus of Calcio Perovskite (CaPv)
        at the considered conditions.

        It corresponds to the twelfth equation in (B5) of Jackson and Rigden (1996).

        Args:
            data: Data holder for the MineralProperties class.
            T: Considered temperature. [K]
            v_capv: Volume of CaPv at considered conditions. [cm^3/mol]

        Returns:
            Isentropic bulk modulus of CaPv. [GPa]
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
            data, T, v_capv, theta_capv, gamma_capv, E_th_capv_0, E_th_capv
        )
        # Isothermal bulk modulus at ambient temperature
        k_v_capv = super()._k_v(v_ratio, data.k_casio3_0, data.k0t_prime_capv)
        # Isothermal bulk modulus
        k_t_capv = self._k_t(
            data, k_v_capv, gamma_capv, v_capv, E_th_capv, E_th_capv_0, E_th_capv_dv
        )
        # Thermal expansion coefficient
        alpha_capv = self._alpha(
            data, T, k_v_capv, theta_capv, gamma_capv, v_capv, E_th_capv, E_th_capv_0,
            E_th_capv_dv
        )
        return super()._k_s(T, alpha_capv, gamma_capv, k_t_capv)
