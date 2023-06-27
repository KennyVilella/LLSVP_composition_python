API: Calculation of spin configuration
======================================

This subpage contains the API documentation for the python file `_spin_configuration.py`.

The main objective of these functions is to determine the average spin state of FeO in Ferropericlase (Fp) as a function of temperature, pressure and FeO content.
The function :ref:`_calc_spin_configuration <calc_spin_configuration>` returns two arrays, the first one provides the average spin state as a function of temperature, volume and FeO content, while the second one gives the associated pressure.
Combining these two arrays is necessary to obtain the average spin state as a function of temperature, pressure and FeO content.

The average spin state is calculated by minimizing the Helmholtz free energy (F), as it corresponds to the most stable configuration.
More specifically, the function searches for the value of the average spin state resulting in a derivative of F with respect to the average spin state equal to zero.
Since multiple local minima may exist, the calculation is performed with three different initial conditions, and the solution with the lowest F value is selected.

This model is based on the work of Sturhahn et al. (2005) and this specific version is described in Vilella et al. (2015).

.. _calc_spin_configuration:
.. autofunction::  src._spin_configuration._calc_spin_configuration

.. autofunction::  src._spin_configuration._energy_equation

.. autofunction::  src._spin_configuration._splitting_energy
