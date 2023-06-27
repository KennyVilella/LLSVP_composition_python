API: Calculation of mineral composition
=======================================

This subpage contains the API documentation for the python file ``_mineral_composition.py``.

The main objective of these functions is to determine the mineral composition of rock assemblages for a wide range of input parameters.
The function :ref:`_calc_mineral_composition <calc_mineral_composition>` calculates the mineral compositions for the entire investigated parameter space, while the function :ref:`_solve_mineral_composition <solve_mineral_composition>` performs the calculation for a single composition.

The six input parameters are the temperature contrast against the ambient mantle, the proportion of Calcio Perovskite (CaPv), the proportion of Bridgmanite (Bm), the FeO content, the alumina content, and the oxidation state of iron in Bm.
These input parameters are used to calculate the molar concentration of FeO in Bm, the molar concentration of FeO in Ferropericlase (Fp), the molar concentration of AlO2 in Bm, the density of Bm, and the density of Fp.

The function :ref:`_calc_mineral_composition <calc_mineral_composition>` writes the results into separate files named after the considered temperature contrast.
It also includes a feature to resume a simulation, allowing for incremental calculations.
This is crucial as the simulation can be computationally intensive, taking up to three weeks on a single process.

Solving the equations governing this problem is challenging due to their highly non-linear nature.
If a solution is not found, a value of 0.0 is returned for each property.
To improve convergence, adjustments to the starting conditions or the non-linear solver are required, both of which can be quite challenging.
In this implementation, the calculation speed is priotized and cases where the solving step is difficult are easily skipped.
This approach is acceptable since having no solution for tens of thousands of cases is not a problem when considering tens of millions of cases.

The theoretical derivation of the equations implemented in these functions is presented in the supplementary material of Vilella et al. (2021).

.. _calc_mineral_composition:
.. autofunction:: src._mineral_composition._calc_mineral_composition

.. _solve_mineral_composition:
.. autofunction:: src._mineral_composition._solve_mineral_composition

.. autofunction:: src._mineral_composition._solve_with_fp

.. autofunction:: src._mineral_composition._set_eqs_with_fp

.. autofunction:: src._mineral_composition._oxides_content_in_bm

.. autofunction:: src._mineral_composition._solve_without_fp

.. autofunction:: src._mineral_composition._set_eqs_without_fp
