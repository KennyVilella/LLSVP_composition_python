API: Classes implementing the equation of state for each mineral
================================================================

This subpage contains the API documentation for the python file `_eos_implementation.py`.

The file consists of an abstract `base class`_ that implements the fundamental formulas required for calculating mineral properties in the simulator.
These base utilities are then applied in separate derived classes specific to Ferropericlase (`EOS_fp`_), Bridgmanite (`EOS_bm`_) and Calcio Perovskite (`EOS_capv`_).

These functions are primarily based on the Mie-Gruneisen-Debye equation of state (EOS), with additional considerations for shear modulus based on the model presented in Bina and Helffrich (1992).
A detailed description of the EOS can be found in Jackson and Rigden (1996) and in the supplementary material of Vilella et al. (2021).

.. _base class:
.. autoclass:: _eos_implementation._EOS
    :members:

.. _EOS_fp:
.. autoclass:: _eos_implementation._EOS_fp
    :members:

.. _EOS_bm:
.. autoclass:: _eos_implementation._EOS_bm
    :members:

.. _EOS_capv:
.. autoclass:: _eos_implementation._EOS_capv
    :members:
