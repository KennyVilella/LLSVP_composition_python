API: Classes implementing the equation of state for each mineral
================================================================

This subpage contains the API documentation for the python file ``_eos_implementation.py``.

The file consists of an abstract base class (:ref:`_EOS) <EOS>`) that implements the fundamental formulas required for calculating mineral properties in the simulator.
These base utilities are then applied in separate derived classes specific to Ferropericlase (:ref:`_EOS_fp <EOS_fp>`), Bridgmanite (:ref:`_EOS_bm <EOS_bm>`) and Calcio Perovskite (:ref:`_EOS_capv <EOS_capv>`).

These functions are primarily based on the Mie-Gruneisen-Debye equation of state (EOS), with additional considerations for shear modulus based on the model presented in `Bina and Helffrich (1992)`_.
A detailed description of the EOS can be found in `Jackson and Rigden (1996)`_ and in the supplementary material of `Vilella et al. (2021)`_.

.. _EOS:
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

.. _Bina and Helffrich (1992): https://doi.org/10.1146/annurev.ea.20.050192.002523
.. _Jackson and Rigden (1996): https://doi.org/10.1016/0031-9201(96)03143-3
.. _Vilella et al. (2021): https://doi.org/10.1016/j.epsl.2020.116685
