Documentation for the LLSVP_composition_python repository
=========================================================

.. note::

    This simulator implements the model developed in `Vilella et al. (2021)`_.
    Please read the associated article for more information on the model.

The aim of this module is to provide some constraints on the composition and temperature of the Large Low Shear Velocity Provinces (LLSVPs) observed in the lowermost mantle.
To do so, this simulator calculates the seismic anomalies associated with a wide range of potential LLSVPs compositions under various temperature conditions.
The subsequent comparison of the calculated seismic anomalies with observations allowed to constrain the potential composition and temperature of LLSVPs.

This process is highly challenging since there are three main sources of uncertainties.
Uncertainties on the seismic signature of the LLSVPs due to inconsistencies between different seismic models but also intrinsic heterogeneity of the LLSVPs.
Uncertainties on the mineral properties since their experimental measurement at lower mantle conditions is challenging.
Uncertainties on the theoretical models used to extrapolate the mineral properties to different pressure and temperature conditions.

As a consequence, the density and seismic wave speeds provided by this simulator, particularly the P-wave and S-wave velocities, are inherently uncertain.
It is therefore essential to approach the results with caution when analyzing specific mineral compositions.
Instead, it is recommended to focus on the overall trends shown in the results, as they are likely to be more robust against the aforementioned uncertainties.

Note that this simulator considers rock assemblages composed only of Ferropericlase, Bridgmanite and Calcio Perovskite and is primarily designed for modeling the lowermost mantle.
For instance, the potential presence of Post-Perovskite or other minerals such as Stishovite is not accounted for.

Contents
--------

A :doc:`quickstart` is provided with instructions to install the simulator and execute simple simulations.
For more information on the code itself or the usage of the simulator, an `API <api.html>`_ documentation is available with information on all classes and functions as well as detailed information on the implementation itself.

.. toctree::
   :maxdepth: 1
   :hidden:

   Home <self>
   quickstart
   General documentation <api>

.. _Vilella et al. (2021): https://doi.org/10.1016/j.epsl.2020.116685
