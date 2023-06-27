API: MineralProperties class
============================

This subpage contains the API documentation for the python file `__init__.py`.

The purpose of this class is to calculate the seismic anomalies produced by a wide range of mineral compositions, assumed to be close to a typical pyrolitic composition, under pressure and temperature conditions suitable for the lowermost mantle.

The main entrypoint of this program is the function `calc_mineral_properties`.
This function first loads the range of mineral compositions investigated by calling the function `_load_conditions`.
The range of composition investigated can be customized by providing a dictionary to the function `calc_mineral_properties`.
The items that can be changed are listed below:

* `P_am`: Pressure of the ambient mantle.
* `T_am`: Temperature of the ambient mantle.
* `p_capv_min`: Minimum proportion of CaPv.
* `p_capv_max`: Maximum proportion of CaPv.
* `delta_capv`: Step value for the proportion of CaPv.
* `p_bm_min`: Minimum proportion of Bm.
* `delta_bm`: Step value for the proportion of Bm.
* `dT_min`: Minimum temperature contrast against the ambient mantle.
* `dT_max`: Maximum temperature contrast against the ambient mantle.
* `delta_dT`: Step value for the temperature contrast against the ambient mantle.
* `iron_content_min`: Minimum FeO content.
* `iron_content_max`: Maximum FeO content.
* `delta_iron_content`: Step value for the FeO content.
* `al_content_min`: Minimum Al2O3 content.
* `al_content_max`: Maximum Al2O3 content.
* `delta_al`: Step value for the Al2O3 content.
* `ratio_fe_bm_min`: Minimum oxidation state in Bm.
* `ratio_fe_bm_max`: Maximum oxidation state in Bm.
* `delta_ratio_fe`: Step value for the oxidation state in Bm.

See the class `MineralProperties`_ for more information on each attributes.

The function `calc_mineral_properties` then calculates the average spin configuration of FeO in Ferropericlase (`_spin_configuration.py <spin_configuration.html>`_), the mineral compositions of the rock assemblages investigated (`_mineral_composition.py <mineral_composition.html>`_), and finally their seismic anomalies (`_seismic_anomalies.py <seismic_anomalies.html>`_).

.. _MineralProperties:
.. autoclass:: src.__init__.MineralProperties
    :members:
