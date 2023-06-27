.. _api:

API
===

This section provides a comprehensive API documentation for the simulator, along with general documentation for each component.
In particular, the parameters of the simulator are described in the :doc: `class MineralProperties <_api/mineral_properties>` documentation.

The simulator can be divided into three main units.
The first unit, implemented in :doc: `Spin configuration <_api/spin_configuration>`, calculates the average spin state of FeO in Ferropericlase (Fp) as a function of temperature, volume, and FeO content.

The second unit, described in :doc: `Mineral composition <_api/mineral_composition>`, is responsible for determining the mineral composition of the rock assemblages.
It takes the investigated mineral compositions and calculates additional properties that provide a comprehensive characterization of the rock assemblages.

The final step is performed in :doc: `Seismic anomalies <_api/seismic_anomalies>`, where the information obtained from the previous units is used to calculate the seismic anomalies of the rock assemblages compared to a specified ambient mantle.

In addition to these three units, :doc: `EOS implementation <_api/eos_implementation>` aggregates all the thermodynamic relationships necessary for calculating the properties of the minerals used in the simulation.

.. toctree::
   :maxdepth: 1

   Home <self>
   class MineralProperties <_api/mineral_properties>
   EOS implementation <_api/eos_implementation>
   Spin configuration <_api/spin_configuration>
   Mineral composition <_api/mineral_composition>
   Seismic anomalies <_api/seismic_anomalies>
