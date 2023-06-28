.. _quickstart:

Quickstart guide
================

Installation
------------

First, clone this repository by running the following command in your terminal:

.. code-block::

   git clone https://github.com/KennyVilella/LLSVP_composition_python

Go to the repository directory and install the Python package:

.. code-block::

   cd LLSVP_composition_python
   pip3 install -e .

This will install the necessary dependencies if they are not already installed.

Launching a simulation
----------------------

An example script to run the simulator is provided in the ``example`` folder.
To execute the script, use the following command:

.. code-block::

   python3 <path_to_repository>/example/sample_script.py

This will run a quick simulation and save the results in the ``results`` folder.


To customize the range of mineral compositions investigated, you can modify the input conditions in the ``conditions`` dictionary.
It is also possible to customize most of parameters used in the simulator by providing a dictionary to the ``MineralProperties`` class.
However, it is generally not recommended to modify these parameters without a thorough understanding of their impact.

Analyzing the results
---------------------

Currently, there is no specific script available for analyzing the results of the simulator.
However, you can easily read the data outputted by the simulator and create your own analysis scripts using your preferred tools.
The final output files, by default located in the ``results`` folder, have "processed" included in their names.

Each line in these files corresponds to an investigated composition and lists the following properties in order: 
temperature contrast against the ambient mantle, proportion of Calcio Perovskite, proportion of Bridgmanite, FeO content, Al2O3 content, oxidation state in Bridgmanite, density anomaly, bulk sound speed anomaly, S-wave speed anomaly, and P-wave speed anomaly.
