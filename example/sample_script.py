"""Example script to launch a short simulation.

The purpose of this script is to provide a usage example of the simulator implemented
in this repository. The range of mineral compositions investigated is modified via an
input dictionnary to produce a short simulation, while defaults parameters are used for
the mineral properties.

Typical usage example:

  python sample_script.py

Copyright, 2023,  Vilella Kenny.
"""
import os
import sys
# Importing path and main class
_filedir = os.path.dirname(__file__)
_rootdir = os.path.abspath(os.path.join(_filedir, '..'))
sys.path.insert(0, _rootdir)
from src.__init__ import MineralProperties
#======================================================================================#
#                                                                                      #
#     Starting implementation of a sample script showing the usage of the simulator    #
#                                                                                      #
#======================================================================================#
# Building simple simulator
LLSVP_compositions_simulator = MineralProperties()

# Setting up simulator conditions
conditions = {
         "dT_min" : 100, "dT_max": 200,
         "p_capv_min": 0.10, "p_capv_max": 0.20,
         "iron_content_min": 0.12, "iron_content_max": 0.14,
         "al_content_min": 0.05, "al_content_max": 0.06,
         "ratio_fe_bm_min": 0.5, "ratio_fe_bm_max":0.6
}

# Calculating mineral proeprties
LLSVP_compositions_simulator.calc_mineral_properties(conditions)
