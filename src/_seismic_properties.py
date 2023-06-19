"""

This file is associated with the article:
"Constraints on the composition and temperature of LLSVPs from seismic properties of
lower mantle minerals" by K. Vilella, T. Bodin, C.-E. Boukare, F. Deschamps, J. Badro,
M. D. Ballmer, and Y. Li

These functions should not be used outside the class MineralProperties.

Typical usage example:


Copyright, 2023,  Vilella Kenny.
"""
import numpy as np
from _eos_implementation import _EOS_fp, _EOS_bm, _EOS_capv
#======================================================================================#
#                                                                                      #
#    Starting implementation of functions used to calculate the seismic properties     #
#                                                                                      #
#======================================================================================#
def _calc_seismic_properties(self, spin_config, P_table):
    """
    """
    # Loading utility class for mineral properties calculation
    fp_eos = _EOS_fp() 
    bm_eos = _EOS_bm()
    capv_eos = _EOS_capv()


def _calc_v_p(self, v_phi, v_s):
    """
    """
    return np.sqrt(v_phi * v_phi + (4/3) * v_s * v_s)
