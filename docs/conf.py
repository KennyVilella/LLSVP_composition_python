import os
import sys
sys.path.insert(0, os.path.abspath('..'))
import src._spin_configuration
import src._mineral_composition
import src._seismic_anomalies

project = "LLSVP composition calculator"
version = "0.0.0"
copyright = "2023,  Vilella Kenny"
author = "Kenny Vilella"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_multiversion",
]

# Include description of private functions/classes/etc.
napoleon_include_private_with_doc = True

# Make parameters list more clear
napoleon_use_ivar = True

templates_path = ["_templates"]
html_theme = "sphinx_rtd_theme"

# Whitelist pattern for tags and branches used for multi-versioning
smv_tag_whitelist = r'^v\d+\.\d+\.\d+$'
smv_branch_whitelist = r'^(main)$'
