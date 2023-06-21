import os
import sys
sys.path.insert(0, os.path.abspath('..'))
import src

project = "LLSVP composition calculator"
version = "0.0.0"
copyright = "2023,  Vilella Kenny"
author = "Kenny Vilella"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
]

templates_path = ["_templates"]
html_theme = "sphinx_rtd_theme"
