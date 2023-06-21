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
    "sphinx_multiversion",
]

templates_path = ["_templates"]
html_sidebars = {
    '**': [
        'versioning.html',
    ],
}
html_theme = "sphinx_rtd_theme"

# Whitelist pattern for tags and branches used for multiversioning
smv_tag_whitelist = r'^v\d+\.\d+$'
smv_branch_whitelist = r'^.*$'
