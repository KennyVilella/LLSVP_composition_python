import os
import sys

# Importing path
_filedir = os.path.dirname(__file__)
_rootdir = os.path.abspath(os.path.join(_filedir, ".."))
_srcdir = os.path.abspath(os.path.join(_filedir, "../src"))
sys.path.insert(0, _rootdir)
sys.path.insert(0, _srcdir)

project = "LLSVP composition calculator"
version = "1.0.0"
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
smv_tag_whitelist = r"^v\d+\.\d+\.\d+$"
smv_branch_whitelist = r"^(main)$"
