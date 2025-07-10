# Configuration file for the Sphinx documentation builder.
import os
import sys
import sphinx_rtd_theme

#sys.path.insert(0, os.path.abspath("../raimitigations"))
sys.path.insert(0, os.path.abspath("../eureka_ml_insights"))
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Eureka ML Insights'
copyright = '2025, MSR AI Frontiers'
author = 'MSR AI Frontiers'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False  # we'll control skipping __init__ differently




# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


extensions = [
                "sphinx.ext.napoleon",  # for Google-style docstrings
                "sphinx_autodoc_typehints",  # optional, for type hints
                "sphinx.ext.autodoc",
                "sphinx.ext.githubpages",
                "sphinx_rtd_theme",
                "nbsphinx",
                "sphinx_gallery.load_style",
                "sphinx.ext.graphviz",
                "sphinx.ext.inheritance_diagram",
                "sphinx.ext.mathjax",
                "IPython.sphinxext.ipython_console_highlighting"
            ]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "prompt_templates",
    "secret_management",
    "tests",
    "utils",
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = "classic"
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_context = {
    'display_github': True,
    "github_repo": "microsoft/responsible-ai-toolbox-mitigations", # Repo name
    "github_version": "main", # Version
    "conf_py_path": "/docs/", # Path in the checkout to the docs root
}