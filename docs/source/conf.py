# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Generalized Isolation Forest"
copyright = "2023, SC, DB"
author = "SC, DB"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "sphinx.ext.doctest",
    "autodoc2",
]

autodoc2_packages = [
    "../../risf",
]

# You need custom extensions to allow fancy stuff like math or images
myst_enable_extensions = ["dollarmath", "amsmath", "html_image"]


# # TO be able to write dollar mathematics!
# dmath_enable = True
# amsmath_enable = True

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"  # Used because of numpydocs
html_static_path = ["_static"]
