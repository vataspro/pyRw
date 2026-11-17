# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyRw'
copyright = '2026, Alexis Provatas'
author = 'Alexis Provatas'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
import sys
import os
sys.path.insert(0, os.path.abspath('../..'))

extensions = [
                'sphinx.ext.autodoc', # automatic documentation from Python source
                'sphinx.ext.mathjax', # include mathjax
                'sphinxcontrib.bibtex' # add LaTeX style references
              ]

templates_path = ['_templates']
exclude_patterns = []

bibtex_bibfiles = ['ref.bib'] # references source


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']


# -- Autodoc settings
autodoc_typehints = "description"
autodoc_docstring_signature = False
