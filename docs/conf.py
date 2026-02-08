# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add the project root directory to sys.path so that Sphinx can find the modules
sys.path.insert(0, os.path.abspath('..'))

# Mock imports for problematic dependencies during documentation build
from unittest.mock import MagicMock

class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

MOCK_MODULES = [
    'tensorflow', 'tensorflow.keras', 'tensorflow.keras.layers',
    'tensorflow.keras.models', 'tensorflow.keras.optimizers',
    'tensorflow.keras.losses', 'tensorflow.keras.metrics',
    'sklearn', 'sklearn.ensemble', 'sklearn.linear_model',
    'sklearn.model_selection', 'sklearn.preprocessing',
    'sklearn.metrics', 'pandas'
]
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

# -- Project information -----------------------------------------------------

project     = 'General Python Utilities'
copyright   = '2025, Maksymilian Kliczkowski'
author      = 'Maksymilian Kliczkowski'
version     = '1.1.0'
release     = '1.1.0'

# -- General configuration ---------------------------------------------------

# The master toctree document
master_doc = 'index'

# Add any Sphinx extension module names here.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages'
]

# Napoleon settings for parsing NumPy/Google style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# AutoDoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns to ignore when looking for source files.
exclude_patterns = ['_build', 'build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages
html_theme = 'sphinx_rtd_theme'

# Theme options
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
}
