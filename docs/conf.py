# Configuration file for the Sphinx documentation builder.

project = "candel"
copyright = "2025, Richard Stiskalek"
author = "Richard Stiskalek"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Theme -------------------------------------------------------------------
html_theme = "sphinx_rtd_theme"

# -- Autodoc -----------------------------------------------------------------
autodoc_member_order = "bysource"

# -- Napoleon ----------------------------------------------------------------
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "numpyro": ("https://num.pyro.ai/en/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}
