CANDEL
======

**CANDEL** (*CA*\libration and *N*\ormalization of the *D*\istanc\ *E* *L*\adder)
is a JAX-based framework for peculiar-velocity inference and distance-ladder
calibration.

CANDEL forward-models distance-indicator observables and redshift while
marginalising over latent variables such as distance and absolute magnitude.
Posterior sampling uses the No-U-Turn Sampler (NUTS) from
`NumPyro <https://github.com/pyro-ppl/numpyro>`_, with JAX providing automatic
differentiation and JIT compilation throughout.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   configuration
   models
   api/index

Tutorials
---------

For practical examples, please refer to the Jupyter notebooks in the
`notebooks/ <https://github.com/Richard-Sti/CANDEL/tree/main/notebooks>`_
directory. These notebooks demonstrate specific workflows, such as:

- **example.ipynb**: A general overview of the package.
- **paper_SH0ES**: Replicating the results from `Stiskalek et al. (2025) <https://arxiv.org/abs/2509.09665>`_.
- **paper_CCHP**: Analysis of the CCHP TRGB calibration.
