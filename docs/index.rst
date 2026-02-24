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
   models
   api/index
