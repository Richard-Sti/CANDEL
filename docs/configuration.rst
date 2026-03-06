Configuration Guide
===================

CANDEL experiments are defined using TOML configuration files. These files
specify the data paths, model parameters, priors, and inference settings.

Basic structure
---------------

A typical configuration file is organized into several sections:

.. code-block:: toml

   # Root directories for outputs and data
   root_main = "./results/my_experiment"
   root_data = "./data"
   fname_output = "samples.h5"

   [model]
   # Model type and parameters
   name = "TRGB"
   Om = 0.3
   r_limits_malmquist = [0.01, 150]
   which_selection = "TRGB_magnitude"

   [model.priors]
   # Priors for free parameters
   H0 = { dist = "uniform", low = 40.0, high = 100.0 }
   M_TRGB = { dist = "normal", mean = -4.0, std = 0.5 }
   sigma_int = { dist = "half_normal", std = 0.1 }

   [inference]
   # Sampler settings
   num_warmup = 500
   num_samples = 1000
   num_chains = 4

Path handling
-------------

- ``root_main``: The directory where all output files (samples, plots, logs)
  will be saved.
- ``root_data``: The base directory for input data files (catalogues, fields).
- Relative paths in the configuration are automatically resolved relative to
  the respective root directory.

Priors
------

Priors are specified in the ``[model.priors]`` section. Supported distributions
include:

- ``uniform``: Requires ``low`` and ``high``.
- ``normal``: Requires ``mean`` and ``std``.
- ``half_normal``: Requires ``std``.
- ``delta``: Fixed value, requires ``value``.
- ``log_normal``: Requires ``mean`` and ``std`` of the underlying normal.

Model settings
--------------

The ``[model]`` section contains settings specific to the distance indicator
and the physical model:

- ``Om``: Matter density parameter :math:`\Omega_m`.
- ``use_reconstruction``: Boolean, whether to use a reconstructed density/velocity field.
- ``which_selection``: Type of selection function to apply (e.g., ``"TRGB_magnitude"``, ``"redshift"``, or ``"none"``).

Inference settings
------------------

The ``[inference]`` section controls the NUTS sampler:

- ``num_warmup``: Number of adaptation steps.
- ``num_samples``: Number of posterior samples to keep per chain.
- ``num_chains``: Number of independent Markov chains.
- ``seed``: Random seed for reproducibility.
