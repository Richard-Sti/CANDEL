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

- ``name``: The name of the PV model class (e.g., ``"TFR"``, ``"SN"``, ``"PantheonPlus"``, ``"FP"``).
- ``which_run``: For H0 models, specifies the pipeline to run (``"CH0"``, ``"CCHP"``, ``"EDD_TRGB"``, ``"CCHP_CSP"``).
- ``Om``: Matter density parameter :math:`\Omega_m`.
- ``use_reconstruction``: Boolean, whether to use a reconstructed density/velocity field.
- ``which_selection``: Type of selection function to apply (e.g., ``"TRGB_magnitude"``, ``"redshift"``, or ``"none"``).

Model and Catalogue Names
-------------------------

When running PV models, the ``[inference]`` and ``[io]`` sections must specify
the model and catalogue:

.. code-block:: toml

   [inference]
   model = "TFRModel"

   [io]
   catalogue_name = "CF4_W1"

For joint inferences, these can be lists:

.. code-block:: toml

   [inference]
   model = ["TFRModel", "SNModel"]

   [io]
   catalogue_name = ["CF4_W1", "Foundation"]

Batch Generation
----------------

For large experiments (e.g., parameter sweeps), CANDEL includes a template-based
configuration generator:

.. code-block:: bash

   python scripts/runs/generate_tasks.py [index]

This script reads a template TOML and applies a grid of overrides defined in the
``manual_overrides`` dictionary within the script. It generates:

1. A directory of ``.toml`` files, one for each combination of parameters.
2. A ``tasks_[index].txt`` file containing the paths to all generated configs.

The task list can be used to launch parallel jobs on a cluster, for example
using a SLURM array.

Inference settings
------------------

The ``[inference]`` section controls the NUTS sampler:

- ``num_warmup``: Number of adaptation steps.
- ``num_samples``: Number of posterior samples to keep per chain.
- ``num_chains``: Number of independent Markov chains.
- ``seed``: Random seed for reproducibility.
