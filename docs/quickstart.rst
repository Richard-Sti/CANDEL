Quickstart Guide
================

This guide demonstrates how to perform a basic peculiar-velocity inference
using CANDEL.

1. Installation
---------------

First, clone the repository and install the package:

.. code-block:: bash

   git clone git@github.com:Richard-Sti/CANDEL.git
   cd CANDEL
   python -m venv venv_candel
   source venv_candel/bin/activate
   pip install -e .

2. Prepare a configuration
--------------------------

Create a TOML file (e.g., ``config.toml``) defining your experiment. Here is a
minimal example for a Tully--Fisher inference:

.. code-block:: toml

   root_main = "/path/to/CANDEL/"
   # root_data and root_results default to <root_main>/data and <root_main>/results
   fname_output = "tfr_test/samples.h5"

   [model]
   name = "TFR"
   Om = 0.3
   r_limits_malmquist = [0.01, 200]
   which_selection = "none"

   [model.priors]
   a_TFR = { dist = "normal", mean = -21.0, std = 0.5 }
   b_TFR = { dist = "normal", mean = -7.5, std = 0.5 }
   sigma_int = { dist = "half_normal", std = 0.2 }
   sigma_v = { dist = "half_normal", std = 150.0 }

   [inference]
   num_warmup = 500
   num_samples = 1000
   num_chains = 2

3. Run the inference
--------------------

Use the provided script to launch the sampler:

.. code-block:: bash

   python scripts/runs/main.py --config config.toml

CANDEL will automatically detect available GPUs and use one per chain if
possible.

4. Inspect the results
----------------------

Once the sampling is complete, results are saved to an HDF5 file. You can
visualise the posterior distributions using the utility functions:

.. code-block:: python

   from candel.util import plot_corner_from_hdf5

   plot_corner_from_hdf5("./results/tfr_test/samples.h5")

This will generate a corner plot of the free parameters in your model.
