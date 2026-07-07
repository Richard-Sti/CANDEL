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

Create a TOML file defining your experiment. For a small Tully--Fisher run,
place the following in ``scripts/runs/configs/quickstart.toml`` so it can
inherit the maintained default configuration:

.. code-block:: toml

   base = ["config.toml"]

   [inference]
   model = "TFRModel"
   num_warmup = 500
   num_samples = 1000
   num_chains = 2

   [io]
   catalogue_name = "CF4_W1"
   fname_output = "results/quickstart/tfr_samples.hdf5"

   [pv_model]
   kind = "Vext"

   [model]
   which_selection = "none"

Machine-local paths such as ``root_main``, ``root_data``, and
``root_results`` should live in ``local_config.toml`` at the repository root,
not in reusable run configs.

3. Run the inference
--------------------

Use the provided script to launch the sampler:

.. code-block:: bash

   python scripts/runs/main.py --config scripts/runs/configs/quickstart.toml

CANDEL will automatically detect available GPUs and use one per chain if
possible.

4. Inspect the results
----------------------

Once the sampling is complete, results are saved to an HDF5 file. You can
visualise the posterior distributions using the utility functions:

.. code-block:: python

   from candel.plotting.corner import plot_corner_from_hdf5

   plot_corner_from_hdf5("./results/quickstart/tfr_samples.hdf5")

This will generate a corner plot of the free parameters in your model.
