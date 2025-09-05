# CANDEL

CANDEL (Calibration and Normalization of the Distance Ladder) is a JAX-based framework for calibrating the cosmic distance ladder and modeling velocity fields using observables such as Tully–Fisher galaxies, supernovae, and cluster scaling relations. It leverages NumPyro for probabilistic programming and provides tools for Bayesian inference and model comparison.


## Installation
```
git clone git@github.com:Richard-Sti/CANDEL.git

# Go to the cloned directory
cd CANDEL

# Create a virtual environment
python -m venv venv_candel
source venv_candel/bin/activate
python -m pip install --upgrade pip && python -m pip install --upgrade setuptools

# Finally install the cloned package
python -m pip install -e .
```

To enable model evidence computation, install the [harmonic](https://github.com/astro-informatics/harmonic) package.


If the package is not recognised in a notebook, add the project directory to the Python path manually, e.g.:
```
import sys
sys.path.insert(0, "/Users/rstiskalek/Projects/candel")
```


## TODO
