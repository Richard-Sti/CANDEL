# CANDEL

CANDEL (Calibration and Normalization of the Distance Ladder) is a JAX-based framework for calibrating the cosmic distance ladder and modeling velocity fields using observables such as Tully–Fisher galaxies, supernovae, and cluster scaling relations. It leverages NumPyro for probabilistic programming and provides tools for Bayesian inference and model comparison.


## TODO

### General
- [ ] Test the effect of galaxy biases. Try linear and quadratic.

$$
1 + b_1 \delta + \frac{b_2}{2} \delta^2
$$


### Hubble Dipole
- Focus the main inference solely on CF4 TFR W1 because of its uniform sky coverage, and then add a set of mock calibration to test if there is any signal.

- [ ] Run the scripts to get results on the data (don't forget to save it in a new folder!)
- [ ] Implement the Boubel likelihood (in part this can explain why they find such lower S8).

### Hubble Calibration
- [ ] Run an inference on the SH0ES calibrated data.
- [ ] Better understand how to implement the calibration.
