# Calibration And Normalization of the DistancE Ladder


## TODO

### General
- [x] Implement an efficient submission script for many runs.

- [ ] Add a strategy to generate CF4 mocks based on sampled CF4 data. This should ideally be a resampling of the true CF4 magnitude and linewidth distributions, at random sky positions. The mocks should also account for both the homogeneous and inhomogeneous Malmquist bias.

### Hubble Dipole
- Focus the main inference solely on CF4 TFR W1 because of its uniform sky coverage, and then add a set of mock calibration to test if there is any signal.
- Implement Boubel likelihood?

### Hubble Calibration

## Completed TODO

### General
- [x] Fix subsampling to preserve calibrator assignment.
- [x] Add evidence computation (Laplace, harmonic mean).
- [x] Add numerically stable treatment of `V_ext` when computing model evidence.
- [x] Add a code, given a model to invert the likelihood.
- [x] Add option to save samples.
- [x] Think about folder organization.
- [x] Break degeneracy when sampling `h` and `a_TFR`
- [x] Add code to remove CF4 outliers
- [x] Add support for interpolating fields.
- [x] Add support for the Carrick linear theory field.
- [x] Implement more efficient/safe grid when marginalising over radial distance.
- [x] Add MNR-compatible versions of likelihoods.
- [x] Add selection in `eta`.
- [x] Add selection in `mag`.

### Hubble Dipole
- [x] Add option for dipole variation in `a_TFR`.

### Hubble Calibration
- [x] Add likelihood term for absolute calibration when available.