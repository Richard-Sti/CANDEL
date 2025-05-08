# Calibration And Normalization of the DistancE Ladder


## TODO

### General
- [x] Add support for interpolating fields.
- [x] Add support for the Carrick linear theory field.
- [x] Implement more efficient/safe grid when marginalising over radial distance.
- [x] Add MNR-compatible versions of likelihoods.

- [ ] Implement an efficient submission script for many runs.
- [ ] Think about a strategy for generating CF4-like mock catalogs.

### Hubble Dipole

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

### Hubble Dipole
- [x] Add option for dipole variation in `a_TFR`.

### Hubble Calibration
- [x] Add likelihood term for absolute calibration when available.