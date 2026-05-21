# CH0 Manticore Evidence-Driver Study

This note mirrors the TRGBH0 single-field diagnostic, but for the Cepheid-only H0 problem we keep the decomposition at the total-likelihood level rather than splitting by individual host.

## Question

The single-field Manticore runs sample over different COLA density and velocity fields.
We want to understand whether the preferred field is preferred because the non-selection likelihood is higher, because the selection normalisation is favourable, or because both effects move in the same direction.

## Diagnostics

For each field we decompose the saved posterior-sample likelihood into:

- `log_likelihood_total`: the full saved total likelihood.
- `log_observed_selection_per_galaxy`: summed over hosts to give the observed-selection contribution.
- `log_selection_integral`: converted to the total selection-normalisation contribution, `-N_host logS`.
- `raw total likelihood`: reconstructed as `log_likelihood_total - observed_selection_total - (-N_host logS)`.

The observed-selection term is constant across the current SN-magnitude-selection single-field runs because it depends on the observed supernova magnitudes and fixed selection threshold, not on the density-field realisation.

## Current Results

The best COLA field is field 21 under all evidence proxies currently checked:

| proxy | best field | value |
| --- | ---: | ---: |
| harmonic lnZ | 21 | -1539.961 |
| Laplace lnZ | 21 | -1553.093 |
| BIC | 21 | 3114.180 |
| AIC | 21 | 3033.302 |

For field 21, relative to the median field, the total-likelihood decomposition is:

| contribution | total delta |
| --- | ---: |
| raw total likelihood | +13.716 |
| observed selection | +0.000 |
| selection normalisation, `-logS` | +14.951 |
| total likelihood | +27.667 |

This means field 21 is preferred because both the non-selection likelihood and the selection normalisation improve relative to the median field.

## Most Favourable Selection Integral

The field with the most favourable selection-normalisation term is field 16:

- `-logS = -10.8255` per host.
- harmonic lnZ: `-1561.373`.

Relative to the median field, field 16 decomposes as:

| contribution | total delta |
| --- | ---: |
| raw total likelihood | -16.202 |
| observed selection | +0.000 |
| selection normalisation, `-logS` | +19.667 |
| total likelihood | +2.464 |

Field 16 shows the same qualitative check as in the TRGB case: a favourable selection normalisation is not sufficient by itself.
It is partly cancelled by a worse raw total likelihood, so it does not become the best-evidence field.

## Reproduction

Analysis script:

`/mnt/users/rstiskalek/CANDEL/notebooks/paper_CH0/plot_ch0_manticore_evidence_drivers.py`

Run:

```bash
/mnt/users/rstiskalek/CANDEL/venv_candel/bin/python \
  /mnt/users/rstiskalek/CANDEL/notebooks/paper_CH0/plot_ch0_manticore_evidence_drivers.py
```

Main outputs:

- `/mnt/users/rstiskalek/CANDEL/results/CH0_paper/single_fields/plots/ch0_manticore_evidence_driver_summary.txt`
- `/mnt/users/rstiskalek/CANDEL/results/CH0_paper/single_fields/plots/ch0_manticore_evidence_driver_summary.csv`
- `/mnt/users/rstiskalek/CANDEL/results/CH0_paper/single_fields/plots/ch0_manticore_raw_likelihood_vs_selection.png`: two-panel raw-likelihood/selection-normalisation/evidence diagnostic.
