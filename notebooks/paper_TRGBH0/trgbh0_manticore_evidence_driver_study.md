# TRGBH0 Manticore Evidence-Driver Study

This note summarises the diagnostic we are running on the TRGB-only H0 single-field Manticore runs.
The current analysis uses the 50 COLA fields in `/mnt/users/rstiskalek/CANDEL/results/TRGBH0_paper/manticore_fields_const_sigv`.

## Question

The evidence marginalisation over BORG/COLA velocity-field samples can be dominated by a small number of fields.
We want to understand whether the preferred fields are preferred because many galaxies are modestly better described, because a few galaxies are extreme outliers, or because the selection normalisation changes coherently across the whole sample.

## Diagnostics

For each field we decompose the saved log likelihood into:

- `log_likelihood_per_galaxy`: the distance-marginalised magnitude/redshift likelihood for each galaxy.
- `log_observed_selection_per_galaxy`: the probability that each observed galaxy enters the sample.
- `log_selection_integral`: the field-level selection normalisation.
- `log_likelihood_per_galaxy_with_selection`: the bookkeeping total including the observed-selection term and the selection normalisation.

The selection term contributes as `-logS`.
A field with a more positive `-logS` is favoured by the selection normalisation, but it can still lose if the per-galaxy magnitude/redshift likelihood or observed-selection terms are worse.

## Current Results

The best COLA field is field 24 under all evidence proxies currently checked:

| proxy | best field | value |
| --- | ---: | ---: |
| harmonic lnZ | 24 | -3974.803 |
| Laplace lnZ | 24 | -3979.061 |
| BIC | 24 | 8011.090 |
| AIC | 24 | 7939.199 |

For field 24, the harmonic-evidence gain is large:

- Delta lnZ relative to the median field: `+81.344`.
- Delta lnZ relative to the next-best field 46: `+36.586`.
- `-logS = -6.0182`.

Relative to the median field, field 24 decomposes as:

| contribution | total delta |
| --- | ---: |
| magnitude/redshift likelihood | -77.108 |
| observed-selection probability | -67.584 |
| selection normalisation, `-logS` | +212.916 |
| total with-selection host likelihood | +70.992 |

This means field 24 is not preferred because each galaxy is slightly better in the raw magnitude/redshift likelihood.
It is preferred because the selection-normalisation term is favourable enough to overcome worse raw per-galaxy likelihood and observed-selection terms.

## Outlier Check

Field 24 does have a small number of large positive raw-likelihood outliers.
The largest positive contributors are `NGC4413`, `NGC4424`, `NGC4526`, and `NGC4328`.
The top 5 galaxies explain `53.9%` of the positive raw magnitude/redshift likelihood improvements, and the top 10 explain `61.3%`.

However, the total raw magnitude/redshift likelihood for field 24 is negative relative to the median field.
Thus the evidence preference is not explained by those outliers alone.
The outliers help locally, but the dominant net effect is the coherent selection-normalisation term.

## More Positive Selection Integral Case

The field with the most positive `-logS` is field 47:

- `-logS = -5.6712`.
- harmonic lnZ: `-4080.169`.
- harmonic-evidence rank: `44 / 50`.

Relative to the median field, field 47 decomposes as:

| contribution | total delta |
| --- | ---: |
| magnitude/redshift likelihood | -313.0 |
| observed-selection probability | -78.8 |
| selection normalisation, `-logS` | +352.1 |
| total with-selection host likelihood | -36.9 |

Field 47 shows that a favourable selection-normalisation term is not sufficient.
The field is penalised because the observed galaxies are much less compatible with the magnitude/redshift likelihood.
Field 24 is closer to the optimum tradeoff: it has a strongly favourable selection term without a catastrophic per-galaxy likelihood penalty.

## Interpretation To Test

The current evidence ranking appears to be driven mainly by a coherent selection-normalisation effect rather than by an average improvement in the raw galaxy likelihood.
The per-galaxy likelihood differences still matter, but they mainly determine whether a field with a favourable selection integral survives the galaxy-level likelihood check.

The key robustness question is whether this selection-normalisation sensitivity is an intended feature of the hierarchical selection model or an undesirable dependence on the finite set of velocity-field samples.

## Reproduction

Analysis script:

`/mnt/users/rstiskalek/CANDEL/notebooks/paper_TRGBH0/plot_trgbh0_manticore_evidence_drivers.py`

Run:

```bash
/mnt/users/rstiskalek/CANDEL/venv_candel/bin/python \
  /mnt/users/rstiskalek/CANDEL/notebooks/paper_TRGBH0/plot_trgbh0_manticore_evidence_drivers.py \
  --field-set cola \
  --results-dir /mnt/users/rstiskalek/CANDEL/results/TRGBH0_paper/manticore_fields_const_sigv \
  --output-dir /mnt/users/rstiskalek/CANDEL/results/TRGBH0_paper/manticore_fields_const_sigv/plots
```

Main outputs:

- `/mnt/users/rstiskalek/CANDEL/results/TRGBH0_paper/manticore_fields_const_sigv/plots/trgbh0_manticore_evidence_driver_summary_cola.txt`
- `/mnt/users/rstiskalek/CANDEL/results/TRGBH0_paper/manticore_fields_const_sigv/plots/trgbh0_manticore_best_field_components_cola.png`
- `/mnt/users/rstiskalek/CANDEL/results/TRGBH0_paper/manticore_fields_const_sigv/plots/trgbh0_manticore_best_field_galaxy_deltas_cola.png`
- `/mnt/users/rstiskalek/CANDEL/results/TRGBH0_paper/manticore_fields_const_sigv/plots/trgbh0_manticore_evidence_driver_metrics_cola.png`
- `/mnt/users/rstiskalek/CANDEL/results/TRGBH0_paper/manticore_fields_const_sigv/plots/trgbh0_manticore_galaxy_likelihood_heatmap_cola.png`
