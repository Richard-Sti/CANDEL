# CH0 Python Script Summary

This file summarises the executable Python scripts in this directory.
The exploratory `.ipynb` notebooks are not covered here.

- `build_paper_figures.py`: regenerates the maintained CH0 paper figures from the current `results/CH0_paper` chains and writes them to `/mnt/users/rstiskalek/Papers/CH0/Figures`.
- `plot_angular_scatter_diagnostics.py`: reads `tasks_CH0_angular_scatter.txt`, compares angular-position-scatter runs against matched no-scatter COLA CIC fields, and plots stacked `H_0` posteriors and per-field `H_0` shifts.
- `plot_cola_mas_comparison.py`: reads `tasks_CH0_single.txt`, compares the 80 matched COLA CIC, PCS, and SPH single-field results, and plots matched-realisation and bulk `H_0-\ln Z` behaviour.
- `plot_leaveoneout_diagnostics.py`: reads `tasks_CH0_leaveoneout.txt`, tolerates still-missing leave-one-out HDF5 outputs, and plots per-host `H_0` shifts, evidence values, and influence summaries relative to the full field-21 run.
- `plot_manticore_evidence_drivers.py`: reads CH0 single-Manticore-field HDF5 outputs with auxiliary likelihood tracking, decomposes total likelihood differences into raw likelihood, observed selection, and selection normalisation, and writes diagnostic summaries and plots.
- `plot_single_field_h0_diagnostics.py`: reads CH0 single-Manticore-field HDF5 outputs from `results/CH0_paper/single_fields`, writes per-field and stacked `H_0` summaries, plots the field-summary histogram and posterior KDEs, and plots per-field `H_0` against Laplace, harmonic, and BIC evidence estimates.
- `plot_single_smoothed_diagnostics.py`: reads `tasks_CH0_single_smoothed.txt`, compares smoothed single-field `H_0` summaries across matched Manticore realisations, adds the uniform-bias/density-field control runs to the summary products, plots broad `H_0` shifts with smoothing scale, and plots `H_0` against harmonic evidence for each model option.
- `plot_single_fixed_bias_diagnostics.py`: reads fixed-bias single-field outputs from `results/CH0_paper/single_fields_fixed_bias`, writes per-field summaries, and plots `H_0`, evidence, likelihood-decomposition, and fixed galaxy-bias diagnostics.
- `plot_swift_sph_cola_cic_comparison.py`: reads tasks `0-109` from `tasks_CH0_single.txt`, compares SWIFT SPH and COLA CIC single-field results, and plots their `H_0-\ln Z` relations and common-field differences.
- `utils.py`: provides a small HDF5 `samples/` reader used by the CH0 notebooks.
