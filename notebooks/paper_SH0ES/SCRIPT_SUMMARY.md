# SH0ES Python Script Summary

This file summarises the executable Python scripts in this directory.
The exploratory `.ipynb` notebooks are not covered here.

- `build_ch0_paper_figures.py`: regenerates the maintained CH0 paper figures from the current `results/CH0_paper` chains and writes them to `/mnt/users/rstiskalek/Papers/CH0/Figures`.
- `plot_ch0_manticore_evidence_drivers.py`: reads CH0 single-Manticore-field HDF5 outputs with auxiliary likelihood tracking, decomposes total likelihood differences into raw likelihood, observed selection, and selection normalisation, and writes diagnostic summaries and plots.
- `plot_ch0_single_field_h0_diagnostics.py`: reads CH0 single-Manticore-field HDF5 outputs from `results/CH0_paper/single_fields`, writes per-field and stacked `H_0` summaries, plots the field-summary histogram and posterior KDEs, and plots per-field `H_0` against Laplace, harmonic, and BIC evidence estimates.
- `utils.py`: provides a small HDF5 `samples/` reader used by the SH0ES notebooks.
