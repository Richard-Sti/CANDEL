# TRGBH0 Paper Plot Scripts

This directory is split by use:

- `paper_figures/` contains figures and tables used by the TRGBH0 paper build.
- `diagnostics/single_fields/` contains run diagnostics for analyses that use individual reconstruction fields.
- `diagnostics/data_checks/` contains auxiliary data-comparison checks that are not part of the main paper build.
- `diagnostics/model_checks/` contains auxiliary posterior/model-comparison checks that are not part of the main paper build.
- `output/` contains generated local plot outputs and is not source.

Shared paths, colours, rc settings, colormaps, and save helpers live in
`trgbh0_plot_style.py`; shared EDD TRGB data-loading helpers live in
`edd_trgb_plot_data.py`.

Build paper figures with:

```bash
/mnt/users/rstiskalek/CANDEL/venv_candel/bin/python \
  /mnt/users/rstiskalek/CANDEL/notebooks/paper_TRGBH0/build_trgbh0_figures.py
```

To also copy generated PDFs into the paper source tree, pass:

```bash
--paper-figdir /mnt/users/rstiskalek/Papers/TRGBH0/Figures
```
