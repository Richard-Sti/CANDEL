Supported models
================

Peculiar-velocity models
------------------------

These models work in units of :math:`h^{-1}\,\mathrm{Mpc}` and can be analysed
jointly via :class:`~candel.model.base_pv.JointPVModel` with user-specified
shared parameters.

- **Tully--Fisher relation** (:class:`~candel.model.model_PV_TFR.TFRModel`):
  2MTF, SFI++, CF4-TFR
- **Type Ia supernovae (SALT2)** (:class:`~candel.model.model_PV_SN.SNModel`):
  LOSS, Foundation
- **Pantheon+** (:class:`~candel.model.model_PV_PantheonPlus.PantheonPlusModel`):
  Pantheon+ with full covariance matrix
- **Fundamental Plane** (:class:`~candel.model.model_PV_FP.FPModel`):
  6dFGS-FP, SDSS-FP

:math:`H_0` inference
---------------------

- **Cepheid-calibrated** :math:`H_0`
  (:class:`~candel.model.model_H0_CH0.CH0Model`):
  35 Cepheid host galaxies from SH0ES
- **TRGB-calibrated** :math:`H_0`
  (:class:`~candel.model.model_H0_TRGB.TRGBModel`):
  Tip of the Red Giant Branch (TRGB) distances from CCHP, EDD, and SH0ES
- **2MTF-calibrated** :math:`H_0`
  (:class:`~candel.model.model_H0_2MTF.MTFModel`):
  Tully--Fisher distances from the 2MTF survey

Package structure
-----------------

- :doc:`candel <api/candel>` -- cosmography, inference, evidence, utilities
- :doc:`candel.model <api/candel.model>` -- forward models for each distance indicator
- :doc:`candel.pvdata <api/candel.pvdata>` -- data loaders for all supported catalogues
- :doc:`candel.cosmo <api/candel.cosmo>` -- growth rate, PV covariance matrices
- :doc:`candel.field <api/candel.field>` -- 3D density/velocity field loading and LOS interpolation
- :doc:`candel.redshift2real <api/candel.redshift2real>` -- observed to cosmological redshift mapping
- :doc:`candel.mock <api/candel.mock>` -- synthetic catalogue generation
