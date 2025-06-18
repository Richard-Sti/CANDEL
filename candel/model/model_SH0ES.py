# Copyright (C) 2025 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Turning the SH0ES framework into a forward model in JAX."""
from abc import ABC

import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular
from numpyro import factor, plate, sample
from numpyro.distributions import (HalfNormal, MultivariateNormal, Normal,
                                   Uniform)

from ..cosmography import (Distance2Distmod, Distmod2Distance,
                           Distmod2Redshift, LogGrad_Distmod2ComovingDistance)
from ..util import (SPEED_OF_LIGHT, fprint, get_nested, load_config,
                    radec_to_cartesian, replace_prior_with_delta)
from .model import JeffreysPrior, MagnitudeDistribution, load_priors, rsample


def mvn_logpdf_cholesky(y, mu, L):
    """
    Log-pdf of a multivariate normal using Cholesky factor L (lower
    triangular).
    """
    z = solve_triangular(L, y - mu, lower=True)
    log_det = jnp.sum(jnp.log(jnp.diag(L)))
    return -0.5 * (len(y) * jnp.log(2 * jnp.pi) + 2 * log_det + jnp.dot(z, z))


def predict_cz(zcosmo, Vrad):
    return SPEED_OF_LIGHT * ((1 + zcosmo) * (1 + Vrad / SPEED_OF_LIGHT) - 1)


class BaseSH0ESModel(ABC):

    def __init__(self, config_path, data):
        config = load_config(config_path)

        # Set unused parameter priors to delta functions.
        config = self.replace_priors(config)
        # Unpack and set the priors.
        priors = config["model"]["priors"]
        self.priors, self.prior_dist_name = load_priors(priors)

        self.config = config
        # Load the data, set attributes, convert to JAX arrays and do
        # any other conversions.
        self.set_data(data)

        # Initialize the interpolators
        Om = get_nested(config, "model/Om0", 0.3)
        self.distmod2redshift = Distmod2Redshift(Om0=Om)
        self.distmod2distance = Distmod2Distance(Om0=Om)

        self.distance2distmod_scalar = Distance2Distmod(Om0=Om, is_scalar=True)
        self.distance2distmod = Distance2Distmod(Om0=Om)

        self.log_grad_distmod2comoving_distance = LogGrad_Distmod2ComovingDistance(Om0=Om)  # noqa

        self.distmod_limits = self.config["model"]["distmod_limits"]

        self.use_SNe_HF_SH0ES = get_nested(config, "model/use_SNe_HF_SH0ES", False)  # noqa
        fprint(f"use_SNe_HF_SH0ES set to {self.use_SNe_HF_SH0ES}")
        self.use_SNe_HF_Bayes = get_nested(config, "model/use_SNe_HF_Bayes", False)  # noqa
        fprint(f"use_SNe_HF_Bayes set to {self.use_SNe_HF_Bayes}")

        if self.use_SNe_HF_SH0ES and self.use_SNe_HF_Bayes:
            raise ValueError(
                "Cannot use both `use_SNe_HF_SH0ES` and `use_SNe_HF_Bayes`.")

        self.use_MNR = get_nested(config, "model/use_MNR", False)
        fprint(f"use_MNR set to {self.use_MNR}")
        self.use_Cepheid_host_redshift = get_nested(
            config, "model/use_Cepheid_host_redshift", False)
        fprint(f"use_Cepheid_host_redshift set to {self.use_Cepheid_host_redshift}")  # noqa
        self.use_uniform_mu_host_priors = get_nested(
            config, "model/use_uniform_mu_host_priors", True)
        fprint(f"use_uniform_mu_host_priors set to {self.use_uniform_mu_host_priors}")  # noqa
        self.use_fiducial_Cepheid_host_PV_covariance = get_nested(
            config, "model/use_fiducial_Cepheid_host_PV_covariance", True)
        fprint(f"use_fiducial_Cepheid_host_PV_covariance set to {self.use_fiducial_Cepheid_host_PV_covariance}")  # noqa
        self.use_PV_covmat_scaling = get_nested(
            config, "model/use_PV_covmat_scaling", False)
        fprint(f"use_PV_covmat_scaling set to {self.use_PV_covmat_scaling}")

        # Precompute min-max for MNR priors.
        self.logP_min = jnp.min(data["logP"])
        self.logP_max = jnp.max(data["logP"])

        self.OH_min = jnp.min(data["OH"])
        self.OH_max = jnp.max(data["OH"])

        if data["Cepheids_only"] and (self.use_SNe_HF_SH0ES or self.use_SNe_HF_Bayes):  # noqa
            raise ValueError(
                "Cannot use SNe_HF with Cepheids only data. Likely because of "
                "imposing a redshift threshold on the Cepheid hosts.")

    def replace_priors(self, config):
        """Replace priors on parameters that are not used in the model."""
        use_SNe = (
            get_nested(config, "model/use_SNe_HF_SH0ES", False)
            or get_nested(config, "model/use_SNe_HF_Bayes", False))
        use_Cepheid_host_redshift = get_nested(
            config, "model/use_Cepheid_host_redshift", False)
        use_PV_covmat_scaling = get_nested(
            config, "model/use_PV_covmat_scaling", False)

        if not use_SNe:
            replace_prior_with_delta(config, "M_B", -19.25)

        if not (use_Cepheid_host_redshift or use_SNe):
            replace_prior_with_delta(config, "H0", 73.04)
            replace_prior_with_delta(config, "Vext", [0., 0., 0.])
            replace_prior_with_delta(config, "sigma_v", 100.0)

        if not use_PV_covmat_scaling:
            replace_prior_with_delta(config, "A_covmat", 1.0)

        return config

    def set_data(self, data):
        attrs_set = []
        for k, v in data.items():
            if k in ["q_names", "host_map"]:
                continue

            if isinstance(v, np.ndarray):
                v = jnp.asarray(v)

            setattr(self, k, v)
            attrs_set.append(k)

            if k.startswith("e_"):
                k = k.replace("e_", "e2_")
                setattr(self, k, v * v)
                attrs_set.append(k)

        if "RA_host" in data and "dec_host" in data:
            fprint("Converting host RA/dec to Cartesian coordinates.")
            rhat = radec_to_cartesian(data["RA_host"], data["dec_host"])
            self.rhat_host = rhat / np.linalg.norm(rhat, axis=1)[:, None]
            attrs_set.append("rhat_host")

        if "RA_SN_HF" in data and "dec_SN_HF" in data:
            fprint("Converting SN_HF RA/dec to Cartesian coordinates.")
            rhat = radec_to_cartesian(data["RA_SN_HF"], data["dec_SN_HF"])
            self.rhat_SN_HF = rhat / np.linalg.norm(rhat, axis=1)[:, None]
            attrs_set.append("rhat_SN_HF")

        fprint(f"set the following attributes: {', '.join(attrs_set)}")

    def sample_host_distmod(self):
        """
        Sample distance moduli for host galaxies, with a uniform prior in the
        distance modulus. The log PDF of the prior if r^2 gets added directly
        within the model. Includes the geometric anchor information for
        NGC 4258 and the LMC.
        """
        dist = Uniform(*self.distmod_limits)

        with plate("hosts", self.num_hosts):
            mu_host = sample("mu_host", dist)

        mu_N4258 = sample("mu_N4258", dist)
        mu_LMC = sample("mu_LMC", dist)
        mu_M31 = sample("mu_M31", dist)

        sample("mu_N4258_ll",
               Normal(self.mu_N4258_anchor, self.e_mu_N4258_anchor),
               obs=mu_N4258)
        sample("mu_LMC_ll",
               Normal(self.mu_LMC_anchor, self.e_mu_LMC_anchor),
               obs=mu_LMC)

        return mu_host, mu_N4258, mu_LMC, mu_M31


class SH0ESModel(BaseSH0ESModel):
    """A version of the SH0ES inference turned into a forward model."""
    def __init__(self, config_path, data):
        super().__init__(config_path, data)

    def __call__(self, ):
        M_B = rsample("M_B", self.priors["M_B"])
        H0 = rsample("H0", self.priors["H0"])
        M_W = rsample("M_W", self.priors["M_W"])
        b_W = rsample("b_W", self.priors["b_W"])
        Z_W = rsample("Z_W", self.priors["Z_W"])
        Vext = rsample("Vext", self.priors["Vext"])
        sigma_v = rsample("sigma_v", self.priors["sigma_v"])
        A_covmat = rsample("A_covmat", self.priors["A_covmat"])

        # HST and Gaia zero-point calibration of MW Cepheids.
        sample("M_W_HST", Normal(M_W, self.e_M_HST), obs=self.M_HST)
        sample("M_W_Gaia", Normal(M_W, self.e_M_Gaia), obs=self.M_Gaia)

        mu_host, mu_N4258, mu_LMC, mu_M31 = self.sample_host_distmod()

        # Distannce moduli for Cepheids, with corrections for LMC.
        dZP = sample("dZP", Normal(0, self.sigma_grnd))
        mu_host_cepheid = jnp.concatenate(
            [mu_host,
             jnp.array([mu_N4258, mu_LMC + dZP, mu_M31])]
            )

        # Distances moduli without any corrections.
        mu_host_all = jnp.concatenate(
            [mu_host, jnp.array([mu_N4258, mu_LMC, mu_M31])]
            )

        if not self.use_uniform_mu_host_priors:
            log_r = self.distmod2distance(
                mu_host_all, h=H0 / 100, return_log=True)
            log_drdmu = self.log_grad_distmod2comoving_distance(
                mu_host_all, h=H0 / 100)
            factor("lp_r2", 2 * log_r + log_drdmu)

        # Now assign these host distances to each Cepheid.
        mu_cepheid = self.L_Cepheid_host_dist @ mu_host_cepheid

        if self.use_MNR:
            # Global hyperpriors for host-level logP mean and std
            mean_logP_all = sample(
                "mean_logP_all", Uniform(self.logP_min, self.logP_max))
            std_logP_all = sample(
                "std_logP_all", Uniform(1e-5, self.logP_max - self.logP_min))
            mean_std_logP = sample(
                "mean_std_logP", Uniform(1e-5, self.logP_max - self.logP_min))

            # Global hyperprior for host-level OH mean and std
            mean_OH_all = sample(
                "mean_OH_all", Uniform(self.OH_min, self.OH_max))
            std_OH_all = sample(
                "std_OH_all", Uniform(1e-5, self.OH_max - self.OH_min))
            mean_std_OH = sample(
                "mean_std_OH", Uniform(1e-5, self.OH_max - self.OH_min))

            # Per-host parameters
            with plate("MNR_Cepheid", len(mu_host_all)):
                mean_logP_per_host = sample(
                    "mean_logP_per_host", Normal(mean_logP_all, std_logP_all))
                std_logP_per_host = sample(
                    "std_logP_per_host", HalfNormal(mean_std_logP))

                mean_OH_per_host = sample(
                    "mean_OH_per_host", Normal(mean_OH_all, std_OH_all))
                std_OH_per_host = sample(
                    "std_OH_per_host", HalfNormal(mean_std_OH))

            # Per-Cepheid parameters
            mean_logP = self.L_Cepheid_host_dist @ mean_logP_per_host
            std_logP = self.L_Cepheid_host_dist @ std_logP_per_host

            mean_OH = self.L_Cepheid_host_dist @ mean_OH_per_host
            std_OH = self.L_Cepheid_host_dist @ std_OH_per_host

            with plate("Cepheid_true_params", self.num_cepheids):
                logP = sample(
                    "logP", Normal(mean_logP, std_logP), obs=self.logP)
                OH = sample(
                    "OH", Normal(mean_OH, std_OH), obs=self.OH)
        else:
            logP = self.logP
            OH = self.OH

        # Predict the Cepheid magnitudes and compute their likelihood.
        mag_cepheid = mu_cepheid + M_W + b_W * logP + Z_W * OH
        factor(
            "ll_cepheid",
            mvn_logpdf_cholesky(self.mag_cepheid, mag_cepheid, self.L_Cepheid)
            )

        if self.use_Cepheid_host_redshift:
            # # SH0ES-Antonio approach
            # Y_Cepheid = mu_host + 5 * jnp.log10(H0)
            # with plate("Cepheid_host_redshift", self.num_hosts):
            #     sample("Y_Cepheid",
            #            Normal(Y_Cepheid, self.Y_Cepheid_new_err),
            #            obs=self.Y_Cepheid_new)

            Vext_rad = jnp.sum(Vext[None, :] * self.rhat_host, axis=1)
            cz_pred = predict_cz(
                self.distmod2redshift(mu_host, h=H0 / 100),
                Vext_rad)

            e2_cz = self.e2_czcmb_cepheid_host + sigma_v**2
            if self.use_fiducial_Cepheid_host_PV_covariance:
                # Because we're adding sigma_v^2 to the diagonal, we cannot
                # use the Cholesky factorization of the covariance matrix.
                C = A_covmat * self.PV_covmat_cepheid_host
                C = C.at[jnp.diag_indices(len(e2_cz))].add(e2_cz)
                sample("cz_pred", MultivariateNormal(cz_pred, C),
                       obs=self.czcmb_cepheid_host)
            else:
                e_cz = jnp.sqrt(e2_cz)
                with plate("Cepheid_anchors_redshift", self.num_hosts):
                    sample("cz_pred", Normal(cz_pred, e_cz),
                           obs=self.czcmb_cepheid_host)

        # Distances to the host that have both supernovae and Cepheids
        if self.use_SNe_HF_SH0ES:
            mu_SN_Cepheid = self.L_SN_Cepheid_dist @ mu_host_all
            mag_SN_Cepheid = mu_SN_Cepheid + M_B

            Y_SN_flow = jnp.ones(self.num_flow_SN) * (M_B - 5 * jnp.log10(H0))
            Y_SN = jnp.concatenate([mag_SN_Cepheid, Y_SN_flow])
            factor(
                "ll_SN",
                mvn_logpdf_cholesky(self.Y_SN, Y_SN, self.L_SN)
                )
        elif self.use_SNe_HF_Bayes:
            # TODO: this is still not fully understood.

            # Distances to the host that have both supernovae and Cepheids
            # and their true apparent magnitudes.
            mu_SN_Cepheid = self.L_SN_Cepheid_dist @ mu_host_all
            mag_true_SN_Cepheid = mu_SN_Cepheid + M_B

            # TODO: add a prior for this in the config. Can we use something
            # without infinite mass at zero?
            e_mu = sample("e_mu", JeffreysPrior(0.001, 0.5))

            # Sample the true apparent magnitudes of the Cepheid hosts, from
            # a r^2 prior effectively.
            with plate("SN_mag", self.num_SN_HF):
                mag_true_HF = sample(
                    "mag_true",
                    MagnitudeDistribution(5, 25, self.Y_SN[77:], e_mu))

            mag_true_SN = jnp.concatenate([mag_true_SN_Cepheid, mag_true_HF])

            factor(
                "ll_SN_HF",
                mvn_logpdf_cholesky(self.Y_SN, mag_true_SN, self.L_SN)
                )

            mu_SN = mag_true_HF - M_B

            # Now the redshift likelihood
            Vext_rad_SN = jnp.sum(Vext[None, :] * self.rhat_SN_HF, axis=1)
            cz_pred_SN = predict_cz(
                self.distmod2redshift(mu_SN, h=H0 / 100),
                Vext_rad_SN)
            e_cz = jnp.sqrt(self.e2_czcmb_SN_HF + sigma_v**2)
            with plate("SN_redshift", self.num_SN_HF):
                sample("cz_pred2", Normal(cz_pred_SN, e_cz),
                       obs=self.czcmb_SN_HF)
