"""
GW170817 parameter estimation with Bilby + JetFit afterglow constraint.

Implements:
  1. Standard BNS PE using IMRPhenomPv2_NRTidal
  2. JetFit EM constraint via GMM on (D_L, theta_obs)
  3. Density-weighted distance prior (Malmquist bias correction)

Usage:
  python run_GW170817.py                  # Quick test (injection, few live points)
  python run_GW170817.py --real-data      # Use GWOSC strain (slow download)
  python run_GW170817.py --nlive 1000     # Production settings
"""
import argparse
import os
import threading
import time

import bilby
import numpy as np
from bilby.gw.utils import ln_i0
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture


class ProgressMonitor:
    """Write sampler progress to a file every `interval` seconds.

    Works by polling the bilby sampler object in a background thread,
    completely independent of dynesty's print_func callback and SLURM
    stdout buffering.
    """

    def __init__(self, path, interval=30):
        self.path = path
        self.interval = interval
        self._sampler = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self, sampler_obj):
        self._sampler = sampler_obj
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()

    def _run(self):
        while not self._stop.wait(self.interval):
            self._write()
        self._write()  # final write

    def _write(self):
        s = self._sampler
        if s is None:
            return
        try:
            it = s.it
            nc = s.ncall
            eff = it / nc * 100 if nc > 0 else 0

            # Compute dlogz = ln(Z_remaining / Z_current), same as dynesty
            logz = s.results.logz[-1] if len(s.results.logz) > 0 else -1e300
            logvol = s.results.logvol[-1] if len(s.results.logvol) > 0 else 0
            loglmax = max(s.live_logl)
            dlogz = np.logaddexp(0, loglmax + logvol - logz)

            ts = time.strftime("%H:%M:%S")
            line = (f"{ts}  iter={it}  ncall={nc:.2e}  "
                    f"eff={eff:.1f}%  dlogz={dlogz:.1f}\n")
            with open(self.path, "a") as f:
                f.write(line)
                f.flush()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRIGGER_TIME = 1187008882.43
RA_NGC4993 = 3.44616       # rad (197.45 deg)
DEC_NGC4993 = -0.4085      # rad (-23.38 deg)
Z_OBS = 0.0099             # observed CMB-frame redshift
ROQ_BASIS_SEGLEN = 128.0   # segment length of the ROQ basis [s]


# ---------------------------------------------------------------------------
# JetFit GMM constraint
# ---------------------------------------------------------------------------

def build_jetfit_gmm(n_samples=50000, seed=42):
    """Build a synthetic GMM approximating the JetFit (D_L, theta_obs) posterior.

    Based on Palmese et al. 2024 (arXiv:2305.19914):
      theta_obs = 0.53 +0.05/-0.03 rad  (~30.4 deg)
      D_L ~ 40 Mpc (weakly constrained by afterglow alone)

    The 2D posterior has a positive correlation: larger D_L requires larger
    theta_obs to match the observed flux.

    The GMM approximates the JetFit *posterior*, not the EM likelihood. To
    extract the EM likelihood, we divide out the JetFit priors (flat in
    theta_obs, flat in D_L) when evaluating the GMM in the combined
    likelihood. See GWplusEMLikelihood for details.
    """
    rng = np.random.default_rng(seed)

    # Marginals from Palmese et al. Fig 3 / Table I
    theta_obs_mean = 0.53  # rad
    theta_obs_std = 0.04   # ~average of +0.05/-0.03

    # Afterglow barely constrains D_L (flux ~ E/D_L^2, degenerate with E)
    dl_mean = 41.0   # Mpc (NGC 4993 luminosity distance)
    dl_std = 25.0    # very broad — afterglow alone barely constrains D_L

    # Correlation: positive (further → need more on-axis to match flux)
    rho = 0.6

    cov = np.array([
        [dl_std**2, rho * dl_std * theta_obs_std],
        [rho * dl_std * theta_obs_std, theta_obs_std**2],
    ])
    mean = np.array([dl_mean, theta_obs_mean])

    samples = rng.multivariate_normal(mean, cov, size=n_samples)
    # Enforce physical bounds
    mask = (samples[:, 0] > 0) & (samples[:, 1] > 0) & (samples[:, 1] < np.pi / 2)
    samples = samples[mask]

    # Use 12 components as in Palmese et al.
    gmm = GaussianMixture(n_components=12, covariance_type="full",
                          random_state=seed)
    gmm.fit(samples)
    return gmm


def load_or_build_jetfit_gmm(data_dir):
    """Load pre-fitted GMM from disk, or build and save it."""
    gmm_path = os.path.join(data_dir, "jetfit_gmm.npz")
    if os.path.exists(gmm_path):
        d = np.load(gmm_path, allow_pickle=True)
        gmm = GaussianMixture(n_components=int(d["n_components"]),
                               covariance_type="full")
        gmm.means_ = d["means"]
        gmm.covariances_ = d["covariances"]
        gmm.weights_ = d["weights"]
        gmm.precisions_cholesky_ = d["precisions_cholesky"]
        return gmm

    gmm = build_jetfit_gmm()

    np.savez(gmm_path,
             n_components=gmm.n_components,
             means=gmm.means_,
             covariances=gmm.covariances_,
             weights=gmm.weights_,
             precisions_cholesky=gmm.precisions_cholesky_)
    print(f"Saved JetFit GMM to {gmm_path}")
    return gmm


# ---------------------------------------------------------------------------
# GW + EM combined likelihood
# ---------------------------------------------------------------------------

class GWplusEMLikelihood(bilby.gw.GravitationalWaveTransient):
    """GW likelihood augmented with JetFit afterglow constraint.

    The EM term is a GMM density over (D_L, theta_obs) fitted to JetFit
    *posterior* samples. To correctly combine GW and EM, we need:

        p(theta | x_GW, x_EM) ~ L_GW(theta) * L_EM(theta) * pi(theta)

    Since the GMM encodes the JetFit posterior (not likelihood):

        GMM(D_L, theta_obs) ~ L_EM * pi_JetFit(D_L) * pi_JetFit(theta_obs)

    we divide out the JetFit priors to recover L_EM. JetFit typically uses:
      - flat prior on D_L (when D_L is held fixed or sampled uniformly)
      - flat prior on theta_obs in [0, 1] rad

    So L_EM ~ GMM / const, and the constant cancels in the posterior.

    Angle folding: GW uses iota in [0, pi]. EM uses
    theta_obs = min(iota, pi - iota) in [0, pi/2].
    """

    def __init__(self, *args, gmm=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gmm = gmm

    def _log_em(self, parameters):
        p = parameters if parameters is not None else self.parameters
        theta_jn = p["theta_jn"]
        dl = p["luminosity_distance"]
        theta_obs = min(theta_jn, np.pi - theta_jn)
        return self.gmm.score_samples(np.array([[dl, theta_obs]]))[0]

    def log_likelihood_ratio(self, parameters=None):
        log_lr_gw = super().log_likelihood_ratio(parameters=parameters)
        if self.gmm is None or not np.isfinite(log_lr_gw):
            return log_lr_gw
        return log_lr_gw + self._log_em(parameters)


class PsiMarginalizedLikelihood(GWplusEMLikelihood):
    """GW+EM likelihood with psi numerically marginalised out.

    For fixed (ra, dec, geocent_time), the antenna patterns rotate as:
        F+(psi) =  F+(0) cos(2psi) + Fx(0) sin(2psi)
        Fx(psi) = -F+(0) sin(2psi) + Fx(0) cos(2psi)

    We compute waveform inner products once and evaluate the likelihood
    at N_psi values of psi cheaply, then logsumexp to marginalise.
    """

    def __init__(self, *args, n_psi=50, marginalize_phase=False,
                 marginalize_time=False, **kwargs):
        if marginalize_time:
            kwargs['time_marginalization'] = True
            kwargs.setdefault('jitter_time', True)
        super().__init__(*args, **kwargs)
        self.n_psi = n_psi
        self.psi_grid = np.linspace(0, np.pi, n_psi, endpoint=False)
        self._marginalize_phase = marginalize_phase
        self._marginalize_time = marginalize_time

    def _inner_product(self, aa, bb, psd, duration):
        """Noise-weighted inner product: 4/T * Re(sum(conj(aa) * bb / S))."""
        return 4.0 / duration * np.sum(np.conj(aa) * bb / psd).real

    def _complex_inner_product(self, aa, bb, psd, duration):
        """Complex noise-weighted inner product: 4/T * sum(conj(aa) * bb / S).

        The real part gives <a|b>, the imaginary part is needed for
        the signal-vs-data terms where we need the full complex value.
        """
        return 4.0 / duration * np.sum(np.conj(aa) * bb / psd)

    def _matched_filter_fft(self, h_pol, data, psd_array, duration):
        """Compute <h|d>(t) for all time shifts via FFT.

        Returns complex array of length N_t = duration * fs / 2.
        """
        n_freq = len(psd_array)
        # Zero where PSD is inf/zero (out-of-band); drop Nyquist
        integrand = np.zeros(n_freq - 1, dtype=complex)
        valid = (psd_array[:-1] > 0) & np.isfinite(psd_array[:-1])
        integrand[valid] = (
            np.conj(h_pol[:-1][valid]) * data[:-1][valid] / psd_array[:-1][valid]
        )
        # FFT gives sum at each time bin; normalise by 4/T
        return 4.0 / duration * np.fft.fft(integrand)

    def log_likelihood_ratio(self, parameters=None):
        if parameters is not None:
            self.parameters.update(parameters)

        # Force psi=0 for waveform generation (psi only enters via antenna)
        self.parameters["psi"] = 0.0

        # Generate waveform polarisations
        wf_pols = self.waveform_generator.frequency_domain_strain(
            self.parameters)
        if wf_pols is None:
            return np.nan_to_num(-np.inf)

        hp = wf_pols["plus"]
        hc = wf_pols["cross"]

        ra = self.parameters["ra"]
        dec = self.parameters["dec"]
        tc = self.parameters["geocent_time"]

        n_det = len(self.interferometers)
        Fp0 = np.zeros(n_det)
        Fc0 = np.zeros(n_det)
        C = np.zeros(n_det)
        D = np.zeros(n_det)
        E = np.zeros(n_det)

        if self._marginalize_time:
            # FFT-based matched filter: compute A(t), B(t) arrays
            # Apply only geometric delay (not geocentric offset) to template;
            # the geocentric time shift is explored via the FFT time grid.
            # Handle time_jitter sub-bin correction
            time_jitter = self.parameters.get("time_jitter", 0.0)
            tc_ref = tc + time_jitter

            A_arrays = []
            B_arrays = []
            for i, ifo in enumerate(self.interferometers):
                psd_full = ifo.power_spectral_density_array
                dur = ifo.strain_data.duration
                data_full = ifo.frequency_domain_strain

                # Geometric delay only (no geocentric offset)
                dt_geo = ifo.time_delay_from_geocenter(ra, dec, tc_ref)
                dt_start = tc_ref - ifo.strain_data.start_time
                time_shift = dt_geo + dt_start
                freqs_full = ifo.frequency_array
                phase_shift = np.exp(-2j * np.pi * freqs_full * time_shift)

                if ifo.calibration_model is not None:
                    cal = ifo.calibration_model.get_calibration_factor(
                        freqs_full[ifo.frequency_mask],
                        prefix=f"recalib_{ifo.name}_",
                        **self.parameters)
                    cal_full = np.ones_like(freqs_full, dtype=complex)
                    cal_full[ifo.frequency_mask] = cal
                else:
                    cal_full = 1.0

                hp_shifted = hp * phase_shift * cal_full
                hc_shifted = hc * phase_shift * cal_full

                # Zero out-of-band frequencies via PSD (set to inf)
                psd_eff = psd_full.copy()
                psd_eff[~ifo.frequency_mask] = np.inf

                A_arrays.append(
                    self._matched_filter_fft(hp_shifted, data_full, psd_eff, dur))
                B_arrays.append(
                    self._matched_filter_fft(hc_shifted, data_full, psd_eff, dur))

                # Template-template products (time-independent, use masked)
                mask = ifo.frequency_mask
                psd = psd_full[mask]
                hp_s = hp_shifted[mask]
                hc_s = hc_shifted[mask]
                C[i] = self._inner_product(hp_s, hp_s, psd, dur)
                D[i] = self._inner_product(hp_s, hc_s, psd, dur)
                E[i] = self._inner_product(hc_s, hc_s, psd, dur)

                Fp0[i] = ifo.antenna_response(ra, dec, tc_ref, 0.0, "plus")
                Fc0[i] = ifo.antenna_response(ra, dec, tc_ref, 0.0, "cross")

            A_array = np.array(A_arrays)  # (n_det, n_times)
            B_array = np.array(B_arrays)

            # Time grid and prior from bilby's setup
            n_times = A_array.shape[1]
            time_prior = self.time_prior_array

            # Evaluate log L at each psi, marginalising over time
            log_ls = np.empty(self.n_psi)
            for k, psi in enumerate(self.psi_grid):
                c2 = np.cos(2 * psi)
                s2 = np.sin(2 * psi)
                Fp = Fp0 * c2 + Fc0 * s2
                Fc = -Fp0 * s2 + Fc0 * c2

                signal_signal = np.sum(
                    Fp**2 * C + 2 * Fp * Fc * D + Fc**2 * E)

                # z(t) = sum_d [Fp_d * A_d(t) + Fc_d * B_d(t)]
                z_t = Fp @ A_array + Fc @ B_array  # (n_times,)

                if self._marginalize_phase:
                    log_l_t = ln_i0(np.abs(z_t)) - 0.5 * signal_signal
                else:
                    log_l_t = z_t.real - 0.5 * signal_signal

                log_ls[k] = logsumexp(log_l_t, b=time_prior)

        else:
            # Original scalar path (no time marginalisation)
            A = np.zeros(n_det, dtype=complex)
            B = np.zeros(n_det, dtype=complex)

            for i, ifo in enumerate(self.interferometers):
                mask = ifo.frequency_mask
                psd = ifo.power_spectral_density_array[mask]
                dur = ifo.strain_data.duration
                data = ifo.frequency_domain_strain[mask]

                dt = ifo.time_delay_from_geocenter(ra, dec, tc)
                dt_geocent = self.parameters["geocent_time"] - ifo.strain_data.start_time
                time_shift = dt + dt_geocent
                freqs = ifo.frequency_array[mask]
                phase_shift = np.exp(-2j * np.pi * freqs * time_shift)

                if ifo.calibration_model is not None:
                    cal = ifo.calibration_model.get_calibration_factor(
                        freqs, prefix=f"recalib_{ifo.name}_",
                        **self.parameters)
                else:
                    cal = 1.0

                hp_s = hp[mask] * phase_shift * cal
                hc_s = hc[mask] * phase_shift * cal

                A[i] = self._complex_inner_product(hp_s, data, psd, dur)
                B[i] = self._complex_inner_product(hc_s, data, psd, dur)
                C[i] = self._inner_product(hp_s, hp_s, psd, dur)
                D[i] = self._inner_product(hp_s, hc_s, psd, dur)
                E[i] = self._inner_product(hc_s, hc_s, psd, dur)

                Fp0[i] = ifo.antenna_response(ra, dec, tc, 0.0, "plus")
                Fc0[i] = ifo.antenna_response(ra, dec, tc, 0.0, "cross")

            # Evaluate log L at each psi value
            log_ls = np.empty(self.n_psi)
            for k, psi in enumerate(self.psi_grid):
                c2 = np.cos(2 * psi)
                s2 = np.sin(2 * psi)
                Fp = Fp0 * c2 + Fc0 * s2
                Fc = -Fp0 * s2 + Fc0 * c2

                signal_signal = np.sum(
                    Fp**2 * C + 2 * Fp * Fc * D + Fc**2 * E)

                if self._marginalize_phase:
                    z = np.sum(Fp * A + Fc * B)
                    log_ls[k] = ln_i0(abs(z)) - 0.5 * signal_signal
                else:
                    signal_data = np.sum(Fp * A + Fc * B).real
                    log_ls[k] = signal_data - 0.5 * signal_signal

        # Marginalise over psi: logsumexp - log(N_psi)
        log_lr = float(np.logaddexp.reduce(log_ls) - np.log(self.n_psi))

        # Add EM term if GMM is present
        if self.gmm is not None and np.isfinite(log_lr):
            log_lr += self._log_em(None)

        return log_lr


class ROQplusEMLikelihood(bilby.gw.likelihood.ROQGravitationalWaveTransient):
    """ROQ likelihood augmented with JetFit afterglow constraint."""

    def __init__(self, *args, gmm=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gmm = gmm

    def log_likelihood_ratio(self, parameters=None):
        log_lr_gw = super().log_likelihood_ratio(parameters=parameters)
        if self.gmm is None or not np.isfinite(log_lr_gw):
            return log_lr_gw
        p = parameters if parameters is not None else self.parameters
        theta_jn = p["theta_jn"]
        dl = p["luminosity_distance"]
        theta_obs = min(theta_jn, np.pi - theta_jn)
        return log_lr_gw + self.gmm.score_samples(
            np.array([[dl, theta_obs]]))[0]


# ---------------------------------------------------------------------------
# Density-weighted distance prior
# ---------------------------------------------------------------------------

def build_density_prior(dl_min=1.0, dl_max=100.0, n_grid=500):
    """Build a density-weighted D_L prior.

    For now, uses a dummy flat density (n_LOS = 1) which reduces to the
    standard D_L^2 prior. Replace n_LOS with the actual CF4 density field
    along the NGC 4993 sightline for production runs.
    """
    dl_grid = np.linspace(dl_min, dl_max, n_grid)

    # Dummy: flat density field (1 + delta = 1 everywhere)
    n_los = np.ones_like(dl_grid)

    # p(D_L) ~ D_L^2 * n(D_L)
    p_dl = dl_grid**2 * n_los

    return bilby.core.prior.Interped(
        xx=dl_grid, yy=p_dl,
        minimum=dl_min, maximum=dl_max,
        name="luminosity_distance",
        latex_label=r"$D_L$", unit="Mpc",
    )


# ---------------------------------------------------------------------------
# Set up interferometers
# ---------------------------------------------------------------------------

def setup_ifos_injection(duration, sampling_frequency, injection_parameters,
                         fmin=40.0):
    """Create interferometers with an injected BNS signal (for testing)."""
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
        parameter_conversion=(
            bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters
        ),
        waveform_arguments={
            "waveform_approximant": "IMRPhenomPv2_NRTidal",
            "reference_frequency": 100.0,
            "minimum_frequency": fmin,
        },
    )

    ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=TRIGGER_TIME - duration + 2,
    )
    ifos.inject_signal(
        waveform_generator=waveform_generator,
        parameters=injection_parameters,
        raise_error=False,
    )

    for ifo in ifos:
        ifo.minimum_frequency = fmin
        ifo.maximum_frequency = 2048.0

    return ifos


def setup_ifos_real_data(duration, sampling_frequency, fmin=23.0,
                         cache_dir=None):
    """Load real GWOSC strain data for GW170817, caching to disk."""
    from gwpy.timeseries import TimeSeries

    post_trigger_duration = 2
    start_time = TRIGGER_TIME - duration + post_trigger_duration
    end_time = start_time + duration
    psd_duration = 32 * duration

    ifos = bilby.gw.detector.InterferometerList([])
    for det_name in ["H1", "L1", "V1"]:
        ifo = bilby.gw.detector.get_empty_interferometer(det_name)

        # Check for cached strain
        cache_file = None
        if cache_dir is not None:
            cache_file = os.path.join(cache_dir, f"{det_name}_strain.hdf5")

        if cache_file and os.path.exists(cache_file):
            print(f"  Loading cached strain for {det_name}")
            data = TimeSeries.read(cache_file)
            # Crop to requested time range (cache may be longer)
            if data.duration.value > duration + 0.01:
                print(f"    Cropping {data.duration.value}s → {duration}s")
                data = data.crop(start_time, end_time)
        else:
            print(f"  Fetching strain for {det_name} from GWOSC...")
            data = TimeSeries.fetch_open_data(det_name, start_time, end_time)
            if cache_file:
                data.write(cache_file, overwrite=True)
                print(f"  Cached to {cache_file}")

        ifo.strain_data.set_from_gwpy_timeseries(data)

        # PSD from off-source data
        psd_cache = None
        if cache_dir is not None:
            psd_cache = os.path.join(
                cache_dir, f"{det_name}_psd_dur{int(duration)}.npz")

        psd_loaded = False

        # Try loading from cache
        if psd_cache and os.path.exists(psd_cache):
            d = np.load(psd_cache)
            if np.any(np.isnan(d["psd"])):
                print(f"  Cached PSD for {det_name} has NaN, removing")
                os.remove(psd_cache)
            else:
                print(f"  Loading cached PSD for {det_name}")
                ifo.power_spectral_density = \
                    bilby.gw.detector.PowerSpectralDensity(
                        frequency_array=d["freq"], psd_array=d["psd"],
                    )
                psd_loaded = True

        # Estimate from off-source data
        if not psd_loaded:
            print(f"  Fetching PSD data for {det_name} from GWOSC...")
            psd_data = TimeSeries.fetch_open_data(
                det_name, start_time - psd_duration, start_time,
            )

            # Some detectors (V1) have NaN gaps in off-source data.
            # Fill gaps with zero so those FFT segments contribute zero
            # power and the median PSD is dominated by valid segments.
            raw = psd_data.value.copy()
            n_nan = np.sum(np.isnan(raw))
            if n_nan > 0:
                nan_frac = n_nan / len(raw)
                print(f"  WARNING: {det_name} off-source data has "
                      f"{nan_frac:.1%} NaN samples, filling with zeros")
                raw[np.isnan(raw)] = 0.0
                psd_data = TimeSeries(raw, dt=psd_data.dt, t0=psd_data.t0)

            psd = psd_data.psd(fftlength=duration, method="median")
            psd_freq = psd.frequencies.value
            psd_val = psd.value

            if np.any(np.isnan(psd_val)):
                print(f"  WARNING: {det_name} PSD still has NaN after "
                      f"gap-filling, using bilby default noise curve")
            else:
                ifo.power_spectral_density = \
                    bilby.gw.detector.PowerSpectralDensity(
                        frequency_array=psd_freq, psd_array=psd_val,
                    )
                if psd_cache:
                    np.savez(psd_cache, freq=psd_freq, psd=psd_val)
                    print(f"  Cached to {psd_cache}")

        ifo.minimum_frequency = fmin
        ifo.maximum_frequency = 2048.0
        ifos.append(ifo)

    return ifos


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------

def build_priors(use_density_prior=False, no_tides=False,
                  roq_scale_factor=None, marginalize_psi=False,
                  aligned_spin=False, marginalize_phase=False):
    """Build priors for GW170817 PE.

    If no_tides=True, use BBHPriorDict (no lambda_1/lambda_2).
    Otherwise use BNSPriorDict with tidal deformabilities.
    """
    if no_tides:
        priors = bilby.gw.prior.BBHPriorDict(aligned_spin=False)
    else:
        priors = bilby.gw.prior.BNSPriorDict(aligned_spin=False)

    if roq_scale_factor is not None:
        # ROQ basis bounds (from params.dat), scaled by 1/s
        s = roq_scale_factor
        m_min = 1.0014 / s + 0.001  # comp-min from basis
        mc_min = 1.4206 / s
        mc_max = 2.6021 / s
        priors["mass_1"] = bilby.prior.Uniform(m_min, 7.7, name="mass_1")
        priors["mass_2"] = bilby.prior.Uniform(m_min, 7.7, name="mass_2")
        priors["chirp_mass"] = bilby.prior.Constraint(
            mc_min, mc_max, name="chirp_mass")
    else:
        priors["mass_1"] = bilby.prior.Uniform(0.5, 7.7, name="mass_1")
        priors["mass_2"] = bilby.prior.Uniform(0.5, 7.7, name="mass_2")
    priors["mass_ratio"] = bilby.prior.Constraint(0.125, 1.0)
    if "chirp_mass" in priors and roq_scale_factor is None:
        del priors["chirp_mass"]

    priors["a_1"] = bilby.prior.Uniform(0, 0.80, name="a_1")
    priors["a_2"] = bilby.prior.Uniform(0, 0.80, name="a_2")

    if aligned_spin:
        priors["tilt_1"] = bilby.prior.DeltaFunction(0.0, name="tilt_1")
        priors["tilt_2"] = bilby.prior.DeltaFunction(0.0, name="tilt_2")
        priors["phi_12"] = bilby.prior.DeltaFunction(0.0, name="phi_12")
        priors["phi_jl"] = bilby.prior.DeltaFunction(0.0, name="phi_jl")
    else:
        priors["tilt_1"] = bilby.prior.Sine(name="tilt_1")
        priors["tilt_2"] = bilby.prior.Sine(name="tilt_2")
        priors["phi_12"] = bilby.prior.Uniform(0, 2 * np.pi, name="phi_12")
        priors["phi_jl"] = bilby.prior.Uniform(0, 2 * np.pi, name="phi_jl")

    if use_density_prior:
        priors["luminosity_distance"] = build_density_prior()
    else:
        priors["luminosity_distance"] = bilby.gw.prior.UniformSourceFrame(
            minimum=1, maximum=100, name="luminosity_distance",
        )

    priors["ra"] = bilby.prior.DeltaFunction(RA_NGC4993, name="ra")
    priors["dec"] = bilby.prior.DeltaFunction(DEC_NGC4993, name="dec")

    priors["theta_jn"] = bilby.prior.Sine(name="theta_jn")

    if marginalize_phase:
        priors["phase"] = bilby.prior.DeltaFunction(0.0, name="phase")
    else:
        priors["phase"] = bilby.prior.Uniform(0, 2 * np.pi, name="phase")

    if marginalize_psi:
        priors["psi"] = bilby.prior.DeltaFunction(0.0, name="psi")
    else:
        priors["psi"] = bilby.prior.Uniform(0, np.pi, name="psi")

    priors["geocent_time"] = bilby.prior.Uniform(
        TRIGGER_TIME - 0.1, TRIGGER_TIME + 0.1, name="geocent_time",
    )

    if not no_tides:
        priors["lambda_1"] = bilby.prior.Uniform(0, 5000, name="lambda_1")
        priors["lambda_2"] = bilby.prior.Uniform(0, 5000, name="lambda_2")

    return priors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_injection_parameters():
    """GW170817-like injection parameters for testing."""
    return dict(
        mass_1=1.5,
        mass_2=1.3,
        luminosity_distance=40.0,
        theta_jn=2.6,    # ~149 deg → theta_obs ~ 31 deg, consistent with JetFit
        ra=RA_NGC4993,
        dec=DEC_NGC4993,
        psi=2.659,
        phase=1.3,
        geocent_time=TRIGGER_TIME,
        a_1=0.02,
        a_2=0.01,
        tilt_1=0.5,
        tilt_2=1.0,
        phi_12=1.7,
        phi_jl=0.3,
        lambda_1=400.0,
        lambda_2=450.0,
    )


def main():
    import logging
    bilby.core.utils.logger.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-data", action="store_true",
                        help="Use GWOSC strain instead of injection")
    parser.add_argument("--nlive", type=int, default=50,
                        help="Number of live points (default: 50 for testing)")
    parser.add_argument("--maxmcmc", type=int, default=500,
                        help="Max MCMC steps per live point proposal")
    parser.add_argument("--nact", type=int, default=10,
                        help="Autocorrelation lengths for act-walk proposals "
                             "(default: 10; bilby default is 2)")
    parser.add_argument("--duration", type=float, default=32.0,
                        help="Segment duration [s] (default: 32 for test, "
                             "128 for real)")
    parser.add_argument("--fmin", type=float, default=None,
                        help="Minimum frequency [Hz] (default: 200 for test, "
                             "23 for real data)")
    parser.add_argument("--sampling-frequency", type=float, default=4096.0)
    parser.add_argument("--no-jetfit", action="store_true",
                        help="Disable JetFit EM constraint")
    parser.add_argument("--density-prior", action="store_true",
                        help="Use density-weighted distance prior")
    parser.add_argument("--roq", action="store_true",
                        help="Use ROQ likelihood with IMRPhenomPv2 (no tides)")
    parser.add_argument("--roq-folder", type=str, default=None,
                        help="Path to ROQ basis folder (default: data/ROQ)")
    parser.add_argument("--marginalize-psi", action="store_true",
                        help="Marginalize over polarisation angle "
                             "(numerical quadrature)")
    parser.add_argument("--n-psi", type=int, default=50,
                        help="Number of psi quadrature points (default: 50)")
    parser.add_argument("--aligned-spin", action="store_true",
                        help="Use aligned-spin waveform "
                             "(fix tilt/phi to 0)")
    parser.add_argument("--marginalize-phase", action="store_true",
                        help="Analytically marginalize over phase "
                             "(requires --aligned-spin)")
    parser.add_argument("--marginalize-time", action="store_true",
                        help="Marginalize over coalescence time (FFT grid)")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--npool", type=int, default=1)
    args = parser.parse_args()

    if args.real_data and args.duration < 16:
        if args.roq:
            args.duration = 106.5  # 128/106.5 = 1.20188, mc_min_eff = 1.18
        else:
            args.duration = 128.0
        print(f"Real data mode: setting duration = {args.duration} s")

    if args.roq:
        # Validate duration * sampling_frequency is integer
        prod = args.duration * args.sampling_frequency
        if abs(prod - round(prod)) > 1e-6:
            raise ValueError(
                f"duration={args.duration} * fs={args.sampling_frequency} "
                f"= {prod} is not integer. For ROQ with basis seglen="
                f"{ROQ_BASIS_SEGLEN}, try duration=106.5 or 106.0")

    if args.fmin is None:
        if args.roq:
            args.fmin = 25.0
        else:
            args.fmin = 23.0 if args.real_data else 200.0

    if args.marginalize_phase and not args.aligned_spin:
        parser.error("--marginalize-phase requires --aligned-spin")
    if args.marginalize_psi and args.roq:
        parser.error("--marginalize-psi is not supported with --roq")
    if args.marginalize_time and not args.marginalize_psi:
        parser.error("--marginalize-time requires --marginalize-psi")
    use_jetfit = not args.no_jetfit

    # Output directory
    label = "GW170817"
    if args.roq:
        label += "_roq"
    if use_jetfit:
        label += "_jetfit"
    if args.density_prior:
        label += "_density"
    if args.marginalize_psi:
        label += "_psimarg"
    if args.aligned_spin:
        label += "_aligned"
    if args.marginalize_phase:
        label += "_phasemarg"
    if args.marginalize_time:
        label += "_timemarg"

    outdir = args.outdir or os.path.join(
        os.path.dirname(__file__), "..", "..", "results", label)
    os.makedirs(outdir, exist_ok=True)

    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data",
                            "GW170817")
    os.makedirs(data_dir, exist_ok=True)

    # ----- Build JetFit GMM -----
    gmm = None
    if use_jetfit:
        print("Building JetFit GMM constraint...")
        gmm = load_or_build_jetfit_gmm(data_dir)
        test_point = np.array([[40.0, 0.53]])
        log_p = gmm.score_samples(test_point)[0]
        print(f"  GMM log-density at (D_L=40, theta_obs=0.53): {log_p:.2f}")

    # ----- Build priors -----
    roq_sf = ROQ_BASIS_SEGLEN / args.duration if args.roq else None
    priors = build_priors(use_density_prior=args.density_prior,
                          no_tides=args.roq,
                          roq_scale_factor=roq_sf,
                          marginalize_psi=args.marginalize_psi,
                          aligned_spin=args.aligned_spin,
                          marginalize_phase=args.marginalize_phase)

    # ----- Set up interferometers -----
    injection_parameters = get_injection_parameters()

    if args.real_data:
        print("Fetching GWOSC strain data (this may take a while)...")
        ifos = setup_ifos_real_data(args.duration, args.sampling_frequency,
                                    fmin=args.fmin, cache_dir=data_dir)
    else:
        print("Using injected BNS signal for testing...")
        ifos = setup_ifos_injection(
            args.duration, args.sampling_frequency, injection_parameters,
            fmin=args.fmin)

    # ----- Waveform generator -----
    if args.roq:
        approximant = "IMRPhenomPv2"
        source_model = bilby.gw.source.binary_black_hole_roq
        param_conversion = (
            bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters
        )
    elif args.aligned_spin:
        approximant = "IMRPhenomD_NRTidal"
        source_model = bilby.gw.source.lal_binary_neutron_star
        param_conversion = (
            bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters
        )
    else:
        approximant = "IMRPhenomPv2_NRTidal"
        source_model = bilby.gw.source.lal_binary_neutron_star
        param_conversion = (
            bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters
        )

    waveform_args = {
        "waveform_approximant": approximant,
        "reference_frequency": 100.0,
        "minimum_frequency": args.fmin,
    }
    if args.roq:
        roq_folder = args.roq_folder or os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "ROQ")
        waveform_args["frequency_nodes_linear"] = np.load(
            os.path.join(roq_folder, "fnodes_linear.npy")) * roq_sf
        waveform_args["frequency_nodes_quadratic"] = np.load(
            os.path.join(roq_folder, "fnodes_quadratic.npy")) * roq_sf

    waveform_generator = bilby.gw.WaveformGenerator(
        duration=args.duration,
        sampling_frequency=args.sampling_frequency,
        frequency_domain_source_model=source_model,
        parameter_conversion=param_conversion,
        waveform_arguments=waveform_args,
    )

    # ----- Likelihood -----
    phase_marg = args.marginalize_phase

    if args.roq:
        roq_scale_factor = roq_sf
        print(f"Using ROQ likelihood from {roq_folder}")
        print(f"  scale_factor = {ROQ_BASIS_SEGLEN}/{args.duration} "
              f"= {roq_scale_factor:.5f}")
        print(f"  mc range (scaled): "
              f"[{1.42/roq_scale_factor:.4f}, {2.60/roq_scale_factor:.4f}]")
        print(f"  flow (scaled): {20*roq_scale_factor:.2f} Hz")

        roq_kwargs = dict(
            interferometers=ifos,
            waveform_generator=waveform_generator,
            priors=priors,
            linear_matrix=os.path.join(roq_folder, "B_linear.npy"),
            quadratic_matrix=os.path.join(roq_folder, "B_quadratic.npy"),
            roq_params=os.path.join(roq_folder, "params.dat"),
            roq_scale_factor=roq_scale_factor,
            phase_marginalization=phase_marg,
        )
        if use_jetfit:
            print("  + JetFit EM constraint")
            likelihood = ROQplusEMLikelihood(**roq_kwargs, gmm=gmm)
        else:
            likelihood = bilby.gw.likelihood.ROQGravitationalWaveTransient(
                **roq_kwargs)
    elif args.marginalize_psi:
        desc = "GW + psi-marginalised"
        if use_jetfit:
            desc += " + JetFit EM"
        print(f"Using {desc} likelihood (n_psi={args.n_psi})")
        likelihood = PsiMarginalizedLikelihood(
            interferometers=ifos,
            waveform_generator=waveform_generator,
            priors=priors,
            gmm=gmm if use_jetfit else None,
            n_psi=args.n_psi,
            marginalize_phase=args.marginalize_phase,
            marginalize_time=args.marginalize_time,
        )
    elif use_jetfit:
        print("Using GW + JetFit EM likelihood")
        likelihood = GWplusEMLikelihood(
            interferometers=ifos,
            waveform_generator=waveform_generator,
            priors=priors,
            gmm=gmm,
            phase_marginalization=phase_marg,
        )
    else:
        print("Using standard GW likelihood")
        likelihood = bilby.gw.GravitationalWaveTransient(
            interferometers=ifos,
            waveform_generator=waveform_generator,
            priors=priors,
            phase_marginalization=phase_marg,
        )

    # ----- Quick likelihood evaluation test -----
    print("\n--- Likelihood sanity check ---")
    test_params = dict(injection_parameters)
    if args.roq:
        test_params.pop("lambda_1", None)
        test_params.pop("lambda_2", None)
    if args.aligned_spin:
        test_params["tilt_1"] = 0.0
        test_params["tilt_2"] = 0.0
        test_params["phi_12"] = 0.0
        test_params["phi_jl"] = 0.0
    log_l = likelihood.log_likelihood(parameters=test_params)
    print(f"  log L at injection: {log_l:.2f}")

    if use_jetfit:
        theta_obs = min(injection_parameters["theta_jn"],
                        np.pi - injection_parameters["theta_jn"])
        log_em = gmm.score_samples(
            np.array([[injection_parameters["luminosity_distance"],
                       theta_obs]])
        )[0]
        print(f"  EM log-density at injection: {log_em:.2f}")
        print(f"  theta_obs at injection: {np.degrees(theta_obs):.1f} deg")

    if not np.isfinite(log_l):
        print("  WARNING: log L is not finite at injection point!")
    print()

    # ----- Run sampler -----
    print(f"Running dynesty with nlive={args.nlive}, maxmcmc={args.maxmcmc}, "
          f"npool={args.npool}")

    # Start a background progress monitor that writes directly to a file
    # every 30s, independent of SLURM stdout buffering.
    progress_log = os.path.join(outdir, "progress.log")
    monitor = ProgressMonitor(progress_log, interval=30)

    # Hook into bilby's Dynesty to start the monitor once the sampler exists
    from bilby.core.sampler.dynesty import Dynesty as _Dynesty
    _orig_run = _Dynesty._run_external_sampler_with_checkpointing

    def _monitored_run(self):
        monitor.start(self.sampler)
        try:
            return _orig_run(self)
        finally:
            monitor.stop()

    _Dynesty._run_external_sampler_with_checkpointing = _monitored_run

    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        nlive=args.nlive,
        npool=args.npool,
        nact=args.nact,
        maxmcmc=args.maxmcmc,
        outdir=outdir,
        label=label,
        conversion_function=(
            bilby.gw.conversion.generate_all_bbh_parameters if args.roq
            else bilby.gw.conversion.generate_all_bns_parameters
        ),
        check_point_delta_t=600,
        resume=True,
    )

    # ----- Report results -----
    print("\n--- Posterior summary ---")
    for key in ["luminosity_distance", "theta_jn", "chirp_mass",
                "mass_ratio", "lambda_tilde"]:
        if key in result.posterior:
            med = np.median(result.posterior[key])
            lo, hi = np.percentile(result.posterior[key], [16, 84])
            print(f"  {key}: {med:.2f} [{lo:.2f}, {hi:.2f}]")

    result.plot_corner(
        parameters=["luminosity_distance", "theta_jn", "chirp_mass",
                     "mass_ratio"],
        filename=os.path.join(outdir, f"{label}_corner.png"),
    )

    # ----- Convergence diagnostics -----
    print("\n--- Convergence diagnostics ---")
    nsamples = len(result.posterior)
    print(f"  nsamples:      {nsamples}")
    print(f"  ln_evidence:   {result.log_evidence:.2f}"
          f" +/- {result.log_evidence_err:.2f}")
    print(f"  ln_bayes_factor: {result.log_bayes_factor:.2f}"
          f" +/- {result.log_evidence_err:.2f}")

    # dynesty stores the final dlogz in the nested_samples metadata
    sampler_kwargs = getattr(result, "sampler_kwargs", {}) or {}
    dlogz_final = sampler_kwargs.get("dlogz", None)
    if dlogz_final is not None:
        print(f"  dlogz (stop):  {dlogz_final}")

    if nsamples < 100:
        print("  WARNING: fewer than 100 posterior samples — likely unconverged")
    if result.log_evidence_err > 1.0:
        print("  WARNING: large evidence uncertainty (> 1.0) — increase nlive")

    print(f"\nResults saved to {outdir}")


if __name__ == "__main__":
    main()
