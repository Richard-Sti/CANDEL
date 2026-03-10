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

import bilby
import numpy as np
from sklearn.mixture import GaussianMixture


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRIGGER_TIME = 1187008882.43
RA_NGC4993 = 3.44616       # rad (197.45 deg)
DEC_NGC4993 = -0.4085      # rad (-23.38 deg)
Z_OBS = 0.0099             # observed CMB-frame redshift


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

    def log_likelihood_ratio(self, parameters=None):
        """GW log-likelihood ratio + EM log-likelihood.

        Only override log_likelihood_ratio; the base class log_likelihood()
        dispatches to self.log_likelihood_ratio() + noise_log_likelihood(),
        so the EM term is automatically included in both code paths without
        double-counting.
        """
        log_lr_gw = super().log_likelihood_ratio(parameters=parameters)
        if self.gmm is None or not np.isfinite(log_lr_gw):
            return log_lr_gw

        p = parameters if parameters is not None else self.parameters
        theta_jn = p["theta_jn"]
        dl = p["luminosity_distance"]

        # Fold inclination to viewing angle [0, pi/2]
        theta_obs = min(theta_jn, np.pi - theta_jn)

        # GMM log-density (encodes JetFit posterior; JetFit priors are flat,
        # so dividing them out is just a constant shift that doesn't affect
        # the posterior shape)
        log_l_em = self.gmm.score_samples(np.array([[dl, theta_obs]]))[0]

        return log_lr_gw + log_l_em


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

def build_priors(use_density_prior=False):
    """Build high_spin_PhenomPNRT priors matching Abbott et al. 2019."""
    priors = bilby.gw.prior.BNSPriorDict(aligned_spin=False)

    # Sample in (mass_1, mass_2) with m1 >= m2
    priors["mass_1"] = bilby.prior.Uniform(0.5, 7.7, name="mass_1")
    priors["mass_2"] = bilby.prior.Uniform(0.5, 7.7, name="mass_2")
    priors["mass_ratio"] = bilby.prior.Constraint(0.125, 1.0)
    del priors["chirp_mass"]

    priors["a_1"] = bilby.prior.Uniform(0, 0.89, name="a_1")
    priors["a_2"] = bilby.prior.Uniform(0, 0.89, name="a_2")
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
    priors["phase"] = bilby.prior.Uniform(0, 2 * np.pi, name="phase")
    priors["psi"] = bilby.prior.Uniform(0, np.pi, name="psi")

    priors["geocent_time"] = bilby.prior.Uniform(
        TRIGGER_TIME - 0.1, TRIGGER_TIME + 0.1, name="geocent_time",
    )

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-data", action="store_true",
                        help="Use GWOSC strain instead of injection")
    parser.add_argument("--nlive", type=int, default=50,
                        help="Number of live points (default: 50 for testing)")
    parser.add_argument("--maxmcmc", type=int, default=500,
                        help="Max MCMC steps per live point proposal")
    parser.add_argument("--nact", type=int, default=5,
                        help="Autocorrelation lengths for act-walk proposals "
                             "(default: 5; bilby default is 2)")
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
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--npool", type=int, default=1)
    args = parser.parse_args()

    if args.real_data and args.duration < 16:
        args.duration = 128.0
        print(f"Real data mode: setting duration = {args.duration} s")

    if args.fmin is None:
        args.fmin = 23.0 if args.real_data else 200.0

    use_jetfit = not args.no_jetfit

    # Output directory
    label = "GW170817"
    if use_jetfit:
        label += "_jetfit"
    if args.density_prior:
        label += "_density"

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
    priors = build_priors(use_density_prior=args.density_prior)

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
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=args.duration,
        sampling_frequency=args.sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
        parameter_conversion=(
            bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters
        ),
        waveform_arguments={
            "waveform_approximant": "IMRPhenomPv2_NRTidal",
            "reference_frequency": 100.0,
            "minimum_frequency": args.fmin,
        },
    )

    # ----- Likelihood -----
    if use_jetfit:
        print("Using GW + JetFit EM likelihood")
        likelihood = GWplusEMLikelihood(
            interferometers=ifos,
            waveform_generator=waveform_generator,
            priors=priors,
            gmm=gmm,
        )
    else:
        print("Using standard GW likelihood")
        likelihood = bilby.gw.GravitationalWaveTransient(
            interferometers=ifos,
            waveform_generator=waveform_generator,
            priors=priors,
        )

    # ----- Quick likelihood evaluation test -----
    print("\n--- Likelihood sanity check ---")
    log_l = likelihood.log_likelihood(parameters=injection_parameters)
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
    print(f"Running dynesty with nlive={args.nlive}, maxmcmc={args.maxmcmc}")
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
        conversion_function=bilby.gw.conversion.generate_all_bns_parameters,
        check_point_delta_t=600,
        resume=False,
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
