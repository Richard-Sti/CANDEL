"""Generate LaTeX results tables from MCMC run outputs.

This script generates multiple LaTeX tables for the cluster anisotropy paper:
- Table 1: Base + dipoles (main results with parameters)
- Table 2: Beyond dipoles summary (LTYT only, evidence comparison)
- Appendix tables: Full parameter tables for pixel/quad/radial models

Tables are output to paper_clusters/tables/ as plain text files.

Usage:
    python paper_clusters/tables.py --folder short

LaTeX requirements:
    \\usepackage{booktabs}  % for \\addlinespace, \\toprule, etc.
    \\usepackage{siunitx}   % optional, for number formatting
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import tomli
from astropy.coordinates import SkyCoord
import astropy.units as u

from config import (
    RESULTS_FOLDER, RESULTS_ROOT, RECON_LABELS_SHORT,
    RECONSTRUCTIONS, get_active_reconstructions,
)
from utils import quadrature


# =============================================================================
# Configuration flags
# =============================================================================

# If True, always report asymmetric errors (e.g., $val^{+upper}_{-lower}$)
# If False, use symmetric errors when upper and lower are within tolerance
REPORT_TWOSIDED = True

# When REPORT_TWOSIDED=False, use symmetric ± if error asymmetry is below this
# fraction of the average error (e.g., 0.1 = 10% tolerance)
SYMMETRIC_TOLERANCE = 0.1


# =============================================================================
# Constraint detection utilities
# =============================================================================

def get_constraint(samples: np.ndarray, prior_min: float, prior_max: float,
                   threshold: float = 0.05) -> tuple:
    """Determine if posterior is well-constrained or hitting prior bounds.

    Parameters
    ----------
    samples : np.ndarray
        1D array of MCMC samples for the parameter.
    prior_min : float
        Lower bound of the prior.
    prior_max : float
        Upper bound of the prior.
    threshold : float
        Fraction of prior range to consider as "hitting the bound".

    Returns
    -------
    tuple
        (value, lower_err, upper_err, is_upper_limit, is_lower_limit)
        - If well-constrained: (median, median-p16, p84-median, False, False)
        - If upper limit: (p95, None, None, True, False)
        - If lower limit: (p5, None, None, False, True)
    """
    p5, p16, p50, p84, p95 = np.percentile(samples, [5, 16, 50, 84, 95])
    prior_range = prior_max - prior_min

    # Check if hitting lower bound → upper limit
    if (p16 - prior_min) / prior_range < threshold:
        return p95, None, None, True, False  # 95% upper limit

    # Check if hitting upper bound → lower limit
    if (prior_max - p84) / prior_range < threshold:
        return p5, None, None, False, True   # 95% lower limit

    # Well-constrained: report median with asymmetric errors
    return p50, p50 - p16, p84 - p50, False, False


def read_samples_from_hdf5(fname: str, param_name: str) -> Optional[np.ndarray]:
    """Read MCMC samples for a parameter from HDF5 file.

    Parameters
    ----------
    fname : str
        Path to HDF5 file.
    param_name : str
        Name of the parameter in the samples group.

    Returns
    -------
    np.ndarray or None
        1D array of samples (flattened if multi-chain), or None if not found.
    """
    try:
        with h5py.File(fname, "r") as f:
            if "samples" not in f:
                return None
            if param_name not in f["samples"]:
                return None
            samples = f[f"samples/{param_name}"][...]
            return samples.flatten()  # Flatten in case of multi-chain
    except Exception:
        return None


def read_prior_bounds_from_toml(fname: str, prior_key: str) -> tuple:
    """Read prior bounds from the TOML config file accompanying the HDF5.

    Parameters
    ----------
    fname : str
        Path to HDF5 file (will replace .hdf5 with .toml).
    prior_key : str
        Key in model.priors section (e.g., "zeropoint_dipole", "Vext").

    Returns
    -------
    tuple
        (prior_min, prior_max) or (None, None) if not found.
    """
    toml_path = Path(fname).with_suffix(".toml")
    if not toml_path.exists():
        return None, None

    try:
        with open(toml_path, "rb") as f:
            config = tomli.load(f)
        prior_config = config.get("model", {}).get("priors", {}).get(prior_key, {})

        if not prior_config:
            return None, None

        prior_min = prior_config.get("low")
        prior_max = prior_config.get("high")

        return prior_min, prior_max
    except Exception:
        return None, None


def format_amplitude_constraint(samples: Optional[np.ndarray],
                                prior_min: Optional[float],
                                prior_max: Optional[float],
                                fmt: str = ".2f",
                                unit: str = "",
                                scale: float = 1.0,
                                threshold: float = 0.05) -> str:
    """Format an amplitude constraint with upper limit detection.

    Parameters
    ----------
    samples : np.ndarray or None
        MCMC samples for the amplitude parameter.
    prior_min, prior_max : float or None
        Prior bounds.
    fmt : str
        Number format string.
    unit : str
        Unit string to append (e.g., r"\\,km/s" or r"\\%").
    scale : float
        Multiplicative scale factor (e.g., 100 for percentages).
    threshold : float
        Threshold for detecting prior rail.

    Returns
    -------
    str
        LaTeX-formatted string.
    """
    if samples is None or prior_min is None or prior_max is None:
        return r"\textemdash"

    val, lower_err, upper_err, is_upper, is_lower = get_constraint(
        samples, prior_min, prior_max, threshold
    )

    # Apply scale
    val = val * scale
    if lower_err is not None:
        lower_err = lower_err * scale
    if upper_err is not None:
        upper_err = upper_err * scale

    if is_upper:
        return f"$< {val:{fmt}}${unit}"
    elif is_lower:
        return f"$> {val:{fmt}}${unit}"
    else:
        # Well-constrained: show median with asymmetric or symmetric errors
        use_symmetric = (
            not REPORT_TWOSIDED and
            abs(lower_err - upper_err) < SYMMETRIC_TOLERANCE * (lower_err + upper_err) / 2
        )
        if use_symmetric:
            # Nearly symmetric errors - use simple ±
            avg_err = (lower_err + upper_err) / 2
            return f"${val:{fmt}} \\pm {avg_err:{fmt}}${unit}"
        else:
            # Asymmetric errors
            return f"${val:{fmt}}^{{+{upper_err:{fmt}}}}_{{-{lower_err:{fmt}}}}${unit}"


def icrs_spherical_to_galactic(phi_samples: np.ndarray,
                                cos_theta_samples: np.ndarray) -> tuple:
    """Convert ICRS spherical coordinates to Galactic (l, b).

    The quadrupole is sampled in ICRS frame with:
    - phi in [0, 2π] radians (corresponds to RA)
    - cos_theta in [-1, 1] (theta measured from pole)

    Parameters
    ----------
    phi_samples : np.ndarray
        Samples of phi in radians.
    cos_theta_samples : np.ndarray
        Samples of cos(theta).

    Returns
    -------
    tuple
        (l_mean, l_std, b_mean, b_std) in degrees.
    """
    # Convert to RA (degrees) and Dec (degrees)
    ra = np.degrees(phi_samples)  # phi -> RA in degrees
    dec = 90.0 - np.degrees(np.arccos(cos_theta_samples))  # theta from pole -> Dec

    # Convert each sample to Galactic
    coords = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    l_samples = coords.galactic.l.degree
    b_samples = coords.galactic.b.degree

    # Handle l wrapping (l is in [0, 360))
    # Use circular mean for l if needed
    l_mean = np.mean(l_samples)
    l_std = np.std(l_samples)
    b_mean = np.mean(b_samples)
    b_std = np.std(b_samples)

    return l_mean, l_std, b_mean, b_std


def get_quadrupole_galactic_direction(fname: str, prefix: str, idx: int) -> tuple:
    """Get Galactic (l, b) for a quadrupole direction from HDF5 samples.

    Parameters
    ----------
    fname : str
        Path to HDF5 file.
    prefix : str
        Parameter prefix (e.g., "Vext_quad" or "zeropoint_quad").
    idx : int
        Direction index (1 or 2).

    Returns
    -------
    tuple
        (l_mean, l_std, b_mean, b_std) or (None, None, None, None) if not found.
    """
    phi_samples = read_samples_from_hdf5(fname, f"{prefix}_phi{idx}")
    cos_theta_samples = read_samples_from_hdf5(fname, f"{prefix}_cos_theta{idx}")

    if phi_samples is None or cos_theta_samples is None:
        return None, None, None, None

    return icrs_spherical_to_galactic(phi_samples, cos_theta_samples)


# Output directory for tables
TABLES_DIR = Path(__file__).parent / "tables"
TABLES_DIR.mkdir(exist_ok=True)


# =============================================================================
# Filename parsing and metadata
# =============================================================================

@dataclass
class RunMetadata:
    """Metadata extracted from a run filename."""
    reconstruction: str  # "Carrick2015", "manticore", "Vext" (NoRecon)
    relation: str  # "LT", "YT", "LTYT"
    model_type: str  # "base", "dipVext", "dipH0", "dipA", etc.
    has_y: bool
    stem: str  # Original filename stem

    @property
    def recon_pretty(self) -> str:
        """Pretty name for reconstruction."""
        return RECON_LABELS_SHORT.get(self.reconstruction, self.reconstruction)


def parse_filename(fname: str) -> Optional[RunMetadata]:
    """Parse a filename to extract run metadata.

    Filename format: {recon}_{relation}_noMNR[_model_flags][_hasY].hdf5
    """
    stem = Path(fname).stem

    # Determine reconstruction
    if stem.startswith("Carrick2015_"):
        recon = "Carrick2015"
        rest = stem[len("Carrick2015_"):]
    elif stem.startswith("manticore_"):
        recon = "manticore"
        rest = stem[len("manticore_"):]
    elif stem.startswith("2mpp_zspace_galaxies_"):
        recon = "2mpp_zspace_galaxies"
        rest = stem[len("2mpp_zspace_galaxies_"):]
    elif stem.startswith("Vext_"):
        recon = "Vext"
        rest = stem[len("Vext_"):]
    else:
        return None

    # Check for hasY suffix
    has_y = rest.endswith("_hasY")
    if has_y:
        rest = rest[:-len("_hasY")]

    # Extract relation
    relation = None
    for rel in ["LTYT", "LT", "YT"]:  # Check LTYT first (longer match)
        if rest.startswith(f"{rel}_noMNR"):
            relation = rel
            rest = rest[len(f"{rel}_noMNR"):]
            break

    if relation is None:
        return None

    # Remove leading underscore if present
    if rest.startswith("_"):
        rest = rest[1:]

    # Determine model type from remaining string
    model_type = parse_model_type(rest) if rest else "base"

    return RunMetadata(
        reconstruction=recon,
        relation=relation,
        model_type=model_type,
        has_y=has_y,
        stem=stem,
    )


def parse_model_type(flags: str) -> str:
    """Parse model type from filename flags.

    Examples:
        "" -> "base"
        "dipVext" -> "dipVext"
        "dipA" -> "dipA"
        "dipH0" -> "dipH0"
        "nodipA" -> "nodipA"  (keep as distinct model type)
        "nodipA_dipVext" -> "nodipA_dipVext"
        "dipA_dipVext" -> "dipA_dipVext"
    """
    if not flags:
        return "base"

    return flags


# Model type to pretty LaTeX name
MODEL_PRETTY_NAMES = {
    "base": "Base",
    "dipVext": r"$\mathbf{V}_{\rm ext}$",
    "dipH0": r"$H_0$",
    "dipA": r"ZP",
    "quadVext": r"$\mathbf{V}_{\rm ext}$",
    "quadA": r"ZP",
    "pixVext": r"$\mathbf{V}_{\rm ext}$",
    "pixA": r"ZP",
    "pixH0": r"$H_0$",
    "radVext": r"$\mathbf{V}_{\rm ext}$",
    "radmagVext": r"$\mathbf{V}_{\rm ext}$",
    "dipA_dipVext": r"ZP + $\mathbf{V}_{\rm ext}$",
    "dipH0_dipVext": r"$H_0$ + $\mathbf{V}_{\rm ext}$",
    "quadH0": r"$H_0$",
}


# =============================================================================
# Data reading utilities
# =============================================================================

def read_gof(fname: str, key: str) -> Optional[float]:
    """Read a GOF statistic from HDF5 file."""
    try:
        with h5py.File(fname, "r") as f:
            if "gof" not in f:
                return None
            if key not in f["gof"]:
                return None
            return float(f[f"gof/{key}"][...])
    except Exception:
        return None


def parse_summary_file(fname: str) -> dict:
    """Parse a _summary.txt file to extract parameter estimates.

    Returns dict with parameter names as keys, each containing:
        mean, std, median, 5%, 95%
    """
    summary_path = Path(fname).with_suffix("").with_suffix("")
    summary_path = Path(str(fname).replace(".hdf5", "_summary.txt"))

    if not summary_path.exists():
        return {}

    params = {}
    with open(summary_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("mean") or line.startswith("20"):
            continue

        # Parse parameter line: name  mean  std  median  5%  95%  n_eff  r_hat
        parts = line.split()
        if len(parts) < 6:
            continue

        param_name = parts[0]
        try:
            params[param_name] = {
                "mean": float(parts[1]),
                "std": float(parts[2]),
                "median": float(parts[3]),
                "p5": float(parts[4]),
                "p95": float(parts[5]),
            }
        except (ValueError, IndexError):
            continue

    return params


@dataclass
class RunResult:
    """Complete results from a single run."""
    meta: RunMetadata
    fname: str
    params: dict = field(default_factory=dict)
    lnZ_harmonic: Optional[float] = None
    err_lnZ_harmonic: Optional[float] = None
    lnZ_laplace: Optional[float] = None
    err_lnZ_laplace: Optional[float] = None
    BIC: Optional[float] = None

    @classmethod
    def from_file(cls, fname: str) -> Optional["RunResult"]:
        """Load results from a file."""
        meta = parse_filename(fname)
        if meta is None:
            return None

        params = parse_summary_file(fname)

        return cls(
            meta=meta,
            fname=fname,
            params=params,
            lnZ_harmonic=read_gof(fname, "lnZ_harmonic"),
            err_lnZ_harmonic=read_gof(fname, "err_lnZ_harmonic"),
            lnZ_laplace=read_gof(fname, "lnZ_laplace"),
            err_lnZ_laplace=read_gof(fname, "err_lnZ_laplace"),
            BIC=read_gof(fname, "BIC"),
        )

    def get_param(self, name: str, stat: str = "mean") -> Optional[float]:
        """Get a parameter statistic."""
        if name not in self.params:
            return None
        return self.params[name].get(stat)

    def get_param_with_err(self, name: str) -> tuple:
        """Get parameter mean and std."""
        if name not in self.params:
            return None, None
        p = self.params[name]
        return p.get("mean"), p.get("std")


def load_all_results(results_dir: Path) -> list[RunResult]:
    """Load all results from a directory."""
    results = []
    for hdf5_file in results_dir.glob("*.hdf5"):
        result = RunResult.from_file(str(hdf5_file))
        if result is not None:
            results.append(result)
    return results


def group_results(results: list[RunResult]) -> dict:
    """Group results by (relation, reconstruction)."""
    groups = {}
    for r in results:
        key = (r.meta.relation, r.meta.reconstruction)
        if key not in groups:
            groups[key] = {}
        groups[key][r.meta.model_type] = r
    return groups


# =============================================================================
# Table generation
# =============================================================================

def compute_delta_lnZ(result: RunResult, base_result: RunResult,
                      is_base: bool = False) -> tuple:
    """Compute delta lnZ and its error relative to base.

    If is_base=True, return (0.0, None) since base is the reference.
    """
    if is_base:
        return 0.0, None

    if result.lnZ_harmonic is None or base_result.lnZ_harmonic is None:
        return None, None

    delta = result.lnZ_harmonic - base_result.lnZ_harmonic
    err = quadrature(result.err_lnZ_harmonic, base_result.err_lnZ_harmonic)
    return delta, err


def format_val_err(val: Optional[float], err: Optional[float],
                   fmt: str = ".2f", show_err: bool = True) -> str:
    """Format a value with optional error."""
    if val is None:
        return r"\textemdash"
    if err is None or not show_err:
        return f"${val:{fmt}}$"
    return f"${val:{fmt}} \\pm {err:{fmt}}$"


def format_angle(val: Optional[float], err: Optional[float],
                  decimals: int = 0) -> str:
    """Format an angle (without degree symbol - put in header)."""
    if val is None:
        return r"\textemdash"
    fmt = f".{decimals}f"
    if err is None:
        return f"${val:{fmt}}$"
    return f"${val:{fmt}} \\pm {err:{fmt}}$"


def wrap_standalone(table_content: str, landscape: bool = False) -> str:
    """Wrap table content in a minimal standalone LaTeX document."""
    geometry = "landscape,margin=1cm" if landscape else "margin=1cm"
    preamble = rf"""\documentclass[11pt]{{article}}
\usepackage[{geometry}]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{amsmath,amssymb}}
\usepackage{{array}}
\usepackage{{multirow}}
\usepackage{{rotating}}

% Define \textemdash if not available
\providecommand{{\textemdash}}{{---}}

% Reconstruction name
\newcommand{{\TWOMZ}}{{2M\texttt{{++}}$\rho(z)$}}

\begin{{document}}
\pagestyle{{empty}}

"""
    postamble = r"""

\end{document}
"""
    return preamble + table_content + postamble


def write_table(content: str, output_path: Path, standalone: bool = True,
                landscape: bool = False) -> None:
    """Write table to file, optionally wrapping as standalone document."""
    if standalone:
        content = wrap_standalone(content, landscape=landscape)

    with open(output_path, "w") as f:
        f.write(content)

    print(f"Generated {output_path}")


# =============================================================================
# Table 1: Main dipole results
# =============================================================================

def generate_table1_dipoles(results: list[RunResult],
                            output_path: Path,
                            relations: list[str] = None) -> None:
    """Generate Table 1: Dipole models with parameters.

    For each relation and reconstruction (NoRecon, Manticore, Carrick):
    - dipVext (with Vext_mag, Vext_ell, Vext_b)
    - dipH0 (with dH_over_H, zeropoint_dipole_ell, zeropoint_dipole_b)
    - dipA (with same as dipH0)

    Uses vertical multirow labels for relation and reconstruction.
    Bold horizontal lines separate the relations.

    Amplitude constraints use upper limit detection: if the posterior is
    railed against the lower prior bound, reports 95% upper limit instead
    of median ± error.

    Args:
        relations: List of relations to include. Default: ["LT", "YT", "LTYT"]
        label: LaTeX label for the table.
    """
    groups = group_results(results)

    if relations is None:
        relations = ["LT", "YT", "LTYT"]
    recons = RECONSTRUCTIONS
    recon_pretty_map = RECON_LABELS_SHORT
    models = ["dipVext", "dipH0", "dipA"]  # No base row

    lines = []
    lines.append(r"\begin{tabular}{|c|c|l|c|c|c|c|}")
    lines.append(r"\hline\hline")
    lines.append(r"Relation & Recon & Model & Amplitude & $\ell$ [$^\circ$] & $b$ [$^\circ$] & $\Delta\ln\mathcal{Z}$ \\")
    lines.append(r"\hline\hline")

    for rel in relations:
        # Count total rows for this relation (all recons * all models)
        rel_row_count = len(recons) * len(models)
        rel_first_row = True

        for recon in recons:
            key = (rel, recon)
            group = groups.get(key, {})
            base = group.get("base")
            recon_pretty = recon_pretty_map[recon]

            # Count rows for this reconstruction (all models)
            recon_row_count = len(models)
            recon_first_row = True

            for model in models:
                r = group.get(model)

                # Relation column (multirow for first row of relation)
                if rel_first_row:
                    rel_col = f"\\multirow{{{rel_row_count}}}{{*}}{{{rel}}}"
                    rel_first_row = False
                else:
                    rel_col = ""

                # Reconstruction column (multirow for first row of recon)
                if recon_first_row:
                    recon_col = f"\\multirow{{{recon_row_count}}}{{*}}{{{recon_pretty}}}"
                    recon_first_row = False
                else:
                    recon_col = ""

                model_col = MODEL_PRETTY_NAMES.get(model, model)

                # Get amplitude and direction based on model type
                if r is None:
                    amp_str = r"\textemdash"
                    ell_str = r"\textemdash"
                    b_str = r"\textemdash"
                elif model == "dipVext":
                    # Read samples for Vext_mag
                    samples = read_samples_from_hdf5(r.fname, "Vext_mag")
                    # Prior: lower bound is always 0, upper bound from TOML
                    prior_min = 0.0
                    _, prior_max = read_prior_bounds_from_toml(r.fname, "Vext")

                    # Format amplitude with upper limit detection
                    amp_str = format_amplitude_constraint(
                        samples, prior_min, prior_max,
                        fmt=".0f", unit=r"\,km/s", scale=1.0
                    )

                    # Direction (keep as before - no upper limit detection)
                    ell, ell_err = r.get_param_with_err("Vext_ell")
                    b, b_err = r.get_param_with_err("Vext_b")
                    ell_str = format_angle(ell, ell_err)
                    b_str = format_angle(b, b_err)

                elif model in ["dipH0", "dipA"]:
                    # Read samples - try new naming first, then legacy
                    if model == "dipH0":
                        samples = read_samples_from_hdf5(r.fname, "H0_dipole_mag")
                        if samples is None:
                            samples = read_samples_from_hdf5(r.fname, "dH_over_H_dipole")
                    else:
                        samples = read_samples_from_hdf5(r.fname, "zeropoint_dipole_mag")

                    # Prior: lower bound is always 0, upper bound from TOML
                    prior_min = 0.0
                    _, prior_max = read_prior_bounds_from_toml(r.fname, "zeropoint_dipole")
                    if prior_max is None:
                        _, prior_max = read_prior_bounds_from_toml(r.fname, "H0_dipole")

                    # Format amplitude with upper limit detection (scale by 100 for %)
                    amp_str = format_amplitude_constraint(
                        samples, prior_min, prior_max,
                        fmt=".1f", unit=r"\%", scale=100.0
                    )

                    # Direction - try new naming first, then legacy
                    ell, ell_err = r.get_param_with_err("H0_dipole_ell")
                    if ell is None:
                        ell, ell_err = r.get_param_with_err("zeropoint_dipole_ell")
                    b, b_err = r.get_param_with_err("H0_dipole_b")
                    if b is None:
                        b, b_err = r.get_param_with_err("zeropoint_dipole_b")
                    ell_str = format_angle(ell, ell_err)
                    b_str = format_angle(b, b_err)
                else:
                    amp_str = r"\textemdash"
                    ell_str = r"\textemdash"
                    b_str = r"\textemdash"

                # Delta lnZ
                if r is not None and base is not None:
                    dlnZ, dlnZ_err = compute_delta_lnZ(r, base, is_base=False)
                    dlnZ_str = format_val_err(dlnZ, dlnZ_err)
                else:
                    dlnZ_str = r"\textemdash"

                lines.append(f"{rel_col} & {recon_col} & {model_col} & {amp_str} & {ell_str} & {b_str} & {dlnZ_str} \\\\")

            # Horizontal line between reconstructions
            if recon != recons[-1]:
                lines.append(r"\cline{2-7}")

        # Bold double line between relations
        if rel != relations[-1]:
            lines.append(r"\hline\hline")

    lines.append(r"\hline\hline")
    lines.append(r"\end{tabular}")

    write_table("\n".join(lines), output_path, landscape=True)


# =============================================================================
# Table 2: Beyond dipoles summary
# =============================================================================

def generate_table2_beyond_dipoles(results: list[RunResult],
                                    output_path: Path) -> None:
    """Generate Table 2: Beyond dipoles summary (LTYT only, evidence only).

    Compact table showing Delta lnZ for pixelised, radial, quadrupole, and mixed models.
    No category labels, just flat list with radial after pixelised.
    """
    groups = group_results(results)

    # Models in desired order: pixelised, radial, quadrupole, mixed
    models_order = [
        "pixVext", "pixA", "pixH0",
        "radVext", "radmagVext",
        "quadVext", "quadA", "quadH0",
        "dipA_dipVext", "dipH0_dipVext",
    ]

    # Specific model names for this table (to distinguish model types)
    model_names = {
        "pixVext": r"$\mathbf{V}_{\rm ext}$ pix.",
        "pixA": r"ZP pix.",
        "pixH0": r"$H_0$ pix.",
        "radVext": r"$\mathbf{V}_{\rm ext}$ rad.",
        "radmagVext": r"$\mathbf{V}_{\rm ext}$ rad. (fix. dir.)",
        "quadVext": r"$\mathbf{V}_{\rm ext}$ dip.+quad.",
        "quadA": r"ZP dip.+quad.",
        "quadH0": r"$H_0$ dip.+quad.",
        "dipA_dipVext": r"ZP dip. + $\mathbf{V}_{\rm ext}$ dip.",
        "dipH0_dipVext": r"$H_0$ dip. + $\mathbf{V}_{\rm ext}$ dip.",
    }

    recons = RECONSTRUCTIONS
    recon_labels = [RECON_LABELS_SHORT[r] for r in RECONSTRUCTIONS]

    lines = []
    lines.append(r"\begin{tabular}{|l|cccc|}")
    lines.append(r"\hline\hline")
    lines.append(r"Model & " + " & ".join(recon_labels) + r" \\")
    lines.append(r"\hline\hline")

    for model in models_order:
        model_name = model_names.get(model, model)

        dlnZ_strs = []
        all_missing = True
        for recon in recons:
            key = ("LTYT", recon)
            if key not in groups:
                dlnZ_strs.append(r"\textemdash")
                continue

            group = groups[key]
            base = group.get("base")
            r = group.get(model)

            if r is None or base is None:
                dlnZ_strs.append(r"\textemdash")
            else:
                all_missing = False
                dlnZ, dlnZ_err = compute_delta_lnZ(r, base)
                dlnZ_strs.append(format_val_err(dlnZ, dlnZ_err))

        lines.append(f"{model_name} & " + " & ".join(dlnZ_strs) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")

    write_table("\n".join(lines), output_path)


# =============================================================================
# Table 3: Full evidence comparison (all models)
# =============================================================================

def generate_table_full_evidence(results: list[RunResult],
                                  output_path: Path) -> None:
    """Generate full evidence table for all models (all relations, reconstructions)."""
    groups = group_results(results)

    relations = ["LT", "YT", "LTYT"]
    recons = RECONSTRUCTIONS
    recon_labels = [RECON_LABELS_SHORT[r] for r in RECONSTRUCTIONS]

    # All model types found
    all_models = set()
    for group in groups.values():
        all_models.update(group.keys())

    # Order models sensibly
    model_order = ["base", "dipVext", "dipH0", "dipA",
                   "quadVext", "quadA", "quadH0",
                   "pixVext", "pixA",
                   "radVext", "radmagVext",
                   "dipA_dipVext", "dipH0_dipVext"]
    models = [m for m in model_order if m in all_models]
    models += sorted(all_models - set(models))  # Add any remaining

    lines = []
    lines.append(r"\begin{tabular}{l|" + "c" * len(recons) + "}")
    lines.append(r"\hline")

    for rel in relations:
        lines.append(f"\\multicolumn{{{len(recons)+1}}}{{c}}{{\\textbf{{{rel}}}}} \\\\")
        lines.append(r"\hline")
        lines.append(r"Model & " + " & ".join(recon_labels) + r" \\")
        lines.append(r"\hline")

        for model in models:
            model_name = MODEL_PRETTY_NAMES.get(model, model.replace("_", r"\_"))

            dlnZ_strs = []
            for recon in recons:
                key = (rel, recon)
                if key not in groups:
                    dlnZ_strs.append(r"\textemdash")
                    continue

                group = groups[key]
                base = group.get("base")
                r = group.get(model)

                if r is None:
                    dlnZ_strs.append(r"\textemdash")
                elif base is None or model == "base":
                    dlnZ_strs.append("$0.00$")
                else:
                    dlnZ, dlnZ_err = compute_delta_lnZ(r, base, is_base=False)
                    dlnZ_strs.append(format_val_err(dlnZ, dlnZ_err))

            lines.append(f"{model_name} & " + " & ".join(dlnZ_strs) + r" \\")

        if rel != relations[-1]:
            lines.append(r"\hline")
            lines.append(r"\addlinespace[5pt]")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")

    write_table("\n".join(lines), output_path, landscape=True)


# =============================================================================
# Appendix: Radial model parameters
# =============================================================================

def generate_appendix_radial(results: list[RunResult],
                              output_path: Path) -> None:
    """Generate appendix table with radial model parameters including uncertainties."""
    groups = group_results(results)

    relations = ["LT", "YT", "LTYT"]
    recons = RECONSTRUCTIONS
    recon_pretty_map = RECON_LABELS_SHORT

    lines = []
    lines.append(r"\begin{tabular}{|ll|ccccc|cc|c|}")
    lines.append(r"\hline\hline")
    lines.append(r"Rel. & Recon & $V_0$ [km/s] & $V_1$ [km/s] & $V_2$ [km/s] & $V_3$ [km/s] & $V_4$ [km/s] & $\ell$ [deg] & $b$ [deg] & $\Delta\ln\mathcal{Z}$ \\")
    lines.append(r"\hline\hline")

    for rel in relations:
        first_rel = True
        for recon in recons:
            key = (rel, recon)
            group = groups.get(key, {})
            base = group.get("base")
            r = group.get("radVext")

            rel_col = rel if first_rel else ""
            first_rel = False

            recon_pretty = recon_pretty_map[recon]

            if r is None:
                # No data - fill with dashes
                mags = [r"\textemdash"] * 5
                ell_str = r"\textemdash"
                b_str = r"\textemdash"
                dlnZ_str = r"\textemdash"
            else:
                # Get magnitude values at knots with uncertainties
                mags = []
                for i in range(5):
                    m, m_err = r.get_param_with_err(f"Vext_rad_mag[{i}]")
                    mags.append(format_val_err(m, m_err, ".0f"))

                # Get direction (average over knots)
                ell_vals, ell_errs = [], []
                b_vals, b_errs = [], []
                for i in range(5):
                    ell, ell_e = r.get_param_with_err(f"Vext_rad_ell[{i}]")
                    b, b_e = r.get_param_with_err(f"Vext_rad_b[{i}]")
                    if ell is not None:
                        ell_vals.append(ell)
                        if ell_e is not None:
                            ell_errs.append(ell_e)
                    if b is not None:
                        b_vals.append(b)
                        if b_e is not None:
                            b_errs.append(b_e)

                if ell_vals:
                    mean_ell = np.mean(ell_vals)
                    mean_b = np.mean(b_vals)
                    # Use mean of errors as representative uncertainty
                    err_ell = np.mean(ell_errs) if ell_errs else None
                    err_b = np.mean(b_errs) if b_errs else None
                    ell_str = format_angle(mean_ell, err_ell)
                    b_str = format_angle(mean_b, err_b)
                else:
                    ell_str = r"\textemdash"
                    b_str = r"\textemdash"

                # Delta lnZ
                if base is not None:
                    dlnZ, dlnZ_err = compute_delta_lnZ(r, base)
                    dlnZ_str = format_val_err(dlnZ, dlnZ_err)
                else:
                    dlnZ_str = r"\textemdash"

            mag_str = " & ".join(mags)
            lines.append(f"{rel_col} & {recon_pretty} & {mag_str} & {ell_str} & {b_str} & {dlnZ_str} \\\\")

        if rel != relations[-1]:
            lines.append(r"\hline")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")

    write_table("\n".join(lines), output_path, landscape=False)


# =============================================================================
# Appendix: Pixel model summary
# =============================================================================

def generate_appendix_pixel(results: list[RunResult],
                             output_path: Path) -> None:
    """Generate appendix table with pixel model parameters (12 HEALPix pixels)."""
    groups = group_results(results)

    relations = ["LT", "YT", "LTYT"]
    recons = RECONSTRUCTIONS
    recon_pretty_map = RECON_LABELS_SHORT

    models = ["pixVext", "pixA", "pixH0"]
    model_names = {
        "pixVext": r"$\mathbf{V}_{\rm ext}$",
        "pixA": r"ZP",
        "pixH0": r"$H_0$",
    }

    # Generate column headers for 12 pixels
    pix_headers = " & ".join([f"$p_{{{i}}}$" for i in range(12)])

    lines = []
    lines.append(r"\begin{tabular}{|c|c|l|" + "c" * 12 + "|c|}")
    lines.append(r"\hline\hline")
    lines.append(f"Rel. & Recon & Model & {pix_headers} & $\\Delta\\ln\\mathcal{{Z}}$ \\\\")
    lines.append(r"\hline\hline")

    for rel in relations:
        # Count all rows (all recons * all models)
        rel_row_count = len(recons) * len(models)
        rel_first_row = True

        for recon in recons:
            key = (rel, recon)
            group = groups.get(key, {})
            base = group.get("base")
            recon_pretty = recon_pretty_map[recon]

            recon_row_count = len(models)
            recon_first_row = True

            for model in models:
                r = group.get(model)
                if r is None and model == "pixVext":
                    r = group.get("nodipA_pixVext")

                if rel_first_row:
                    rel_col = f"\\multirow{{{rel_row_count}}}{{*}}{{{rel}}}"
                    rel_first_row = False
                else:
                    rel_col = ""

                if recon_first_row:
                    recon_col = f"\\multirow{{{recon_row_count}}}{{*}}{{{recon_pretty}}}"
                    recon_first_row = False
                else:
                    recon_col = ""

                model_col = model_names.get(model, model)

                if r is None:
                    # No data - fill with dashes
                    pix_vals = [r"\textemdash"] * 12
                    dlnZ_str = r"\textemdash"
                else:
                    pix_vals = []
                    # For H0 pixel models, try reading entire array first (new format)
                    h0_pix_array = read_samples_from_hdf5(r.fname, "H0_pix")
                    for i in range(12):
                        if model == "pixVext":
                            v, v_err = r.get_param_with_err(f"Vext_pix[{i}]")
                            pix_vals.append(format_val_err(v, v_err, ".0f"))
                        else:
                            # Try new array format first
                            if h0_pix_array is not None and h0_pix_array.ndim == 2:
                                pix_samples = h0_pix_array[:, i]
                                v = float(np.median(pix_samples))
                                v_err = float((np.percentile(pix_samples, 84) - np.percentile(pix_samples, 16)) / 2)
                            else:
                                # Legacy format: individual parameters
                                v, v_err = r.get_param_with_err(f"dH_over_H_pix[{i}]")
                                if v is None:
                                    v, v_err = r.get_param_with_err(f"A_pix[{i}]")
                                    if v is not None:
                                        frac = 10 ** (0.5 * v) - 1.0
                                        if v_err is not None:
                                            frac_err = v_err * 0.5 * np.log(10) * 10 ** (0.5 * v)
                                        else:
                                            frac_err = None
                                        v, v_err = frac, frac_err
                            if v is not None:
                                v = v * 100
                                v_err = v_err * 100 if v_err is not None else None
                                pix_vals.append(format_val_err(v, v_err, ".1f") + r"\%")
                            else:
                                pix_vals.append(r"\textemdash")

                    if base is not None:
                        dlnZ, dlnZ_err = compute_delta_lnZ(r, base)
                        dlnZ_str = format_val_err(dlnZ, dlnZ_err)
                    else:
                        dlnZ_str = r"\textemdash"

                pix_str = " & ".join(pix_vals)
                lines.append(f"{rel_col} & {recon_col} & {model_col} & {pix_str} & {dlnZ_str} \\\\")

            if recon != recons[-1]:
                lines.append(r"\cline{2-16}")

        if rel != relations[-1]:
            lines.append(r"\hline\hline")

    lines.append(r"\hline\hline")
    lines.append(r"\end{tabular}")

    write_table("\n".join(lines), output_path, landscape=False)


# =============================================================================
# Appendix: Quadrupole parameters
# =============================================================================

def generate_appendix_quadrupole(results: list[RunResult],
                                  output_path: Path) -> None:
    """Generate appendix table with quadrupole model parameters.

    Includes quadVext and combination models with full flow parameters.
    Structure similar to appendix_dipoles with all relations.
    """
    groups = group_results(results)

    relations = ["LT", "YT", "LTYT"]
    recons = RECONSTRUCTIONS
    recon_pretty_map = RECON_LABELS_SHORT

    # Models to include: quadVext, quadA, quadH0
    models = ["quadVext", "quadA", "quadH0"]

    model_names = {
        "quadVext": r"$\mathbf{V}_{\rm ext}$",
        "quadA": r"ZP",
        "quadH0": r"$H_0$",
    }

    lines = []
    lines.append(r"\begin{tabular}{|c|c|c|c|cc|c|cccc|c|}")
    lines.append(r"\hline\hline")
    lines.append(r"Rel. & Recon & Model & $A_{\rm dip}$ & $\ell$ [$^\circ$] & $b$ [$^\circ$] & $A_{\rm quad}$ & $\ell_1$ [$^\circ$] & $b_1$ [$^\circ$] & $\ell_2$ [$^\circ$] & $b_2$ [$^\circ$] & $\Delta\ln\mathcal{Z}$ \\")
    lines.append(r"\hline\hline")

    for rel in relations:
        # Count all rows (all recons * all models)
        rel_row_count = len(recons) * len(models)
        rel_first_row = True

        for recon in recons:
            key = (rel, recon)
            group = groups.get(key, {})
            base = group.get("base")
            recon_pretty = recon_pretty_map[recon]

            # Count rows for this reconstruction (all models)
            recon_row_count = len(models)
            recon_first_row = True

            for model in models:
                r = group.get(model)

                # Relation column
                if rel_first_row:
                    rel_col = f"\\multirow{{{rel_row_count}}}{{*}}{{{rel}}}"
                    rel_first_row = False
                else:
                    rel_col = ""

                # Recon column
                if recon_first_row:
                    recon_col = f"\\multirow{{{recon_row_count}}}{{*}}{{{recon_pretty}}}"
                    recon_first_row = False
                else:
                    recon_col = ""

                model_col = model_names.get(model, model)

                # Get parameters depending on model type
                if r is None:
                    # No data - fill with dashes
                    vdip_str = r"\textemdash"
                    vquad_str = r"\textemdash"
                    ell_str = r"\textemdash"
                    b_str = r"\textemdash"
                    l1_str = r"\textemdash"
                    b1_str = r"\textemdash"
                    l2_str = r"\textemdash"
                    b2_str = r"\textemdash"
                    dlnZ_str = r"\textemdash"
                elif model == "quadVext":
                    # Vext quadrupole model has Vext dipole + quad
                    # Use upper limit detection for amplitudes (no units in table)
                    vdip_samples = read_samples_from_hdf5(r.fname, "Vext_mag")
                    _, vdip_prior_max = read_prior_bounds_from_toml(r.fname, "Vext")
                    vdip_str = format_amplitude_constraint(
                        vdip_samples, 0.0, vdip_prior_max,
                        fmt=".0f", unit="", scale=1.0
                    )

                    vquad_samples = read_samples_from_hdf5(r.fname, "Vext_quad_mag")
                    _, vquad_prior_max = read_prior_bounds_from_toml(r.fname, "Vext_quad")
                    vquad_str = format_amplitude_constraint(
                        vquad_samples, 0.0, vquad_prior_max,
                        fmt=".0f", unit="", scale=1.0
                    )

                    ell, ell_err = r.get_param_with_err("Vext_ell")
                    b, b_err = r.get_param_with_err("Vext_b")
                    # Get quadrupole directions in Galactic coordinates
                    l1, l1_err, b1, b1_err = get_quadrupole_galactic_direction(r.fname, "Vext_quad", 1)
                    l2, l2_err, b2, b2_err = get_quadrupole_galactic_direction(r.fname, "Vext_quad", 2)

                    ell_str = format_val_err(ell, ell_err, ".0f") if ell is not None else r"\textemdash"
                    b_str = format_val_err(b, b_err, ".0f") if b is not None else r"\textemdash"
                    l1_str = format_val_err(l1, l1_err, ".0f") if l1 is not None else r"\textemdash"
                    b1_str = format_val_err(b1, b1_err, ".0f") if b1 is not None else r"\textemdash"
                    l2_str = format_val_err(l2, l2_err, ".0f") if l2 is not None else r"\textemdash"
                    b2_str = format_val_err(b2, b2_err, ".0f") if b2 is not None else r"\textemdash"

                    # Delta lnZ
                    if base is not None:
                        dlnZ, dlnZ_err = compute_delta_lnZ(r, base)
                        dlnZ_str = format_val_err(dlnZ, dlnZ_err)
                    else:
                        dlnZ_str = r"\textemdash"
                else:
                    # Zeropoint/H0 models have different parameter names
                    # Use upper limit detection for amplitudes (no units in table)
                    # Try new naming first (H0_dipole_mag), then legacy (dH_over_H_dipole)
                    vdip_samples = read_samples_from_hdf5(r.fname, "H0_dipole_mag")
                    if vdip_samples is None:
                        vdip_samples = read_samples_from_hdf5(r.fname, "dH_over_H_dipole")
                    if vdip_samples is None:
                        vdip_samples = read_samples_from_hdf5(r.fname, "zeropoint_dipole_mag")
                    _, vdip_prior_max = read_prior_bounds_from_toml(r.fname, "H0_dipole")
                    if vdip_prior_max is None:
                        _, vdip_prior_max = read_prior_bounds_from_toml(r.fname, "zeropoint_dipole")
                    vdip_str = format_amplitude_constraint(
                        vdip_samples, 0.0, vdip_prior_max,
                        fmt=".1f", unit="", scale=100.0
                    )

                    vquad_samples = read_samples_from_hdf5(r.fname, "dH_over_H_quad")
                    if vquad_samples is None:
                        vquad_samples = read_samples_from_hdf5(r.fname, "zeropoint_quad_mag")
                    _, vquad_prior_max = read_prior_bounds_from_toml(r.fname, "zeropoint_quad")
                    vquad_str = format_amplitude_constraint(
                        vquad_samples, 0.0, vquad_prior_max,
                        fmt=".1f", unit="", scale=100.0
                    )

                    # Try new naming first, then legacy
                    ell, ell_err = r.get_param_with_err("H0_dipole_ell")
                    if ell is None:
                        ell, ell_err = r.get_param_with_err("zeropoint_dipole_ell")
                    b, b_err = r.get_param_with_err("H0_dipole_b")
                    if b is None:
                        b, b_err = r.get_param_with_err("zeropoint_dipole_b")
                    # Get quadrupole directions in Galactic coordinates
                    l1, l1_err, b1, b1_err = get_quadrupole_galactic_direction(r.fname, "zeropoint_quad", 1)
                    l2, l2_err, b2, b2_err = get_quadrupole_galactic_direction(r.fname, "zeropoint_quad", 2)

                    ell_str = format_val_err(ell, ell_err, ".0f") if ell is not None else r"\textemdash"
                    b_str = format_val_err(b, b_err, ".0f") if b is not None else r"\textemdash"
                    l1_str = format_val_err(l1, l1_err, ".0f") if l1 is not None else r"\textemdash"
                    b1_str = format_val_err(b1, b1_err, ".0f") if b1 is not None else r"\textemdash"
                    l2_str = format_val_err(l2, l2_err, ".0f") if l2 is not None else r"\textemdash"
                    b2_str = format_val_err(b2, b2_err, ".0f") if b2 is not None else r"\textemdash"

                    # Delta lnZ
                    if base is not None:
                        dlnZ, dlnZ_err = compute_delta_lnZ(r, base)
                        dlnZ_str = format_val_err(dlnZ, dlnZ_err)
                    else:
                        dlnZ_str = r"\textemdash"

                lines.append(f"{rel_col} & {recon_col} & {model_col} & {vdip_str} & {ell_str} & {b_str} & {vquad_str} & {l1_str} & {b1_str} & {l2_str} & {b2_str} & {dlnZ_str} \\\\")

            # Line between reconstructions
            if recon != recons[-1]:
                lines.append(r"\cline{2-12}")

        # Bold line between relations
        if rel != relations[-1]:
            lines.append(r"\hline\hline")

    lines.append(r"\hline\hline")
    lines.append(r"\end{tabular}")

    write_table("\n".join(lines), output_path, landscape=False)


# =============================================================================
# Appendix: Mixed dipoles (Vext + A/H0)
# =============================================================================

def generate_appendix_mixed_dipoles(results: list[RunResult],
                                    output_path: Path) -> None:
    """Generate appendix table with mixed dipole model parameters."""
    groups = group_results(results)

    relations = ["LT", "YT", "LTYT"]
    recons = RECONSTRUCTIONS
    recon_pretty_map = RECON_LABELS_SHORT

    models = ["dipA_dipVext", "dipH0_dipVext"]
    model_names = {
        "dipA_dipVext": r"ZP + $\mathbf{V}_{\rm ext}$",
        "dipH0_dipVext": r"$H_0$ + $\mathbf{V}_{\rm ext}$",
    }

    lines = []
    lines.append(r"\begin{tabular}{|c|c|l|c|cc|c|cc|c|}")
    lines.append(r"\hline\hline")
    lines.append(r"Rel. & Recon & Model & $A_{V}$ & $\ell$ [$^\circ$] & $b$ [$^\circ$] & $A_{\rm ZP}$ & $\ell$ [$^\circ$] & $b$ [$^\circ$] & $\Delta\ln\mathcal{Z}$ \\")
    lines.append(r"\hline\hline")

    for rel in relations:
        # Count all rows (all recons * all models)
        rel_row_count = len(recons) * len(models)
        rel_first_row = True

        for recon in recons:
            key = (rel, recon)
            group = groups.get(key, {})
            base = group.get("base")
            recon_pretty = recon_pretty_map[recon]

            recon_row_count = len(models)
            recon_first_row = True

            for model in models:
                r = group.get(model)

                if rel_first_row:
                    rel_col = f"\\multirow{{{rel_row_count}}}{{*}}{{{rel}}}"
                    rel_first_row = False
                else:
                    rel_col = ""

                if recon_first_row:
                    recon_col = f"\\multirow{{{recon_row_count}}}{{*}}{{{recon_pretty}}}"
                    recon_first_row = False
                else:
                    recon_col = ""

                model_col = model_names.get(model, model)

                if r is None:
                    # No data - fill with dashes
                    vdip_str = r"\textemdash"
                    ell_str = r"\textemdash"
                    b_str = r"\textemdash"
                    ah0_str = r"\textemdash"
                    ah0_ell_str = r"\textemdash"
                    ah0_b_str = r"\textemdash"
                    dlnZ_str = r"\textemdash"
                else:
                    # Vext dipole amplitude with upper limit detection (no units)
                    vdip_samples = read_samples_from_hdf5(r.fname, "Vext_mag")
                    _, vdip_prior_max = read_prior_bounds_from_toml(r.fname, "Vext")
                    vdip_str = format_amplitude_constraint(
                        vdip_samples, 0.0, vdip_prior_max,
                        fmt=".0f", unit="", scale=1.0
                    )

                    ell, ell_err = r.get_param_with_err("Vext_ell")
                    b, b_err = r.get_param_with_err("Vext_b")

                    # Zeropoint/H0 dipole amplitude with upper limit detection (no units)
                    # Try new naming first (H0_dipole_mag), then legacy
                    ah0_samples = read_samples_from_hdf5(r.fname, "H0_dipole_mag")
                    if ah0_samples is None:
                        ah0_samples = read_samples_from_hdf5(r.fname, "dH_over_H_dipole")
                    if ah0_samples is None:
                        ah0_samples = read_samples_from_hdf5(r.fname, "zeropoint_dipole_mag")
                    _, ah0_prior_max = read_prior_bounds_from_toml(r.fname, "H0_dipole")
                    if ah0_prior_max is None:
                        _, ah0_prior_max = read_prior_bounds_from_toml(r.fname, "zeropoint_dipole")
                    ah0_str = format_amplitude_constraint(
                        ah0_samples, 0.0, ah0_prior_max,
                        fmt=".1f", unit="", scale=100.0
                    )

                    # Try new naming first, then legacy
                    ah0_ell, ah0_ell_err = r.get_param_with_err("H0_dipole_ell")
                    if ah0_ell is None:
                        ah0_ell, ah0_ell_err = r.get_param_with_err("zeropoint_dipole_ell")
                    ah0_b, ah0_b_err = r.get_param_with_err("H0_dipole_b")
                    if ah0_b is None:
                        ah0_b, ah0_b_err = r.get_param_with_err("zeropoint_dipole_b")

                    ell_str = format_val_err(ell, ell_err, ".0f") if ell is not None else r"\textemdash"
                    b_str = format_val_err(b, b_err, ".0f") if b is not None else r"\textemdash"

                    ah0_ell_str = format_val_err(ah0_ell, ah0_ell_err, ".0f") if ah0_ell is not None else r"\textemdash"
                    ah0_b_str = format_val_err(ah0_b, ah0_b_err, ".0f") if ah0_b is not None else r"\textemdash"

                    if base is not None:
                        dlnZ, dlnZ_err = compute_delta_lnZ(r, base)
                        dlnZ_str = format_val_err(dlnZ, dlnZ_err)
                    else:
                        dlnZ_str = r"\textemdash"

                lines.append(
                    f"{rel_col} & {recon_col} & {model_col} & {vdip_str} & "
                    f"{ell_str} & {b_str} & {ah0_str} & {ah0_ell_str} & {ah0_b_str} & {dlnZ_str} \\\\"
                )

            if recon != recons[-1]:
                lines.append(r"\cline{2-10}")

        if rel != relations[-1]:
            lines.append(r"\hline\hline")

    lines.append(r"\hline\hline")
    lines.append(r"\end{tabular}")

    write_table("\n".join(lines), output_path, landscape=False)


# =============================================================================
# Main entry point
# =============================================================================

def main(results_folder: Optional[str] = None):
    """Generate all tables from results folder."""
    if results_folder is None:
        results_folder = RESULTS_FOLDER

    results_dir = RESULTS_ROOT / results_folder

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    print(f"Loading results from {results_dir}")
    results = load_all_results(results_dir)
    print(f"Loaded {len(results)} result files")

    # Generate main tables
    # Table 1: LTYT only (main paper)
    generate_table1_dipoles(results, TABLES_DIR / "table1_dipoles.tex",
                            relations=["LTYT"])
    # Appendix: All relations (LT, YT, LTYT)
    generate_table1_dipoles(results, TABLES_DIR / "appendix_dipoles.tex",
                            relations=["LT", "YT", "LTYT"])
    generate_table2_beyond_dipoles(results, TABLES_DIR / "table2_beyond_dipoles.tex")

    # Generate appendix tables
    generate_appendix_radial(results, TABLES_DIR / "appendix_radial.tex")
    generate_appendix_pixel(results, TABLES_DIR / "appendix_pixel.tex")
    generate_appendix_quadrupole(results, TABLES_DIR / "appendix_quadrupole.tex")
    generate_appendix_mixed_dipoles(results, TABLES_DIR / "appendix_mixed_dipoles.tex")

    # Skipping for now:
    # generate_table_full_evidence(results, TABLES_DIR / "table_full_evidence.tex")

    print(f"\nAll tables written to {TABLES_DIR}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from MCMC results")
    parser.add_argument("--folder", default=RESULTS_FOLDER,
                        help="Results folder name (default: config RESULTS_FOLDER)")
    args = parser.parse_args()

    main(args.folder)
