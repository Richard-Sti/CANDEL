"""
Convergence test of the 3D selection integral for CH0 / TRGB H0 models.

Computes
    log S(H0; R) = logsumexp_j[ log P_sel,j ] + log dV_grid - 3 log h
on a uniform-density (n=1) spherical grid of half-side R Mpc/h, with no
reconstruction and no peculiar velocity.  The selection kernels are identical
to the production code (log_prob_integrand_sel).

Selection types
---------------
  mag   P_sel = int N(m_obs | mu(r)+M_abs, e_mag) Phi((m_lim-m_obs)/w) dm_obs
  cz    P_sel = int N(cz_obs | cz_cosmo(r), sv)   Phi((cz_lim-cz_obs)/w) dcz_obs

Sample presets (--sample)
--------------------------
  SH0ES     SN B-band magnitude selection.
  EDD_TRGB  TRGB F814W magnitude selection.
Any individual argument overrides the preset.

Reported quantities
-------------------
log S(R)              Integral value at each truncation radius.
delta_logS(R)         Truncation residual vs R_max  (negative = missing tail).
bias_proxy(R)         N_hosts * H0-spread(delta_logS) / dH0  [1/(km/s/Mpc)].
                      Multiply by sigma_H0 to get approx H0 bias in sigma units.

All radii are in Mpc/h (matching the production config key
`model.selection_integral_grid_radius`).
"""
import argparse

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from candel.cosmo.cosmography import Distance2Distmod, Distance2Redshift
from candel.model.utils import log_prob_integrand_sel
from candel.util import SPEED_OF_LIGHT

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS = {
    "SH0ES": dict(
        selection="mag",
        M_abs=[-19.3],
        mag_lim=14.5,
        mag_width=0.15,
        e_mag=0.15,
        cz_lim=3330.0,
        cz_width=300.0,
        sigma_v=200.0,
        N_hosts=37,
        H0=[60, 70, 80, 90, 100],
        radii=[25, 50, 75, 100, 125, 150, 200],
    ),
    "EDD_TRGB": dict(
        selection="mag",
        M_abs=[-4.05],
        mag_lim=27.0,
        mag_width=0.5,
        e_mag=0.07,
        cz_lim=3330.0,
        cz_width=300.0,
        sigma_v=250.0,
        N_hosts=480,
        H0=[60, 70, 80, 90, 100],
        radii=[10, 25, 50, 75, 100],
    ),
}


# ---------------------------------------------------------------------------
# Grid and cosmography
# ---------------------------------------------------------------------------

def build_voxel_radii(R_max, dx):
    """Voxel-centre radii (Mpc/h) inside a sphere of half-side R_max.

    Central voxel floored at 0.25*dx, matching _volume_density_geometry.
    """
    n = int(np.ceil(2 * R_max / dx))
    if n % 2 == 0:
        n += 1
    c = (np.arange(n) - (n - 1) / 2) * dx
    x, y, z = np.meshgrid(c, c, c, indexing="ij")
    r = np.maximum(np.sqrt(x*x + y*y + z*z), 0.25 * dx)
    return r[r <= R_max].astype(np.float64)


def precompute_cosmo(r_mpch, Om0):
    r = jnp.asarray(r_mpch)
    return (np.asarray(Distance2Distmod(Om0=Om0)(r, h=1.0)),
            np.asarray(Distance2Redshift(Om0=Om0)(r, h=1.0)))


# ---------------------------------------------------------------------------
# Integral kernels — match H0ModelBase._compute_volume_log_S_{mag,cz}
# ---------------------------------------------------------------------------

def _log_S_from_log_P(log_P, log_dV, H0):
    h = H0 / 100.0
    return float(logsumexp(jnp.asarray(log_P)) + log_dV - 3.0 * jnp.log(h))


def eval_log_S_mag(mu_h1, log_dV, H0, M_abs, e_mag, mag_lim, mag_width):
    h = H0 / 100.0
    mu = jnp.asarray(mu_h1) - 5.0 * jnp.log10(h)
    return _log_S_from_log_P(
        log_prob_integrand_sel(mu + M_abs, e_mag, mag_lim, mag_width),
        log_dV, H0)


def eval_log_S_cz(z_cosmo, log_dV, H0, sigma_v, cz_lim, cz_width):
    cz = SPEED_OF_LIGHT * jnp.asarray(z_cosmo)
    return _log_S_from_log_P(
        log_prob_integrand_sel(cz, sigma_v, cz_lim, cz_width),
        log_dV, H0)


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def compute_table(combos, selection, radii, r_full, mu_h1_full, z_cosmo_full,
                  log_dV, **sel_kwargs):
    """Return log S array of shape (n_combos, n_radii)."""
    masks = {R: r_full <= R for R in radii}
    table = np.empty((len(combos), len(radii)))
    for i, combo in enumerate(combos):
        H0 = combo[0]
        for j, R in enumerate(radii):
            m = masks[R]
            if selection == "mag":
                table[i, j] = eval_log_S_mag(
                    mu_h1_full[m], log_dV, H0, combo[1], **sel_kwargs)
            else:
                table[i, j] = eval_log_S_cz(
                    z_cosmo_full[m], log_dV, H0, **sel_kwargs)
    return table


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def combo_label(combo, selection):
    if selection == "mag":
        return f"{combo[0]:6.0f}  {combo[1]:+6.2f}"
    return f"{combo[0]:6.0f}"


def header_prefix(selection):
    return "    H0   M_abs" if selection == "mag" else "    H0"


def print_table(title, combos, radii, data, selection, fmt="{:10.4f}"):
    print(f"\n{title}")
    R_hdr = "\t".join(f"R={R:g}" for R in radii)
    print(f"{header_prefix(selection)}  {R_hdr}")
    for combo, row in zip(combos, data):
        vals = "\t".join(fmt.format(v) for v in row)
        print(f"{combo_label(combo, selection)}  {vals}")


def print_bias_proxy(M_abs_vals, combos, radii, delta, N_hosts, dH0, selection):
    print(f"\nbias_proxy(R) = N_hosts * H0_spread(delta_logS) / dH0")
    print(f"  [1/(km/s/Mpc);  bias_in_sigma ~= bias_proxy * sigma_H0]")
    print(f"  N_hosts={N_hosts}, dH0={dH0:.0f} km/s/Mpc")
    R_hdr = "\t".join(f"R={R:g}" for R in radii)
    if selection == "mag":
        print(f"{'M_abs':>6}  {R_hdr}")
        for M in M_abs_vals:
            idx = [i for i, c in enumerate(combos) if c[1] == M]
            sub = delta[np.ix_(idx, range(len(radii)))]
            proxy = N_hosts * (sub.max(axis=0) - sub.min(axis=0)) / dH0
            print(f"{M:+6.2f}  " + "\t".join(f"{v:10.3e}" for v in proxy))
    else:
        proxy = N_hosts * (delta.max(axis=0) - delta.min(axis=0)) / dH0
        print("        " + "\t".join(f"{v:10.3e}" for v in proxy))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_floats(s):
    return [float(x) for x in s.split(",")]


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--sample", choices=list(PRESETS),
                   help="Load defaults for a known sample.")
    p.add_argument("--selection", choices=["mag", "cz"])
    p.add_argument("--radii", type=parse_floats, metavar="R1,R2,...")
    p.add_argument("--dx", type=float, default=1.0,
                   help="Voxel size (Mpc/h). Manticore 1024^3 ~0.665.")
    p.add_argument("--H0", type=parse_floats, metavar="H0_1,...")
    p.add_argument("--N_hosts", type=int)
    p.add_argument("--M_abs", type=parse_floats, metavar="M1,...",
                   help="Absolute magnitude(s). SH0ES: ~-19.3. TRGB: ~-4.05.")
    p.add_argument("--mag_lim", type=float)
    p.add_argument("--mag_width", type=float)
    p.add_argument("--e_mag", type=float)
    p.add_argument("--cz_lim", type=float)
    p.add_argument("--cz_width", type=float)
    p.add_argument("--sigma_v", type=float)
    p.add_argument("--Om0", type=float, default=0.3)
    p.add_argument("--output", default=None, help="Save results to .npz file.")
    args = p.parse_args()

    preset = PRESETS.get(args.sample, {})
    def get(attr, fallback=None):
        v = getattr(args, attr, None)
        return v if v is not None else preset.get(attr, fallback)

    selection = get("selection")
    if selection is None:
        p.error("--selection is required when --sample is not given.")

    radii    = get("radii",    [25, 50, 75, 100, 125, 150, 200])
    H0       = get("H0",       [60, 70, 80, 90, 100])
    N_hosts  = get("N_hosts",  1)
    M_abs    = get("M_abs",    [-19.3])
    mag_lim  = get("mag_lim",  14.5)
    mag_width= get("mag_width",0.15)
    e_mag    = get("e_mag",    0.15)
    cz_lim   = get("cz_lim",  3330.0)
    cz_width = get("cz_width", 300.0)
    sigma_v  = get("sigma_v",  200.0)

    R_max = max(radii)
    print(f"sample    : {args.sample or '(custom)'}")
    print(f"selection : {selection}")
    print(f"radii     : {radii} Mpc/h  (units: Mpc/h)")
    print(f"dx        : {args.dx} Mpc/h,  R_max = {R_max} Mpc/h")
    print(f"grid      : {int(np.ceil(2*R_max/args.dx))}^3 cube at R_max")

    r_full = build_voxel_radii(R_max, args.dx)
    print(f"voxels    : {len(r_full):,} inside sphere of R_max")
    if selection == "mag":
        print(f"params    : M_abs={M_abs}, mag_lim={mag_lim}, "
              f"e_mag={e_mag}, mag_width={mag_width}")
    else:
        print(f"params    : cz_lim={cz_lim}, cz_width={cz_width}, "
              f"sigma_v={sigma_v}")
    print(f"N_hosts   : {N_hosts}")

    log_dV = float(3.0 * np.log(args.dx))
    mu_h1_full, z_cosmo_full = precompute_cosmo(r_full, args.Om0)

    combos = ([(h, m) for h in H0 for m in M_abs] if selection == "mag"
              else [(h,) for h in H0])
    sel_kwargs = (dict(M_abs=None, e_mag=e_mag, mag_lim=mag_lim,
                       mag_width=mag_width)
                  if selection == "mag"
                  else dict(sigma_v=sigma_v, cz_lim=cz_lim,
                             cz_width=cz_width))
    # M_abs is passed per-combo for mag; remove the placeholder key.
    if selection == "mag":
        del sel_kwargs["M_abs"]

    log_S  = compute_table(combos, selection, radii, r_full,
                           mu_h1_full, z_cosmo_full, log_dV, **sel_kwargs)
    delta  = log_S - log_S[:, -1:]
    dH0    = H0[-1] - H0[0] if len(H0) > 1 else 1.0

    print_table("log S(R):", combos, radii, log_S, selection)
    print_table("delta_logS(R) = logS(R) - logS(R_max):", combos, radii,
                delta, selection, fmt="{:+10.4f}")
    print_bias_proxy(M_abs, combos, radii, delta, N_hosts, dH0, selection)

    if args.output is not None:
        np.savez(args.output,
                 selection=selection, radii=np.asarray(radii),
                 dx=args.dx, Om0=args.Om0,
                 H0=np.asarray(H0), M_abs=np.asarray(M_abs),
                 N_hosts=N_hosts,
                 mag_lim=mag_lim, mag_width=mag_width, e_mag=e_mag,
                 cz_lim=cz_lim, cz_width=cz_width, sigma_v=sigma_v,
                 log_S=log_S, delta_logS=delta)
        print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()
