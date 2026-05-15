# Notes on radially dependent Vext

This note summarises the current CANDEL support for radially dependent external
velocity models and the main interpolation tradeoffs.

## Current implementation

The PV model stack supports two radial external-velocity modes through
`pv_model.which_Vext`:

- `radial`: samples a full 3D external velocity vector at each radial knot.
- `radial_magnitude`: samples one sky direction and a radius-dependent
  magnitude.

The knots are not chosen automatically.
They are read from the relevant prior block:

```toml
[model.priors.Vext_radial]
dist = "vector_radial_uniform"
rknot = [0, 250, 600]
method = "cubic"

[model.priors.Vext_radial_magnitude]
dist = "vector_radialmag_uniform"
rknot = [0, 50, 150]
method = "cubic"
```

For `radial`, the sampler draws one magnitude and one isotropic direction at
each knot.
The implementation then interpolates the magnitude with `interpax.interp1d`
and interpolates the direction with SLERP.
For `radial_magnitude`, the sampler draws one isotropic direction and one
magnitude per knot, then interpolates only the scalar magnitude.

The interpolated external velocity is projected onto each object's line of
sight at every radial integration point and added to the reconstructed velocity
field in the redshift likelihood.

## Issues noticed in review

The `radial` interpolation is not the same as Cartesian vector interpolation.
It treats the amplitude and direction as separate fields.
This creates several edge cases:

- Cubic interpolation of positive knot magnitudes can undershoot below zero,
  which effectively flips the interpolated direction locally.
- Direction interpolation is fragile near zero-magnitude knots because the
  direction is not well-defined.
- The diagnostic profile plot currently interpolates magnitude, Galactic
  longitude, and Galactic latitude separately, so it need not match the vector
  trajectory used in the likelihood.
- There is little validation that `rknot` is sorted, has enough points, or is
  compatible with the interpolation method.

The `radial_magnitude` mode is conceptually cleaner if the intended model is a
fixed dipole direction with varying amplitude.
It still inherits the usual risks of sparse cubic interpolation, such as ringing
between knots.

## Cartesian interpolation option

A cleaner default for `radial` would be to interpolate the Cartesian vector
components directly:

```text
sample Vext_x(r_k), Vext_y(r_k), Vext_z(r_k)
interpolate x(r), y(r), z(r) independently
project Vext(r) onto each line of sight
```

The least invasive version would keep the current isotropic knot prior:

```text
mag_k ~ Uniform(low, high)
direction_k ~ isotropic
Vext_k = mag_k * direction_k
```

but then interpolate `Vext_k[:, 0]`, `Vext_k[:, 1]`, and `Vext_k[:, 2]`
directly.
Magnitudes and directions would be derived only for summaries and plots.

This avoids undefined directions at zero magnitude and removes the possibility
of a negative interpolated magnitude.
It also makes the plotted radial profile easier to make consistent with the
likelihood.

## Tradeoffs of Cartesian interpolation

Cartesian interpolation is not free of modelling assumptions.
If neighbouring knots have large, oppositely directed velocities, the
interpolated vector can pass through small magnitude by cancellation.
Magnitude-plus-direction interpolation instead tends to keep the amplitude large
while rotating the direction.
Which behaviour is preferred depends on the intended physical interpretation of
`Vext(r)`.

The prior over the continuous function between knots also changes.
Even if the knot prior remains isotropic, Cartesian interpolation induces a
different prior between knots than separate magnitude and direction
interpolation.

Cubic Cartesian interpolation can still overshoot in individual components.
For sparse knots, linear interpolation is the safer default unless smoothness is
scientifically important.

## Recommendation

For a free 3D external velocity vector as a function of radius, Cartesian
component interpolation is the cleaner default.
Keep the isotropic magnitude-direction prior at the knots if preserving the
existing knot-level prior is important, but interpolate the resulting Cartesian
vectors.

If the scientific model is instead explicitly "a dipole with smoothly rotating
direction and separately smooth amplitude", then the current structure is closer
in spirit, but it should be hardened around zero magnitudes and the diagnostics
should use the exact same interpolation as the likelihood.
