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
"""Shared helpers for reconstruction-derived preprocessing products."""
import math
from os.path import splitext

from ..util import get_nested


def validate_field_smoothing_scale(
        value, label="model.density_3d_smoothing_scale"):
    """Return a non-zero Gaussian field smoothing scale, or ``None``."""
    if value is None:
        return None
    scale = float(value)
    if not math.isfinite(scale) or scale < 0:
        raise ValueError(
            f"`{label}` must be a finite non-negative number in Mpc/h, "
            f"got {value!r}.")
    if scale == 0:
        return None
    return scale


def field_smoothing_scale_from_config(config):
    """Read the optional 3D field smoothing scale from a config."""
    return validate_field_smoothing_scale(
        get_nested(config, "model/density_3d_smoothing_scale", None))


def field_smoothing_cache_payload(field_smoothing_scale):
    """Return the cache-key payload for optional field smoothing."""
    scale = validate_field_smoothing_scale(field_smoothing_scale)
    if scale is None:
        return {}
    return {"field_smoothing_scale": scale}


def field_smoothing_tag(field_smoothing_scale):
    """Return a stable filename tag for a non-zero field smoothing scale."""
    scale = validate_field_smoothing_scale(field_smoothing_scale)
    if scale is None:
        return None
    return f"field_smooth_R{scale:g}"


def field_smoothed_los_path(path, field_smoothing_scale):
    """Append the field-smoothing suffix to a LOS file path if needed."""
    tag = field_smoothing_tag(field_smoothing_scale)
    if path is None or tag is None:
        return path
    root, ext = splitext(path)
    return f"{root}_{tag}{ext}"


def resolve_los_data_path(path, which_los=None, field_smoothing_scale=None):
    """Resolve ``<X>`` and optional field-smoothing LOS filename suffix."""
    if path is None:
        return None
    if which_los is not None:
        path = path.replace("<X>", which_los)
    return field_smoothed_los_path(path, field_smoothing_scale)


# Backwards-compatible names. The config key is intentionally unchanged, but
# the product now smooths every field carried by the cache/LOS product.
validate_density_smoothing_scale = validate_field_smoothing_scale
density_smoothing_scale_from_config = field_smoothing_scale_from_config
density_smoothing_cache_payload = field_smoothing_cache_payload
density_smoothing_tag = field_smoothing_tag
density_smoothed_los_path = field_smoothed_los_path
