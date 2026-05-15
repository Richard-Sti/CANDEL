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
"""
Data loading and preprocessing utilities for peculiar-velocity catalogues.

Provides dataframe-like containers, LOS interpolation helpers, covariance
assembly, and catalogue I/O wired to the project config files.
"""
from ..util import SPEED_OF_LIGHT
from .field_cache import (  # noqa: F401
    _FIELD_CACHE_VERSION,
    _ArrayShapeOnly,
    _field_cache_dir_from_config,
    _field_cache_enabled_from_config,
    _field_cache_float_tag,
    _field_cache_indices_tag,
    _field_cache_mpi_comm,
    _field_cache_path,
    _field_cache_payload_digest,
    _field_cache_slug,
    _field_source_metadata,
    _h0_volume_cache_filename,
    _jsonable,
    _parse_field_cache_indices_tag,
    _pv_volume_density_cache_filename,
    _read_field_cache,
    _read_field_cache_mpi_part,
    _read_h0_volume_cache_superset,
    _slice_h0_volume_cache_fields,
    _write_field_cache,
    _write_field_cache_mpi_part,
)
from .volume_density import (  # noqa: F401
    _SPHERE_RADIUS_DX_WARN_MIN,
    _cached_h0_volume_result,
    _cached_pv_volume_density_result,
    _choose_voxel_subsample_indices,
    _density_unit_normalization,
    _extract_subcube,
    _h0_density_fields_from_rho,
    _h0_log_radius_from_r,
    _h0_volume_cache_sampling_payload,
    _h0_volume_runtime_result,
    _load_h0_volume_data_from_config,
    _load_one_pv_volume_density_field,
    _load_volume_data_for_H0,
    _load_volume_data_for_H0_mpi,
    _load_volume_density_3d,
    _load_volume_density_3d_fields,
    _load_volume_density_3d_fields_mpi,
    _precompute_cosmo_3d,
    _prepare_pv_volume_density_arrays,
    _pv_mpi_placeholder,
    _reconstruction_omega_m,
    _sphere_voxel_weights,
    _subsample_h0_volume_arrays,
    _validate_voxel_subsample_fraction,
    _validate_voxel_subsample_seed,
    _volume_density_geometry,
    _volume_density_mode,
    _warn_coarse_sphere_radius,
)
from .los import (  # noqa: F401
    _compute_r_grid,
    _filter_data,
    _zcmb_blat_mask,
    effective_rank_entropy,
    load_los,
    precompute_pixel_projection,
)
from .catalogues import (  # noqa: F401
    _CATALOGUE_LOADERS,
    _edd_col_float,
    _edd_col_str,
    _load_EDD_TRGB_from_config_common,
    _load_LOSS_Foundation,
    _load_edd_trgb_core,
    _parse_edd_trgb_txt,
    arcsec_to_radian,
    load_2MTF,
    load_6dF_FP,
    load_CCHP_from_config,
    load_CF4_data,
    load_CF4_mock,
    load_CSP,
    load_CSP_from_config,
    load_EDD_2MTF,
    load_EDD_2MTF_from_config,
    load_EDD_TRGB,
    load_EDD_TRGB_from_config,
    load_EDD_TRGB_grouped,
    load_EDD_TRGB_grouped_from_config,
    load_Foundation,
    load_LOSS,
    load_PantheonPlus,
    load_PantheonPlus_Lane,
    load_SDSS_FP,
    load_SFI,
    load_SH0ES,
    load_SH0ES_calibration,
    load_SH0ES_from_config,
    load_SH0ES_separated,
    load_generic,
    match_cchp_to_csp,
)
from .frame import PVDataFrame, load_PV_dataframes  # noqa: F401
