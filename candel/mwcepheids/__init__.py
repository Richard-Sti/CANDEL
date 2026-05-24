# Copyright (C) 2026 Richard Stiskalek
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

from ._logging import setup_logging  # noqa
from .config import load_config, load_local_config  # noqa
from .data import (FEH_TO_OH, AnchorCepheidData, CepheidData,  # noqa
                   load_anchor_data, load_data)
from .distributions import (DiskPrior, DistanceModulusPrior,  # noqa
                            sample_disk_sightlines)
from .dust import (AKS_TO_AH, R_H_BAYESTAR,  # noqa
                   postprocess_extinction_profiles, query_AH, query_AH_grid,
                   query_reddening)
from .inference import (build_output_dir, drop_fixed_samples,  # noqa
                        get_log_density, run_inference, save_corner,
                        save_samples, save_summary)
from .distance_marg import log_likelihood_marg_distance  # noqa
from .evidence import (BIC_AIC, dict_samples_to_array,  # noqa
                       harmonic_evidence, laplace_evidence)
from .model import MWCepheidModel  # noqa
from .ppc import generate_ppc, plot_ppc  # noqa
from .prior import load_priors  # noqa
from .selection import (C22SelectionConfig, C27SelectionConfig,  # noqa
                        SelectionMCData, log_AH_selection,
                        log_probit_selection, sample_selection_params,
                        selection_correction, spiral_log_factor)
from .simpson import ln_simpson  # noqa
from .spiral import compute_dist_sq_per_arm, get_drimmel_arm_traces  # noqa
from .utils import (plot_corner, plot_corner_from_hdf5,  # noqa
                    plot_corner_getdist, plot_distance_comparison, plot_trace,
                    print_summary)
