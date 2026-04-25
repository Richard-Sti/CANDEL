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

from .utils import (                                                            # noqa
    load_priors,                                                                # noqa
    log_prior_r_empirical,                                                      # noqa
    smoothclip_nr,                                                              # noqa
    )
from .pv_utils import (                                                         # noqa
    interp_cartesian_vector,                                                    # noqa
    lp_galaxy_bias,                                                             # noqa
    )
from .base_pv import BasePVModel, JointPVModel                                  # noqa
from .model_PV_TFR import TFRModel                                              # noqa
from .model_PV_SN import SNModel                                                # noqa
from .model_PV_PantheonPlus import PantheonPlusModel                            # noqa
from .model_PV_FP import FPModel                                                # noqa
from .model_H0_CH0 import CH0Model                                              # noqa
from .model_H0_TRGB import TRGBModel                                            # noqa
from .dev.model_H0_2MTF import EDD2MTFModel                                     # noqa
from .model_H0_maser import MaserDiskModel, JointMaserModel                     # noqa
from .interp import LOSInterpolator                                             # noqa
from .integration import (ln_simpson, ln_trapz_precomputed,                     # noqa
                          simpson_log_weights, trapz_log_weights)               # noqa

from ..util import fprint, fsection, get_nested, load_config


def _model_section_title(name, cat):
    return f"Model: {name} ({cat})" if cat else f"Model: {name}"


def name2model(name, shared_param=None, config=None):
    mapping = {
        "TFRModel": TFRModel,
        "SNModel": SNModel,
        "PantheonPlusModel": PantheonPlusModel,
        "FPModel": FPModel,
        }

    cats = None
    if config is not None:
        cfg = load_config(config, replace_none=False, replace_los_prior=False,
                          fill_paths=False)
        cats = get_nested(cfg, "io/catalogue_name", None)

    if isinstance(name, str):
        if name not in mapping:
            raise ValueError(f"Model name `{name}` not recognized.\n"
                             f"Available models: {list(mapping.keys())}")
        cat = cats if isinstance(cats, str) else None
        fsection(_model_section_title(name, cat))
        return mapping[name](config)

    if isinstance(name, list):
        unknown = [n for n in name if n not in mapping]
        if unknown:
            raise ValueError(f"Unknown model names: {unknown}\n"
                             f"Available models: {list(mapping.keys())}")
        if shared_param is None:
            shared_param = []
        else:
            fprint(f"using shared parameters: `{shared_param}`")

        cat_list = (cats if isinstance(cats, list)
                    else [None] * len(name))
        submodels = []
        for n, cat in zip(name, cat_list):
            fsection(_model_section_title(n, cat))
            submodels.append(mapping[n](config))
        return JointPVModel(submodels, shared_param)

    raise TypeError("`name` must be a string or a list of strings.")
