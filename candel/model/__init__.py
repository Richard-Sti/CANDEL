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

from .model import (                                                            # noqa
    load_priors,                                                                # noqa
    TFRModel,                                                                   # noqa
    TFRModel_DistMarg,                                                          # noqa
    PantheonPlusModel_DistMarg,                                                 # noqa
    ClustersModel_DistMarg,                                                          # noqa
    FPModel_DistMarg,                                                           # noqa
    JointPVModel,                                                               # noqa
    )
from .magnitude_selection import (                                              # noqa
    MagnitudeSelection,                                                         # noqa
    log_magnitude_selection,                                                    # noqa
    )
from .SH0ES_model import SH0ESModel                                             # noqa
from .interp import LOSInterpolator                                             # noqa
from .simpson import ln_simpson                                                 # noqa

from ..util import fprint


def name2model(name, shared_param=None, config=None):
    mapping = {
        "TFRModel": TFRModel,
        "TFRModel_DistMarg": TFRModel_DistMarg,
        "PantheonPlusModel_DistMarg": PantheonPlusModel_DistMarg,
        "ClustersModel_DistMarg": ClustersModel_DistMarg,
        "FPModel_DistMarg": FPModel_DistMarg
        }

    if isinstance(name, str):
        if name not in mapping:
            raise ValueError(f"Model name `{name}` not recognized.\n"
                             f"Available models: {list(mapping.keys())}")
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

        submodels = [mapping[n](config) for n in name]
        return JointPVModel(submodels, shared_param)

    raise TypeError("`name` must be a string or a list of strings.")
