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
Thin common base class for all forward models (PV and H0).
"""
from abc import ABC, abstractmethod

from ..cosmography import (Distance2Distmod, Distance2Redshift,
                           Distmod2Distance, Redshift2Distance)
from ..util import get_nested, load_config
from .utils import load_priors


class ModelBase(ABC):
    """Common ancestor for PV and H0 model hierarchies."""

    def __init__(self, config_path):
        config = load_config(config_path, replace_los_prior=False)
        # SH0ES configs use "Om0", PV configs use "Om".
        self.Om = get_nested(config, "model/Om",
                             get_nested(config, "model/Om0", 0.3))
        self.distance2distmod = Distance2Distmod(Om0=self.Om)
        self.distance2redshift = Distance2Redshift(Om0=self.Om)
        self.redshift2distance = Redshift2Distance(Om0=self.Om)
        self.distmod2distance = Distmod2Distance(Om0=self.Om)
        self.config = config

    def _load_and_set_priors(self):
        """Load priors from config and store as attributes."""
        priors = self.config["model"]["priors"]
        self.priors, self.prior_dist_name = load_priors(priors)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass
