#!/bin/bash -l
# Local machine profile. Sourced by _submit_lib.sh when
# `machine = "local"` in local_config.toml. Intentionally a no-op:
# there is no batch backend to configure and no cluster-specific
# environment to set up. Exists so that the per-cluster profile
# sourcing in _submit_lib.sh works uniformly across machines.
