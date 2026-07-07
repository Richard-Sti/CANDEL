#!/bin/bash -l
# Glamdring cluster profile. Sourced by _submit_lib.sh on the login node
# before submission; addqueue -s inherits this environment.

# Build LD_LIBRARY_PATH from gpu_ld_library_path in local_config.toml.
# Expects CANDEL_ROOT to be set by _submit_lib.sh.
if [[ -n "${CANDEL_ROOT:-}" && -f "$CANDEL_ROOT/local_config.toml" ]]; then
    _ldpath=$(awk '
        /^gpu_ld_library_path[[:space:]]*=[[:space:]]*\[/ { in_block = 1; next }
        in_block && /\]/ { exit }
        in_block {
            gsub(/[[:space:]]*,?[[:space:]]*$/, "")
            gsub(/^[[:space:]]*["'\'']|["'\'']$/, "")
            if (length) print
        }
    ' "$CANDEL_ROOT/local_config.toml" | paste -sd: -)
    # Also add the libpython directory from the venv's Python build.
    if [[ -n "${CANDEL_PYTHON:-}" ]]; then
        _pylib=$("$CANDEL_PYTHON" -c \
            "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))" \
            2>/dev/null)
        [[ -n "$_pylib" ]] && _ldpath="${_pylib}${_ldpath:+:$_ldpath}"
        unset _pylib
    fi
    if [[ -n "$_ldpath" ]]; then
        export LD_LIBRARY_PATH="$_ldpath${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    unset _ldpath
fi
