#!/bin/bash
# Remove job output files from the preprocess directory.
rm -f "$(dirname "$0")"/python-*.out
echo "Cleared job output files."
