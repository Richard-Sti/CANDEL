#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <package_dir>"
  exit 1
fi

PKG="$1"
if [ ! -d "$PKG" ]; then
  echo "Directory '$PKG' not found"
  exit 1
fi

VENV=$(ls -d venv_* 2>/dev/null | head -1)
if [ -z "$VENV" ]; then
  echo "No venv_ directory found"
  exit 1
fi

PYTHON="$VENV/bin/python"
echo "Using $PYTHON on $PKG"

echo "Running isort..."
find "$PKG" -name "*.py" ! -name "__init__.py" -exec "$PYTHON" -m isort {} +

echo "Running flake8..."
find "$PKG" -name "*.py" ! -name "__init__.py" -exec "$PYTHON" -m flake8 {} +
