#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <dir>[,<dir>,...]"
  exit 1
fi

IFS=',' read -ra DIRS <<< "$1"
for PKG in "${DIRS[@]}"; do
  if [ ! -d "$PKG" ]; then
    echo "Directory '$PKG' not found"
    exit 1
  fi
done

VENV=$(ls -d venv_* 2>/dev/null | head -1)
if [ -z "$VENV" ]; then
  echo "No venv_ directory found"
  exit 1
fi

PYTHON="$VENV/bin/python"
echo "Using $PYTHON on ${DIRS[*]}"

echo "Running isort..."
find "${DIRS[@]}" -name "*.py" ! -name "__init__.py" -exec "$PYTHON" -m isort {} +

echo "Running flake8..."
find "${DIRS[@]}" -name "*.py" ! -name "__init__.py" -exec "$PYTHON" -m flake8 {} +
