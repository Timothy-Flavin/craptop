#!/usr/bin/env bash
set -euo pipefail

# Build sdist and wheel, then upload to PyPI using twine.
# Requires either:
# - environment variable PYPI_API_TOKEN set to a PyPI API token (recommended), or
# - TWINE_USERNAME and TWINE_PASSWORD set for legacy credentials.

if [ -z "${PYPI_API_TOKEN-}" ]; then
  if [ -z "${TWINE_USERNAME-}" ] || [ -z "${TWINE_PASSWORD-}" ]; then
    echo "Error: set PYPI_API_TOKEN or TWINE_USERNAME/TWINE_PASSWORD in the environment." >&2
    exit 1
  fi
else
  export TWINE_USERNAME="__token__"
  export TWINE_PASSWORD="$PYPI_API_TOKEN"
fi

rm -rf dist build *.egg-info
python -m pip install --upgrade build twine
python -m build --sdist --wheel
python -m twine check dist/*
python -m twine upload dist/*

echo "Published to PyPI (if credentials provided)."
