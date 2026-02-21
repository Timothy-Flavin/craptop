#Requires -Version 5.1
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Build sdist and wheel, then upload to PyPI using twine.
# Requires either:
# - environment variable PYPI_API_TOKEN set to a PyPI API token (recommended), or
# - TWINE_USERNAME and TWINE_PASSWORD set for legacy credentials.

if (-not $env:PYPI_API_TOKEN) {
    if (-not $env:TWINE_USERNAME -or -not $env:TWINE_PASSWORD) {
        Write-Error "Error: set PYPI_API_TOKEN or TWINE_USERNAME/TWINE_PASSWORD in the environment."
        exit 1
    }
} else {
    $env:TWINE_USERNAME = "__token__"
    $env:TWINE_PASSWORD = $env:PYPI_API_TOKEN
}

# Clean previous build artifacts
foreach ($dir in @('dist', 'build', 'wheelhouse')) {
    if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }
}
Get-ChildItem -Filter '*.egg-info' -Directory | Remove-Item -Recurse -Force

python -m pip install --upgrade build twine

# Build sdist + wheel
python -m build --sdist --wheel

# On Windows there is no auditwheel; just use the wheel from dist/ directly.
# Copy it to wheelhouse/ to keep the upload step consistent.
New-Item -ItemType Directory -Force -Path wheelhouse | Out-Null
Copy-Item dist\*.whl wheelhouse\

# Sanity-check everything we are about to ship
python -m twine check dist\*.tar.gz wheelhouse\*.whl

# Upload: sdist from dist/, wheel from wheelhouse/
python -m twine upload dist\*.tar.gz wheelhouse\*.whl

Write-Host "Published to PyPI."
