#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="3.12"
VENV_DIR=".venv"

echo "[build.sh] Starting build process..."

if [ -f "pyproject.toml" ]; then
    echo "[build.sh] Found pyproject.toml, extracting package version..."
    PACKAGE_VERSION=$(sed -nE "s/^[[:space:]]*version[[:space:]]*=[[:space:]]*['\"]([^'\"]+)['\"].*/\1/p" pyproject.toml | head -n1)
else
    echo "[build.sh] Error: pyproject.toml not found."
    exit 1    
fi

echo "[build.sh] Removing old build artifacts..."
find ./src/pixtreme_source -name "*.cpp" -exec rm -f {} +
rm -rf ./dist

echo "[build.sh] Version: ${PACKAGE_VERSION}"
echo "[build.sh] Python version: ${PYTHON_VERSION}"

mkdir -p ./src/pixtreme

if [ -d "${VENV_DIR}" ]; then
    echo "[build.sh] Virtual environment already exists."
else
    echo "[build.sh] Creating virtual environment..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv python pin ${PYTHON_VERSION}
    uv sync --extra dev
    echo "[build.sh] Virtual environment created."
fi

if [[ "$OSTYPE" == "msys"* ]]; then
    source ${VENV_DIR}/Scripts/activate
else
    source ${VENV_DIR}/bin/activate
fi

# if not "*.pyi" in src/pixtreme
if ! ls src/pixtreme/*.pyi 1> /dev/null 2>&1; then
    echo "[build.sh] No .pyi files found in src/pixtreme."
    python setup.py build_ext --inplace
    #uv build
    #python -m build
fi


export CIBW_BUILD="cp310-* cp311-* cp312-* cp313-*"
export CIBW_BUILD_FRONTEND="build[uv]"
python -m build --sdist --installer uv
#cibuildwheel --output-dir dist
cibuildwheel dist/pixtreme-${PACKAGE_VERSION}.tar.gz --output-dir dist
cibuildwheel --platform linux dist/pixtreme-${PACKAGE_VERSION}.tar.gz --output-dir dist

echo "[build.sh] Build process completed successfully."

twine check dist/*
# twine upload --repository testpypi --skip-existing dist/*
# twine upload --repository pypi --skip-existing dist/*

