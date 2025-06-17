
#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="3.12"
RETRY_COUNT=5
VENV_DIR=".venv"

for i in {1..${RETRY_COUNT}}; do
    if [ -d "${VENV_DIR}" ]; then
        echo "[setup.sh] Virtual environment already exists."
        rm -rf ${VENV_DIR}
        sleep ${i}
    fi
done

echo "[setup.sh] Creating virtual environment..."
curl -LsSf https://astral.sh/uv/install.sh | sh

uv python pin ${PYTHON_VERSION}
uv sync --extra dev

if [[ "$OSTYPE" == "msys"* ]]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

echo "[setup.sh] Virtual environment created and activated."
python setup.py build_ext --inplace
