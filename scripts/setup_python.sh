#!/usr/bin/env bash
# Create the Python virtualenv the sidecar runs in.
# Run once on the Mac before building/launching the app.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$HERE/.venv"
PY="${PYTHON:-python3}"

echo "==> creating venv at $VENV"
"$PY" -m venv "$VENV"

echo "==> installing sidecar deps (this builds/downloads the mjbatch + mujoco wheels)"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r "$HERE/python/requirements.txt"

echo "==> sanity import"
"$VENV/bin/python" - <<'PY'
import mujoco, mjbatch, PIL, websockets
print("mujoco", mujoco.__version__)
print("mjbatch OK, Batch =", mjbatch.Batch)
print("pillow", PIL.__version__, "websockets", websockets.__version__)
PY

echo "==> done. venv python: $VENV/bin/python"
