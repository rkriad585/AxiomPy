#!/usr/bin/env bash
set -euo pipefail

# AxiomPy installer — uses uv when available, falls back to pip.
# Usage: bash installer.sh [--dev]

if ! command -v python3 &>/dev/null; then
    echo "Error: python3 is required." >&2
    exit 1
fi

if command -v uv &>/dev/null; then
    echo "==> Installing with uv..."
    uv sync ${1:+--$1}
else
    echo "==> uv not found, falling back to pip..."
    python3 -m pip install --upgrade pip
    python3 -m pip install -e . ${1:+$1}
fi

echo "==> AxiomPy installed. Try: python -c 'from axiompy import Axiom; print(Axiom)'"
