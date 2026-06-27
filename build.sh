#!/usr/bin/env bash
set -euo pipefail

echo "==> Installing dependencies..."
uv sync

echo "==> Building wheel..."
uv build

echo "==> Done. Wheel is in dist/"
