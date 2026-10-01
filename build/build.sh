#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_dir/dist"

# src/capi.mojo is the single entry point; -I src resolves the ported.*
# module import that holds the kernels.
export MODULAR_HOME="${MODULAR_HOME:-$(dirname "$(dirname "$(command -v mojo)")")/share/max}"

mojo build --emit shared-lib -I "$repo_dir/src" \
    "$repo_dir/src/capi.mojo" \
    -o "$repo_dir/dist/libmojo-stable-baselines3.so"
