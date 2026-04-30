#!/usr/bin/env bash
# Phase B end-to-end demo (PIVOT_ROADMAP §B.1, §B.2, §B.4, §B.16).
#
# Wraps phase_b_demo.py in the right conda env so the harness is always
# invoked with the project's pinned interpreter. Exits with the Python
# script's status code so CI / pre-commit hooks can gate on it.

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate medvlm-base
fi

exec python "$REPO_ROOT/scripts/phase_b_demo.py" "$@"
