#!/usr/bin/env bash
# MILESTONE A end-to-end demo (PIVOT_ROADMAP §A.MILESTONE).
#
# Wraps milestone_a_demo.py in the right conda env so the harness is
# always invoked with the project's pinned interpreter. Exits with the
# Python script's status code so CI / pre-commit hooks can gate on it.

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate medvlm-base
fi

exec python "$REPO_ROOT/scripts/milestone_a_demo.py" "$@"
