#!/usr/bin/env bash
# scripts/guard-no-hardcoded-model.sh — NFR1 grep guard
# Ensures no hardcoded MODEL="gpt-..." literals exist in pageindex_mutil/.
# Model names must come from ConfigLoader (config.yaml / MODEL_NAME env).
set -euo pipefail

if grep -rnE 'MODEL\s*=\s*"gpt-' pageindex_mutil/ --include='*.py'; then
  echo "FAIL: hardcoded MODEL=\"gpt-...\" found in pageindex_mutil/ (must use ConfigLoader)"
  exit 1
fi

echo "OK: no hardcoded MODEL=\"gpt-...\" in pageindex_mutil/"
