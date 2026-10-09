#!/usr/bin/env bash
# Linux check without GitHub Actions: copy this working tree to the deploy server and run the tests
# and the plugin validation in a throwaway python:3.11 container (CPU PyTorch; the tests load no model).
# Usage: [SSH_KEY=path] scripts/ci-linux.sh user@docker-host
set -euo pipefail
cd "$(dirname "$0")/.."
TARGET=${1:?usage: scripts/ci-linux.sh user@docker-host}
SSH="ssh ${SSH_KEY:+-i $SSH_KEY} -o BatchMode=yes $TARGET"
DIR=/tmp/memlab-ci-$$
git ls-files -co --exclude-standard | tar czf - -T - | $SSH "mkdir -p $DIR && tar xzf - -C $DIR"
$SSH "docker run --rm -v $DIR:/src -w /src python:3.11-slim bash -c '
  set -e
  apt-get update -qq && apt-get install -y -qq git nodejs npm >/dev/null
  git config --global user.email ci@memlab && git config --global user.name ci
  pip install -q torch --index-url https://download.pytorch.org/whl/cpu
  pip install -q -e \".[test,research]\"
  python -m pytest -q tests
  npm install -g -s @anthropic-ai/claude-code >/dev/null
  claude plugin validate . && claude plugin validate .claude-plugin/plugin.json && claude plugin validate mods/memlab-root
'; status=\$?; rm -rf $DIR; exit \$status"
