#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

if ! command -v tectonic >/dev/null 2>&1; then
  echo "tectonic is required to compile this dissertation." >&2
  exit 127
fi

tectonic --keep-logs --keep-intermediates thesis.tex
