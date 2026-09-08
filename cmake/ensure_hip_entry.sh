#!/usr/bin/env bash
set -euo pipefail

src="$1"
dst="$2"

if [[ -f "$dst" ]]; then
  exit 0
fi

if [[ ! -f "$src" ]]; then
  echo "missing source entry: $src" >&2
  exit 1
fi

cp "$src" "$dst"
