#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

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
