#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TECTONIC_BIN="${TECTONIC:-$(command -v tectonic || true)}"
if [[ -z "$TECTONIC_BIN" && -x /tmp/bfl-tectonic-tool/tectonic ]]; then
  TECTONIC_BIN=/tmp/bfl-tectonic-tool/tectonic
fi
if [[ -z "$TECTONIC_BIN" ]]; then
  printf 'Tectonic is unavailable; set TECTONIC=/path/to/binary.\n' >&2
  exit 1
fi
OUT_DIR="${1:-$PROJECT}"
mkdir -p "$OUT_DIR"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/bfl-tectonic-cache}" \
  "$TECTONIC_BIN" --keep-logs --outdir "$OUT_DIR" "$PROJECT/PAPER.tex"
printf 'PDF generated: %s/PAPER.pdf\n' "$OUT_DIR"
