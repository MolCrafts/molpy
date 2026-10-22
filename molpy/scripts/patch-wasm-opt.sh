#!/usr/bin/env bash
# Patch emsdk wasm-opt so Binaryen 3.1.58 (Pyodide 0.27) accepts output from
# modern rustc, which advertises bulk-memory-opt / call-indirect-overlong.
set -euo pipefail

rustup target add wasm32-unknown-emscripten

patch_one() {
  local real="$1"
  [[ -e "$real" ]] || return 0
  # Already a shim pointing at .real
  if [[ -f "${real}.real" ]]; then
    return 0
  fi
  # Only patch real binaries / non-shim scripts once.
  if head -c 4 "$real" 2>/dev/null | grep -q $'\x7fELF' || file "$real" 2>/dev/null | grep -qi 'executable\|ELF'; then
    :
  elif [[ -L "$real" ]]; then
    :
  else
    # Might already be a text shim from a prior run with different layout.
    if grep -q 'wasm-opt.real' "$real" 2>/dev/null; then
      return 0
    fi
  fi
  mv "$real" "${real}.real"
  cat >"$real" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
args=()
for a in "$@"; do
  case "$a" in
    --enable-bulk-memory-opt|--disable-bulk-memory-opt|\
    --enable-call-indirect-overlong|--disable-call-indirect-overlong)
      ;;
    *) args+=("$a") ;;
  esac
done
exec "$(dirname "$0")/wasm-opt.real" "${args[@]}"
SH
  chmod +x "$real"
  echo "patched wasm-opt → ${real}.real"
}

if [[ -n "${EMSDK:-}" ]]; then
  patch_one "${EMSDK}/upstream/bin/wasm-opt"
fi

# cibuildwheel / pyodide-build cache and in-tree xbuildenv
search_roots=("$HOME/.cache" "${CIBW_CACHE_PATH:-}" /tmp "$(pwd)" "$(pwd)/..")
for root in "${search_roots[@]}"; do
  [[ -n "$root" && -d "$root" ]] || continue
  while IFS= read -r -d '' p; do
    patch_one "$p"
  done < <(find "$root" -path '*/emsdk/upstream/bin/wasm-opt' \( -type f -o -type l \) -print0 2>/dev/null || true)
done
