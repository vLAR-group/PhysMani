#!/usr/bin/env bash
set -euo pipefail

HF_REPO="${PHYS_MANI_HF_REPO:-vLAR/PhysMani-Bench}"
HF_DATA_PREFIX="${PHYS_MANI_HF_DATA_PREFIX:-data}"
BASE_URL="${PHYS_MANI_DATA_BASE_URL:-https://huggingface.co/datasets/${HF_REPO}/resolve/main/${HF_DATA_PREFIX}}"
DOWNLOAD_BACKEND="${PHYS_MANI_DOWNLOAD_BACKEND:-hf}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_ROOT="${PHYS_MANI_3DDA_ROOT:-${REPO_ROOT}/third_party/updated_3d_diffuser_actor}"
DATA_ROOT="${PHYS_MANI_DATA_ROOT:-${CODE_ROOT}/data/rmt/physmani_bench}"
ARCHIVE_DIR="${PHYS_MANI_ARCHIVE_DIR:-${DATA_ROOT}/_archives}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/download_physmani_bench.sh [default|test|val|train|wm|instructions|all] [--extract|--extract-only]

Default downloads only test, val, and instructions archives to limit bandwidth.
Use "train" or "all" explicitly for large training archives.
Use "--extract-only" to extract existing archives without downloading.

Download backend:
  The script prefers Hugging Face CLI/Xet for faster public downloads. If `hf` is
  not available, it asks whether to install it. If installation is skipped or
  unavailable, it falls back to curl.

Environment overrides:
  PHYS_MANI_HF_REPO                    Hugging Face dataset repo
  PHYS_MANI_HF_DATA_PREFIX             data path inside the dataset repo
  PHYS_MANI_DOWNLOAD_BACKEND           hf|curl
  PHYS_MANI_AUTO_INSTALL_HF_CLI        set to 1 to install hf CLI without prompting
  HF_XET_HIGH_PERFORMANCE              defaults to 1 for hf backend
  HF_XET_NUM_CONCURRENT_RANGE_GETS     defaults to 64 for hf backend
  PHYS_MANI_DATA_BASE_URL              curl URL prefix
  PHYS_MANI_3DDA_ROOT                  path to third_party/updated_3d_diffuser_actor
  PHYS_MANI_DATA_ROOT                  final dataset root
  PHYS_MANI_ARCHIVE_DIR                archive download directory
EOF
}

find_hf_cli() {
  if [ -n "${PHYS_MANI_HF_CLI:-}" ] && [ -x "${PHYS_MANI_HF_CLI}" ]; then
    printf '%s\n' "${PHYS_MANI_HF_CLI}"
    return 0
  fi
  if command -v hf >/dev/null 2>&1; then
    command -v hf
    return 0
  fi
  if [ -x "${HOME}/.local/bin/hf" ]; then
    printf '%s\n' "${HOME}/.local/bin/hf"
    return 0
  fi
  return 1
}

install_hf_cli() {
  local python_bin="${PYTHON:-python3}"
  echo "[install] Installing Hugging Face CLI and hf-xet with ${python_bin} -m pip --user" >&2
  "${python_bin}" -m pip install --user -U huggingface_hub hf_xet
}

ensure_hf_cli() {
  local hf_cli
  if hf_cli="$(find_hf_cli)"; then
    printf '%s\n' "${hf_cli}"
    return 0
  fi

  if [ "${PHYS_MANI_AUTO_INSTALL_HF_CLI:-0}" = "1" ]; then
    install_hf_cli
  elif [ -t 0 ]; then
    printf 'Hugging Face CLI `hf` was not found. Install it now for faster downloads? [y/N] ' >&2
    local answer
    read -r answer
    case "${answer}" in
      y|Y|yes|YES) install_hf_cli ;;
      *) return 1 ;;
    esac
  else
    return 1
  fi

  hf_cli="$(find_hf_cli)" || return 1
  printf '%s\n' "${hf_cli}"
}

hf_download_one() {
  local file="$1"
  local remote_path="${HF_DATA_PREFIX}/${file}"
  local hf_cli="$2"
  local tmp_dir
  tmp_dir="$(mktemp -d "${ARCHIVE_DIR}/.hf_download.XXXXXX")"

  echo "[download:hf] ${HF_REPO}/${remote_path}"
  HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}" \
  HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-64}" \
    "${hf_cli}" download "${HF_REPO}" "${remote_path}" \
      --repo-type dataset \
      --local-dir "${tmp_dir}"

  if [ ! -f "${tmp_dir}/${remote_path}" ]; then
    echo "[error] hf download did not create ${tmp_dir}/${remote_path}" >&2
    rm -rf "${tmp_dir}"
    return 1
  fi
  mv -f "${tmp_dir}/${remote_path}" "${ARCHIVE_DIR}/${file}"
  rm -rf "${tmp_dir}"
}

curl_download_one() {
  local file="$1"
  local url="${BASE_URL}/${file}"
  echo "[download:curl] ${url}"
  curl -L --fail --continue-at - --retry 3 --retry-delay 5 \
    -o "${ARCHIVE_DIR}/${file}" "${url}"
}

download_one() {
  local file="$1"
  mkdir -p "${ARCHIVE_DIR}"

  if [ "${DOWNLOAD_BACKEND}" = "curl" ]; then
    curl_download_one "${file}"
    return
  fi

  local hf_cli
  if hf_cli="$(ensure_hf_cli)"; then
    hf_download_one "${file}" "${hf_cli}"
  else
    echo "[warn] Hugging Face CLI is unavailable; falling back to slower curl download." >&2
    curl_download_one "${file}"
  fi
}

extract_one() {
  local file="$1"
  local target="${DATA_ROOT}"
  if [ "${file}" = "physmani_bench_instructions.tar.gz" ]; then
    target="${CODE_ROOT}"
  fi
  echo "[extract] ${ARCHIVE_DIR}/${file} -> ${target}"
  mkdir -p "${target}"
  tar -xf "${ARCHIVE_DIR}/${file}" -C "${target}"
}

group="${1:-default}"
extract=0
extract_only=0
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  usage
  exit 0
fi
if [ "${2:-}" = "--extract" ] || [ "${1:-}" = "--extract" ]; then
  extract=1
  [ "${1:-}" = "--extract" ] && group="default"
fi
if [ "${2:-}" = "--extract-only" ] || [ "${1:-}" = "--extract-only" ]; then
  extract=1
  extract_only=1
  [ "${1:-}" = "--extract-only" ] && group="default"
fi

case "${group}" in
  default)
    files=(
      physmani_bench_test.tar.gz
      physmani_bench_val_package.tar.gz
      physmani_bench_val_wm_predictions.tar.gz
      physmani_bench_instructions.tar.gz
    )
    ;;
  test)
    files=(physmani_bench_test.tar.gz)
    ;;
  val)
    files=(
      physmani_bench_val_package.tar.gz
      physmani_bench_val_wm_predictions.tar.gz
    )
    ;;
  train)
    files=(
      physmani_bench_train_package.tar.gz
      physmani_bench_train_wm_predictions_part1.tar.gz
      physmani_bench_train_wm_predictions_part2.tar.gz
      physmani_bench_train_wm_predictions_part3.tar.gz
    )
    ;;
  wm)
    files=(
      physmani_bench_train_wm_predictions_part1.tar.gz
      physmani_bench_train_wm_predictions_part2.tar.gz
      physmani_bench_train_wm_predictions_part3.tar.gz
      physmani_bench_val_wm_predictions.tar.gz
    )
    ;;
  instructions)
    files=(physmani_bench_instructions.tar.gz)
    ;;
  all)
    files=(
      physmani_bench_test.tar.gz
      physmani_bench_train_package.tar.gz
      physmani_bench_train_wm_predictions_part1.tar.gz
      physmani_bench_train_wm_predictions_part2.tar.gz
      physmani_bench_train_wm_predictions_part3.tar.gz
      physmani_bench_val_package.tar.gz
      physmani_bench_val_wm_predictions.tar.gz
      physmani_bench_instructions.tar.gz
    )
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

for file in "${files[@]}"; do
  if [ "${extract_only}" = "1" ]; then
    if [ ! -f "${ARCHIVE_DIR}/${file}" ]; then
      echo "[error] Missing archive for --extract-only: ${ARCHIVE_DIR}/${file}" >&2
      echo "[hint] Run without --extract-only first to download the archive." >&2
      exit 1
    fi
  else
    download_one "${file}"
  fi
  if [ "${extract}" = "1" ]; then
    extract_one "${file}"
  fi
done

cat <<EOF

Downloaded ${#files[@]} archive(s) to:
  ${ARCHIVE_DIR}

Expected dataset root for PhysMani scripts:
  ${DATA_ROOT}
EOF
