#!/usr/bin/env bash
set -euo pipefail

HF_REPO="${PHYS_MANI_HF_REPO:-vLAR/PhysMani-Bench}"
HF_CKPT_PREFIX="${PHYS_MANI_HF_CKPT_PREFIX:-checkpoints}"
DOWNLOAD_BACKEND="${PHYS_MANI_DOWNLOAD_BACKEND:-hf}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_ROOT="${PHYS_MANI_3DDA_ROOT:-${REPO_ROOT}/third_party/updated_3d_diffuser_actor}"
CKPT_ROOT="${PHYS_MANI_CKPT_ROOT:-${CODE_ROOT}/train_logs}"
ARCHIVE_DIR="${PHYS_MANI_CKPT_ARCHIVE_DIR:-${CKPT_ROOT}/_archives}"
BASE_URL="${PHYS_MANI_CKPT_BASE_URL:-https://huggingface.co/datasets/${HF_REPO}/resolve/main/${HF_CKPT_PREFIX}}"
CKPT_FILE="physmani_release_ckpts.tar.gz"
DIFFUSER_ACTOR_PERACT_FILE="diffuser_actor_peract.pth"

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
  local remote_path="$1"
  local output_path="$2"
  local hf_cli="$3"
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
  mkdir -p "$(dirname "${output_path}")"
  mv -f "${tmp_dir}/${remote_path}" "${output_path}"
  rm -rf "${tmp_dir}"
}

curl_download_one() {
  local file="$1"
  local output_path="$2"
  local url="${BASE_URL}/${file}"
  echo "[download:curl] ${url}"
  curl -L --fail --continue-at - --retry 3 --retry-delay 5 \
    -o "${output_path}" "${url}"
}

download_one() {
  local file="$1"
  local output_path="$2"
  mkdir -p "${ARCHIVE_DIR}" "${CKPT_ROOT}"

  if [ "${DOWNLOAD_BACKEND}" = "curl" ]; then
    curl_download_one "${file}" "${output_path}"
    return
  fi

  local hf_cli
  if hf_cli="$(ensure_hf_cli)"; then
    hf_download_one "${HF_CKPT_PREFIX}/${file}" "${output_path}" "${hf_cli}"
  else
    echo "[warn] Hugging Face CLI is unavailable; falling back to slower curl download." >&2
    curl_download_one "${file}" "${output_path}"
  fi
}

mkdir -p "${ARCHIVE_DIR}" "${CKPT_ROOT}"

download_one "${CKPT_FILE}" "${ARCHIVE_DIR}/${CKPT_FILE}"

if [ "${1:-}" = "--extract" ]; then
  echo "[extract] ${ARCHIVE_DIR}/${CKPT_FILE} -> ${CKPT_ROOT}"
  tar -xf "${ARCHIVE_DIR}/${CKPT_FILE}" -C "${CKPT_ROOT}"

  download_one "${DIFFUSER_ACTOR_PERACT_FILE}" "${CKPT_ROOT}/${DIFFUSER_ACTOR_PERACT_FILE}"
fi

cat <<EOF

Downloaded checkpoint archive to:
  ${ARCHIVE_DIR}

Expected checkpoint root for PhysMani scripts:
  ${CKPT_ROOT}
EOF
