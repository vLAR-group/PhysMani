#!/usr/bin/env bash
set -eo pipefail

source /opt/conda/etc/profile.d/conda.sh

DEFAULT_3DDA_CUDA_ARCH_LIST="8.0;8.6+PTX"
DEFAULT_LEROBOT_CUDA_ARCH_LIST="8.0;8.6;8.9+PTX"
PHYS_MANI_CUDA_ARCH_LIST="${PHYS_MANI_CUDA_ARCH_LIST:-}"
PHYS_MANI_3DDA_CUDA_ARCH_LIST="${PHYS_MANI_3DDA_CUDA_ARCH_LIST:-${PHYS_MANI_CUDA_ARCH_LIST:-${DEFAULT_3DDA_CUDA_ARCH_LIST}}}"
PHYS_MANI_LEROBOT_CUDA_ARCH_LIST="${PHYS_MANI_LEROBOT_CUDA_ARCH_LIST:-${PHYS_MANI_CUDA_ARCH_LIST:-${DEFAULT_LEROBOT_CUDA_ARCH_LIST}}}"
PHYS_MANI_MAX_JOBS="${PHYS_MANI_MAX_JOBS:-10}"
PHYS_MANI_WARMUP_GSPLAT="${PHYS_MANI_WARMUP_GSPLAT:-1}"
PHYS_MANI_REPO_ROOT="${PHYS_MANI_REPO_ROOT:-/workspace}"

if [ ! -d "${PHYS_MANI_REPO_ROOT}/third_party/updated_3d_diffuser_actor" ]; then
  echo "Invalid PHYS_MANI_REPO_ROOT=${PHYS_MANI_REPO_ROOT}" >&2
  echo "Expected: \${PHYS_MANI_REPO_ROOT}/third_party/updated_3d_diffuser_actor" >&2
  echo "For mounted code, pass e.g. -e PHYS_MANI_REPO_ROOT=/usr/app/Code/PhysMani" >&2
  exit 1
fi

THIRD_PARTY_DIR="${PHYS_MANI_REPO_ROOT}/third_party"

echo "Using PHYS_MANI_3DDA_CUDA_ARCH_LIST=${PHYS_MANI_3DDA_CUDA_ARCH_LIST}"
echo "Using PHYS_MANI_LEROBOT_CUDA_ARCH_LIST=${PHYS_MANI_LEROBOT_CUDA_ARCH_LIST}"
echo "Using PHYS_MANI_MAX_JOBS=${PHYS_MANI_MAX_JOBS}"
echo "Using PHYS_MANI_REPO_ROOT=${PHYS_MANI_REPO_ROOT}"

export MAX_JOBS="${PHYS_MANI_MAX_JOBS}"
export PHYS_MANI_REPO_ROOT
export PYTHONPATH="${THIRD_PARTY_DIR}/updated_3d_diffuser_actor:${THIRD_PARTY_DIR}/RLBench:${THIRD_PARTY_DIR}/PyRep:${PYTHONPATH:-}"

ensure_cuda_runtime_paths() {
  local cuda_runtime_include
  cuda_runtime_include="$(python - <<'PY'
from pathlib import Path
import sys

prefix = Path(sys.prefix)
candidates = sorted(prefix.glob("lib/python*/site-packages/nvidia/cuda_runtime/include"))
if candidates:
    print(candidates[0])
PY
)"

  mkdir -p "${CONDA_PREFIX}/lib64"
  if [ -e "${CONDA_PREFIX}/lib/libcudart.so" ]; then
    ln -sf ../lib/libcudart.so "${CONDA_PREFIX}/lib64/libcudart.so"
  fi
  if [ -e "${CONDA_PREFIX}/lib/libcudart.so.12" ]; then
    ln -sf ../lib/libcudart.so.12 "${CONDA_PREFIX}/lib64/libcudart.so.12"
  fi

  export CPATH="${CONDA_PREFIX}/targets/x86_64-linux/include:${CONDA_PREFIX}/include:${cuda_runtime_include}:${CPATH:-}"
  export LIBRARY_PATH="${CONDA_PREFIX}/lib64:${CONDA_PREFIX}/lib:${CONDA_PREFIX}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib64:${CONDA_PREFIX}/lib:${CONDA_PREFIX}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
}

patch_simple_knn_cfloat() {
  local simple_knn="$1"
  if ! grep -q '#include <cfloat>' "$simple_knn"; then
    sed -i '1i#include <cfloat>' "$simple_knn"
  fi
}

install_gaussian_stack() {
  python -m pip install -e diff-gaussian-rasterization-extentions --no-build-isolation

  patch_simple_knn_cfloat Deformable-3D-Gaussians/submodules/simple-knn/simple_knn.cu
  python -m pip install Deformable-3D-Gaussians/submodules/simple-knn --no-build-isolation

  python -m pip install -e gsplat --no-build-isolation
}

clear_editable_package_state() {
  local package_name="$1"
  local module_name="$2"
  local site_packages

  site_packages="$(python - <<'PY'
import site

print(site.getsitepackages()[0])
PY
)"

  rm -f "${site_packages}/${module_name}.egg-link"
  rm -f "${site_packages}/__editable__.${package_name}"*.pth
  rm -f "${site_packages}/__editable___${package_name}"*_finder.py
  rm -rf "${site_packages}/__pycache__/__editable___${package_name}"*_finder.*
  rm -rf "${site_packages}/${package_name}"-*.dist-info
}

install_runtime_rlbench_pyrep() {
  echo "Installing PyRep/RLBench with python=$(which python)"
  echo "pip=$(python -m pip --version)"

  clear_editable_package_state "PyRep" "PyRep"
  clear_editable_package_state "pyrep" "pyrep"
  clear_editable_package_state "rlbench" "rlbench"

  python -m pip install -r "${THIRD_PARTY_DIR}/PyRep/requirements.txt"
  python -m pip install -e "${THIRD_PARTY_DIR}/PyRep" --no-build-isolation
  (
    cd "${THIRD_PARTY_DIR}/PyRep"
    python setup.py build_ext --inplace
  )

  python -m pip install -r "${THIRD_PARTY_DIR}/RLBench/requirements.txt"
  python -m pip install -e "${THIRD_PARTY_DIR}/RLBench" --no-build-isolation
  python -c "import rlbench, pyrep; print('rlbench=', rlbench.__file__); print('pyrep=', pyrep.__file__)"
}

verify_3dda_runtime_imports() {
  python - <<'PY'
import diffusers
import h5py
import msgpack
import pyrep
import rlbench
import websockets.asyncio.server

print("3d_diffuser_actor runtime imports ok")
print("rlbench=", rlbench.__file__)
print("pyrep=", pyrep.__file__)
print("diffusers=", diffusers.__version__)
PY
}

verify_lerobot_runtime_imports() {
  python - <<'PY'
import diffusers
import gsplat
import msgpack
import pyrep
import pytorch3d
import rlbench
import websockets.asyncio.server

print("lerobot runtime imports ok")
print("rlbench=", rlbench.__file__)
print("pyrep=", pyrep.__file__)
print("diffusers=", diffusers.__version__)
PY
}

warmup_gsplat() {
  if [ "${PHYS_MANI_WARMUP_GSPLAT}" != "1" ]; then
    echo "Skipping gsplat warmup because PHYS_MANI_WARMUP_GSPLAT=${PHYS_MANI_WARMUP_GSPLAT}"
    return
  fi

  python - <<'PY'
import inspect

import torch
from gsplat.rendering import rasterization

if not torch.cuda.is_available():
    print("Skipping gsplat warmup because CUDA is not available")
    raise SystemExit(0)

print("gsplat rasterization signature:", inspect.signature(rasterization))

device = "cuda"
num_gaussians, num_cameras, height, width = 8, 1, 32, 32

means = torch.randn(num_gaussians, 3, device=device)
quats = torch.randn(num_gaussians, 4, device=device)
quats = quats / quats.norm(dim=-1, keepdim=True)
scales = torch.ones(num_gaussians, 3, device=device) * 0.01
opacities = torch.ones(num_gaussians, device=device) * 0.5
colors = torch.rand(num_gaussians, 3, device=device)

viewmats = torch.eye(4, device=device)[None].repeat(num_cameras, 1, 1)
Ks = torch.tensor(
    [[[30.0, 0.0, width / 2], [0.0, 30.0, height / 2], [0.0, 0.0, 1.0]]],
    device=device,
)

with torch.no_grad():
    rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
    )

torch.cuda.synchronize()
print("gsplat warmup ok")
PY
}

conda activate 3d_diffuser_actor
export TORCH_CUDA_ARCH_LIST="${PHYS_MANI_3DDA_CUDA_ARCH_LIST}"
ensure_cuda_runtime_paths
install_runtime_rlbench_pyrep
python -m pip install diffusers==0.34.0 h5py msgpack websockets
python -m pip install flash-attn==2.5.9.post1 --no-build-isolation
cd "${THIRD_PARTY_DIR}/updated_3d_diffuser_actor"
python -m pip install submodules/fps --no-build-isolation
cd "${THIRD_PARTY_DIR}"
install_gaussian_stack
rm -rf pytorch3d/build
python -m pip install -e pytorch3d --no-build-isolation
verify_3dda_runtime_imports

conda activate lerobot
export TORCH_CUDA_ARCH_LIST="${PHYS_MANI_LEROBOT_CUDA_ARCH_LIST}"
ensure_cuda_runtime_paths
install_runtime_rlbench_pyrep
cd "${THIRD_PARTY_DIR}"

install_gaussian_stack
python -m pip install -e pytorch3d --no-build-isolation
python -m pip install updated_3d_diffuser_actor/submodules/fps --no-build-isolation
verify_lerobot_runtime_imports
warmup_gsplat

conda clean -afy
