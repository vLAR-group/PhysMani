# PhysMani: Physics-principled 3D World Model for Dynamic Object Manipulation

**ECCV 2026**

[License: CC BY-NC-SA 4.0](./LICENSE)

## Overview

PhysMani is a framework for manipulating fast and dynamically moving targets in unstructured 3D environments. It couples a physics-principled 3D Gaussian world model (inspired by [FreeGave](https://github.com/vLAR-group/FreeGave) with a future-aware action policy, enabling physically grounded future-dynamics prediction and low-latency action generation.

This pre-release provides the minimal code, Docker environment, data links, and scripts needed to run PhysMani training and simulation evaluation on PhysMani-Bench.

## Release Scope

Included in this pre-release:

- PhysMani training and evaluation code.
- PhysMani-Bench data download script.
- Checkpoint download script.

Not included yet:

- 3DDA / 3DFA / ManiGaussian / Pi0 baseline.
- Refactored code layout.

The exact release versions are recorded in `[reproducibility/versions.json](reproducibility/versions.json)`.

## Clone

```bash
git clone --recursive https://github.com/vLAR-group/PhysMani.git
cd PhysMani
export PHYSMANI_ROOT_DIR="$(pwd)"
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```



## Docker

Build the image from the PhysMani repository root:

```bash
cd ${PHYSMANI_ROOT_DIR}
docker build -t physmani:v0.1 .
```

Create a GPU container from the PhysMani repository root. Set `PHYS_MANI_CUDA_ARCH_LIST` to the target GPU architecture to reduce CUDA extension compilation time. Keep `--shm-size=64g` for PyTorch dataloaders and distributed training; the Docker default 64 MB shared memory can cause `SIGBUS` failures.

```bash
docker run -dit --gpus all \
  --name physmani_v01 \
  --shm-size=16g \
  -v "$PWD":/usr/app/Code/PhysMani \
  -e PHYS_MANI_REPO_ROOT=/usr/app/Code/PhysMani \
  -e PHYS_MANI_CUDA_ARCH_LIST="8.6" \
  physmani:v0.1
```

Common CUDA arch values:


| GPU            | `PHYS_MANI_CUDA_ARCH_LIST` |
| -------------- | -------------------------- |
| RTX 3090       | `8.6`                      |
| RTX 4090 / L20 | `8.9`                      |
| A100 / A800    | `8.0`                      |


Enter the container:

```bash
docker exec -it physmani_v01 bash
```

Inside the container, run the runtime setup script from the mounted PhysMani repository:

```bash
cd ${PHYS_MANI_REPO_ROOT}
bash ${PHYS_MANI_REPO_ROOT}/docker/setup-runtime.sh
```

The runtime setup installs the mounted runtime repositories and compiles CUDA extensions for the selected architecture. It is expected to take several minutes on first run.



## Data

PhysMani scripts expect the dataset under:

```text
third_party/updated_3d_diffuser_actor/data/rmt/physmani_bench/
```

Run data download commands on the host from the PhysMani repository root, not inside the Docker container. The download script prefers Hugging Face CLI/Xet for faster public downloads. If `hf` is not available, it asks whether to install it; if installation is skipped or unavailable, it falls back to slower `curl`. 

Download the data archives from Hugging Face (https://huggingface.co/datasets/vLAR/PhysMani-Bench):

```bash
cd ${PHYSMANI_ROOT_DIR}
bash scripts/download_physmani_bench.sh default
```

The default group downloads only the test set, validation packaged data, validation world-model data, and instructions. This keeps bandwidth low for environment and evaluation checks.

Available groups:

```bash
bash scripts/download_physmani_bench.sh test
bash scripts/download_physmani_bench.sh val
bash scripts/download_physmani_bench.sh train
bash scripts/download_physmani_bench.sh wm
bash scripts/download_physmani_bench.sh instructions
bash scripts/download_physmani_bench.sh all
```

Add `--extract` to download and extract archives into the dataset root. If the archives have already been downloaded, use `--extract-only` to skip downloading and only extract existing files:

```bash
bash scripts/download_physmani_bench.sh default --extract
bash scripts/download_physmani_bench.sh default --extract-only
```

Rebuild the RLBench-style test indices after extracting the test set:

```bash
# Inside the container
conda activate 3d_diffuser_actor
cd third_party/updated_3d_diffuser_actor
rm -r data/rmt/physmani_bench/test/*/variation*
python data_preprocessing/rearrange_rlbench_demos.py \
  --root_dir $(pwd)/data/rmt/physmani_bench/test
```

After extraction, verify the expected paths:

```bash
cd third_party/updated_3d_diffuser_actor
test -d data/rmt/physmani_bench/test
test -d data/rmt/physmani_bench/train_package_compressed
test -d data/rmt/physmani_bench/val_package_compressed
test -f instructions/rmt/rmt_instructions_v5_rldyna19task_genvel_withstr.pkl
```

## Train PhysMani

Full training requires the training package, validation package, world-model predictions, instructions, and the PerAct checkpoint used by 3DARF pretraining. Prepare them on the host from the PhysMani repository root before starting training:

```bash
cd ${PHYSMANI_ROOT_DIR}
bash scripts/download_physmani_bench.sh all --extract
bash scripts/download_physmani_checkpoints.sh --extract
```

Checkout to target commit:

```bash
cd third_party/updated_3d_diffuser_actor
TRAIN_COMMIT=$(python3 -c 'import json; print(json.load(open("../../reproducibility/versions.json"))["repositories"]["physmani"]["training_commit"])')
git fetch origin release/physmani_train
git checkout "${TRAIN_COMMIT}"
```

Inside the container:

```bash
# Following commands should be executed in your docker container
conda activate 3d_diffuser_actor
# pretrain 100k steps 3drf
bash scripts/exp/physmani_train/train_3darf.sh
# continue train 100k steps physmani
bash scripts/exp/physmani_train/train_3dafdprf+velattn.sh
```

The training script defaults to `ngpus=3` and writes logs/checkpoints under:

```text
train_logs/physmani/
```

## Evaluate PhysMani

## Checkpoints

The evaluation script expects PhysMani checkpoints under:

```text
third_party/updated_3d_diffuser_actor/train_logs/
```

Run checkpoint download commands on the host from the PhysMani repository root, not inside the Docker container. The checkpoint script uses the same Hugging Face CLI/Xet fast path and `curl` fallback as the data script.

Download the release checkpoints from Hugging Face (https://huggingface.co/datasets/vLAR/PhysMani-Bench):

```bash
cd ${PHYSMANI_ROOT_DIR}
bash scripts/download_physmani_checkpoints.sh --extract
```

```text
third_party/updated_3d_diffuser_actor/train_logs/
└── physmani/
    └── diffusion_multitask-C120-B18-lr1e-4-2-H3-DT100/
        ├── epoch_79999.pth
        ├── epoch_84999.pth
        ├── epoch_89999.pth
        ├── epoch_94999.pth
        └── epoch_99999.pth
```

Before launching the evaluation terminals, switch the 3DDA submodule to the PhysMani evaluation commit recorded in `reproducibility/versions.json`:

```bash
cd third_party/updated_3d_diffuser_actor
EVAL_COMMIT=$(python3 -c 'import json; print(json.load(open("../../reproducibility/versions.json"))["repositories"]["physmani"]["evaluation_commit"])')
git fetch origin release/physmani_eval
git checkout "${EVAL_COMMIT}"
```
Inside the container, run the full PhysMani evaluation from a third terminal:

```bash
# Following commands should be executed in your docker container
conda activate 3d_diffuser_actor
cd /usr/app/Code/PhysMani/third_party/updated_3d_diffuser_actor
bash scripts/exp/physmani_eval/gpu0.sh
```

Inside the container, start the policy server in one terminal:

```bash
# Following commands should be executed in your docker container
cd /usr/app/Code/PhysMani/third_party/updated_3d_diffuser_actor
bash scripts/exp/physmani_eval/eval_sim_3dafdprf_policy_server.sh 0 0.0.0.0 8765
```

Inside the container, start the world-model server in another terminal:

```bash
# Following commands should be executed in your docker container
cd /usr/app/Code/PhysMani/third_party/updated_3d_diffuser_actor
bash scripts/exp/physmani_eval/eval_sim_3dafdprf_world_server.sh 0 0.0.0.0 8866
```

The full script evaluates 5 checkpoints, 16 tasks, and 100 episodes per task.

Evaluation logs are written under:

```text
eval_logs/
```



## Reproducibility

The release manifest is [reproducibility/versions.json](reproducibility/versions.json). Key entries:


| Component      | Commit                                     |
| -------------- | ------------------------------------------ |
| PhysMani train | `341e69161ad5ed8f4a7905dac87cf8e9aed6cb4a` |
| PhysMani eval  | `d6e15abbbde45e92cf885671448600a3ff1b6d06` |
| RLBench        | `0cc1948f59eda6e8bbbe32b759c869aa8be63ecd` |
| PyRep          | `7b7f6328a22c35262b4e93446563a0e68a31b870` |
| Dataset        | `physmani_bench`                           |




## TODO

- Release 3DDA / 3DFA / ManiGaussian / Pi0 baselines.
- Refactor the codebase after the pre-release reproduction path is stable.



## Acknowledgements

This release builds on the following open-source projects included under `third_party/`:

- [FreeGave](https://github.com/vLAR-group/FreeGave)
- [3D Diffuser Actor](https://github.com/nickgkan/3d_diffuser_actor)
- [RLBench](https://github.com/stepjam/RLBench)
- [PyRep](https://github.com/stepjam/PyRep)
- [Deformable-3D-Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians)
- [diff-gaussian-rasterization-extentions](https://github.com/ingra14m/diff-gaussian-rasterization-extentions)
- [gsplat](https://github.com/nerfstudio-project/gsplat)
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d)

We thank the authors and contributors of these projects for making their code publicly available.


## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{yun2026physmani,
  title     = {{PhysMani}: Physics-principled 3D World Model for Dynamic Object Manipulation},
  author    = {Yun, Peng and Huang, Shouwang and Li, Hao and Li, Jinxi and Wang, Jianan and Yang, Bo},
  booktitle = {European Conference on Computer Vision},
  year      = {2026}
}
```



## License

This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](./LICENSE).
