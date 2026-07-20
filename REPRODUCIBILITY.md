# PIR-SBFR Reproducibility Protocol

This document provides a complete execution path from public-data preparation and three-seed training through evaluation, paired bootstrap analysis, and deployment-efficiency calibration. See [`docs/PAPER_SPEC.md`](docs/PAPER_SPEC.md) for the equation-by-equation mapping between the paper and the official source code.

## 1. Protocol scope

Interpret results and experiment records using three distinct categories:

- **Paper-level definition:** a formula, value, data split, result, or test procedure stated in the PIR-SBFR paper.
- **Official source definition:** a release-level architectural, numerical, or execution detail fixed by this repository where the paper presents a higher-level description.
- **External artifact boundary:** an experiment that depends on data or a per-sample realization that is not redistributed in the public repository.

This repository is the **official implementation released by the PIR-SBFR paper authors**. The paper and source code are complementary: the paper defines the method and reports the experiments, while this repository is authoritative for release-level network topology, framework settings, executable configurations, and numerical conventions. The DIOR and AI-TOD-v2 experiments can be rerun from the public code and datasets using the protocol below.

Two reported evaluations depend on artifacts that are not redistributed:

- **Real flight experiments:** the eight videos, annotations, and synchronized altitude/attitude/IMU records remain private research data.
- **Archived joint OOD realization:** the paper reports "2.5x sampling + unseen PSF + unseen noise," but the exact per-image PSF realization used for the reported table is not included in the repository.

The mixed-degradation metadata-control images and their per-image manifest are also not redistributed. The official scripts provide deterministic protocol-equivalent generation paths for new evaluations, but a newly generated sample set should not be presented as the archived sample set used for the paper's numerical table.

## 2. Environment

The paper reports Ubuntu 22.04, CUDA 12.8, PyTorch 2.8.0, and one 24 GB RTX 4090D for training; efficiency tests use an RTX 4090. This repository pins PyTorch 2.8.0, torchvision 0.23.0, and Ultralytics 8.3.0.

Python 3.11 is recommended:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Check package versions and CUDA availability:

```bash
python -c "import torch, ultralytics; print(torch.__version__, torch.version.cuda, ultralytics.__version__); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The paper identifies the baseline as "Ultralytics YOLO11, Version 11.0.0." The official public code standardizes the executable dependency on `ultralytics==8.3.0`, the Python package release containing the YOLO11 modules used by this repository.

## 3. Data preparation

### 3.1 DIOR

Paper-fixed image-identifier counts:

| Split | Images | Original split record |
|---|---:|---|
| train | 5,862 | `train.txt` |
| val | 5,863 | `val.txt` |
| test | 11,738 | `test.txt` |

Prepare the YOLO directory used by this repository:

```bash
python scripts/prepare_dior.py \
  --source DATA/DIOR \
  --output datasets/DIOR \
  --mode symlink
```

The converter reads official split, image, and XML annotation locations by default. Override split/image/XML paths explicitly for nonstandard layouts; `--source-root` and `--output-root` are compatibility aliases for `--source` and `--output`. Use `--mode copy` to copy rather than symlink. The converter is idempotent and refuses to overwrite conflicting content silently.

The converter interprets XML coordinates as the official VOC one-based inclusive convention. `difficult` instances are retained in training labels by default and recorded as `difficult/ignore` with `iscrowd=1` in generated COCO ground truth so stock COCOeval truly ignores them. This is the official source convention. If `--exclude-difficult` is used, apply the same conversion to every model and all three seeds, and disclose the deviation from the default protocol.

Inspect the generated configuration:

```bash
python scripts/prepare_dior.py --help
python -c "import yaml; print(yaml.safe_load(open('configs/datasets/dior.yaml', encoding='utf-8')))"
```

### 3.2 AI-TOD-v2

Paper-fixed image-identifier counts:

| Split | Images | Original split record |
|---|---:|---|
| train | 11,214 | `aitodv2_train.json` |
| val | 5,607 | `aitodv2_val.json` |
| test | 11,215 | `aitodv2_test.json` |

```bash
python scripts/prepare_aitodv2.py \
  --source DATA/AI-TOD-v2 \
  --output datasets/AI-TOD-v2 \
  --mode symlink
```

The script discovers common official JSON names under `source/annotations`. Pass paths explicitly to remove version ambiguity:

```bash
python scripts/prepare_aitodv2.py \
  --source DATA/AI-TOD-v2 \
  --output datasets/AI-TOD-v2 \
  --train-json DATA/AI-TOD-v2/annotations/aitodv2_train.json \
  --val-json DATA/AI-TOD-v2/annotations/aitodv2_val.json \
  --test-json DATA/AI-TOD-v2/annotations/aitodv2_test.json \
  --mode symlink
```

The AI-TOD-v2 training route target merges very-tiny and tiny into P3, assigns small to P4, and assigns medium to P5; evaluation retains all four original intervals.

By default, AI-TOD-v2 instances with `iscrowd=1` are excluded from YOLO training labels while the official COCO ground truth is preserved; `--include-crowd` changes this choice. COCO category IDs are sorted numerically and mapped to contiguous YOLO class IDs. The authoritative mapping is stored in the generated `category_mapping.json`.

## 4. Paper-fixed training protocol

### 4.1 Optimization and augmentation

| Setting | Paper-fixed value |
|---|---|
| Initialization | Train from scratch without pretrained weights |
| Input | 640 x 640 |
| epochs | 200 |
| optimizer | SGD |
| Initial learning rate | 0.005 |
| momentum | 0.937 |
| weight decay | `5e-4` |
| warmup | 3 epochs |
| Source batch size | 16; each source produces clean and degraded forwards |
| Mosaic | probability 1.0 for epochs 1-180; disabled for the last 20 epochs |
| HSV hue | `U(-0.015, 0.015)` |
| HSV saturation multiplier | `U(0.30, 1.70)` |
| HSV value multiplier | `U(0.60, 1.40)` |
| affine scale | `U(0.50, 1.50)` |
| Translation | up to 0.10 of image width or height |
| horizontal flip | 0.50 |
| Disabled | vertical flip, rotation, shear, perspective, MixUp, Copy-Paste |
| Main-experiment seeds | 2023, 2024, 2025 |

Spatial and color augmentation must be sampled once for the clean/degraded pair. Both views share the crop, scale, flip, and labels. Degradation randomness is keyed by `training seed + epoch + image identifier` so different models use paired random numbers.

The official CLI enables AMP and `deterministic=True` by default, uses eight dataloader workers, and sets `nbs=batch=16` to avoid additional gradient accumulation. AMP can be disabled with `--no-amp`, but the choice must remain identical across all three seeds and every comparison.

The official source complements the paper's initial learning rate and warmup duration with a fully pinned schedule: linear decay (`cos_lr=false`), `lrf=0.01`, `warmup_momentum=0.8`, `warmup_bias_lr=0.1`, and `box/cls/dfl=7.5/0.5/1.5`. These are canonical release settings, not estimates inferred from the reported results.

### 4.2 Paired degradation

Each source image produces one clean view and one degraded view at a 1:1 ratio. The two detection losses are averaged before one optimizer update.

clean reference:

```text
relative GSD = 1
MTF = 0.50
SNR = 30 dB
```

degraded view:

| Mode | Probability | Relative GSD | MTF | SNR |
|---|---:|---:|---:|---:|
| blur only | 0.35 | 1 | `U(0.15, 0.45)` | reference 30 dB |
| noise only | 0.35 | 1 | reference 0.50 | `U(10, 28)` dB |
| blur + noise | 0.30 | 1 | `U(0.15, 0.45)` | `U(10, 28)` dB |

A dash in paper Table 3 means that the corresponding operator is not applied; its descriptor value uses the clean reference. Each GSD/MTF/SNR field is independently masked with `p_drop=0.25`. Relative GSD remains 1 during training. Resolution changes, motion blur, disk PSF, anisotropic blur, speckle noise, and stripe noise are not used for training.

The official `PairedDegradationGenerator` shares one field-level dropout mask across a clean/degraded pair so the consistency loss does not also mix different missingness patterns. Gaussian blur uses reflection padding. Both behaviors are source-defined parts of the public protocol.

The official RNG key uses the dataloader-provided `im_file` string, usually an absolute path. Moving or renaming the dataset root therefore changes the degradation sequence, so all three seeds and model comparisons must use the same normalized dataset path.

The paper fixes the image-formation order as sampling, blur, then shot noise:

1. Relative GSD `r`: antialiased bicubic downsampling to `round(640/r)`, followed by bicubic restoration to 640.
2. Gaussian PSF: `sigma = sqrt(-2*log(mtf))/pi`, with kernel width `2*ceil(3*sigma)+1`.
3. Poisson shot noise: `lambda = mean(x)*10^(snr/10)/mean(x^2)`, output `Poisson(lambda*x)/lambda`, clipped to `[0,1]`.

Controlled and OOD transforms remain `float32 [0,1]` from this point to the model tensor and are never round-trip quantized to uint8. Source images retain their native precision only during decoding.

### 4.3 Fixed routing and loss values

```text
eta = 1
tau = 1
K_visual = 4
lambda_scale = 0.1
lambda_consistency = 0.1
p_drop = 0.25
```

The official `PIRSBFRLoss` averages the scale KL over clean and degraded views. This is the canonical public training behavior.

## 5. Three-seed training

Inspect the final CLI arguments first:

```bash
pir-train --help
```

DIOR:

```bash
pir-train --data configs/datasets/dior.yaml --scale-mode dior --seed 2023
pir-train --data configs/datasets/dior.yaml --scale-mode dior --seed 2024
pir-train --data configs/datasets/dior.yaml --scale-mode dior --seed 2025
```

The default run directories are `runs/pir_sbfr/dior_seed{2023,2024,2025}`.

The full model reads `configs/pir_sbfr.yaml` by default. Pass an official public ablation configuration explicitly, for example:

```bash
pir-train \
  --data configs/datasets/dior.yaml \
  --config configs/ablations/a6_analytic_all.yaml \
  --scale-mode dior \
  --seed 2023
```

`configs/ablations/` provides the official public configurations for A0, A1, A2, A3, A4, A5, A6, A9, and A10. A7 direct concatenation and A8 FiLM are reported comparison controls but are not exposed as standalone YAML configurations in this release.

AI-TOD-v2:

```bash
pir-train --data configs/datasets/aitodv2.yaml --scale-mode aitodv2 --seed 2023
pir-train --data configs/datasets/aitodv2.yaml --scale-mode aitodv2 --seed 2024
pir-train --data configs/datasets/aitodv2.yaml --scale-mode aitodv2 --seed 2025
```

The default run directories are `runs/pir_sbfr/aitodv2_seed{2023,2024,2025}`. Use different `--device` values for parallel runs, and never allow multiple processes to write to the same `--project/--name`.

Retain the following information for every run:

- fully resolved training arguments;
- SHA-256 hashes of split files;
- Python, PyTorch, CUDA, Ultralytics, and GPU model;
- seed, checkpoint path, and per-epoch validation metrics;
- parameter count, GFLOP profiler name, and profiler version.

The paper reports the arithmetic mean and sample standard deviation over three runs, using `n-1` in the denominator. Do not confuse population standard deviation with the paper's `mean +/- sample SD`.

## 6. Inference and evaluation

### 6.1 Single-checkpoint prediction

```bash
pir-predict \
  --weights runs/PIR_RUN/weights/best.pt \
  --source datasets/DIOR/images/test
```

With per-image acquisition descriptors:

```bash
pir-predict \
  --weights runs/PIR_RUN/weights/best.pt \
  --source datasets/DIOR/images/test \
  --metadata metadata.json
```

The metadata JSON uses an absolute path, file name, or stem as each key and a descriptor object as its value:

```json
{
  "000001.jpg": {
    "gsd": 1.0,
    "mtf": 0.5,
    "snr": 30.0,
    "availability": [1, 1, 1]
  },
  "000002": {
    "gsd": 2.0,
    "availability": [1, 0, 0]
  }
}
```

When `availability` is omitted, the CLI derives the mask from the presence of `gsd/mtf/snr` keys. An image with no record receives reference values and an all-zero mask. The model tensor interface is fixed as `metadata=[gsd, mtf_or_psf_sharpness, snr_db]` plus an identically shaped `availability`. Missing fields must be marked with the mask; never use a zero value as a fake valid measurement. When all three fields are missing, the physical prior is `[1,1,1]` and the model reduces to visual routing.

### 6.2 COCO-style evaluation

DIOR:

```bash
pir-eval \
  --weights runs/PIR_RUN/weights/best.pt \
  --annotations datasets/DIOR/annotations/test.json \
  --images datasets/DIOR/images/test \
  --dataset dior
```

AI-TOD-v2:

```bash
pir-eval \
  --weights runs/PIR_RUN/weights/best.pt \
  --annotations datasets/AI-TOD-v2/annotations/test.json \
  --images datasets/AI-TOD-v2/images/test \
  --category-mapping datasets/AI-TOD-v2/category_mapping.json \
  --dataset aitodv2
```

Use `pir-eval --help` as the authority for CLI paths and optional output names. AP in the paper is COCO-style AP over IoU 0.50:0.95, not AP50.

The official evaluation path defaults to `conf=0.001` and NMS IoU `0.70`; the visualization-oriented `pir-predict` defaults to `conf=0.25`. Formal evaluation must use `pir-eval` or record explicit overrides, and the two defaults must not be mixed.

Evaluate each of the three DIOR checkpoints:

```bash
for seed in 2023 2024 2025; do
  pir-eval \
    --weights "runs/pir_sbfr/dior_seed${seed}/weights/best.pt" \
    --annotations datasets/DIOR/annotations/test.json \
    --images datasets/DIOR/images/test \
    --dataset dior \
    --output "output/dior_seed${seed}_predictions.json" \
    --metrics-output "output/dior_seed${seed}_metrics.json"
done
```

Evaluate the three AI-TOD-v2 seeds:

```bash
for seed in 2023 2024 2025; do
  pir-eval \
    --weights "runs/pir_sbfr/aitodv2_seed${seed}/weights/best.pt" \
    --annotations datasets/AI-TOD-v2/annotations/test.json \
    --images datasets/AI-TOD-v2/images/test \
    --category-mapping datasets/AI-TOD-v2/category_mapping.json \
    --dataset aitodv2 \
    --output "output/aitodv2_seed${seed}_predictions.json" \
    --metrics-output "output/aitodv2_seed${seed}_metrics.json"
done
```

Compute the arithmetic mean and sample SD over three seeds, with an optional paired-t interval using matching seed order:

```bash
python scripts/summarize_seeds.py \
  --candidate output/dior_seed2023_metrics.json output/dior_seed2024_metrics.json output/dior_seed2025_metrics.json \
  --output output/dior_three_seed_summary.json
```

When comparing two models, pass three `--reference` files in the same seed order.

Scale intervals:

- DIOR: small `[0,32^2)`, medium `[32^2,96^2)`, and large `[96^2,+inf)`, measured as pixel area on the 640 x 640 model input.
- AI-TOD-v2: very-tiny `[2^2,8^2)`, tiny `[8^2,16^2)`, small `[16^2,32^2)`, medium `[32^2,64^2)`.

DIOR COCO ground truth retains original-image coordinates. Before applying the 32^2/96^2 thresholds, the evaluator multiplies GT/DT diagnostic area by `gain^2`, where the letterbox gain is `min(640/width,640/height)`. Box matching remains in original-image coordinates. Default `maxDets` values are `[1,10,100]` for DIOR and `[1,100,1500]` for AI-TOD-v2. The latter must not use COCO's 100-detection cap.

### 6.3 Paired bootstrap

The paper resamples two models jointly by image identifier with replacement. Repeatedly sampled images must be duplicated while remapping ground truth and both models' predictions, followed by a full COCOeval rerun. Averaging a supposed "per-image AP" is invalid.

```bash
python scripts/bootstrap_paired_coco.py \
  --annotations datasets/DIOR/annotations/test.json \
  --predictions-a outputs/yolo_dsf_seed2023.json \
  --predictions-b outputs/pir_full_seed2023.json \
  --area-mode dior \
  --dior-input-size 640 \
  --replicates 10000 \
  --seed 20260718 \
  --confidence 0.95 \
  --output outputs/bootstrap_dior.json
```

The paper uses 10,000 replicates, seed `20260718`, and a 2.5%/97.5% percentile interval. Flight experiments resample complete videos rather than frames, but the source flight data is not public.

## 7. Robustness protocol

### 7.1 Complete 27-cell factorial grid

Evaluate every combination:

```text
GSD = {1, 2, 3}
MTF = {0.50, 0.30, 0.15}
SNR = {30, 20, 10} dB
```

Average seeds 2023/2024/2025 within each cell. Across YOLOv11n, YOLO-DSF, and PIR-SBFR Full, the paper obtains 81 model-condition means. Reporting only the worst cell or a one-variable slice is insufficient.

The 27 cells for one checkpoint can be evaluated after loading the model once:

```bash
python scripts/robustness_grid.py \
  --weights runs/pir_sbfr/dior_seed2023/weights/best.pt \
  --annotations datasets/DIOR/annotations/test.json \
  --images datasets/DIOR/images/test \
  --dataset dior \
  --seed 20260718 \
  --output-dir output/grid/seed2023
```

The complete three-seed PIR-SBFR Full grid can also be run with the explicit loop below. `pir-eval` fixes Poisson randomness using the image ID and `--degradation-seed`, then sends the current cell's correct descriptor to the router:

```bash
for seed in 2023 2024 2025; do
  for gsd in 1 2 3; do
    for mtf in 0.50 0.30 0.15; do
      for snr in 30 20 10; do
        tag="seed${seed}_g${gsd}_q${mtf}_s${snr}"
        pir-eval \
          --weights "runs/pir_sbfr/dior_seed${seed}/weights/best.pt" \
          --annotations datasets/DIOR/annotations/test.json \
          --images datasets/DIOR/images/test \
          --dataset dior \
          --gsd "${gsd}" --mtf "${mtf}" --snr "${snr}" \
          --degradation-seed 20260718 \
          --output "output/grid/${tag}_predictions.json" \
          --metrics-output "output/grid/${tag}_metrics.json"
      done
    done
  done
done
```

Each baseline must also generate COCO predictions for identical cells and per-image degradation seeds. Evaluate them uniformly with `pir-eval --predictions BASELINE.json --annotations ... --dataset dior` to form the paper's 81 model-condition observations. The official YOLOv11n baseline training entry point is `scripts/train_yolo11n_baseline.py`, and `configs/ablations/a0_yolo_dsf.yaml` is the public YOLO-DSF baseline configuration for this release.

### 7.2 Nine conditions outside the training distribution

| Condition | Fixed setting | Public regeneration behavior |
|---|---|---|
| disk defocus | radius 3 px | Deterministic official disk-kernel and reflection-border implementation |
| motion blur | 9 px, random orientation | Deterministic for a fixed command seed; realized orientation is recorded |
| anisotropic Gaussian | `sigma_x=2.5, sigma_y=0.6`, random orientation | Deterministic for a fixed command seed; realized orientation is recorded |
| speckle | `sigma=0.12` | Deterministic official multiplicative-noise implementation |
| stripe + read noise | amplitude 0.08, read-noise `sigma=0.02` | Deterministic official sinusoid-plus-read-noise implementation |
| unseen GSD | 1.5x | Deterministic for a fixed command seed |
| unseen GSD | 2.5x | Deterministic for a fixed command seed |
| unseen GSD | 4x | Deterministic for a fixed command seed |
| joint | 2.5x + unseen PSF + unseen noise | Deterministic for a fixed command seed; newly generated images are distinct from the non-redistributed archived realization |

These conditions must not influence training, early stopping, hyperparameter selection, or model selection.

Official entry point for regenerating the nine OOD conditions:

```bash
python scripts/evaluate_unseen.py \
  --weights runs/pir_sbfr/dior_seed2023/weights/best.pt \
  --annotations datasets/DIOR/annotations/test.json \
  --images datasets/DIOR/images/test \
  --dataset dior \
  --seed 20260718 \
  --output-dir output/unseen/seed2023
```

Each result records `approximate=true` to distinguish a newly synthesized stress-test realization from the non-redistributed archived evaluation images. The flag does **not** mean that the PIR-SBFR model or public implementation is approximate. The output also records the command seed, realized random orientation, and equivalent MTF/SNR mapping.

### 7.3 Mixed-degradation metadata controls

If already degraded images and correct per-image descriptors of the same type as the paper are available, use the command below. Replace the example values after `--training-mean` with the actual training-set mean:

```bash
python scripts/metadata_controls.py \
  --weights runs/pir_sbfr/dior_seed2023/weights/best.pt \
  --annotations /path/to/mixed/annotations.json \
  --images /path/to/mixed/images \
  --metadata /path/to/mixed/metadata.json \
  --training-mean 1.0 0.35 23.4 \
  --dataset dior \
  --output-dir output/metadata_controls/seed2023
```

When the non-redistributed mixed set is unavailable, pass `--synthesize-controlled` to construct a deterministic, explicitly labeled synthetic control set. The script covers Table 12's correct descriptors, +/-10/20/50% multiplicative errors, 25/50/100% missingness, training-set mean, and cross-image shuffling. It reports only within-set differences relative to the correct condition and does not present regenerated samples as the archived samples behind the paper table.

## 8. Parameter, GFLOP, and latency calibration

Paper deployment anchors at 640 x 640:

| Model | Params (M) | GFLOPs | FP16 latency (ms) | FPS | AP |
|---|---:|---:|---:|---:|---:|
| YOLOv11n | 2.584 | 6.47 | 2.347 | 426.1 | 57.31 |
| YOLO-DSF | 3.628 | 8.43 | 3.079 | 324.8 | 62.98 |
| DSF + scale supervision + P5 bypass | 3.691 | 8.51 | 3.117 | 320.8 | 63.69 |
| PIR-SBFR Full | 3.942 | 8.82 | 3.313 | 301.8 | 65.52 |

Direct THOP measurements for the repository's default `PIRSBFRModel(nc=20)` on a real 640 x 640 input:

```text
parameters = 3,944,613 = 3.944613 M
THOP MACs = 4,417,637,728
GFLOPs (2 x MACs) = 8.835275456
```

The official release model profiles at 3.944613M parameters and 8.835275456 direct-profiled GFLOPs at 640. These values are 0.0026M and 0.0153G above the rounded 3.942M/8.82G paper entries, a 0.17% difference attributable to reporting precision and profiler convention. Ultralytics `get_flops(model,imgsz=640)` first profiles 32 x 32 and multiplies by 400 according to image area. This extrapolation incorrectly multiplies **per-image fixed costs**, including post-GAP fully connected layers and MoE gates/experts, by 400 and therefore returns an inflated 9.0798848 GFLOPs. Do not use that extrapolated value as the primary release measurement.

GFLOPs still depend on how the profiler counts multiply-adds, interpolation, activations, and dynamic branches. Every report must state the input size, profiler name, and profiler version. Never alter only the displayed value to "match" the paper.

Verification command:

```bash
python -c "import torch; from thop import profile; from pir_sbfr.models import PIRSBFRModel; m=PIRSBFRModel(nc=20).eval(); macs,params=profile(m,inputs=(torch.zeros(1,3,640,640),),verbose=False); print(int(params)); print('MACs',int(macs)); print('GFLOPs',2*macs/1e9)"
```

The paper's latency protocol uses FP16, batch 1, 640 x 640, a preallocated GPU input, 200 warm-up iterations, and five runs of 1,000 forwards each. CUDA is synchronized before and after each timed run, and the median of the five run-wise means is reported:

```bash
python scripts/benchmark.py \
  --factory pir_sbfr.models:PIRSBFRModel \
  --checkpoint runs/PIR_RUN/weights/best.pt \
  --allow-pickled-module \
  --device cuda:0 \
  --image-size 640 \
  --warmup 200 \
  --runs 5 \
  --iterations 1000 \
  --output outputs/benchmark.json
```

An Ultralytics trainer checkpoint contains a pickled model object. Use `--allow-pickled-module` only when the checkpoint source is trusted. Omit `--checkpoint` and this flag when benchmarking architecture-only latency.

The timing includes only the network forward and excludes disk access, image decoding, resize, host-to-device transfer, NMS, and serialization. Latency measured on different GPUs, power limits, drivers, or clock states should not be compared directly with the paper's RTX 4090 value.

## 9. Minimum acceptance checklist

- [ ] Split image counts match paper Table 2.
- [ ] All three seeds are trained from scratch, with configuration and environment saved.
- [ ] Clean/degraded views share geometric and color augmentation, and degradation RNG is reproducible.
- [ ] Metadata dropout operates independently per field rather than dropping the whole descriptor.
- [ ] Fully missing metadata gives `rho_phy == 1` with no NaNs.
- [ ] Equal weights `w=[1/3,1/3,1/3]` satisfy `3*w_i*P_i == P_i`.
- [ ] As degradation increases, analytic P3 reliability decreases faster than P4/P5 reliability.
- [ ] The P5 bypass is added after the FPN-PAN F5 output.
- [ ] AP uses COCO 0.50:0.95 and dataset-appropriate scale intervals.
- [ ] Three-seed results report sample SD.
- [ ] Bootstrap is paired by image ID and reruns COCOeval.
- [ ] Parameter/GFLOP reports identify the profiler; latency follows the forward-only protocol exactly.
- [ ] Non-redistributed boundaries for flight, mixed-control, and joint OOD experiments are clearly disclosed.
