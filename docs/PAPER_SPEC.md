# PIR-SBFR Official Implementation Specification

This document maps the July 20, 2026 PIR-SBFR paper revision to the official public implementation. The paper PDF is not redistributed in this repository. The source code and configurations are authoritative for release-level details that the paper summarizes rather than enumerates.

## 1. Implementation conclusions

The paper defines the following components sufficiently:

- P3/P4/P5 strides;
- DRFB channel compression, two dilation branches, and residual form;
- acquisition descriptors, degradation coordinates, and the analytic reliability prior;
- the visual-residual mixture form, scale-distribution supervision, and logit-space fusion;
- `3*w_i` feature reweighting and the post-FPN-PAN P5 bypass;
- the abstract FACH formulas for the channel gate, dynamic mixture, identity path, and final task separation;
- paired degradation, the total loss, fixed hyperparameters, and major experimental protocols.

For release-level components that the paper summarizes at method level, the official source code provides the complete executable definitions:

- complete executable YOLO-DSF topology, layer indices, and width/depth settings;
- the public visual-only SBFR baseline configuration;
- DRFB `Conv_cat`, exact insertion locations, count, convolution groups, and final activation;
- channel-alignment output width and operator;
- visual-expert `T_k` depth, hidden width, activation, and numeric interpretation of "bounded";
- FACH expert count, `J_j`, `C_coupled`, cross-level sharing, and classification/regression tower details;
- cross-view positive matching and the class-probability definition used by consistency KL.

The following experiment artifacts are not redistributed in the public repository and are treated as data-availability boundaries rather than implementation gaps:

- pretrained checkpoints;
- sample-level degradation records for the mixed-degradation control;
- private flight videos, annotations, and telemetry;
- the joint-OOD PSF orientation.

The public implementation is the canonical executable form of the paper method. The paper supplies the method-level presentation and reported results; the official source supplies the release-level architectural and numerical definitions. Differences between rounded paper resource counts and direct local profiling are documented below.

## 2. Overall computation graph

The connectivity shown in paper Figures 1 and 3 can be written as:

```text
I
  -> YOLO11n + DRFB backbone
  -> P3(stride=8), P4(stride=16), P5(stride=32)
       |                  |
       |                  +------------------------------+
       |                                                 |
       +-> channel alignment A3/A4/A5                    |
           -> visual residual delta_vis                  |
           -> analytic reliability rho_phy               |
           -> logit-space fusion                         |
           -> weights w3,w4,w5, sum=1                    |
           -> P_i_rw = 3*w_i*A_i(P_i)                    |
           -> original FPN-PAN                           |
           -> F3, F4, F5 -------------------------------(+) <- projection(P5)
                                                        |
                                                        v
                                                       FACH
                                                        |
                                                        v
                                                   cls + reg
```

The P5 structural bypass does not add P5 before routing or after FACH. It starts from the original P5, passes through a projection that is not multiplied by `w5`, and is added elementwise after FPN-PAN has produced F5. Official code locations:

- backbone: `src/pir_sbfr/models/blocks.py::YOLO11DRFBBackbone`
- PIR router and FPN-PAN: `src/pir_sbfr/models/router.py::PIRSBFRNeck`
- FACH: `src/pir_sbfr/models/blocks.py::FACH`
- YOLO Detect: `src/pir_sbfr/models/detector.py::PIRSBFRModel`

## 3. Equations 1-24 mapping

| Paper equation | Meaning | Official implementation | Agreement and source definitions |
|---|---|---|---|
| (1)-(2) | `I -> B_DRFB -> {P_i} -> N_PIR -> {F_i} -> H_FACH` | `PIRSBFRModel._forward_detector` | Connection order matches; final decoding reuses Ultralytics `Detect` |
| (3) | `1x1 C->C/2` channel compression | `DRFB.reduce` | Conv+BN+SiLU |
| (4) | 3x3 branches with dilation 2/3 | `DRFB.branches` | Parallel standard/group convolutions with source-fixed group counts |
| (5) | Compression of concatenated responses | `DRFB.compress` | Official release uses 1x1 Conv+BN+SiLU |
| (6) | Restore to C and add identity | `DRFB.restore`, `DRFB.forward` | Official release uses BN without pre-add activation, followed by post-add SiLU |
| (7) | `m=[g,q,s]`, `a=[a_g,a_q,a_s]` | `PhysicalReliabilityPrior` input | Code names should be interpreted as GSD/MTF/SNR to avoid confusion with scale target `q` |
| (8) | Three non-negative degradation coordinates | `degradation_coordinates` | Mathematically aligned; missing values are replaced by references before operations to prevent NaNs |
| (9)-(10) | Stride-monotonic analytic prior | `PhysicalReliabilityPrior.forward` | No trainable parameters; fixed kappa |
| (11) | Three-level channel alignment + GAP concatenation | `PIRSBFRNeck.align`, `VisualResidualRouter.forward` | Official aligned width is 64 |
| (12) | K-branch visual mixture | `VisualResidualRouter`, `ResidualExpert` | Official release fixes K=4, MLP hidden width 280, and tanh bound 4 |
| (13) | `q_hat=softmax(delta_vis)` | `VisualResidualRouter.forward` | Matches |
| (14) | Labeled scale distribution | `object_scale_distribution` | `epsilon=1e-6`; computed from boxes after augmentation |
| (15) | `D_KL(q || q_hat)` | `scale_kl_divergence` | Batch mean; official loss averages clean and degraded views |
| (16) | Visual + physical logit fusion | `PIRSBFRNeck.forward` | Code uses `log(clamp_min(rho,eps))`; paper writes `log(rho+eps)` |
| (17) | `P_i_rw=3*w_i*Pbar_i` | `PIRSBFRNeck.forward` | Matches |
| (18) | Original FPN-PAN | `FPNPAN` | Official release defines the complete topology directly in Python |
| (19) | `F5 <- F5+A5(P5)` | `p5_projection` | Independent projection; does not share weights with alignment A5 |
| (20) | Symmetric KL + normalized-box L1 | `PIRSBFRLoss.consistency_loss` | Clean TaskAligned foreground defines shared R; class softmax; SKL includes 0.5 |
| (21) | Paired detection + scale + consistency | `PIRSBFRLoss.__call__` | Equation weights match; multiplied by batch size for the Ultralytics loss-return convention |
| (22) | FACH channel gate | `FeatureAwareCoupling.channel_gate` | Independent Linear(C,C)+sigmoid per level |
| (23) | Dynamic experts + identity | `FeatureAwareCoupling.expert_gate/experts` | Three source-defined separable-conv experts per level |
| (24) | Classification/regression separation after coupled feature | `FeatureAwareCoupling.coupled` + `Detect` | FACH creates shared features; official Detect supplies cls/reg branches and DFL |

## 4. DRFB specification

For input:

\[
X^{(l-1)}\in\mathbb{R}^{H\times W\times C}
\]

Paper equation (3):

\[
X_{red}^{(l)}=\sigma\left(BN\left(Conv_{1\times1}^{C\rightarrow C/2}(X^{(l-1)})\right)\right)
\]

Equation (4):

\[
D_r^{(l)}=\sigma\left(BN\left(Conv_{3\times3,dilation=r}(X_{red}^{(l)})\right)\right),\quad r\in\{2,3\}
\]

Equations (5)-(6):

\[
X_{ctx}^{(l)}=Conv_{cat}([D_2^{(l)},D_3^{(l)}])
\]

\[
X^{(l)}=X^{(l-1)}+\phi_{1\times1}^{C/2\rightarrow C}(X_{ctx}^{(l)})
\]

### 4.1 Paper-level structure

```text
HxWxC
 -> 1x1 Conv C->C/2 + BN + SiLU/ReLU
 -> parallel 3x3 dilation=2 and dilation=3 + BN + SiLU/ReLU
 -> concatenation/compression
 -> 1x1 Conv C/2->C + BN + SiLU/ReLU
 -> residual add with original input
 -> HxWxC
```

The paper presents the module at operator level. The official source supplies release-level definitions for:

- the role, kernel, normalization, and activation of `Conv_cat`;
- standard/grouped convolution behavior in each dilation branch;
- activation placement around the residual addition;
- the number of DRFB modules and their exact YOLO11n insertion points.

### 4.2 Official source definition

- Place one DRFB at each P3 and P4 stage output; none at P5.
- `P3 channels=128` with branch groups `(1,2)`.
- `P4 channels=128` with branch groups `(2,2)`.
- Implement `Conv_cat` as `1x1: C -> C/2` Conv+BN+SiLU.
- Implement restore as `1x1: C/2 -> C` Conv+BN with no pre-add activation.
- Apply SiLU after identity addition.

These group counts and operator placements are canonical settings of the official public release and should be cited as source-defined values.

## 5. YOLO11n backbone and FPN-PAN

The official backbone materializes nano scaling directly:

```text
3 -> Conv s2, 16
16 -> Conv s2, 32 -> C3k2, 64
64 -> Conv s2, 64 -> C3k2, 128 -> DRFB -> P3
P3 -> Conv s2, 128 -> C3k2, 128 -> DRFB -> P4
P4 -> Conv s2, 256 -> C3k2, 256 -> SPPF -> C2PSA -> P5
```

P3/P4/P5 channel counts are `(128,128,256)`, with strides `(8,16,32)`.

After PIR alignment to 64 channels, the official FPN-PAN is:

```text
P5 upsample + P4 -> C3k2 -> TD4(128)
TD4 upsample + P3 -> C3k2 -> F3(64)
down(F3) + TD4 -> C3k2 -> F4(128)
down(F4) + aligned P5 -> C3k2 -> F5(256)
```

The paper summarizes this stage as "original FPN-PAN operations"; the topology and channel widths above are the authoritative release definition in this repository.

## 6. Analytic physical reliability

### 6.1 Inputs and reference

Paper equation (7):

\[
\mathbf{m}=[g,q,s]^T,\quad \mathbf{a}=[a_g,a_q,a_s]^T\in\{0,1\}^3
\]

- `g`: GSD or relative GSD;
- `q`: Nyquist MTF or PSF-derived sharpness;
- `s`: SNR dB;
- `a`: field-level availability mask.

Fixed reference:

\[
g_{ref}=1,\quad q_{ref}=0.50,\quad s_{ref}=30
\]

### 6.2 Degradation coordinates

Equation (8):

\[
d_g=\max(0,\log(g/g_{ref}))
\]

\[
d_q=\max(0,1-q/q_{ref})
\]

\[
d_s=\max(0,(s_{ref}-s)/s_{ref})
\]

The code first replaces fields whose mask is zero with reference values, then performs logarithms and division, and finally multiplies by the mask. This matches the paper for valid fields and avoids `log(0)` when a missing GSD uses a zero placeholder.

### 6.3 level-wise prior

Equations (9)-(10):

\[
\rho_i^{phy}=\exp[-\kappa_i^T(\mathbf{a}\odot\mathbf{d})]
\]

\[
\kappa_i=\frac{8}{r_i}[1,1,1]^T,\quad r_i\in\{8,16,32\}
\]

Therefore:

```text
kappa_3 = [1.00, 1.00, 1.00]
kappa_4 = [0.50, 0.50, 0.50]
kappa_5 = [0.25, 0.25, 0.25]
```

This module has no trainable parameters. When every field is missing, `a=0` gives exactly `rho_phy=[1,1,1]`.

## 7. Scale-supervised visual residual

Equation (11):

\[
\bar P_i=A_i(P_i),\quad
z=[GAP(\bar P_3);GAP(\bar P_4);GAP(\bar P_5)]
\]

Each official `A_i` is an independent Ultralytics `1x1 Conv+BN+SiLU` that outputs 64 channels, so `z` has dimension 192.

Equations (12)-(13):

\[
\pi=softmax(W_\pi z)
\]

\[
\delta^{vis}=\sum_{k=1}^{K}\pi_kT_k(z),\quad K=4
\]

\[
\hat q=softmax(\delta^{vis})
\]

The paper represents `T_k` abstractly. The official source instantiates each expert as:

```text
Linear(192,280) -> SiLU -> Linear(280,3) -> tanh -> multiply by 4
```

The `tanh*4` operation makes the paper's "bounded visual correction" concrete. Bound 4 and hidden width 280 are canonical source-level settings of this release.

### 7.1 DIOR scale target

For a 640 x 640 input:

```text
small:  area < 32^2            -> P3
medium: 32^2 <= area < 96^2    -> P4
large:  area >= 96^2           -> P5
```

Equation (14):

\[
q=\frac{[n_S,n_M,n_L]^T+\epsilon}{n_S+n_M+n_L+3\epsilon}
\]

### 7.2 AI-TOD-v2 scale target

```text
very-tiny 2-8 px + tiny 8-16 px -> P3
small 16-32 px                  -> P4
medium 32-64 px                 -> P5
```

The code computes pixel area/absolute size from normalized `xywh` on the actual post-augmentation network input. `epsilon=1e-6`; images without objects receive a uniform target.

Equation (15):

\[
L_{scale}=D_{KL}(q\|\hat q)
\]

The official loss takes a batch mean and then averages the clean and degraded `L_scale` values.

### 7.3 Evaluation-area coordinates and maxDets

DIOR COCO conversion retains original-image coordinates, while the paper's small/medium/large definitions are measured after 640 x 640 resize/letterbox. `src/pir_sbfr/evaluation/coco.py` uses per-image `gain=min(640/W,640/H)`, multiplies GT/DT diagnostic area by `gain^2`, and then applies the 32^2/96^2 thresholds. Bounding-box matching remains in original-image coordinates.

Evaluator defaults:

```text
DIOR maxDets = [1,10,100]
AI-TOD-v2 maxDets = [1,100,1500]
```

AI-TOD-v2 is a dense tiny-object dataset and must not use COCO's default 100-detection cap per image. The underlying `COCOeval` treats both area-range endpoints as inclusive, so instances exactly on boundaries such as 32^2 or 96^2 may enter two adjacent scale diagnostics. Overall AP is unaffected. This is an explicit convention inherited from official COCO matching and accumulation.

## 8. Logit-space fusion and P5 bypass

Equation (16):

\[
w_i=
\frac{\exp[(\delta_i^{vis}+\eta\log(\rho_i^{phy}+\epsilon))/\tau]}
{\sum_{j=3}^{5}\exp[(\delta_j^{vis}+\eta\log(\rho_j^{phy}+\epsilon))/\tau]}
\]

Fixed values are `eta=1` and `tau=1`. The official code uses:

```python
physical_logits = eta * log(rho_phy.clamp_min(eps))
weights = softmax((delta_vis + physical_logits) / temperature)
```

`clamp_min(eps)` is the official source-level stabilization for the paper's `rho+eps` expression and prevents extreme exponential underflow.

Equation (17):

\[
P_i^{rw}=3w_i\bar P_i
\]

The factor 3 preserves feature magnitude when `w_i=1/3`. In the official implementation, each `w_i` is a `[B]` image-level scalar broadcast over channels and spatial positions.

Equations (18)-(19):

\[
(F_3,F_4,F_5)=N_{FPN-PAN}(P_3^{rw},P_4^{rw},P_5^{rw})
\]

\[
F_5\leftarrow F_5+A_5(P_5)
\]

Figure 3 labels both channel alignment and bypass projection as `A5`, while the text calls it a "separate P5 projection." The code uses an independent `Conv(256,256,1)` that does not share weights with alignment `Conv(256,64,1)`; their output dimensions also differ.

## 9. Missing metadata and paired consistency

During training, each acquisition field and its mask are independently dropped with `p_drop=0.25`. This random process belongs to the dataloader/trainer, not `PhysicalReliabilityPrior`. The model is responsible for safe mask consumption and a neutral fallback.

The official `PairedDegradationGenerator` samples one field-level mask for a clean/degraded pair and shares it across both views. Consistency loss therefore does not also mix metadata-missingness differences.

Equation (20):

\[
L_{cons}=\frac{1}{|R|}\sum_{r\in R}
[D_{SKL}(p_r,p'_r)+\|b_r-b'_r\|_1]
\]

The paper gives the consistency objective at method level. The official source defines the following execution details:

- construction of shared set `R` under TaskAligned assignment;
- conversion of YOLO class scores into the distribution used by KL;
- the `1/2` convention in symmetric KL;
- the coordinate representation and normalization used by box L1.

Official source definitions:

- Run TaskAlignedAssigner on the clean view and use its foreground-anchor set as shared `R` for both views.
- Apply softmax to foreground class logits.
- `D_SKL=0.5*(KL(p||p')+KL(p'||p))`;
- Normalize decoded boxes by image width/height and compute four-coordinate L1.
- Return a differentiable zero when no foreground exists.

Equation (21):

\[
L=\frac12(L_{det}(I)+L_{det}(I'))+
\lambda_{scale}L_{scale}+\lambda_{cons}L_{cons}
\]

Fixed values are `lambda_scale=lambda_cons=0.1`. The official detection loss, DFL, and assigner come from Ultralytics `v8DetectionLoss`. Auxiliary losses are multiplied by batch size when added so they remain consistent with Ultralytics' internal loss-reduction and Trainer conventions.

## 10. FACH specification

For neck level `F_i in R^(H_i x W_i x C)`, equation (22) is:

\[
z_i=GAP(F_i)
\]

\[
F_i^c=F_i\odot\sigma(W_cz_i+b_c)
\]

Equation (23):

\[
\phi_i=softmax(W_rz_i+b_r)
\]

\[
F_i^{out}=F_i+\sum_{j=1}^{K_h}\phi_{ij}J_j(F_i^c)
\]

Equation (24):

\[
G_i=C_{coupled}(F_i^{out}),\quad
y_i^{cls}=H_{cls}(G_i),\quad
y_i^{reg}=H_{reg}(G_i)
\]

### 10.1 Structure visible in Figure 4

```text
neck feature
  -> GAP + FC -> routing weights
  -> feature-aware coupling
       -> multiple dynamic branches phi_1 ... phi_k
       -> identity path
       -> dynamic mixture
  -> shared/coupled representation
  -> final cls/reg separation
```

### 10.2 Official source definition

- The three levels use channel widths `(64,128,256)`.
- Each level uses an independent channel gate and expert gate.
- `K_h=3`.
- Each `J_j` is `3x3 depthwise Conv+BN+SiLU` followed by `1x1 Conv+BN+SiLU`.
- Identity is added to the weighted expert mixture before coupled convolution.
- Coupled kernels are `(1,1,3)` for P3/P4/P5 respectively.
- FACH does not generate logits directly; Ultralytics `Detect` subsequently provides cls/reg branches, DFL, and decoding.
- FACH does not receive metadata and has the same structure under correct, noisy, and missing metadata.

These details define the canonical runnable model released with the paper. They complement the paper's abstract FACH equations and are authoritative for public-code execution.

## 11. Official YOLO-DSF and ablation configurations

The paper defines YOLO-DSF as:

```text
DRFB backbone + visual-only SBFR + FACH
```

The official public A0 configuration instantiates this visual-only baseline with the following switches:

```python
PIRSBFRModel(
    router_config={
        "use_physical": False,
        "use_visual": True,
        "p5_bypass": False,
    },
    lambda_scale=0.0,
    lambda_consistency=0.0,
)
```

This is the canonical YOLO-DSF baseline exposed by the official repository and the supported starting point for public ablation runs.

The repository records these choices in `configs/ablations/`:

```text
a0_yolo_dsf.yaml
a1_degradation_augmentation.yaml
a2_scale_p5.yaml
a3_gsd.yaml
a4_mtf.yaml
a5_snr.yaml
a6_analytic_all.yaml
a9_no_consistency.yaml
a10_full.yaml
```

A7 direct concatenation and A8 FiLM are reported comparison controls but are not exposed as standalone YAML configurations in this public release. Their paper-reported values remain part of the experimental record; the executable configuration set focuses on A0-A6, A9, and A10.

The public configurations use the following authoritative mapping. A0 disables paired degradation, the physical route, the bypass, and both auxiliary losses. A1 enables only paired degradation. A2 disables paired degradation but enables the visual route, scale loss, and bypass; its clean-only loss branch still computes scale KL while physical routing and consistency remain disabled. A3-A6 disable the visual route and both auxiliary losses, retain only the specified analytic prior and bypass, and inherit `degradation.enabled=true`, so detection loss still averages clean/degraded pairs. A9 and A10 both use dual routes and differ only in `lambda_consistency=0/0.1`. When the bypass is disabled, the constructor does not register the 66,048 inactive parameters of `p5_projection`.

Paper Table 10 and the official YAML files together define the ablation family; rows are controlled configurations rather than a strictly cumulative sequence:

| ID | Paper name | Official public setting | Release note |
|---|---|---|---|
| A0 | YOLO-DSF | visual-only, no PIR, no bypass | Public baseline: `a0_yolo_dsf.yaml` |
| A1 | + degradation augmentation | A0 structure plus paired degradation | Public configuration: `a1_degradation_augmentation.yaml` |
| A2 | + scale supervision + P5 bypass | visual scale loss + bypass | Public configuration: `a2_scale_p5.yaml` |
| A3-A5 | single physical reliability | matching one-hot `physical_fields`, with bypass | Public GSD/MTF/SNR configurations |
| A6 | analytic GSD+MTF+SNR prior | three-field physical prior, with bypass | Public configuration: `a6_analytic_all.yaml` |
| A7-A8 | concat / FiLM controls | paper-reported metadata-conditioning controls | No standalone YAML in this release |
| A9 | analytic prior + visual residual | dual PIR routes, with bypass | Public configuration: `a9_no_consistency.yaml` |
| A10 | Full | A9 + consistency | Public configuration: `a10_full.yaml` |

## 12. Parameter and GFLOP calibration

### 12.1 Paper anchors

| Model | Params (M) | GFLOPs | Interpretable increment from previous structure |
|---|---:|---:|---|
| YOLOv11n | 2.584 | 6.47 | base |
| PAN-FPN + DRFB/FACH | 3.462 | 8.08 | Aggregate difference from DRFB/FACH and the controlled neck |
| YOLO-DSF | 3.628 | 8.43 | Visual-only SBFR aggregation adds approximately `+0.166M/+0.35G` |
| scale supervision + P5 bypass | 3.691 | 8.51 | `+0.063M/+0.08G` over DSF; scale loss itself has no deployment parameters |
| PIR-SBFR Full | 3.942 | 8.82 | `+0.314M/+0.39G` over DSF |

Paper Table 8 lists the no-consistency control as 3.897M/8.76G and Full as 3.942M/8.82G. Section 4.9 states that the stored no-consistency record used incompatible resource counts, so Table 17 omits that deployment row. Because consistency is training-only, the two variants have the same deployed graph in the official release.

### 12.2 Official release measurements

Default `nc=20`:

| Component | Parameters |
|---|---:|
| YOLO11DRFBBackbone | 1,508,320 |
| PIRSBFRNeck | 1,027,184 |
| FACH | 974,537 |
| Detect | 434,572 |
| **Total** | **3,944,613** |

Selected modules:

| Submodule | Parameters |
|---|---:|
| DRFB P3 | 80,640 |
| DRFB P4 | 62,208 |
| channel alignments | 33,152 |
| visual router | 220,304 |
| analytic prior | 0 |
| FPN-PAN | 707,680 |
| P5 projection | 66,048 |

Direct THOP profiling on a real `(1,3,640,640)` input gives 4,417,637,728 MACs. Counting one multiply-add as two FLOPs gives **8.835275456 GFLOPs**. The direct values are 0.0153G and 0.0026M above the rounded 8.82-GFLOP and 3.942M paper entries, a 0.17% reporting/profiler difference rather than a different model identity.

Ultralytics `get_flops(imgsz=640)` currently returns 9.0798848 GFLOPs, but that helper profiles 32 x 32 and multiplies by 400 according to image area. Post-GAP fully connected layers and MoE gates/experts are fixed per-image costs and must not scale with spatial area. Thus 9.0799G is an inapplicable extrapolation for this architecture and is not the primary calibration value. Every report must state the real profiling input, MAC-to-FLOP convention, and profiler version; differences must not be hidden by changing display precision.

## 13. Paper-fixed experimental settings

### 13.1 Data conversion and input coordinates

The paper provides split identifiers and a 640 x 640 input. The official source defines the complete data conventions as follows:

- Interpret DIOR VOC XML as one-based inclusive coordinates, then convert to zero-based half-open boxes. Retain `difficult` in YOLO training labels by default and record `ignore/difficult` as `iscrowd=1` in COCO ground truth to obtain stock COCOeval's true ignore behavior.
- Exclude AI-TOD-v2 `iscrowd=1` instances from YOLO training labels by default while retaining official COCO ground truth. Sort category IDs numerically, map them to contiguous class IDs, and write `category_mapping.json`.
- Use an aspect-ratio-preserving square letterbox with padding value 114 for inference.
- Use `conf=0.25` for ordinary prediction and `conf=0.001`, NMS IoU 0.70 for COCO evaluation.

These conventions must remain fixed across all three seeds and every comparison.

### 13.2 Training

```text
PyTorch 2.8.0
input 640x640
from scratch
200 epochs
SGD lr=0.005, momentum=0.937, weight_decay=5e-4
warmup=3 epochs
source batch=16, clean/degraded pair=1:1
Mosaic p=1 for epochs 1-180, disabled for final 20
seeds={2023,2024,2025}
eta=1, tau=1, K=4
lambda_scale=0.1, lambda_consistency=0.1
p_drop=0.25 per field
```

The official CLI enables AMP by default, sets `deterministic=True`, uses eight dataloader workers, and sets `nbs=batch=16` to avoid additional gradient accumulation. It explicitly pins `lrf=0.01`, a linear scheduler, `warmup_momentum=0.8`, `warmup_bias_lr=0.1`, and `box/cls/dfl=7.5/0.5/1.5`. These source-level settings complete the paper's compact training description. The degradation RNG key contains the full `im_file` path, so all compared runs should use the same normalized dataset root to preserve identical degradation sequences.

See the root [`REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) for the full augmentation schedule and commands.

### 13.3 controlled grid

```text
relative GSD = {1,2,3}
MTF = {0.50,0.30,0.15}
SNR = {30,20,10} dB
```

All 27 cells must be covered.

### 13.4 unseen OOD

```text
disk radius=3
motion kernel=9, random orientation
anisotropic Gaussian sigma_x=2.5, sigma_y=0.6, random orientation
speckle sigma=0.12
stripe amplitude=0.08 + read noise sigma=0.02
GSD={1.5,2.5,4}
joint=GSD 2.5 + unseen PSF + unseen noise
```

The public script deterministically generates a joint condition for a fixed command seed. The per-image PSF realization behind the archived paper table is not redistributed, so newly generated images must be labeled as a new realization rather than the archived evaluation set.

`src/pir_sbfr/data/degradations.py` provides the authoritative public conventions:

- Apply controlled GSD on a square input already letterboxed to network size. PyTorch bicubic reduction uses `antialias=True`, followed by bicubic restoration to the original network size, matching the paper's sampling order. Results remain `float32 [0,1]` and enter the model without extra quantization.
- Use reflection borders for Gaussian, disk, motion, and anisotropic filters.
- Normalize a binary disk kernel whose pixel centers lie within the radius.
- Require the caller to provide motion/anisotropic angles explicitly; seeded experiment scripts generate and record those angles.
- Define speckle as `x + x*N(0,sigma)`.
- Define stripes as a vertical sinusoid with four periods and random phase, followed by Gaussian read noise.

These definitions make every newly generated OOD run deterministic for a recorded seed. Dataset provenance should still distinguish newly generated stress-test images from non-redistributed archived evaluation images.

## 14. Boundaries of the non-redistributed flight experiment

The paper's flight collection contains eight complete videos, altitudes from 5.2 to 30.0 m, 5,370 annotated frames, and 38,947 instances. It is not a public benchmark, and the paper's Data Availability statement provides no download location.

Relative GSD is obtained from attitude-corrected slant range:

\[
g=\frac{H/\cos\theta}{H_{ref}/\cos\theta_{ref}}
\]

MTF/PSF and SNR are not recorded, so their masks must be zero and must not be inferred from labels or imagery. The numerical values in Tables 14/15 are the authoritative paper record; the public repository preserves the official inference API and missing-field logic but does not redistribute the private videos, annotations, or telemetry needed to rerun those tables.

## 15. External implementations and references

Base code repository directly cited by the paper:

- Jocher, G.; Qiu, J. *Ultralytics YOLO11, Version 11.0.0* (2024): `https://github.com/ultralytics/ultralytics`

The same repository is also used for the paper's YOLOv8 citation. This repository is the official PIR-SBFR release; a separate YOLO-DSF checkpoint and source package are not bundled.

Related method references:

- *Multi-Scale Context Aggregation by Dilated Convolutions*: dilation background for DRFB; the paper provides no DOI or code URL.
- *Deep Residual Learning for Image Recognition*: identity/residual background for DRFB; the paper provides no DOI or code URL.
- *Dynamic Convolution: Attention over Convolution Kernels*, *Dynamic Head*, and *TOOD*: dynamic-coupling background for FACH; the paper provides no code URL.
- GSDDet: DOI `10.1109/TGRS.2023.3309838`.
- FiLM: DOI `10.1609/aaai.v32i1.11671`.
- AI-TOD-v2: DOI `10.1016/j.isprsjprs.2022.06.002`.

The paper PDF and this official repository are complementary release artifacts. The official `scripts/bootstrap_paired_coco.py` implements the documented image-paired evaluation protocol, while the source tree and configurations are authoritative for executable details.
