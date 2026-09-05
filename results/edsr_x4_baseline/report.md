# Phase 5: Model Evaluation & Benchmark

## 1. Experiment Overview

This phase evaluates the trained EDSR-style convolutional neural network for 4× single-image super-resolution.

The model receives a 48×48 low-resolution image and reconstructs a 192×192 high-resolution image.

## 2. Model Configuration

| Property | Value |
|---|---:|
| Architecture | EDSR-style CNN |
| Scale factor | 4× |
| Feature channels | 64 |
| Residual blocks | 16 |
| Residual scaling | 0.1 |
| Trainable parameters | 1,517,571 |
| Checkpoint size | 17.44 MB |

## 3. Dataset and Evaluation Split

The evaluation uses the DIV2K dataset.

The local test set contains 100 held-out images selected from the publicly available DIV2K training images.

The official DIV2K test high-resolution images are not publicly available, so this project does not claim an official DIV2K test benchmark.

Validation and test degradation are deterministic to make evaluation reproducible.

## 4. Quantitative Results

### Held-out Test Set

| Metric | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| PSNR | 22.164 dB | 3.439 | 14.724 | 33.335 |
| SSIM | 0.5414 | 0.1555 | 0.1938 | 0.8770 |
| LPIPS | 0.5126 | 0.1391 | 0.1451 | 1.0108 |

PSNR and SSIM are higher-is-better metrics.

LPIPS is a lower-is-better perceptual similarity metric.

## 5. Validation Results

The validation evaluation produced:

| Metric | Mean |
|---|---:|
| PSNR | 23.445 dB |
| SSIM | 0.6068 |

The held-out test performance is lower than validation performance, which is expected because the two splits contain different images.

## 6. Inference Benchmark

Benchmark configuration:

- GPU: NVIDIA Tesla T4
- Batch size: 1
- Input: 48×48 RGB
- Output: 192×192 RGB
- Warm-up iterations: 10
- Timed iterations: 100
- CUDA synchronization enabled

Results:

| Metric | Result |
|---|---:|
| Mean latency | 4.325 ms |
| Median latency | 4.315 ms |
| Throughput | 231.19 images/sec |

The benchmark measures neural-network inference only and excludes image loading, preprocessing, and result saving.

## 7. Qualitative Evaluation

Representative comparisons were generated for:

- Best-performing example: `best_0143.png`
- Typical/median example: `median_0301.png`
- Worst-performing example: `worst_0700.png`

Each comparison contains:

1. Low-resolution input
2. EDSR super-resolved output
3. High-resolution ground truth

The LR image is enlarged only for visualization. Metric calculations use the original tensors.

## 8. Interpretation

The trained model demonstrates successful 4× spatial reconstruction using a compact EDSR-style CNN with approximately 1.52 million trainable parameters.

The model provides low-latency GPU inference while maintaining measurable reconstruction quality across the held-out test set.

The difference between validation and test results demonstrates why evaluation on a separate held-out set is important.

LPIPS provides an additional perceptual-quality measurement beyond traditional pixel-level metrics such as PSNR and SSIM.

## 9. Limitations

1. The test set is a project-specific held-out subset derived from the public DIV2K training images.
2. The reported GPU latency is hardware-dependent.
3. The current model was trained with a fixed ×4 output scale.
4. Results depend on the synthetic degradation pipeline used to generate LR inputs.
5. No claim is made that the model represents the state of the art in super-resolution.

## 10. Reproducibility

The project stores:

- Dataset split definitions
- Degradation configuration
- Model configuration
- Training configuration
- Evaluation code
- Benchmark code

Large raw datasets and trained checkpoints are intentionally excluded from Git version control.

The trained checkpoint can be supplied separately for evaluation.

## 11. Final Test Result

**EDSR ×4 CNN**

**22.16 dB PSNR | 0.5414 SSIM | 0.5126 LPIPS**

with:

**1.52M parameters | 17.44 MB checkpoint | 4.325 ms mean GPU inference latency**
