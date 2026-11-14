# Assignment B: Reproducible Semantic Segmentation Experiment (`CS107203CS`)

Controlled architectural ablation comparing a 4-level same-padding **U-Net** baseline against a **Boundary-Gated U-Net** (adding a single Sobel-guided Gated Shape-Stream Boundary Branch) on **PASCAL VOC 2012** under a strictly locked training and evaluation budget.

---

## 1. Repository Structure

```text
CS107203CS-AssignmentB-Segmentation/
├── src/
│   ├── dataset.py             # PASCAL VOC 2012 data contract & ignore-aware contour targets
│   ├── models.py              # BaselineUNet & BoundaryGatedUNet architectures
│   └── losses_metrics.py      # Ignore-aware CE + Soft Dice + Boundary loss & confusion matrix
├── logs/
│   ├── baseline_train_log.csv # Epoch-by-epoch training/validation metrics (Baseline)
│   ├── boundary_train_log.csv # Epoch-by-epoch training/validation metrics (Boundary-Gated)
│   └── final_evaluation.txt   # Complete per-class IoU, BIoU, parameter, and latency report
├── outputs/
│   ├── training_curves.png    # Loss and validation mIoU convergence curves
│   ├── successes/             # Qualitative visual comparisons on high-IoU validation samples
│   └── failures/              # Qualitative visual comparisons on worst-case validation failures
├── train.py                   # Deterministic FP16 mixed-precision training script
├── eval.py                    # Standalone evaluation, BIoU, and CUDA latency profiler
├── requirements.txt           # Pinned Python and PyTorch environment versions
└── README.md                  # Experiment documentation and reproduction instructions
```

---

## 2. Data Contract

* **Dataset:** PASCAL VOC 2012 (`[R26]`), comprising **$1,464$ training images** and **$1,449$ validation images** with pixel-level human annotations (no algorithm-generated pseudo-ground truth is used).
* **Semantic Class IDs ($C = 21$):**
  * `0`: background, `1`: aeroplane, `2`: bicycle, `3`: bird, `4`: boat, `5`: bottle, `6`: bus, `7`: car, `8`: cat, `9`: chair, `10`: cow, `11`: diningtable, `12`: dog, `13`: horse, `14`: motorbike, `15`: person, `16`: pottedplant, `17`: sheep, `18`: sofa, `19`: train, `20`: tvmonitor.
* **Ignored Labels:** Pixel value `255` (`void` / ambiguous object boundary border) is preserved strictly as `ignore_index = 255` and excluded from Cross-Entropy loss, Soft Dice loss, $3 \times 3$ morphological boundary target generation, and confusion-matrix IoU accumulation.
* **Image and Mask Transforms:**
  * **Training Split:** Resized to $288 \times 288$ using **bilinear interpolation** for RGB images and **nearest-neighbour interpolation** for integer class masks (preventing label averaging), followed by random horizontal flip ($p = 0.5$), random crop to $256 \times 256$, and ImageNet channel normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
  * **Validation Split:** Deterministically resized to $256 \times 256$ (`BILINEAR` for RGB images, `NEAREST` for integer masks) and normalized with ImageNet statistics.

---

## 3. Controlled Experiment & Locked Budget

### Architectures
1. **Baseline (`BaselineUNet`, `[R04]`):** 4-level same-padding encoder–decoder with $2 \times 2$ max-pooling, $2 \times 2$ transposed-convolution upsampling, and channel concatenation skips across resolutions $256^2 \rightarrow 128^2 \rightarrow 64^2 \rightarrow 32^2$ (channel widths $64, 128, 256, 512$).
2. **Controlled Change (`BoundaryGatedUNet`, `[R14]`):** Holds the entire encoder–decoder backbone, skip connections, and hyperparameters identical while adding **only one mechanism**—a Gated Shape-Stream Boundary Branch:
   * Extracts a $1$-channel Sobel image-gradient magnitude cue $G(X)$ from the input image and concatenates it with low-level encoder features `Enc1` ($64$ channels), projecting to shape features $s \in \mathbb{R}^{32 \times 256 \times 256}$.
   * Gates $s$ using upsampled decoder context $r \in \mathbb{R}^{128 \times 256 \times 256}$ from `Dec2` via residual gating:
     $$\alpha = \sigma\big(\text{Conv}_{1\times 1}([s \,\|\, r])\big), \qquad \tilde{s} = \text{Conv}_{3\times 3}\big((s \odot \alpha) + s\big)$$
   * Supervises boundary logits $\hat{B}$ with ignore-aware $3 \times 3$ morphological contours ($\lambda_{\text{bnd}} = 0.5$) and sharpens finest decoder features via $D_1 \odot (1 + \sigma(\hat{B}))$.

### Locked Comparison Budget
* **Random Seed:** `42` (Python, NumPy, PyTorch CPU/CUDA, deterministic CuDNN)
* **Input Resolution:** $256 \times 256$
* **Training Schedule:** Fixed $20$ epochs, `batch_size = 32`, FP16 Automatic Mixed Precision (`torch.amp`)
* **Optimizer & LR Decay:** `AdamW` ($\text{lr} = 1 \times 10^{-3}$, $\text{weight\_decay} = 1 \times 10^{-4}$) with `CosineAnnealingLR` ($T_{\max} = 20$)
* **Stopping Rule:** Best model weights saved at peak validation mIoU (`checkpoints/<arch>_best.pt`)

---

## 4. Quantitative Results & Measured Latency Conditions

### Latency Measurement Protocol
* **Hardware & Runtime:** NVIDIA Tesla T4 GPU ($16\text{ GB}$), PyTorch `2.14.1+cu130`, CUDA `13.0`
* **Evaluation Precision & Input:** `FP32` (`torch.no_grad()`), $256 \times 256 \times 3$ input tensor
* **Timing Protocol:** $50$ warm-up forward passes followed by $300$ CUDA-synchronized (`torch.cuda.synchronize()`) timed forward passes

### Overall Model Comparison

| Model | Trainable Params | Latency (`Batch=1`) | Latency (`Batch=8`) | Pixel Acc | Val mIoU | 3px Boundary IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline U-Net** | $7,701,461$ | $16.79\text{ ms/img}$ | $21.22\text{ ms/img}$ | $73.71\%$ | $5.65\%$ | $4.90\%$ |
| **Boundary-Gated U-Net** | $7,718,070$ ($+0.22\%$) | $19.40\text{ ms/img}$ | $22.62\text{ ms/img}$ | $73.61\%$ | $5.60\%$ | **$5.73\%$ ($+0.83\%$)** |

### Per-Class Validation IoU ($\%$) Across All 21 Classes

| ID | Class Name | Baseline U-Net | Boundary-Gated U-Net | Difference |
| :---: | :--- | :---: | :---: | :---: |
| `00` | `background` | $75.86\%$ | $75.82\%$ | $-0.04\%$ |
| `01` | `aeroplane` | $0.00\%$ | $0.00\%$ | $+0.00\%$ |
| `02` | `bicycle` | $0.00\%$ | $0.00\%$ | $+0.00\%$ |
| `03` | `bird` | $0.00\%$ | $0.00\%$ | $+0.00\%$ |
| `04` | `boat` | $0.00\%$ | $0.00\%$ | $+0.00\%$ |
| `05` | `bottle` | $0.00\%$ | $0.00\%$ | $+0.00\%$ |
| `06` | `bus` | $0.28\%$ | **$1.74\%$** | **$+1.46\%$** |
| `07` | `car` | $0.64\%$ | **$1.02\%$** | **$+0.37\%$** |
| `08` | `cat` | $11.62\%$ | $10.65\%$ | $-0.96\%$ |
| `09` | `chair` | $0.00\%$ | $0.00\%$ | $+0.00\%$ |
| `10` | `cow` | $0.00\%$ | $0.00\%$ | $+0.00\%$ |
| `11` | `diningtable` | $0.00\%$ | $0.00\%$ | $+0.00\%$ |
| `12` | `dog` | $0.01\%$ | $0.00\%$ | $-0.01\%$ |
| `13` | `horse` | $0.00\%$ | $0.00\%$ | $+0.00\%$ |
| `14` | `motorbike` | $10.98\%$ | $10.85\%$ | $-0.13\%$ |
| `15` | `person` | $18.64\%$ | $17.46\%$ | $-1.18\%$ |
| `16` | `pottedplant` | $0.52\%$ | $0.10\%$ | $-0.43\%$ |
| `17` | `sheep` | $0.00\%$ | $0.00\%$ | $+0.00\%$ |
| `18` | `sofa` | $0.00\%$ | $0.00\%$ | $+0.00\%$ |
| `19` | `train` | $0.00\%$ | $0.00\%$ | $+0.00\%$ |
| `20` | `tvmonitor` | $0.00\%$ | $0.00\%$ | $+0.00\%$ |

---

## 5. What This Experiment Does Not Establish

1. **Asymptotic Convergence Beyond 20 Epochs:** Training from random initialization for $20$ epochs is insufficient for low-frequency PASCAL VOC object classes to overcome extreme background dominance ($75.86\%$ background IoU); a $100$-epoch schedule or an ImageNet-pretrained encoder may produce a different mIoU difference.
2. **Multi-Seed Statistical Variance:** Both models were trained under a single deterministic seed (`42`), so run-to-run variance and confidence intervals across multiple random seeds are not established.
3. **Cross-Domain Generalization:** Results at $256 \times 256$ resolution on object-centric PASCAL VOC imagery do not establish performance on high-resolution dense urban scene-parsing datasets (such as Cityscapes `[R25]`).

---

## 6. Reproduction Instructions

1. **Install Pinned Environment:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Prepare Dataset:** Download PASCAL VOC 2012 (`1,464` train / `1,449` val) into `data/voc2012/{train,val}/{images,masks}` using either `lance-format/pascal-voc-2012-segmentation-lance` or the official `VOCtrainval_11-May-2012.tar` archive.
3. **Train Both Models Under the Locked Budget:**
   ```bash
   python train.py --arch baseline --epochs 20 --batch_size 32 --lr 1e-3 --seed 42
   python train.py --arch boundary --epochs 20 --batch_size 32 --lr 1e-3 --seed 42
   ```
4. **Run Deterministic Evaluation, Latency Profiler, and Visual Export:**
   ```bash
   python eval.py | tee logs/final_evaluation.txt
   ```

---

## 7. References

* **`[R04]`** Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI. `arXiv:1505.04597`.
* **`[R14]`** Takikawa, T., Acuna, D., Jampani, V., & Fidler, S. (2019). *Gated-SCNN: Gated Shape CNNs for Semantic Segmentation*. ICCV. `arXiv:1907.05740`.
* **`[R25]`** Cordts, M., et al. (2016). *The Cityscapes Dataset for Semantic Urban Scene Understanding*. CVPR.
* **`[R26]`** Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J., & Zisserman, A. *The PASCAL Visual Object Classes Challenge 2012 (VOC2012) Results*.