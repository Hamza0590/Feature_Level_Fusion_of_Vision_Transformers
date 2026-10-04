# Feature-Level Fusion of Vision Transformers for Medical Image Classification

This project fuses features from three frozen, pretrained Vision Transformer backbones (one general-purpose, two medical) into a single lightweight classifier. It then tests the approach on several medical imaging datasets.

Only the projection layers and the classification head are trained (about 0.56–0.76 M parameters out of about 100 M). That keeps training cheap, so each run takes 5 epochs on a single GPU.

The repository also includes:
- an **ablation study** of every combination of backbones, and
- a **parallel and distributed computing (PDC) experiment** that runs the three backbones on separate CUDA streams and measures the speedup over sequential execution.

---

## Architecture

```
                    ┌──────────────────────────┐
                    │  Input image 3×224×224   │
                    └────────────┬─────────────┘
          ┌──────────────────────┼──────────────────────┐
          ▼                      ▼                      ▼
 ┌─────────────────┐   ┌──────────────────┐   ┌──────────────────┐
 │ DeiT-Base/16    │   │ MedViT-V2 tiny   │   │ MedFormer tiny   │
 │ (ImageNet)      │   │ (ISIC2018 ckpt)  │   │ (Brain-Tumor ckpt)│
 │ CLS token → 768 │   │ pooled → 384     │   │ GAP → 256        │
 └────────┬────────┘   └────────┬─────────┘   └────────┬─────────┘
   FROZEN │                     │ FROZEN               │ FROZEN
          ▼                     ▼                      ▼
   Linear 768→256        Linear 384→256         Linear 256→256      ← trainable
          └──────────────────────┼──────────────────────┘
                                 ▼
                     Concatenate → 768-d fused vector
                                 ▼
               Linear 768→512 → ReLU/GELU → Dropout(0.3)
                                 ▼
                        Linear 512→num_classes
```

| Backbone | Source | Feature extraction | Dim | GFLOPs (1 image) |
|---|---|---|---|---|
| DeiT-Base (`deit_base_patch16_224`) | `timm`, ImageNet pretrained | `forward_features(x)[:, 0]` (CLS token) | 768 | 16.86 |
| MedViT-V2 tiny | MedViTV2 repo + `MedViT_tiny_ISIC2018.pth` | `proj_head` replaced by `Identity` | 384 | 1.22 |
| MedFormer tiny | MedFormer + `Medformer_tiny_BT.pth` | `forward_features` → global average pool | 256 | 0.42 |
| Fusion and head | trained from scratch | 3 projections + MLP | — | ≈0.0006 |
| **Total** | | | | **18.51** |

**Training setup (all experiments):** AdamW, lr = 1e‑4, cross-entropy loss, 5 epochs, images resized to 224×224, batch size 16 (32 for MedMNIST). Custom-folder datasets use a stratified **80 / 10 / 10** train/val/test split with `random_state=42`.

---

## Repository structure

```
.
├── fusion_testing.ipynb          # Main notebook: every experiment, with saved outputs
├── Sequential_Fusion/
│   └── ISIC_2018.py              # Three-backbone fusion on ISIC 2018 (7-class skin lesions)
├── MEDMNIST/
│   └── medminst_training.py      # Three-backbone fusion on MedMNIST (set DATASET_NAME)
├── ABLATION/
│   └── ablation_STUDY.py         # 7-configuration ablation on Brain Tumor MRI
└── README.md
```

### Notebook contents (`fusion_testing.ipynb`)

| Section | What it does |
|---|---|
| ISIC 2018 | Fusion on ISIC 2018, plus a confusion matrix |
| BT | Fusion on Brain Tumor MRI (4 classes), plus FLOPs and a confusion matrix |
| Pneumonia | Fusion on `hf-vision/chest-xray-pneumonia` (Hugging Face), plus FLOPs and a confusion matrix |
| Parallel Execution → "BT Dataset Sequential" | Sequential fusion baseline with latency, throughput and per-epoch timing (**runs on PathMNIST**, despite the heading) |
| Parallel Execution → "BT Dataset Parallel" | `StreamFusionModel`: each backbone runs on its own `torch.cuda.Stream`; reports speedup and efficiency (**PathMNIST**) |
| PathMNIST / PneumoniaMNIST / DermaMNIST / OCTMNIST | Fusion on four MedMNIST datasets |
| Ablation study | Same as `ABLATION/ablation_STUDY.py`, with saved outputs and a LaTeX table |
| Training curves | Loss and validation-accuracy plots for BT, ISIC 2018 and Pneumonia |

---

## Results

All numbers below are copied from the saved outputs in `fusion_testing.ipynb` and are **test-set** results after 5 epochs.

### Three-backbone fusion across datasets

| Dataset | Classes | Train / Val / Test | Accuracy | Precision | Recall | F1 | Train time |
|---|---|---|---|---|---|---|---|
| Chest X-ray Pneumonia (HF) | 2 | 4684 / 586 / 586 | **95.73%** | 0.9483 | 0.9428 | 0.9455 | 312 s |
| Brain Tumor MRI | 4 | 5618 / 702 / 703 | 92.46% | 0.9233 | 0.9207 | 0.9195 | 202 s |
| PathMNIST (RGB, sequential run) | 9 | 89996 / 10004 / 7180 | 89.37% | 0.8552 | 0.8496 | 0.8499 | 2562 s |
| PneumoniaMNIST | 2 | 4708 / 524 / 624 | 88.62% | 0.8975 | 0.8603 | 0.8733 | 132 s |
| PathMNIST (grayscale, MedMNIST script) | 9 | 89996 / 10004 / 7180 | 85.75% | 0.8123 | 0.8313 | 0.8168 | 2597 s |
| ISIC 2018 | 7 | 9221 / 1153 / 1153 | 78.58% | 0.7739* | 0.7858* | 0.7644* | 520 s |
| DermaMNIST | 7 | 7007 / 1003 / 2005 | 72.82% | 0.3631 | 0.3172 | 0.3309 | 204 s |
| OCTMNIST | 4 | 97477 / 10832 / 1000 | 67.70% | 0.7743 | 0.6770 | 0.6208 | 2763 s |

Precision, recall and F1 are **macro**-averaged, except ISIC 2018 (marked \*), which uses **weighted** averaging.

### Ablation study — Brain Tumor MRI

| Configuration | Acc | Prec | Rec | F1 | Time (s) | Trainable params |
|---|---|---|---|---|---|---|
| DeiT only | 0.9132 | 0.9112 | 0.9092 | 0.9097 | 140.3 | 197,892 |
| MedViT-V2 only | 0.8037 | 0.7915 | 0.7938 | 0.7877 | 97.9 | 99,588 |
| MedFormer only | 0.7852 | 0.7796 | 0.7772 | 0.7767 | 97.1 | 33,412 |
| **DeiT + MedViT-V2** | **0.9403** | **0.9374** | **0.9371** | **0.9370** | 178.1 | 427,780 |
| DeiT + MedFormer | 0.9317 | 0.9288 | 0.9284 | 0.9286 | 164.7 | 395,012 |
| MedViT-V2 + MedFormer | 0.8549 | 0.8497 | 0.8475 | 0.8471 | 119.2 | 296,708 |
| Full fusion (all three) | 0.9317 | 0.9290 | 0.9288 | 0.9288 | 200.8 | 756,996 |

What the ablation shows:
- **Fusion helps.** Every pair of backbones beats each of its members used alone. DeiT + MedViT-V2 improves on DeiT alone by +2.7 accuracy points.
- **DeiT matters most.** Every configuration that includes DeiT scores above 91%. Without DeiT, the best score is 85.5%.
- **The third backbone adds no accuracy here.** Full fusion (93.17%) ties DeiT + MedFormer and falls slightly below DeiT + MedViT-V2, while costing the most training time.

### PDC experiment — sequential vs. CUDA-stream parallel fusion (PathMNIST)

| Metric | Sequential | CUDA streams (3) |
|---|---|---|
| Forward latency (batch 16) | 70.43 ms | 70.46 ms |
| Inference throughput | 227.2 img/s | 227.1 img/s |
| Training throughput | ~195.7 img/s | ~196.9 img/s |
| Total training time (5 epochs) | 2561.77 s | 2540.61 s |
| **Speedup** | — | **1.008×** |
| **Parallel efficiency** | — | **33.6%** |
| Test accuracy / F1 | 89.37% / 0.8499 | 87.48% / 0.8321 |

**Why streams barely help:** DeiT-Base accounts for about 91% of the FLOPs (16.86 of 18.51 GFLOPs). Its kernels alone already saturate the GPU, so the small MedViT and MedFormer kernels have almost no idle compute to overlap with. Splitting the work across streams cannot shorten the critical path, which is DeiT's run time.

The two runs also use slightly different heads: the stream model adds `LayerNorm` and uses `GELU`. The difference in accuracy therefore comes from the head, not from parallelism.

---

## Setup

### 1. Environment

```bash
pip install torch torchvision timm thop scikit-learn numpy pillow tqdm matplotlib seaborn medmnist datasets
```

The saved outputs were produced with PyTorch 2.10 (CUDA 12.8) on a single NVIDIA GPU. The stream-parallel experiment **requires CUDA**.

### 2. External model code and checkpoints

The MedViT-V2 and MedFormer model definitions are **not** included in this repository. Clone them separately and put the checkpoints in one folder:

```
<somewhere>/
├── MedViTV2-main/                 # must provide MedViT.py  (from MedViT import MedViT_tiny)
├── MedFormer-main/                # must provide models/medformer.py (medformer_tiny)
└── Pretrained Models/
    ├── MedViT_tiny_ISIC2018.pth
    └── Medformer_tiny_BT.pth
```

DeiT-Base downloads automatically through `timm`.

### 3. Datasets

| Dataset | How it's loaded | Expected layout |
|---|---|---|
| Brain Tumor MRI | image folders | `DATA_DIR/{train,val}/{glioma,meningioma,notumor,pituitary}/*.jpg` (train and val are merged, then re-split 80/10/10) |
| ISIC 2018 | image folders | `ROOT/ISIC2018_Train/Categorized/<class>/` and `ROOT/ISIC2018_Test/Categorized/<class>/` (merged, then re-split) |
| Chest X-ray Pneumonia | Hugging Face `datasets` | downloaded automatically: `load_dataset("hf-vision/chest-xray-pneumonia")` |
| MedMNIST | `medmnist` package | downloaded automatically |

### 4. Update the paths

Every script has **hardcoded absolute Windows paths** from the original author's machine (`C:\Users\hassa\...`). Before running, change these variables at the top of each script or cell to match your setup:

- `MEDVIT_DIR`, `MEDFORMER_DIR`
- `PRETRAINED` / `MEDVIT_CKPT` / `MEDFORMER_CKPT`
- `DATA_DIR` (Brain Tumor) / `ROOT` (ISIC 2018)

---

## Running

```bash
# ISIC 2018 fusion (prints metrics and shows a confusion matrix)
python Sequential_Fusion/ISIC_2018.py

# MedMNIST fusion: first set DATASET_NAME in the script to one of
#   pathmnist | dermamnist | octmnist | pneumoniamnist
python MEDMNIST/medminst_training.py

# Ablation on Brain Tumor MRI: trains 7 configurations, then prints a summary table and a LaTeX table
python ABLATION/ablation_STUDY.py
```

Run the Pneumonia (Hugging Face) experiment, the Brain Tumor experiment, the PDC sequential vs. stream experiment, and the training curves from `fusion_testing.ipynb`.

---

## Known limitations

- **ChestMNIST is not supported.** It is a multi-label dataset and would need BCE loss. The MedMNIST script refuses to run on it (an `assert` blocks it).
- **The MedMNIST script converts every image to grayscale** (`transforms.Grayscale(num_output_channels=3)`), including the RGB datasets PathMNIST and DermaMNIST. This throws away color information. It is the likely reason that PathMNIST scores 85.75% with this script but 89.37% in the notebook's RGB run.
- **DermaMNIST has poor macro scores** (F1 0.33) because its classes are severely imbalanced and nothing compensates for it: there is no class weighting and no resampling. Scikit-learn warns that some classes are never predicted.
- **Checkpoints load with `strict=False`.** If a checkpoint's keys don't match the model, loading fails silently. `Sequential_Fusion/ISIC_2018.py` also passes the raw checkpoint straight to `load_state_dict`, without first unwrapping a `"model"`/`"state_dict"` key as the other scripts do. Check that the weights actually loaded.
- **Possible domain overlap in the pretrained checkpoints.** MedFormer was pretrained on a Brain Tumor dataset and MedViT-V2 on ISIC 2018. If those checkpoints saw images that land in this project's re-split test sets, the Brain Tumor and ISIC results may be optimistic.
- **Results come from one run each.** Every experiment uses one seed and runs for only 5 epochs, with no hyperparameter tuning.

---

## Acknowledgements

- **DeiT:** Touvron et al., *Training data-efficient image transformers & distillation through attention* (via `timm`).
- **MedViT-V2:** Manzari et al., *MedViT V2: Medical Image Classification with KAN-Integrated Transformers and Dilated Neighborhood Attention*.
- **MedFormer:** hierarchical medical vision transformer.
- **MedMNIST v2:** Yang et al., *MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification*.
- **Datasets:** ISIC 2018 Challenge; Brain Tumor MRI; Chest X-ray Pneumonia (Kermany et al., via `hf-vision/chest-xray-pneumonia`).
