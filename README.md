# Retinal Image Robustness

A study of how image degradation affects diabetic-retinopathy classification and whether corruption-aware training can improve robustness.

## Overview

Deep-learning models for medical imaging are usually evaluated on clean test data, but real images may be affected by blur, sensor noise, illumination changes, contrast shifts, or image compression.

This project studies how these corruptions affect a retinal-image classifier trained on the IDRiD diabetic-retinopathy dataset.

I trained two ImageNet-pretrained ResNet-18 models:

1. a **baseline model** using ordinary image augmentation;
2. a **corruption-trained model** exposed during training to additional synthetic degradations.

Both models use the same architecture, image size, batch size, training/validation split, optimizer, and evaluation pipeline. The main experimental difference is the corruption-aware augmentation used by the robust model.

The final experiment evaluates both fixed models on exactly the same corrupted versions of the official IDRiD test images.

---

## Video Walkthrough

I created a complete video walkthrough of the project covering the motivation, experimental design, implementation, robustness testing, and final results.

### Full Project Playlist

[Watch the complete project walkthrough on YouTube](https://www.youtube.com/playlist?list=PLNgnEuulhlis)

The video series includes:

1. **Project Introduction** — motivation, research question, dataset, and experimental design
2. **Baseline Model (`dataset.py`)** — preprocessing, ResNet-18 training, validation, and evaluation
3. **Corruption-Aware Training (`train_robust.py`)** — Gaussian noise, blur, JPEG compression, and robust training
4. **Robustness Evaluation (`robustness.py`)** — controlled comparison of the baseline and robust models
5. **Results and Conclusion** — robustness gains, tradeoffs, limitations, and conclusions

---

## Dataset

The project uses the **IDRiD disease-grading dataset**.

The original training set contains 413 retinal fundus images across five diabetic-retinopathy severity grades:

| Grade | Images |
|---|---:|
| 0 | 134 |
| 1 | 20 |
| 2 | 136 |
| 3 | 74 |
| 4 | 49 |

The dataset is strongly imbalanced, particularly for grade 1.

The official training data is divided into a stratified training and validation split, while the official IDRiD test set (103 images) is reserved for final evaluation.

Because the test set has 103 images, each image is worth about **0.97 percentage points** of accuracy.

---

## Model

Both classifiers use an ImageNet-pretrained **ResNet-18**.

The original 1000-class ImageNet output layer is replaced with a five-class classifier for diabetic-retinopathy severity.

### Shared configuration

- Architecture: ResNet-18
- Input size: **384 × 384**
- Batch size: **8**
- Optimizer: AdamW
- Learning rate: `1e-4`
- Loss: cross-entropy
- Epochs: 15
- Model selection: best validation accuracy

Before entering the network, retinal images are cropped to remove much of the black background and are normalized using ImageNet channel statistics.

---

## Baseline Model

The baseline training pipeline uses mild augmentation:

- horizontal flips
- small rotations
- small brightness changes
- small contrast changes
- small saturation changes

The selected baseline model achieved:

**54.4% clean test accuracy**

For reference, always predicting the most common test grade (grade 0, 34 of 103 images) would give **33.0%**.

---

## Corruption-Aware Model

The second model uses the same basic pipeline but receives additional randomized corruptions during training, including:

- Gaussian blur
- Gaussian noise
- JPEG compression
- stronger appearance variation

The purpose is to encourage the network to learn features that remain useful when image quality degrades.

Its clean test accuracy was:

**51.5%**

This is only **2.9 percentage points lower** than the baseline.

---

# Robustness Results

## Gaussian Blur

![Gaussian blur robustness](figures/gaussian_blur_robustness.png)

| Blur Radius | Baseline | Robust | Change |
|---:|---:|---:|---:|
| 0 | 54.4% | 51.5% | -2.9 |
| 0.5 | 56.3% | 51.5% | -4.9 |
| 1 | 48.5% | 48.5% | 0.0 |
| 2 | 43.7% | 50.5% | +6.8 |
| 4 | 14.6% | 31.1% | **+16.5** |
| 8 | 13.6% | 18.4% | +4.9 |

The baseline model becomes highly sensitive to stronger blur.

At blur radius 4, corruption-aware training improves accuracy from **14.6% to 31.1%**.

---

## Gaussian Noise

![Gaussian noise robustness](figures/gaussian_noise_robustness.png)

Noise is added to pixel values on the 0–255 scale.

| Noise Std. | Baseline | Robust | Change |
|---:|---:|---:|---:|
| 0 | 54.4% | 51.5% | -2.9 |
| 5 | 45.6% | 53.4% | +7.8 |
| 10 | 20.4% | 47.6% | **+27.2** |
| 20 | 17.5% | 41.7% | **+24.3** |
| 30 | 16.5% | 35.0% | +18.4 |
| 50 | 12.6% | 20.4% | +7.8 |

Gaussian noise produced the strongest robustness gains.

At noise standard deviation 10, the robust model improves from **20.4% to 47.6%**.

At standard deviation 20, accuracy improves from **17.5% to 41.7%**.

---

## JPEG Compression

![JPEG compression robustness](figures/jpeg_compression_robustness.png)

| JPEG Quality | Baseline | Robust | Change |
|---:|---:|---:|---:|
| 100 | 54.4% | 51.5% | -2.9 |
| 90 | 53.4% | 54.4% | +1.0 |
| 70 | 49.5% | 56.3% | +6.8 |
| 50 | 45.6% | 55.3% | +9.7 |
| 30 | 41.7% | 48.5% | +6.8 |
| 10 | 18.4% | 12.6% | -5.8 |

The corruption-trained model improves robustness to moderate and heavy JPEG compression, although extremely severe compression remains difficult.

---

## Brightness

![Brightness robustness](figures/brightness_robustness.png)

| Brightness Factor | Baseline | Robust | Change |
|---:|---:|---:|---:|
| 0.25 | 39.8% | 54.4% | **+14.6** |
| 0.50 | 53.4% | 56.3% | +2.9 |
| 0.75 | 56.3% | 53.4% | -2.9 |
| 1.00 | 54.4% | 51.5% | -2.9 |
| 1.25 | 53.4% | 53.4% | 0.0 |
| 1.50 | 50.5% | 48.5% | -1.9 |
| 1.75 | 45.6% | 48.5% | +2.9 |

The largest brightness improvement occurs for very dark images.

---

## Contrast

![Contrast robustness](figures/contrast_robustness.png)

| Contrast Factor | Baseline | Robust | Change |
|---:|---:|---:|---:|
| 0.25 | 44.7% | 55.3% | **+10.7** |
| 0.50 | 49.5% | 52.4% | +2.9 |
| 0.75 | 52.4% | 56.3% | +3.9 |
| 1.00 | 54.4% | 51.5% | -2.9 |
| 1.25 | 53.4% | 50.5% | -2.9 |
| 1.50 | 55.3% | 45.6% | -9.7 |
| 2.00 | 51.5% | 41.7% | -9.7 |

The robust model improves performance under low contrast but performs worse under unusually high contrast.

---

# Key Findings

Corruption-aware training produced a favorable but non-universal robustness tradeoff.

Clean accuracy changed only slightly:

**54.4% → 51.5% (-2.9 points)**

while several degraded conditions improved substantially:

| Condition | Baseline | Robust | Improvement |
|---|---:|---:|---:|
| Noise std 10 | 20.4% | 47.6% | **+27.2** |
| Noise std 20 | 17.5% | 41.7% | **+24.3** |
| Blur radius 4 | 14.6% | 31.1% | **+16.5** |
| Brightness 0.25 | 39.8% | 54.4% | **+14.6** |
| Contrast 0.25 | 44.7% | 55.3% | **+10.7** |
| JPEG quality 50 | 45.6% | 55.3% | **+9.7** |

The strongest gains occurred under Gaussian noise.

The experiment also shows that robustness training is not universally beneficial. High-contrast images and extremely compressed JPEG images sometimes performed worse.

---

## Project Structure

```text
retina-robustness/
├── dataset.py
├── train_robust.py
├── robustness.py
├── README.md
├── requirements.txt
├── results/
│   └── robustness_results.txt
└── figures/
    ├── brightness_robustness.png
    ├── contrast_robustness.png
    ├── gaussian_blur_robustness.png
    ├── gaussian_noise_robustness.png
    └── jpeg_compression_robustness.png
```

- `dataset.py` trains the baseline classifier and saves best_model.pth.
- `train_robust.py` trains the corruption-aware classifier and saves `best_robust_model.pth`.
- `robustness.py` loads both fixed models and evaluates them under identical corruption conditions.
- `results/robustness_results.txt` stores the final numerical results.
- `figures/` contains visualizations of the major robustness experiments.

The IDRiD image data and trained model checkpoints are excluded from version control.

---

## Limitations

This project is an exploratory robustness study, not a clinically validated diagnostic system.

Important limitations include:

- only 413 original training images;
- strong class imbalance;
- only 20 grade-1 examples;
- poor performance on grade 1;
- a relatively small official test set (103 images), so differences of about 1 point correspond to a single image;
- a single training run per model (seed 42), so run-to-run variance is not measured;
- one model architecture;
- synthetic corruptions rather than naturally collected degraded medical images;
- corruption-specific robustness gains that do not necessarily transfer to every distribution shift.

---

## Conclusion

The baseline ResNet-18 classifier was highly sensitive to several kinds of image degradation, especially Gaussian noise and strong blur.

Corruption-aware training preserved most clean-image performance while substantially improving robustness under several severe corruptions.

The largest improvement occurred under Gaussian noise, where accuracy increased by **27.2 percentage points** at noise standard deviation 10.

These results demonstrate that clean test accuracy alone can hide important failure modes and that robustness-oriented training can substantially reduce some of those failures without necessarily improving every operating condition.