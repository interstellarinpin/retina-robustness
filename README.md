# Retinal Image Robustness

A study of how image corruptions affect deep-learning-based diabetic retinopathy classification and whether corruption-aware training can improve robustness.

## Overview

Deep learning models for medical imaging are usually evaluated on clean test images, but real-world images can be degraded by blur, noise, changes in illumination, contrast variation, or image compression.

This project investigates how these corruptions affect a ResNet-18 classifier trained to predict diabetic retinopathy severity from retinal fundus photographs.

I first trained a baseline classifier on the IDRiD dataset and evaluated it under five types of image corruption:

- Gaussian blur
- Brightness changes
- Contrast changes
- Gaussian noise
- JPEG compression

I then trained a second model using randomized corruption augmentation and evaluated both models under exactly the same corruption benchmark.

The results show a tradeoff: corruption-aware training decreases clean accuracy but substantially improves robustness to several severe corruptions, particularly Gaussian noise, blur, and JPEG compression.

---

## Dataset

This project uses the **IDRiD (Indian Diabetic Retinopathy Image Dataset)** disease grading dataset.

The training set contains **413 retinal images** classified into five diabetic retinopathy grades:

| Grade | Training Images |
|---|---:|
| 0 | 134 |
| 1 | 20 |
| 2 | 136 |
| 3 | 74 |
| 4 | 49 |

The strong class imbalance, particularly the small number of grade 1 examples, is an important limitation of the experiment.

Images are cropped to remove black background regions and resized to **384 × 384** before being passed to the model.

---

## Baseline Model

The classifier is based on an ImageNet-pretrained **ResNet-18** with its final classification layer replaced by a five-class output layer.

The baseline model achieved:

**60.2% clean test accuracy**

Its clean-test confusion matrix showed substantial differences between classes, including difficulty recognizing the underrepresented grade 1 class.

---

## Robustness Benchmark

The baseline model was evaluated without retraining under progressively stronger image corruptions.

### Gaussian Blur

![Gaussian blur robustness](figures/gaussian_blur_robustness.png)

Performance was relatively stable under very mild blur but collapsed as blur became stronger.

| Blur Radius | Baseline Accuracy | Robust Accuracy |
|---:|---:|---:|
| 0 | 60.2% | 53.4% |
| 0.5 | 60.2% | 53.4% |
| 1 | 57.3% | 48.5% |
| 2 | 42.7% | 50.5% |
| 4 | 16.5% | 31.1% |
| 8 | 12.6% | 17.5% |

At radius 4, corruption-aware training improved accuracy by **14.6 percentage points**.

---

### Gaussian Noise

![Gaussian noise robustness](figures/gaussian_noise_robustness.png)

Gaussian noise produced one of the most severe baseline failure modes.

| Noise Std. | Baseline Accuracy | Robust Accuracy |
|---:|---:|---:|
| 0 | 60.2% | 53.4% |
| 5 | 49.5% | 55.3% |
| 10 | 21.4% | 50.5% |
| 20 | 12.6% | 42.7% |
| 30 | 12.6% | 34.0% |
| 50 | 12.6% | 21.4% |

At noise standard deviation 20, corruption-aware training improved accuracy from **12.6% to 42.7%**, a **30.1-point improvement**.

---

### JPEG Compression

![JPEG compression robustness](figures/jpeg_compression_robustness.png)

The baseline model also became increasingly unreliable as JPEG compression became stronger.

| JPEG Quality | Baseline Accuracy | Robust Accuracy |
|---:|---:|---:|
| 100 | 60.2% | 53.4% |
| 90 | 63.1% | 58.3% |
| 70 | 53.4% | 53.4% |
| 50 | 47.6% | 53.4% |
| 30 | 36.9% | 49.5% |
| 10 | 14.6% | 12.6% |

At JPEG quality 30, robust training improved accuracy by **12.6 percentage points**.

Very severe compression remained challenging for both models.

---

## Brightness and Contrast

The models were also evaluated under brightness and contrast changes.

The baseline model was comparatively tolerant to moderate brightness and contrast changes. Corruption-aware training did not consistently improve these conditions.

Examples include:

| Condition | Baseline | Robust | Change |
|---|---:|---:|---:|
| Brightness 0.25 | 43.7% | 54.4% | +10.7 |
| Brightness 1.75 | 44.7% | 48.5% | +3.9 |
| Contrast 0.25 | 52.4% | 58.3% | +5.8 |
| Contrast 1.50 | 58.3% | 49.5% | -8.7 |

This suggests that the benefits of corruption-aware training depend strongly on the type and severity of distribution shift.

---

## Corruption-Aware Training

A second ResNet-18 was trained with randomized image degradations applied during training.

The augmentation pipeline included mild-to-moderate:

- Gaussian blur
- Brightness variation
- Contrast variation
- Gaussian noise
- JPEG compression

The goal was not simply to train on the most extreme benchmark conditions, but to expose the model to realistic variation and test whether this generalized to stronger corruptions.

### Clean-Accuracy Tradeoff

The final benchmark produced:

| Model | Clean Accuracy |
|---|---:|
| Baseline ResNet-18 | **60.2%** |
| Corruption-trained ResNet-18 | **53.4%** |

Corruption-aware training therefore reduced clean accuracy by **6.8 percentage points**.

However, it substantially reduced degradation under several more severe corruptions.

---

## Key Results

Some of the largest robustness improvements were:

| Condition | Baseline | Robust | Improvement |
|---|---:|---:|---:|
| Gaussian noise, std 20 | 12.6% | 42.7% | **+30.1** |
| Gaussian noise, std 10 | 21.4% | 50.5% | **+29.1** |
| Gaussian noise, std 30 | 12.6% | 34.0% | **+21.4** |
| Blur radius 4 | 16.5% | 31.1% | **+14.6** |
| JPEG quality 30 | 36.9% | 49.5% | **+12.6** |
| Brightness 0.25 | 43.7% | 54.4% | **+10.7** |

Overall, the experiment demonstrates a **clean-accuracy versus corruption-robustness tradeoff**.

The corruption-trained model is not universally superior. It performs worse on clean images and under several mild corruptions, but it is substantially less brittle under several severe degradations.

---

## Project Structure

```text
retina-robustness/
├── dataset.py
├── train_robust.py
├── robustness.py
├── results.txt
├── figures/
│   ├── gaussian_blur_robustness.png
│   ├── gaussian_noise_robustness.png
│   └── jpeg_compression_robustness.png
└── README.md
```

- `dataset.py` — baseline IDRiD training pipeline
- `train_robust.py` — corruption-aware training pipeline
- `robustness.py` — side-by-side corruption benchmark
- `results.txt` — final experimental results
- `figures/` — robustness plots

The IDRiD images and trained model checkpoints are excluded from the repository.

---

## Limitations

This is a small-scale robustness experiment rather than a clinically validated system.

Important limitations include:

- The training dataset contains only **413 images**.
- The class distribution is highly imbalanced.
- Grade 1 has only **20 training examples**.
- Both models struggled substantially with grade 1.
- Accuracy can vary considerably on a test set of this size.
- Only one model architecture was studied.
- The synthetic corruptions are simplified approximations of real image-quality degradation.
- Improved robustness to one corruption does not necessarily imply robustness to other distribution shifts.

These results therefore should not be interpreted as evidence of clinical performance.

---

## Conclusion

The baseline retinal classifier performed reasonably on clean images but was highly vulnerable to several forms of image degradation. Strong Gaussian blur and noise produced particularly severe failures.

Training with randomized corruption augmentation changed this behavior. Although clean accuracy decreased from **60.2% to 53.4%**, the robust model retained substantially more accuracy under several severe corruptions, including a **30.1-point improvement under Gaussian noise** and a **14.6-point improvement under strong blur**.

The results illustrate that evaluating only clean accuracy can hide important model failure modes and that robustness interventions can involve meaningful tradeoffs rather than universally improving performance.