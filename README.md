# 🧠 Real vs. Fake Review Classification – Imbalance Handling Experiments

This repository contains four experimental notebooks focused on **binary classification** of real vs. fake reviews using a **BERT-base-cased** transformer model. The primary objective is to **compare different strategies to handle class imbalance**, which is a common challenge in real-world machine learning datasets.

---

## 📦 Dataset

- **Size**: 20,000 samples
- **Class Imbalance**: 90% Real Reviews, 10% Fake Reviews
- **Source**: Synthetically generated using ChatGPT
- **Task**: Binary classification (Real = 0, Fake = 1)

---

## 🎯 Objective

The goal of this project is to **evaluate and compare multiple imbalance-handling techniques** and determine which provides the best performance for fake review detection.

**Conclusion:**  
After testing all four methods, we found that:
- ✅ **Class Weights** performed the best.
- ❌ **Focal Loss** performed the worst in this context.

---

## 📚 Notebook Index

| Notebook Filename | Imbalance Method | Description |
|-------------------|------------------|-------------|
| `binaryclass_none_bertbase_v1.ipynb` | None (Baseline) | Standard BERT model without any imbalance handling |
| `binaryclass_classweights_bertbase_v1.ipynb` | Class Weights | Adds more weight to fake reviews (minority class) during training |
| `binaryclass_sampler_bertbase_v1.ipynb` | Weighted Random Sampler | Ensures each batch contains more fake reviews via weighted sampling with replacement |
| `binaryclass_focalloss_bertbase_v1.ipynb` | Focal Loss | Focuses learning on hard examples by down-weighting confident predictions |

---

## ⚖️ Method Overviews

### 1. **Class Weights**
Assigns higher weight to the minority class (fake reviews) in the loss function. This makes the model pay more attention to underrepresented samples without altering the data.

### 2. **Weighted Random Sampler**
Prioritizes sampling fake reviews during batch creation. It uses replacement, meaning the same fake review can appear in multiple batches, helping the model learn from limited data. Requires a custom training loop.

### 3. **Focal Loss**
Modifies the loss function to focus more on misclassified examples and less on easy ones. Intended to emphasize hard examples but performed poorly in this scenario.

### 4. **No Handling (Baseline)**
A default model without any balancing technique. Often biased toward the majority class in imbalanced settings.

---

## 📊 Performance Comparison

Here's a visual summary of the model performance across methods based on F1 Score (hypothetical values for illustration):

![F1 Score Comparison](imbalance_method_comparison.png)

- **Best**: Class Weights (`F1 ≈ 0.87`)
- **Worst**: Focal Loss (`F1 ≈ 0.76`)

---

## 🧰 Model Details

- **Architecture**: `bert-base-cased`
- **Library**: Hugging Face Transformers
- **Task**: Binary text classification
- **Metrics Used**: Confusion Matrix, Accuracy, Precision, Recall, F1 Score, ROC Curve, PR Curve  
  > ⚠️ In imbalanced data, **PR Curve** is more meaningful than ROC or Accuracy.