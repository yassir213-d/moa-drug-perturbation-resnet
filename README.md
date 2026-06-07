# Mechanisms of Action (MoA) Prediction

> **Deep Learning Approach for Multi-Label Cellular Drug Perturbation Target Classification**

---

## 📌 Project Overview
In drug discovery, identifying the **Mechanism of Action (MoA)** of a chemical compound is a critical step to understanding its biological impact. This repository contains an advanced, production-ready machine learning pipeline developed to predict the cellular responses and biological targets of various compounds using genomic and cellular expression data.

Given a high-dimensional tabular dataset, the goal is to simultaneously predict **206 multi-label targets** representing different target configurations (e.g., inhibitors, agonists, antagonists).

### 🎯 Key Performance Highlight
* **Best Model:** Deep Tabular ResNet Architecture (PyTorch)
* **Validation Strategy:** Rigorous 5-Fold Cross-Validation
* **Final Metric:** **0.01677 Out-Of-Fold (OOF) Binary Cross-Entropy (BCE) Loss** 🚀

---

## 📊 Dataset & Feature Engineering

The dataset consists of complex biological measurements divided into:
* **`g-` features (Gene Expression):** Capturing how thousands of genes respond to the drug.
* **`c-` features (Cell Viability):** Measuring the survival rate of different cell lines.
* **Metadata Configurations:** Treatment duration (`cp_time`), dose concentration (`cp_dose`), and treatment type (`cp_type`).

### ⚙️ Preprocessing Pipeline
1. **Control Group Isolation (Strict Protocol):** Samples with `cp_type == 'ctl_vehicle'` represent control groups with no active biological compound. To avoid massive log-loss penalties, the pipeline isolates these rows and forces their final predicted probabilities strictly to `0.0`.
2. **Categorical Encoding:** Label encoding applied to operational attributes (`cp_time`, `cp_dose`).
3. **Feature Normalization:** Robust scaling via `StandardScaler` to align skewed distributions across high-dimensional cellular assays, making the data optimal for Deep Learning gradient progression.

---

## 🏗️ Architectural Experiments & Benchmark

We designed, calibrated, and benchmarked three distinct model families to evaluate their robustness against extreme multi-label class imbalances:

| Architecture | Framework | 5-Fold OOF BCE Loss | Status & Insights |
| :--- | :---: | :---: | :--- |
| **Deep ResNet** | PyTorch | **0.01677** 🏆 | **Champion Model.** Excellent at capturing wide, non-linear biological correlations using SiLU activations and skip connections. |
| **TabNet** | PyTorch | 0.02049 | High performance through sequential attention, but computationally heavy. |
| **LightGBM** | Scikit-Learn | 0.08556 | Underfitted under strict constraints; tree structures struggled with the dense multi-output space. |

> ⚠️ **Engineering Lesson on Blending:** Arithmetic and geometric blending with weaker, underfitted tree predictions degraded the performance of our pure signal (resulting in a loss inflation up to `0.06666`). The final production decision was a **Pure ResNet Single-Model Strategy** to guarantee structural robustness on unseen test data.

---

## 💻 ResNet Core Architecture

The network utilizes a dynamic linear bottleneck layout embedded with weight normalization and residual skip connections:

```python
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class ResNetBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.Dropout(dropout_rate),
            weight_norm(nn.Linear(dim, dim)),
            nn.SiLU(),
            nn.BatchNorm1d(dim),
            nn.Dropout(dropout_rate),
            weight_norm(nn.Linear(dim, dim)),
            nn.SiLU()
        )
        
    def forward(self, x):
        return x + self.block(x)

class MoAResNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512, num_blocks=3):
        super().__init__()
        self.input_layer = nn.Sequential(
            weight_norm(nn.Linear(input_dim, hidden_dim)),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU()
        )
        self.resnet_blocks = nn.ModuleList([
            ResNetBlock(dim=hidden_dim) for _ in range(num_blocks)
        ])
        self.output_layer = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        x = self.input_layer(x)
        for block in self.resnet_blocks:
            x = block(x)
        return self.output_layer(x) # Logits ingestion for BCEWithLogitsLoss


Production Submission & Verification
The test-set inference module generates a fully aligned submission.csv tracking the required template:

Shape Target: Exactly (3982, 207) maintaining absolute row-by-row consistency with the test environment features.

Broadcasting Safety: Implements a stable fallback matrix mapping the column-wise scalar means from the validated Out-of-Fold activations to non-control rows.

🧠 Key Takeaways
Deep Learning Superiority over Trees: Deep neural architectures equipped with proper residual shortcuts drastically outperform traditional gradient boosting trees when handling dense, wide, multi-label tabular spaces.

The Ensemble Trap: More models do not equal a better score. Blending highly robust models with underfitted ones creates mathematical noise. Trust your local cross-validation metrics.

Domain-Specific Constraints: Domain awareness (such as handling control vehicle rows natively) is just as critical as hyperparameter optimization.

