

#  Our Approach — Amazon ML Challenge 2025

##  Feature Engineering

### **Numeric Features**
- `unit_qty`, `pack_count`, `total_qty`
- `num_bullet_points`, `num_product_desc`
- `total_chars_bullet_points`, `total_chars_product_desc`
- `avg_bullet_point_len`
- *(Model 5 added)* `info_density`

### **Categorical Features**
- `unit`, `brand_name`, `category`
- *(Model 5 added)* `product_type_dynamic`

### **Text Features**
- `catalog_content`
  - Models 1–3: `SentenceTransformer('stsb-roberta-base')`
  - Model 4: `Lajavaness/bilingual-embedding-large`
  - Model 5: Combined `SigLIP` embeddings (text + image)

### **Image Features**
- *(Introduced in Model 5)* via **SigLIP**

### **Preprocessing Pipeline**

| Data Type | Technique |
|------------|------------|
| Numeric | `StandardScaler` |
| Categorical | `TargetEncoder` |
| Text / Image | Transformer / SigLIP embeddings |
| Integration | Concatenation of all features (~778–900 total) |

---

##  Model Evolution

| Version | Architecture | Embedding | Normalization | Dropout | Optimizer | Val SMAPE | Test SMAPE | Notes |
|----------|--------------|------------|----------------|----------|------------|------------|------------|--------|
| Model 1 | 4-layer MLP | `stsb-roberta-base` | ❌ | 0.1–0.4 | RMSprop | 0.45687 | 82 | Baseline |
| Model 2 | Optimized MLP | `stsb-roberta-base` | ✅ BatchNorm | 0.15–0.3 | Adam | **0.33986** | **67** | Major improvement |
| Model 3 | Residual MLP | `stsb-roberta-base` | ✅ LayerNorm | 0.2–0.3 | Adam (LR decay) | 0.37304 | 71 | Slightly over-regularized |
| Model 4 | Optimized MLP | `bilingual-embedding-large` | ✅ BatchNorm | 0.15–0.3 | Adam | **0.33250** | **64** | Best model |
| Model 5 | Optimized MLP | `SigLIP` (text + image) | ✅ BatchNorm | 0.15–0.3 | Adam | 0.37556 | — | Multimodal experiment |

---

##  Core Architecture

```python
Input → Dense(512, LeakyReLU) → [BatchNorm / LayerNorm] → Dropout(0.3)
       → Dense(256, LeakyReLU) → [Residuals or Norm] → Dropout(0.25)
       → Dense(128, LeakyReLU) → Dropout(0.2)
       → Dense(64, ReLU)
       → Output(1)
````

**Loss:** Custom SMAPE Loss
**Metric:** MAE
**Optimizer:** Adam / RMSprop
**Learning Rate:** 1e-3 (with decay in Model 3)

---

##  Training Configuration

| Parameter        | Value                                 |
| ---------------- | ------------------------------------- |
| Epochs           | 100                                   |
| Batch Size       | 128                                   |
| Validation Split | 0.2                                   |
| Callback         | ModelCheckpoint                       |
| Framework        | TensorFlow 2.x                        |
| Python           | 3.11                                  |
| Hardware         | Colab (T4 GPU) / Lightning (L4OS GPU) |

---

##  Key Insights

* Transformer-based text embeddings significantly impacted accuracy.
* BatchNorm + LeakyReLU stabilized training.
* Residuals + GaussianNoise improved generalization.
* Multimodal SigLIP showed potential for richer representation.
* Dropout between 0.15–0.3 provided best regularization.

---

##  Future Scope

* Ensemble (Model 2 + Model 4) for robust prediction.
* Further SigLIP fine-tuning for image-text synergy.
* Experiment with transformer-based regression heads.
* Category-wise cross-validation for domain-specific tuning.

