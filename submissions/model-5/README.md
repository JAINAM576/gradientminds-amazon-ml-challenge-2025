
# Model 5 — Optimized MLP with Combined Image + Catalog Embeddings (SigLIP)

###  Folder Contents

```
model-5/
├── model_preprocess_training.ipynb        # Data preprocessing + model training
├── model.h5          # Trained MLP model weights
├── process_pipeline.pkl            # Saved preprocessing pipeline (scaler + encoder)
└── README.md                              # Documentation for this model
```

>  Note: No test predictions or training curve image are available due to session timeout on Lightning AI.

---

##  Overview

Model 5 builds upon **Model 2’s optimized MLP architecture** but enhances the feature set by combining **product image embeddings and catalog content embeddings** using **SigLIP**.
This model was trained on **Lightning AI L4OS GPU**.
Due to session timeout, test submission was not completed, and final leaderboard evaluation is unavailable.

---

##  Features Used

Numeric and categorical features remain the same, with added **image embeddings**.

###  **Numeric Features**

* `unit_qty`
* `pack_count`
* `total_qty`
* `num_bullet_points`
* `num_product_desc`
* `total_chars_bullet_points`
* `total_chars_product_desc`
* `avg_bullet_point_len`
* `info_density`

###  **Categorical Features**

* `unit`
* `brand_name`
* `category`
* `product_type_dynamic`

###  **Text Feature**

* `catalog_content` → encoded via **SigLIP**

###  **Image Feature**

* `image_link` → processed via **SigLIP** and combined with text embeddings

###  **Preprocessing**

* Numeric: `StandardScaler`
* Categorical: `TargetEncoder`
* Text + Image: SigLIP embeddings averaged for image + text
* Combined with structured features for final input

---

##  Model Architecture (Optimized MLP)

| Layer   | Units | Activation     | Normalization | Dropout |
| :------ | :---: | :------------- | :-----------: | :-----: |
| Dense_1 |  512  | LeakyReLU(0.1) |       ✅       |   0.3   |
| Dense_2 |  256  | LeakyReLU(0.1) |       ✅       |   0.25  |
| Dense_3 |  128  | LeakyReLU(0.1) |       ✅       |   0.2   |
| Dense_4 |   64  | ReLU           |       —       |   0.15  |
| Output  |   1   | Linear         |       —       |    —    |

**Optimizer:** Adam
**Learning Rate:** 0.001
**Loss Function:** Custom SMAPE Loss
**Metric:** Mean Absolute Error (MAE)

---

##  Training Configuration

| Parameter        | Value                 |
| :--------------- | :-------------------- |
| Epochs           | 100                   |
| Batch Size       | 128                   |
| Validation Split | 0.2                   |
| Callbacks        | ModelCheckpoint       |
| Environment      | Lightning AI L4OS GPU |
| Python Version   | 3.11                  |
| Framework        | TensorFlow 2.x        |

---

##  Validation Performance

| Metric                |    Value    |
| :-------------------- | :---------: |
| Validation SMAPE Loss | **0.37556** |

> ⚠️ No training curve image available due to session timeout

---

## Evaluation on Test Data

| Metric                    |               Value               |
| :------------------------ | :-------------------------------: |
| SMAPE (Amazon Evaluation) | ❌ Not available (session timeout) |

> **Note:**
>
> * SMAPE (Symmetric Mean Absolute Percentage Error) is measured in **percent (%)**.
> * Its range is typically **0 – 200 %**, where **lower values indicate better model performance**.

---

##  Insights

* Incorporating **image embeddings + catalog text embeddings** via SigLIP improved feature richness.
* Architecture remains stable using Model 2’s design with BatchNormalization and LeakyReLU.
* Moderate dropout continues to prevent overfitting.
* Model was **fully trained**, but final leaderboard evaluation could not be performed.
* Validation SMAPE shows reasonable performance: **0.37556**

---

##  Artifacts

* [`best_mlp_smape_model_image.h5`](./model.h5) — trained model weights
* [`process_pipeline_image1.pkl`](./process_pipeline.pkl) — preprocessing pipeline
* [`model_preprocess_training.ipynb`](./model_preprocess_training.ipynb) — notebook

---

##  Leaderboard Result

> ❌ Not submitted / no leaderboard score due to session timeout

---

###  Author

**Team GradientMinds**

* Sanghavi Jainam Pankajbhai (Leader)
* Priyank Zezariya
* Prajapati Kenilkumar Sureshbhai
* Aryan Mukeshkumar Dave


