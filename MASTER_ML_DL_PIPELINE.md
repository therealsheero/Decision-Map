# MASTER ML/DL PIPELINE & DECISION MAP

> This document is a practical decision manual, pipeline map, and study reference for building ML/DL models end-to-end.

---

# PART 0 — MASTER MAP: PROBLEM FAMILY OVERVIEW

```text
WHAT TYPE OF PROBLEM AM I SOLVING?
│
├── 1. TABULAR / STRUCTURED DATA
│   ├── TARGET IS CONTINUOUS
│   │   └── Regression
│   ├── TARGET IS CATEGORICAL
│   │   ├── 2 Classes → Binary Classification
│   │   ├── >2 Mutually Exclusive → Multiclass Classification
│   │   └── >2 Non-Exclusive → Multilabel Classification
│   ├── NO TARGET (DISCOVERY)
│   │   ├── Grouping → Clustering
│   │   ├── Compression/Vis → Dimensionality Reduction
│   │   └── Outlier Hunt → Anomaly Detection
│   └── MODEL FAMILY CHOICE
│       ├── Classical/Tabular Supervised ML (Trees, Boosting, Linear)
│       └── Neural-network-based tabular learning (MLP, TabNet)
│
├── 2. IMAGE / SPATIAL DATA (Computer Vision / CNN)
│   ├── What is in this image? → Image Classification
│   ├── Where are the objects? → Object Detection
│   └── Which pixels belong to what? → Image Segmentation
│
├── 3. SEQUENCE / TEMPORAL DATA
│   ├── Sequence Modelling (General time series)
│   ├── RNN/LSTM/GRU (Stateful, sequential memory)
│   └── Transformers (Attention-based, parallelizable)
│
├── 4. TEXT / LANGUAGE (NLP)
│   ├── Discriminative (Classification, NER, Extractive QA)
│   └── Generative (LLM training/fine-tuning)
│
├── 5. MULTIPLE MODALITIES (Multimodal Learning)
│   ├── Aligning spaces → CLIP-style contrastive learning
│   └── Generative/Reasoning → VLM training/fine-tuning
│
└── 6. TRAINING & OPTIMIZATION STRATEGIES
    ├── No Labels? → Self-supervised learning
    ├── Starting from scratch? → Pretraining
    ├── Have existing weights? → Transfer learning
    └── Adapting existing weights? → Fine-tuning (Full or PEFT)
```

## Pipeline Component Lifecycle Breakdown

| Component Type | Shared | Different / Modality-Specific | Optional |
|---|---|---|---|
| **Before Training** | Data splitting, batching, defining loss/optimizer | Tokenization (Text), Augmentation (Images/Audio), Scaling (Tabular) | Dimensionality reduction, Synthetic data |
| **Inside Training Loop** | Forward pass, loss calculation, backward pass, optimizer step, zero_grad | Teacher forcing (RNNs), Contrastive pairs (CLIP), Gradient accumulation | Gradient clipping, Mixed precision (AMP), EMA |
| **After Training** | Validation metric calculation, Checkpointing | NMS (Object Detection), Generation decoding (LLMs) | Error analysis, Distillation, Pruning |
| **Inference Only** | Forward pass, post-processing | KV Caching (LLMs), Dynamic batching | TensorRT/ONNX export, Quantization |

---

# PART 1 — UNIVERSAL ML LIFECYCLE

### RAW DATA
**What happens here?** Unprocessed data is ingested from disk, DB, or API.
**Available techniques:** Pandas `read_csv`, PyTorch `ImageFolder`, HuggingFace `load_dataset`.
**When to use:** Always.
**Common mistakes:** Loading entire multi-TB dataset into RAM.
**What comes next?** Data Understanding.

### DATA UNDERSTANDING
**What happens here?** Profiling the data to find schema, types, and raw statistics.
**Available techniques:** `df.info()`, `df.describe()`, Pandas Profiling.
**Trade-offs:** Deep profiling takes time but prevents downstream crashes.
**What comes next?** Data Validation.

### DATA VALIDATION
**What happens here?** Asserting data meets expectations (schema, ranges, no impossible values).
**Available techniques:** Pydantic, Great Expectations, simple asserts.
**Failure cases:** Pipeline crashes later because `age = -5` caused a log-transform NaN.
**What comes next?** Data Cleaning.

### DATA CLEANING
**What happens here?** Handling missing values, duplicates, and invalid entries.
**Available techniques:** Imputation, dropping rows, capping outliers.
**Typical values/ranges:** Drop feature if >30-50% missing.
**What comes next?** Data Reformation.

### DATA REFORMATION
**What happens here?** Changing data structure (e.g., aggregating rows, parsing dates, JSON unnesting).
**Available techniques:** `groupby`, `pd.json_normalize`, regex extraction.
**What comes next?** Data Transformation.

### DATA TRANSFORMATION
**What happens here?** Scaling numericals, handling skewness, encoding categoricals.
**Available techniques:** Log transform, StandardScaler, OneHotEncoding.
> [!WARNING]
> Transformations requiring statistics (mean, variance, vocab) MUST be fit on the training set ONLY.
**What comes next?** Feature/Label Processing.

### FEATURE/LABEL PROCESSING
**What happens here?** Creating model-ready inputs and targets (X and y).
**Available techniques:** Feature crosses, embeddings, label encoding (target only).
**What comes next?** Splitting.

### SPLITTING
**What happens here?** Dividing data into Train, Validation, and Test sets.
**Available techniques:** Random, Stratified, Time-based, Group-based.
**Typical values/ranges:** 80/10/10 or 70/15/15.
**What comes next?** Augmentation.

### AUGMENTATION
**What happens here?** Artificially expanding the training set by applying transformations.
**Available techniques:** Random crop (Image), synonym replacement (Text), SMOTE (Tabular - use with caution).
**When to use:** ONLY on training data.
**What comes next?** Baseline.

### BASELINE
**What happens here?** Establishing the "dumbest possible" successful prediction to measure against.
**Available techniques:** Predict mean (regression), predict majority class (classification), simple linear model.
**When to use:** Always. Never skip.
**What comes next?** Model Selection.

### MODEL SELECTION & INITIALIZATION
**What happens here?** Choosing the architecture and initializing weights.
**Available techniques:** ResNet vs ViT, XGBoost vs MLP, Xavier/Kaiming initialization.
**What comes next?** Objective/Loss.

### OBJECTIVE/LOSS
**What happens here?** Defining how the model is penalized for being wrong.
**Available techniques:** MSE (Regression), CrossEntropy (Classification), Contrastive (Similarity).
**What comes next?** Optimization.

### OPTIMIZATION
**What happens here?** Defining how weights are updated based on the loss gradient.
**Available techniques:** Adam, AdamW, SGD with Momentum.
**Typical values:** LR `1e-3` to `1e-5`.
**What comes next?** Training.

### TRAINING & VALIDATION
**What happens here?** Iterative loop of forward/backward passes and periodic validation checks.
**Available techniques:** Epoch-based, step-based, early stopping.
**Common mistakes:** Not zeroing gradients (`optimizer.zero_grad()`), evaluating with `model.train()` instead of `model.eval()`.
**What comes next?** Hyperparameter Optimization.

### HYPERPARAMETER OPTIMIZATION (HPO)
**What happens here?** Tuning the non-learned configurations.
**Available techniques:** Grid Search, Random Search, Bayesian Optimization (Optuna).
**What comes next?** Error Analysis.

### ERROR ANALYSIS & ITERATION
**What happens here?** Looking at *where* the model failed on the validation set.
**Available techniques:** Confusion matrix, highest-loss examples review, residual plots.
**What comes next?** Fix data/model, loop back to Training.

### FINAL TEST & MODEL SELECTION
**What happens here?** Evaluating the final tuned model on the untouched test set.
**What comes next?** Checkpoint/Export.

### CHECKPOINT/EXPORT & INFERENCE
**What happens here?** Saving weights and deploying.
**Available techniques:** `torch.save(model.state_dict())`, ONNX export, TensorRT, vLLM serving.
**What comes next?** Monitoring.

### MONITORING
**What happens here?** Tracking model performance in production to detect drift.
**Available techniques:** Tracking input distributions, prediction confidence, business metrics.

---

# PART 2 — DATA UNDERSTANDING PIPELINE

Run these BEFORE modeling, BEFORE splitting, and BEFORE transformation. 

```python
import pandas as pd
import numpy as np
import cv2 # For images

# 1. Shape and structure
print(df.shape)          # rows, columns
display(df.head())       # Sanity check values
display(df.tail())

# 2. Feature Types
print(df.dtypes)         # int, float, object, datetime
print(df.info())         # memory usage, non-null counts

# 3. Target Inspection
print(df[target].dtype)
display(df[target].value_counts(dropna=False)) # Check for hidden NaNs or '?'

# 4. Numerical Features
num_cols = df.select_dtypes(include=np.number).columns
display(df[num_cols].describe()) # Mean, std, min, max, quartiles
df[num_cols].hist(bins=50, figsize=(15, 10)) # Distributions

# 5. Categorical & Ordinal Features
cat_cols = df.select_dtypes(include=['object', 'category']).columns
print(df[cat_cols].nunique()) # Cardinality
for col in cat_cols:
    print(df[col].value_counts(normalize=True).head(10)) # Top classes

# 6. Missing & Invalid Values
print(df.isna().sum())           # Absolute missing
print(df.isna().mean() * 100)    # % missing
# Domain-specific invalid (e.g. age cannot be negative)
print((df['age'] < 0).sum())     

# 7. Duplicates
print(df.duplicated().sum())     # Exact row duplicates
# Entity duplicates (e.g., same user, conflicting rows)
print(df.duplicated(subset=['user_id']).sum())

# 8. Inconsistent Formatting
# Example: strip whitespace, make lowercase to check true cardinality
df['city'] = df['city'].str.strip().str.lower()

# 9. Outliers (Z-score & IQR)
# Quick IQR check for a column
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['price'] < (Q1 - 1.5 * IQR)) | (df['price'] > (Q3 + 1.5 * IQR))]
print(len(outliers))

# 10. Class/Target Distribution
df[target].value_counts(normalize=True).plot(kind='bar') # Classification
df[target].hist(bins=50) # Regression

# 11. Image specific (if dataset is image paths)
# Check shape, channels, dtype
img = cv2.imread(df['image_path'].iloc[0])
print(img.shape, img.dtype) # e.g., (224, 224, 3), uint8
```

> [!WARNING] 
> **Leakage Indicators:** If a feature correlates >0.95 with the target, or if a feature is conceptually measured *after* the target event occurs, drop it. It is target leakage.

---

# PART 3 — MISSING VALUE DECISION TREE

## Missing Feature Values

### For Numerical Features

```text
Numerical feature is missing
             │
      What % is missing?
      │      │      │
    <5%    5-30%  >30-50%
     │       │      │
     A       B      C

A (<5% missing):
├── Remove rows: If MCAR (Missing Completely At Random) and dataset is huge.
├── Mean imputation: Normal distribution, no outliers.
└── Median imputation: Skewed distribution, outliers present.

B (5-30% missing):
├── Iterative Imputation (MICE): Features are correlated.
├── KNN Imputation: Features are correlated, compute is affordable.
├── Group-wise median: E.g., fill missing 'Salary' by 'Job Title' median.
└── Interpolation / Forward Fill: If Time Series.

C (>30-50% missing):
├── Drop Feature: Usually contains too little signal.
├── Model-based imputation: If the feature is critical domain knowledge.
└── Missing Indicator: Create a binary column `is_missing`. The fact it's missing is the signal.
```
*Note: Tree-based models (XGBoost, LightGBM) handle NaNs natively. You can just leave them!*

### For Categorical Features

```text
Categorical feature is missing
             │
├── Treat as new category: Fill with "Unknown" or "MISSING". (Safest, often best).
├── Mode imputation: If <5% missing and one class dominates.
├── Group-wise mode: E.g., missing 'City' filled by mode of 'State'.
└── Predictive imputation: Train a classifier to predict the missing category.
```

## Missing Target Values

**Rule:** Imputing a target is fundamentally different from imputing a feature. A supervised loss function uses the target as the ground truth. If you guess the ground truth, you are training the model to predict your guesses.

```text
Target has missing values (e.g., '?')
                  │
        What does '?' mean?
        │         │        │
      Truly     Refused   Another
     unknown   to answer   class
        │         │        │
        X         Y        Z

X (Truly unknown - e.g. sensor failed):
├── Remove rows: Standard practice for supervised learning.
├── Use Unlabeled Data: Move to semi-supervised learning.
└── Pseudo-labeling: Train on labeled, predict on '?', add confident preds back.

Y (Refused to answer - e.g. survey data):
└── Treat as new class: 'Refused' is a valid behavior pattern.

Z (Another class - e.g. target is "Churn", missing means "Didn't churn"):
└── Map to appropriate existing label (e.g., 0).
```

---

# PART 4 — DATA REFORMATION & TRANSFORMATION

## Numerical Data Transformations

| Technique | When Useful | When NOT Needed | Model Requirements | Outlier Sensitivity |
|---|---|---|---|---|
| **StandardScaler** | Linear models, SVM, KNN, PCA, NNs | Tree models | Assumes normal-ish distribution | Sensitive |
| **MinMaxScaler** | Image pixels (0-255 → 0-1), bounded outputs | Tree models | Preserves zero values | Very sensitive |
| **RobustScaler** | Data with severe outliers | Tree models | Uses median/IQR | **Not sensitive** |
| **Log transform** | Right-skewed data (e.g. Income, Price) | Already normal | Requires > 0 (use log1p for >=0) | Handles extremes |
| **Box-Cox** | Force normality for regression assumptions | Trees | Strictly > 0 | Handles extremes |
| **Yeo-Johnson** | Same as Box-Cox | Trees | Handles negative/zero values | Handles extremes |
| **Clipping** | Known physical bounds, sensor errors | When outliers are real | N/A | Removes them |

> [!TIP]
> **Tree models** (Random Forest, XGBoost) are invariant to monotonic transformations. Scaling numbers does not change how a decision tree splits them. Do not waste compute scaling features for XGBoost.

## Categorical Data Encoding

```text
Categorical feature
         │
    What is the nature of the categories?
    │               │              │               │
 Ordinal         Binary     Low Cardinality   High Cardinality
(Low/Med/High)   (Y/N)        (<15 unique)     (>15 unique, e.g. ZIP code)
    │               │              │               │
OrdinalEncoder  LabelEncoder   OneHotEncoder       │
                                                   ├── Tree Model → Frequency/Target Encoding
                                                   └── Neural Net → Embedding Layer / Hashing
```

> [!CAUTION]
> **NEVER use LabelEncoder for nominal variables if using a linear model, SVM, or NN.**
> If `Mumbai=0, Delhi=1, Chennai=2`, a linear model assumes `Chennai` is exactly twice as much as `Delhi`. Use One-Hot Encoding instead.

## Text Data Pipeline

```text
Raw Text
  │
Cleaning (lowercase, remove HTML, normalize unicode)
  │
Tokenization Strategy Choice:
  ├── Word (split by space) - huge vocab, handles out-of-vocab (OOV) poorly
  ├── Subword (BPE, WordPiece, SentencePiece) - MODERN STANDARD (BERT, GPT)
  └── Character - tiny vocab, long sequences, slow
  │
Build Vocabulary (Fit on TRAIN text ONLY)
  │
Token → ID mapping
  │
Padding (Pad to max length in BATCH, not global dataset max)
  │
Attention Masks (1 for real token, 0 for pad)
  │
Embeddings (nn.Embedding)
```

## Image Data Pipeline

```text
Raw Image File
  │
Decode (cv2, PIL) -> Validate Channels (RGB vs BGR)
  │
Resize Strategy:
  ├── Aspect-agnostic Resize (squishes image)
  ├── Resize + Center Crop (Standard for evaluation)
  └── Random Resized Crop (Standard for training)
  │
Tensor Conversion (HWC [Height, Width, Channels] → CHW [Channels, Height, Width] for PyTorch)
  │
Normalization:
  (Pixel / 255.0) - mean / std
  * Use ImageNet stats (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if using pretrained model.
  * Fit on train set if training from scratch.
  │
Augmentation (TRAINING ONLY - Flips, Color Jitter, etc.)
  │
Batch
```

---

# PART 5 — TRAIN/VALIDATION/TEST SPLITTING

## Why Three Sets?

```text
TRAIN SET                VALIDATION SET                 TEST SET
    │                          │                           │
Learn the weights          Tune hyperparameters        Final unbiased estimate
Update gradients           Select best architecture    of generalization error
(Seen inside epoch)        (Evaluated between epochs)  (Used ONCE at the end)
```

## Splitting Strategies

| Strategy | When to Use | Scikit-Learn | Warning |
|---|---|---|---|
| **Random Split** | IID data, balanced classes | `train_test_split` | Fails on imbalanced data |
| **Stratified** | Classification tasks, imbalanced | `train_test_split(stratify=y)` | Must use for classification |
| **Group Split** | Multiple rows belong to 1 entity (e.g. 5 scans per patient) | `GroupShuffleSplit` | If random split, patient is in both train & test (Leakage!) |
| **Time-based** | Stock prices, forecasting | Temporal manual split | NEVER random split time series |

## Cross-Validation Strategies

| Strategy | When to Use | Scikit-Learn |
|---|---|---|
| **K-Fold** | General tabular regression | `KFold(n_splits=5)` |
| **Stratified K-Fold** | General tabular classification | `StratifiedKFold` |
| **Group K-Fold** | Groups must not cross folds | `GroupKFold` |
| **Time Series Split**| Temporal data evaluation | `TimeSeriesSplit` |

## CRITICAL: Preprocessing Must Fit Only on Train

**Data Leakage** occurs when information from the Validation or Test set leaks into the Training process. The most common cause is scaling or imputing *before* splitting.

```python
# ❌ WRONG (LEAKAGE):
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # Test data mean/std influences training data!
X_train, X_test = train_test_split(X_scaled, test_size=0.2)

# ✅ CORRECT:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Fit ONLY on Train
X_test_scaled = scaler.transform(X_test)       # Transform Test
```

**Best Practice:** Use Scikit-Learn Pipelines to prevent this automatically.
```python
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', Ridge())
])
pipe.fit(X_train, y_train) # Safe: fits steps on train only
pipe.predict(X_test)       # Safe: applies fitted steps to test
```


---

# PART 6 — REGRESSION PIPELINE

```text
DATA
  ↓
CLEANING (Impute NaNs, remove impossible values)
  ↓
TRANSFORMATIONS (Log-transform skewed targets, robust scale features)
  ↓
FEATURE ENGINEERING (Polynomials, interactions, embeddings)
  ↓
SPLIT (Random or Time-based, 80/10/10)
  ↓
BASELINE (Predict mean of training target)
  ↓
MODELS (See decision map below)
  ↓
LOSS (MSE or Huber)
  ↓
OPTIMIZATION (Analytic solution or Adam/SGD)
  ↓
VALIDATION (RMSE / MAE / R²)
  ↓
ERROR ANALYSIS (Residual plots, heteroscedasticity checks)
```

## Regression Model Decision Map

| Model | When to Use | Scaling Req? | Interpretability | Typical Hyperparameters | Common Failure Modes |
|---|---|---|---|---|---|
| **Linear Regression** | Simple baseline, strictly linear relationships | Yes | Very High (Coefficients) | None | Fails on non-linear data; sensitive to collinearity. |
| **Ridge (L2)** | Linear, but many correlated features | Yes | High | `alpha=0.1 to 100` (1.0 is default, higher = more penalty) | Underfitting if alpha is too high. |
| **Lasso (L1)** | Need feature selection (pushes coeffs to 0) | Yes | High (Selects features) | `alpha=0.01 to 10` | Drops correlated features arbitrarily. |
| **Elastic Net** | Mix of Ridge (L2) and Lasso (L1) | Yes | High | `alpha`, `l1_ratio` (0 = Ridge, 1 = Lasso, 0.5 = Mix) | Hard to tune both parameters without GridSearch. |
| **Polynomial Reg.** | Known non-linear physics/business logic | Yes | Medium | `degree=2 or 3` | Explodes massively if `degree > 3` (Overfitting). |
| **Decision Tree** | Highly non-linear, interacting features | No | Very High (Tree plot) | `max_depth=3 to 10` | Massive overfitting if depth is unconstrained. |
| **Random Forest** | Strong baseline, little tuning needed | No | Medium (Feature imp) | `n_estimators=100-500`, `max_depth` | Very large model size in memory; slow inference. |
| **XGBoost/LightGBM** | **Industry standard for Tabular data** | No | Medium (SHAP) | `learning_rate=0.01-0.1`, `max_depth=3-8` | Overfits small datasets quickly. |
| **Neural Network** | Huge datasets, multi-modal features (text+tab) | Yes | Low (Black box) | `hidden_layers`, `lr=1e-3`, `batch_size` | Hard to optimize for tabular without massive data. |

---

# PART 7 — REGRESSION LOSS + EVALUATION

> [!NOTE]
> **Training Loss** is optimized by the algorithm (e.g., MSE is mathematically easy to differentiate).
> **Evaluation Metric** is read by humans/business to understand performance (e.g., MAE is easy to explain in dollars).

## Loss & Metric Decision Table

| Metric | Formula Concept | Use Case | Outlier Sensitivity |
|---|---|---|---|
| **MAE** (Mean Absolute Error) | Average of absolute differences | Business reporting ("We are off by $5 on average") | Low. Treats all errors linearly. |
| **MSE** (Mean Squared Error) | Average of squared differences | Default training loss for NNs/Linear models | High. Punishes large errors quadratically. |
| **RMSE** (Root Mean Squared Error) | Square root of MSE | Standard evaluation metric, same units as target | High. |
| **Huber Loss** | Quadratic for small errors, linear for large | Training loss when dataset has unremovable outliers | Low/Robust. |
| **R²** (Coefficient of Determination) | Variance explained relative to mean baseline | Explaining model quality (Max 1.0, <0 means worse than mean) | High. |
| **RMSLE** (Log Error) | RMSE of log-transformed targets | Target spans multiple orders of magnitude (e.g., Housing prices) | Low to large absolute errors, punishes ratio errors. |

### Error Analysis: Residual Analysis
Always plot `Predictions vs. Actuals` and `Predictions vs. Residuals (Actual - Predicted)`.
- **Good:** Residuals look like random noise centered around 0.
- **Bad (Heteroscedasticity):** Residual variance grows as predictions grow (e.g., off by $10 on cheap houses, off by $100k on mansions) → **Fix:** Log-transform the target.

---

# PART 8 — CLASSIFICATION PIPELINE

## Architectures & Problem Types

```text
WHAT IS THE LABEL STRUCTURE?
  │
  ├── BINARY (e.g., Spam vs Not Spam)
  │   ├── Target representation: [0, 1]
  │   └── Loss: Binary Cross-Entropy (BCE)
  │
  ├── MULTICLASS (e.g., Cat vs Dog vs Bird)
  │   ├── Target: One-hot [0, 1, 0] or Integer ID
  │   └── Loss: Categorical Cross-Entropy (Softmax output)
  │
  └── MULTILABEL (e.g., Movie genres: Action AND Sci-Fi)
      ├── Target: Multi-hot [1, 0, 1]
      └── Loss: BCE with Logits Loss (Independent Sigmoids)
```

## Imbalance Handling Decision Tree

```text
CLASSES ARE IMBALANCED (e.g. 99% Negative, 1% Positive)
  │
  ├── 1. Data-Level Fixes
  │   ├── Collect more minority data (Best if possible)
  │   ├── Undersample majority (If dataset is massive)
  │   ├── Oversample minority / SMOTE (Often causes overfitting, do AFTER splitting)
  │   └── Augmentation (Images/Text minority classes)
  │
  ├── 2. Model-Level Fixes
  │   ├── Class Weights (Pass `class_weight='balanced'` in sklearn)
  │   └── Focal Loss (Down-weights easy examples in NNs)
  │
  └── 3. Post-Processing Fixes
      └── Threshold Tuning (Move decision boundary from 0.5 to e.g. 0.1)
```
> [!WARNING]
> If you use SMOTE to create synthetic minority examples, you **MUST** apply it only to the training set. If you apply SMOTE before Train/Test split, synthetic data leaks into the test set resulting in 99% fake accuracy.

## Classification Model Decision Map

| Model | Preprocessing | Strengths | Typical Ranges |
|---|---|---|---|
| **Logistic Regression** | Scale features, One-Hot | Strong baseline, outputs probabilities | `C=0.01 to 100` (Inverse regularization) |
| **Naive Bayes** | None | Extremely fast, great for text/TF-IDF | `alpha=1.0` (Smoothing) |
| **SVM** | MUST scale | High accuracy on complex boundaries | `C=0.1-10`, `kernel='rbf'` |
| **XGBoost/LightGBM** | Label encode categories | Dominates tabular classification | `lr=0.01-0.1`, `max_depth=3-8` |
| **CNN / Transformer** | Resize/Tokenize | Images, text, audio classification | `lr=1e-4`, `batch_size=32-128` |

---

# PART 9 & 10 — CLASSIFICATION LOSS + METRICS

Do not rely on Accuracy. Accuracy on a 99% imbalanced dataset is 99% if the model just predicts "0" every time.

## Metric Decision Tree

```text
CLASSIFICATION EVALUATION
  │
  ├── Are classes balanced?
  │   ├── YES → Accuracy is fine.
  │   └── NO → DO NOT USE ACCURACY. Move below.
  │
  ├── What is the cost of mistakes?
  │   ├── False Positives are expensive (e.g. Spam filter deleting real email)
  │   │   └── Optimize for PRECISION.
  │   │
  │   ├── False Negatives are expensive (e.g. Cancer detection missing a tumor)
  │   │   └── Optimize for RECALL (Sensitivity).
  │   │
  │   └── Both matter equally
  │       └── F1-Score (Harmonic mean of Precision and Recall)
  │
  └── Are you evaluating model probability quality (ranking)?
      ├── Balanced Data → ROC-AUC
      └── Imbalanced Data → PR-AUC (Precision-Recall Area Under Curve)
```

## Threshold Tuning & Calibration
By default, `model.predict()` uses a 0.5 threshold.
If Recall is critical, don't change the loss function immediately. Evaluate `model.predict_proba()`.
If `P(Cancer) > 0.1`, output 1 (Cancer). This drastically increases Recall at the cost of Precision.

**Probability Calibration:** Models like Random Forest or SVM output scores that aren't true probabilities. Use `CalibratedClassifierCV` (Platt scaling or Isotonic regression) if you need the output to mean "There is exactly a 70% chance of rain."

---

# PART 11 — UNSUPERVISED PIPELINES

## Clustering Pipeline

```text
DATA → SCALE (Crucial) → DIMENSIONALITY REDUCTION (Optional) → CHOOSE K → CLUSTER → EVALUATE
```

### Clustering Decision Map

| Model | When to use | Hyperparameters | Trade-offs |
|---|---|---|---|
| **K-Means** | Spherical clusters, large datasets | `n_clusters=K` | Assumes equal variance, sensitive to outliers. Must scale data. |
| **Hierarchical** | Need a taxonomy/dendrogram | `linkage='ward'` | Doesn't scale to large N (O(N³)). |
| **DBSCAN** | Clusters of arbitrary shape, outlier detection | `eps`, `min_samples` | No need to pick K. Struggles with varying densities. |
| **GMM** | Soft clustering (probabilities), overlapping groups | `n_components`, `covariance_type` | Harder to fit, prone to local minima. |

**Evaluation:**
Since there are no labels, automated metrics are flawed proxies.
- **Silhouette Score:** Measures cohesion vs separation (-1 to 1). Higher is better.
- **Davies-Bouldin / Calinski-Harabasz:** Variance ratios.
- **Domain Validation:** Ultimately, you must look at the clusters and ask domain experts if they make sense.

## Dimensionality Reduction

### Decision Map

| Model | Purpose | Where it belongs in pipeline | Trade-offs |
|---|---|---|---|
| **PCA** | Feature reduction, removes collinearity | After scaling, before model training | Linear only, loses feature interpretability. |
| **t-SNE** | 2D/3D Visualization of high-dim data | Standalone analysis, never for feature eng | Slow, does not preserve global distances. |
| **UMAP** | 2D/3D Visualization AND reduction | Standalone analysis or feature eng | Faster than t-SNE, preserves global structure better. |

> [!WARNING]
> DO NOT use t-SNE as a preprocessing step before training a classifier. It does not learn an explicit mapping function `f(X)` that can be applied to new test data. Use PCA or UMAP for pipeline reduction.


---

# PART 12 — NEURAL NETWORK MASTER PIPELINE

This is the standard deep learning loop. Every framework (PyTorch, TensorFlow, JAX) follows this logic.

```text
DATASET (Defines how to load 1 item)
      ↓
DATALOADER (Batches items, shuffles, multiprocessing)
      ↓
(LOOP START) 
GET BATCH (Move to GPU: X.to(device), y.to(device))
      ↓
FORWARD PASS (Predictions = model(X))
      ↓
CALCULATE LOSS (loss = loss_fn(Predictions, y))
      ↓
ZERO GRADIENTS (optimizer.zero_grad() — CRITICAL)
      ↓
BACKPROPAGATION (loss.backward() — compute gradients)
      ↓
GRADIENT CLIPPING (Optional stability step)
      ↓
OPTIMIZER STEP (optimizer.step() — update weights)
      ↓
EMA UPDATE (Optional: update Exponential Moving Average weights)
      ↓
NEXT BATCH
(LOOP END)
      ↓
VALIDATION (with torch.no_grad(), model.eval())
      ↓
SCHEDULER STEP (Adjust LR based on epoch/val loss)
      ↓
CHECKPOINT (Save model.state_dict())
```

## Pipeline Modifiers & Where They Go

| Modifier | Where it goes | Purpose |
|---|---|---|
| **Weight Initialization** | Before training loop | E.g. Xavier/Kaiming. Prevents vanishing/exploding gradients at step 1. |
| **Mixed Precision (AMP)** | Wraps forward pass & loss | Speeds up training, saves VRAM by using FP16/BF16. |
| **Gradient Accumulation** | Divides loss, calls `.step()` every N batches | Simulates large batch sizes on small VRAM GPUs. |
| **Gradient Clipping** | Between `.backward()` and `.step()` | Prevents exploding gradients (common in RNNs/Transformers). |
| **Weight Decay (L2)** | Inside Optimizer definition | Penalizes large weights to prevent overfitting. |
| **Dropout** | Inside Model Architecture | Randomly zeroes activations during training to force redundancy. |

---

# PART 13 — NEURAL NETWORK ARCHITECTURE DECISION TREE

> **“Why would I use this architecture instead of that one?”**

```text
WHAT IS THE UNDERLYING STRUCTURE OF THE DATA?
│
├── No spatial or sequential relationship (Tabular/Simple Vectors)
│   └── MULTILAYER PERCEPTRON (MLP)
│
├── Local spatial relationships matter (Images, Audio Spectrograms)
│   └── CONVOLUTIONAL NEURAL NETWORK (CNN)
│
└── Sequential relationships matter (Text, Time Series, DNA)
    ├── Short/Medium context, sequential dependence
    │   └── RNN / LSTM / GRU
    └── Long context, highly parallelizable, complex interactions
        └── TRANSFORMER
```

### Depth & Width Decision Map

- **When to add 1 Convolutional Layer?** When detecting simple, low-level edges/colors.
- **When to add multiple layers?** When you need to compose simple features into complex ones (edges → textures → object parts).
- **When to increase channels (width)?** When the network needs to memorize/represent more variations of features at a given spatial resolution.
- **When to downsample (stride/pooling)?** When you want to increase the *receptive field* (look at a wider area of the original image) and reduce computational memory.
- **When to use Residual Connections?** Whenever the network is deeper than ~10-15 layers. It prevents vanishing gradients and allows the network to easily learn the identity function (skipping a layer if it isn't needed).

---

# PART 14 — CNN PIPELINE

```text
IMAGE PATHS
  ↓
VALIDATE & DECODE (Check corruptions)
  ↓
RESIZE (e.g., 256x256)
  ↓
CROP (e.g., RandomCrop to 224x224)
  ↓
AUGMENTATION (Flip, Rotate, Color Jitter - Train Only!)
  ↓
TENSOR (Scale to [0, 1])
  ↓
NORMALIZATION (Mean/Std subtraction)
  ↓
BATCHING (B, C, H, W)
  ↓
=== CNN ARCHITECTURE ===
[ Conv Layers (Extract Features) ]
  ↓
[ Feature Maps (C channels, smaller H, W) ]
  ↓
[ Global Average Pooling (Flattens H,W to 1x1) ]
  ↓
[ Classifier Head (Linear Layer(s)) ]
=========================
  ↓
LOSS → BACKPROP → OPTIMIZER → VALIDATION
```

## Historic & Modern Architecture Evolution

| Architecture | Problem it Addressed | Core Idea | When to use |
|---|---|---|---|
| **LeNet (1998)** | Digit recognition | Convolutions + Pooling | Historical reference only. |
| **AlexNet (2012)** | ImageNet scale | ReLU, Dropout, GPU training | Historical reference only. |
| **VGG (2014)** | Finding ideal kernel sizes | Stack tiny 3x3 kernels to simulate large ones | When you need a simple, uniform feature extractor. Slow & heavy. |
| **Inception (2014)** | Varying object scales | Apply 1x1, 3x3, 5x5 in parallel (Inception module) | Historical reference. |
| **ResNet (2015)** | Vanishing gradients in deep nets | Residual (Skip) Connections | **Default starting point** for any image task (ResNet18/34/50). |
| **DenseNet (2017)** | Feature reuse | Connect every layer to every subsequent layer | Medical imaging (highly efficient parameter usage). |
| **EfficientNet (2019)**| Scaling up networks | Scale depth, width, and resolution simultaneously | **Industry standard** for high accuracy per FLOP. |
| **ConvNeXt (2022)** | Transformers beating CNNs | Modernized ResNet using ViT training tricks | Modern SOTA CNN alternative to ViTs. |

---

# PART 15 — CNN ARCHITECTURE COMPONENT DECISION MAP

## 1. Convolution
- **Kernel Size:** Default is `3x3`. Use `7x7` or `5x5` only in the very first layer (stem) to aggressively downsample raw pixels. Use `1x1` to reduce channel dimensions without spatial change (bottleneck).
- **Stride:** Default is `1`. Use `2` to downsample spatial dimensions (H, W) by half.
- **Padding:** Usually `padding = kernel_size // 2` to keep spatial dimensions identical.
- **Dilation:** Use for Segmentation (e.g. DeepLab). Increases receptive field without increasing parameters or losing resolution.

## 2. Pooling
- **Max Pooling:** Preserves the most prominent feature (e.g., highest activation). Used early in networks to aggressively downsample.
- **Average Pooling:** Smooths features.
- **Adaptive Average / Global Average Pooling (GAP):** Used at the very end of the network. Converts `[Batch, Channels, H, W]` to `[Batch, Channels, 1, 1]`. Eliminates the need for massive flattened Linear layers, making the network agnostic to input image size!

## 3. Normalization (WHERE does it go?)
```text
STANDARD PATTERN: Conv → BatchNorm → Activation (ReLU)
MODERN PATTERN (Pre-activation): BatchNorm → Activation → Conv
```
- **BatchNorm:** Default for CNNs. Normalizes across the batch dimension. *Fails if batch size is very small (e.g., < 8).*
- **LayerNorm:** Normalizes across the channel dimension. Default for Transformers and ConvNeXt. Unaffected by batch size.
- **GroupNorm:** Divides channels into groups. Use for CNNs when batch size must be small (e.g., Medical 3D imaging, large object detectors).

## 4. Activation
- **ReLU:** Default. Fast, simple.
- **LeakyReLU:** Prevents "dead ReLUs" (neurons that output 0 forever).
- **GELU:** Smoother than ReLU. Standard for Transformers, ViTs, and ConvNeXt.
- **Sigmoid:** Output layer for Binary Classification or Multilabel.
- **Softmax:** Output layer for Multiclass Classification.

---

# PART 16 — “WHEN DO I FREEZE LAYERS?” DECISION MAP

## Transfer Learning Decision Tree

```text
Do you have a pre-trained backbone? (e.g. ResNet trained on ImageNet)
         │
    How much target data do you have?
    │                           │
 SMALL DATASET              LARGE DATASET
    │                           │
How similar is              Fine-tune the entire network
the target domain?          (Full fine-tuning). Use a very
    │      │                small learning rate (e.g., 1e-5).
  VERY   VERY
 SIMILAR DIFFERENT
    │      │
    A      B

A (Small Data, Similar Domain - e.g., Dogs vs Cats):
├── Freeze entire backbone.
├── Replace classifier head.
├── Train head only with normal LR (1e-3).
└── (Optional) Unfreeze last block of backbone and fine-tune with tiny LR (1e-5).

B (Small Data, Different Domain - e.g., X-ray classification using ImageNet backbone):
├── Freezing the backbone is risky because ImageNet features don't look like X-rays.
├── Freeze early layers (basic edges/textures transfer well).
├── Train later layers + head.
└── Expect high risk of overfitting. Use heavy augmentation.
```

### PyTorch Implementation: Freezing

```python
import torch.nn as nn
import torchvision.models as models

# 1. Load Pretrained
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 2. Freeze Backbone
for param in model.parameters():
    param.requires_grad = False

# 3. Replace Head (New layers have requires_grad=True by default)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2) # e.g., Binary classification

# 4. Only pass unfrozen parameters to optimizer
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
```

### Gradual Unfreezing Strategy
If performance plateaus after training the head:
1. Load best checkpoint.
2. Unfreeze the last convolutional block.
3. Lower learning rate significantly (e.g., `1e-3` → `1e-5`).
4. Re-train to fine-tune the high-level feature extractors without destroying them (Catastrophic Forgetting).


---

# PART 17 — RNN / LSTM / GRU PIPELINE

```text
SEQUENCE DATA
  ↓
TOKENIZATION (or time-step chunking)
  ↓
EMBEDDING
  ↓
HIDDEN STATE INITIALIZATION (Usually zeros)
  ↓
FOR t IN 1...T (Unrolled loop):
    Output_t, Hidden_t = RNNCell(Input_t, Hidden_{t-1})
  ↓
LOSS CALCULATION
  ↓
BACKPROPAGATION THROUGH TIME (BPTT)
```

## Sequence Modeling Decision Tree

```text
SEQUENCE TASK
      │
      ├── Is the sequence highly parallelizable and very long (>500 steps)?
      │   └── TRANSFORMER (e.g. LLM, Vision Transformer). Default for modern NLP.
      │
      └── Is memory/compute highly restricted, or data truly continuous time-series?
          │
          ├── Is long-term dependency crucial?
          │   ├── NO → Standard RNN (Prone to vanishing gradients).
          │   └── YES → LSTM or GRU
          │
          └── LSTM vs GRU?
              ├── LSTM: Has Cell State + Hidden State, 3 gates (Forget, Input, Output). Highly expressive.
              └── GRU: Only Hidden State, 2 gates (Reset, Update). Faster, less VRAM, often similar accuracy.
```

### Key Concepts
- **Bidirectional:** Reads sequence left-to-right AND right-to-left. Use for translation/classification. **Cannot be used for causal generation (next-word prediction).**
- **Teacher Forcing:** During training, feed the *true* previous token to step `t` instead of the model's generated token from `t-1`. Prevents early mistakes from destroying the rest of the sequence during training.

---

# PART 18 — TRANSFORMER PIPELINE

```text
INPUT (Text or Image Patches)
  ↓
TOKENIZATION (e.g., BPE, SentencePiece)
  ↓
TOKEN IDs
  ↓
EMBEDDINGS (Vector lookup)
  ↓
POSITIONAL INFORMATION (Sinusoidal or RoPE to inject order awareness)
  ↓
=== TRANSFORMER BLOCK (Repeated N times) ===
    [ Q, K, V Generation (Linear projections) ]
      ↓
    [ Self-Attention (or Masked for Decoder) ]
      ↓
    [ Multi-Head Attention (Concat heads) ]
      ↓
    [ Residual Add + Normalization (LayerNorm/RMSNorm) ]
      ↓
    [ Feed Forward Network (FFN) ]
      ↓
    [ Residual Add + Normalization ]
============================================
  ↓
OUTPUT (Logits over vocabulary)
```

## Architecture Decisions

| Component | Choice | When to Use |
|---|---|---|
| **Macro Architecture** | **Encoder-Only (BERT)** | Understanding, classification, NER, embeddings. Sees whole sequence at once (Bidirectional). |
| | **Decoder-Only (GPT, Llama)**| Generation, chat, reasoning. Left-to-right autoregressive (Causal Masking). |
| | **Encoder-Decoder (T5)** | Translation, Summarization. Reads source completely, generates target sequentially. |
| **Attention** | **Self-Attention** | Tokens attending to other tokens in the *same* sequence. |
| | **Cross-Attention** | Decoder attending to the Encoder's output, or Text attending to Image tokens. |
| | **Causal Masking** | Prevents token `t` from looking at `t+1`. Mandatory for decoder-only models. |
| **Position** | **Absolute / Sinusoidal** | Fixed encodings (Older models like original Transformer). |
| | **RoPE (Rotary)** | Encodes relative distance between tokens via rotation matrices. Standard in Llama/Mistral. Extrapolates better to longer contexts. |

---

# PART 19 — TRANSFORMER TRAINING

## Objectives
- **Masked Language Modeling (MLM):** Hide 15% of words, predict them. (BERT).
- **Causal Language Modeling (CLM):** Predict next token. (GPT/LLaMA).
- **Sequence-to-Sequence (Seq2Seq):** Input sequence → output sequence. (T5, Translation).
- **Contrastive:** Push similar texts together in vector space, pull apart dissimilar. (CLIP/Embeddings).

## Optimization Pipeline

```text
LOSS → BACKPROP → GRADIENT CLIPPING → OPTIMIZER → LR SCHEDULE
```

### Essential Hyperparameters
- **Optimizer:** `AdamW`. (Weight decay decoupled from gradient updates).
- **Learning Rate Warmup:** Start LR at 0 and linearly increase for the first ~5% of steps. Without this, Adam's initial variance estimates are wild and the model diverges (NaN loss).
- **Learning Rate Decay:** Cosine decay is standard. Drops LR to ~10% of max value by the end of training.
- **Gradient Clipping:** Clip gradients to a max norm (usually `1.0`). Prevents exploding gradients which plague self-attention.
- **Mixed Precision:** BF16 (Brain Float 16) is the modern standard (handles larger ranges than FP16 without underflow).

---

# PART 20 — PRETRAINING VS FINE-TUNING DECISION MAP

```text
Do I have a pretrained model?
        │
    YES ──────────── (NO: Train from scratch - requires massive compute/data)
        │
Is my domain extremely different? (e.g. Legal, Medical, Code)
    │           │
   YES          NO (I just want it to do a specific task/chat)
    │           │
    │      How many GPUs/Compute do I have?
    │        │                         │
    │     PLENTY (Clusters)         LIMITED (1-4 GPUs)
    │        │                         │
Continued Pretraining         Parameter-Efficient Fine-Tuning (PEFT)
(Predict next token on       │
 massive raw corpus)         ├── LoRA (Low-Rank Adaptation)
                             │   Freeze base model, train tiny adapter matrices.
                             │
                             └── QLoRA (Quantized LoRA)
                                 Load base model in 4-bit, train adapters in 16-bit.
                                 (Can train a 7B model on a single 24GB GPU).
```

### Distinctions:
- **Pretraining:** Learning to speak English. (Huge raw text).
- **Supervised Fine-Tuning (SFT) / Instruction Tuning:** Learning to answer questions politely. (Format: Prompt/Response pairs).
- **RLHF/DPO:** Learning which polite answer humans actually prefer.

---

# PART 21 — MULTIMODAL / CLIP PIPELINE (Dual Encoders)

```text
IMAGE                    TEXT
  ↓                        ↓
Vision Encoder           Text Encoder (Tokenizer + Transformer)
(e.g., ViT)              (e.g., BERT-like)
  ↓                        ↓
Image Embedding          Text Embedding
  ↓                        ↓
Projection Head          Projection Head (Map to same dimension)
  ↓                        ↓
L2 Normalize             L2 Normalize
  ↘                        ↙
   Dot Product (Cosine Similarity)
                 ↓
            Temperature (Learned parameter to scale logits)
                 ↓
      Contrastive Loss (InfoNCE)
(Maximize similarity of (Image_i, Text_i), minimize (Image_i, Text_j))
```

**What is it doing?** Learning a shared vector space. The text "A photo of a dog" and an actual photo of a dog should produce nearly identical vectors.
**When to use:** Image retrieval, zero-shot classification.

---

# PART 22 — VLM (VISION-LANGUAGE MODEL) PIPELINE

A modern VLM (like LLaVA or GPT-4V) acts like an LLM that can "see."

```text
IMAGE
  ↓
Vision Encoder (usually frozen CLIP backbone)
  ↓
Visual Features (Grid of patch embeddings)
  ↓
CONNECTOR / PROJECTOR (Crucial step!)
(Projects visual dimension to LLM hidden dimension.
 E.g., a simple Linear layer or a small MLP)
  ↓
Multimodal Tokens (Image tokens look exactly like Word tokens now)
  ↓
Combine with Tokenized Text Prompt
  ↓
LLM (Decoder-only Transformer)
  ↓
OUTPUT TEXT
```

### VLM Training Pipeline
1. **Pretraining (Modality Alignment):** Freeze Vision Encoder + Freeze LLM. Train ONLY the Connector using huge amounts of simple Image-Caption pairs. (Teaches the LLM what the vision tokens mean).
2. **Visual Instruction Tuning (SFT):** Keep Vision Encoder frozen. Unfreeze Connector + Unfreeze LLM (or use LoRA on LLM). Train on complex instructions ("Why is this meme funny?").

---

# PART 23 — DATA CURATION FOR LARGE-SCALE TRAINING

> **Data beats algorithms. Always.**

```text
RAW DATA (Scraped web, docs, PDFs)
  ↓
INGESTION (Extract raw text/images)
  ↓
VALIDATION (Check encoding, corrupt images)
  ↓
FILTERING (Language id, NSFW removal, length heuristics)
  ↓
DEDUPLICATION (MinHash/LSH for exact/fuzzy overlap - CRITICAL to prevent memorization)
  ↓
QUALITY SCORING (Use a smaller classifier to rate document quality)
  ↓
LEAKAGE CHECKS (Remove benchmark/test-set data from training corpus)
  ↓
SPLITTING (Train/Val)
  ↓
SHARDING (Split dataset into 1000s of chunks for distributed loading)
  ↓
STREAMING (Dataloader streams shards to GPUs to avoid RAM exhaustion)
```

**Where does this happen?** ENTIRELY before the training loop starts. Distributed training will stall (GPU utilization = 0%) if data curation and streaming are not highly optimized.


---

# PART 24 — OPTIMIZATION MASTER MAP

## A. Model & Architecture Optimization
| Parameter | Purpose | Typical Range | Increase When | Decrease When | Symptoms if Wrong |
|---|---|---|---|---|---|
| **Depth** (Layers) | Learn complex hierarchies | 3 to 100+ | Underfitting, data is very complex | Overfitting, slow inference | Too deep w/o residuals = vanishing gradients. |
| **Width** (Channels/Dim) | Memorize more variations | 64 to 4096 | Model misses nuances, underfits | VRAM OOM, overfitting | Too wide = massive parameter bloat. |
| **Num Heads** (Attn) | Focus on different parts of input | 4 to 32 | Complex relationships needed | Memory limits | Usually tied to `hidden_dim / 64`. |

## B. Objective & Optimizer Optimization
| Parameter | Purpose | Typical Range | Increase When | Decrease When | Symptoms if Wrong |
|---|---|---|---|---|---|
| **Learning Rate (LR)** | Step size for updates | `1e-5` to `1e-2` | Loss barely moves (Underfitting) | Loss jumps around, NaNs | NaNs or sudden spikes = LR too high. |
| **Weight Decay** | L2 Regularization inside AdamW | `1e-4` to `0.1` | Overfitting | Underfitting | If 0, model overfits; if too high, model can't learn. |
| **Momentum** (SGD) | Accelerate past local minima | `0.9` | SGD learning is too slow | Loss oscillations | N/A |
| **Warmup Steps** | Stabilize Adam early on | 1% to 5% of total steps | NaNs at step 10-100 | Wasting compute | Immediate NaNs in Transformers. |

## C. Learning Rate Schedules
- **Constant:** Good for simple models. Bad for deep learning (plateaus).
- **Step Decay / ReduceLROnPlateau:** Drops LR by 10x when validation loss stops improving. Standard for CNNs.
- **Cosine Decay:** Smoothly drops LR to ~0 by end of training. Standard for Transformers/LLMs.

## D. Memory Optimization
| Technique | What it does | When to use | Trade-off |
|---|---|---|---|
| **Batch Size** | Number of samples per step | Always maximize to fit VRAM | Small batch = noisy gradients; Large = VRAM OOM. |
| **Gradient Accumulation** | `step()` every N batches | When max batch size is too small | Slower wall-clock time. |
| **Mixed Precision (AMP)** | Use FP16/BF16 | Almost always | Can cause NaNs if FP16 scales poorly (use BF16). |
| **Activation Checkpointing** | Recomputes forward pass during backprop | Massive models, OOM issues | Saves VRAM but increases training time by ~30%. |

---

# PART 25 — NORMALIZATION MASTER MAP

> [!CAUTION]
> **DO NOT CONFUSE DATA NORMALIZATION WITH ARCHITECTURAL NORMALIZATION.**

### 1. Data Preprocessing Normalization (Before Model)
Applied to RAW DATA to ensure inputs are on the same scale (usually ~[0, 1] or mean 0, std 1).
- **StandardScaler:** Tabular data (Neural Nets, SVM, Linear models).
- **MinMaxScaler:** Pixel values (divide by 255).
- **RobustScaler:** Tabular data with extreme outliers.

### 2. Neural Network Architectural Normalization (Inside Model)
Applied to ACTIVATIONS inside the neural network during the forward pass to keep gradients stable.
- **BatchNorm:** Standard for CNNs. Normalizes across the batch. **Fails if batch size < 8.**
- **LayerNorm:** Standard for Transformers. Normalizes across the channels of a single token. Unaffected by batch size.
- **GroupNorm:** Modern CNNs/Medical Imaging. Splits channels into groups. Unaffected by batch size.

---

# PART 26 — REGULARIZATION (OVERFITTING DECISION TREE)

```text
OVERFITTING (Training Loss goes down, Validation Loss goes UP or plateaus)
│
├── 1. DATA FIXES (Always try these first)
│   ├── Get more data (The ultimate regularizer)
│   ├── Data Augmentation (Crop, flip, color jitter, text synonym swap)
│   └── Clean the data (Remove mislabeled validation examples)
│
├── 2. ARCHITECTURE FIXES
│   ├── Smaller model (Reduce depth/width)
│   ├── Dropout (Usually 0.1 to 0.5. Add before Linear/Dense layers)
│   └── Transfer Learning (Use pre-trained weights, freeze the backbone)
│
└── 3. OPTIMIZATION FIXES
    ├── Early Stopping (Stop when Val Loss rises for N epochs)
    ├── Weight Decay (Increase L2 penalty in AdamW)
    └── Label Smoothing (Target becomes [0.9, 0.1] instead of [1, 0]. Prevents overconfidence)
```

---

# PART 27 — DATA AUGMENTATION MASTER MAP

> **Augmentation belongs in the DATA PIPELINE (DataLoader), not the model architecture.**

### Image Augmentation
- **Geometric:** Random Crop, Flip (Horizontal only for most objects), Rotation.
- **Color:** Jitter (brightness/contrast), Gaussian Blur.
- **Advanced:** Cutout (black box over image), MixUp (blend 2 images and their labels), CutMix (paste patch of image A onto image B).

### Text Augmentation
- Back-translation (English → French → English).
- Synonym replacement, Random word deletion. (Careful: Can destroy sentence logic).

> [!WARNING]
> **When Augmentation is Harmful:**
> - Vertical Flip on cars/pedestrians (they don't walk upside down).
> - Color jitter on medical tumors where color implies malignancy.
> - Rotation on digits (a 6 becomes a 9).

---

# PART 28 — IMBALANCE DECISION TREE

```text
CLASSES ARE HEAVILY IMBALANCED
│
├── Can you get more minority data?
│   └── YES → Do it.
│
├── NO. What model are you using?
│   ├── TREE/XGBOOST → Use `scale_pos_weight` or `class_weight='balanced'`.
│   │
│   ├── LOGISTIC/LINEAR → Use `class_weight='balanced'`, maybe SMOTE.
│   │
│   └── NEURAL NETWORK
│       ├── Use Weighted Cross-Entropy Loss.
│       ├── Over-sample minority class in the DataLoader (RandomSampler).
│       └── Use Focal Loss (downweights easy majority examples).
│
└── EVALUATION
    └── NEVER use Accuracy. Use PR-AUC, F1, or Recall depending on business need.
```

---

# PART 29 — OUTLIER DECISION TREE

```text
OUTLIERS DETECTED (e.g., Age = 200, or Salary = $50M)
│
├── Is it an ERROR? (Age = 200)
│   └── YES → Remove row, or impute with median.
│
└── Is it a LEGITIMATE rare observation? (Salary = $50M)
    ├── Are you predicting it? (Regression target)
    │   └── Use Log-Transform or Huber Loss to prevent it from dominating gradients.
    │
    └── Is it an input feature?
        ├── Tree Model → Do nothing. Trees isolate outliers easily.
        └── Linear/NN → Use RobustScaler, Winsorize (cap at 99th percentile).
```

---

# PART 30 — LABEL QUALITY DECISION TREE

```text
LABELS ARE BAD
│
├── MISSING: Use semi-supervised learning or drop.
├── INCONSISTENT (Annotator A says Cat, B says Dog): Use Soft Labels (Cat 0.5, Dog 0.5).
├── CORRUPTED/NOISY: Use robust loss functions (e.g., Label Smoothing, MAE instead of MSE).
└── AMBIGUOUS: Can it be both? Move to Multilabel classification.
```

---

# PART 31 — LEAKAGE DECISION TREE

Leakage means your model has information during training that it won't have in production. It leads to 99% accuracy in testing and 50% in reality.

| Type | How it Happens | How to Detect | How to Fix |
|---|---|---|---|
| **Target Leakage** | Feature is a direct proxy for the target (e.g., predicting "Will cancel account", feature is "Cancellation date"). | Feature importance shows 1 feature at 99%. | Drop the feature. |
| **Preprocessing Leakage** | `StandardScaler.fit()` on whole dataset before splitting. | Not obvious from metrics. | Use Sklearn Pipeline. Fit on train only. |
| **Temporal Leakage** | Random split on time-series data. Model learns from the future. | Extremely high accuracy on stock/time data. | Split chronologically (Train: 2020, Test: 2021). |
| **Entity/Duplicate Leakage** | Patient A has 5 scans. 3 in train, 2 in test. Model memorizes Patient A's anatomy, not the disease. | High test accuracy, fails on new patients. | Use `GroupShuffleSplit` on Patient ID. |
| **Augmentation Leakage** | Augmented test images end up in training. | Validation score is suspiciously identical to train. | Split train/val FIRST, then augment train only. |

---

# PART 32 — TRAINING FAILURE DEBUGGING TREE

```text
WHAT IS THE FAILURE?
│
├── 1. LOSS = NaN
│   ├── Check Learning Rate (usually too high).
│   ├── Check Mixed Precision (FP16 underflow → Switch to BF16).
│   ├── Check Data (NaNs or Infs in input).
│   └── Exploding Gradients → Add Gradient Clipping.
│
├── 2. LOSS DOES NOT DECREASE AT ALL
│   ├── Check Labels (Are they randomized/detached from inputs?).
│   ├── Check Optimizer (Is `optimizer.step()` being called?).
│   ├── Check Learning Rate (Too low = 1e-8, or too high = 10.0).
│   └── Check Model Capacity (Is a linear model trying to learn a spiral?).
│
├── 3. TRAINING GOOD, VALIDATION BAD
│   └── OVERFITTING. Check leakage, add dropout, add weight decay, augment data.
│
├── 4. BOTH TRAINING AND VALIDATION BAD
│   └── UNDERFITTING. Model is too weak, LR is wrong, or data is garbage.
│
├── 5. GPU UTILIZATION LOW (< 50%)
│   ├── DataLoader Bottleneck: CPU is too slow at augmenting/loading.
│   │   → Fix: Increase `num_workers`, use faster storage (NVMe), pre-process offline.
│   └── Batch size too small.
│
└── 6. CUDA OUT OF MEMORY (OOM)
    ├── Decrease Batch Size.
    ├── Decrease Image Resolution / Sequence Length.
    ├── Use Mixed Precision (AMP).
    ├── Use Gradient Accumulation.
    └── Use Activation Checkpointing.
```


---

# PART 33 & 34 — EVALUATION PIPELINE & COMPUTER VISION EVALUATION

> **Evaluation is not just one number. It is a multi-stage process.**

## Metric Distinctions
- **Training Metric:** Smooth, differentiable loss (e.g., Cross-Entropy, MSE).
- **Validation Metric:** Human-readable (e.g., Accuracy, F1, RMSE) to guide early stopping and HPO.
- **Test Metric:** Unbiased evaluation on hold-out set.
- **Business Metric:** The actual ROI (e.g., "Dollars saved," "Click-through rate"). The ML metric must act as a proxy for this.

## Computer Vision Specific Metrics
| Task | Metric | Meaning Operationally |
|---|---|---|
| **Detection** | **IoU (Intersection over Union)** | Area of Overlap / Area of Union between predicted box and ground truth. Threshold usually 0.5. |
| | **mAP (Mean Average Precision)** | Area under the PR curve across all classes and multiple IoU thresholds. The gold standard for detectors (YOLO, DETR). |
| | **NMS (Non-Max Suppression)** | Not a metric, but post-processing. Removes duplicate bounding boxes overlapping the same object. |
| **Segmentation** | **Dice Score / F1** | `2 * Overlap / (Total Pixels Pred + Total Pixels True)`. Penalizes false positives. |
| | **Pixel Accuracy** | % of pixels classified correctly. Misleading if background is 99% of the image. |

---

# PART 35 — GENERATIVE / FOUNDATION MODEL EVALUATION

Automated metrics for generative AI are fundamentally flawed because language is flexible.

| Metric | Task | What it measures | Limitation |
|---|---|---|---|
| **BLEU** | Translation | Exact N-gram overlap | Penalizes valid synonyms ("The cat sat" vs "A feline rested"). |
| **ROUGE** | Summarization | Recall-focused N-gram overlap | Same as BLEU. |
| **METEOR** | Translation | Overlap, but includes stemming/synonyms | Better than BLEU, still brittle. |
| **Perplexity** | Language Modeling | How "surprised" the model is by a real test set | Lower is better, but doesn't guarantee factual accuracy or coherence. |
| **FID (Fréchet Inception Distance)** | Image Gen (GAN/Diffusion) | Distance between feature vectors of real vs fake images | Requires huge sample sizes to be statistically valid. |

**Modern Evaluation:** LLM-as-a-Judge (using GPT-4 to grade outputs), human preference voting (Elo rating / Chatbot Arena), and strict benchmark harnesses (MMLU, HumanEval).

---

# PART 36 — INFERENCE + MODEL OPTIMIZATION

```text
TRAINED MODEL (PyTorch .pt)
  ↓
EXPORT (ONNX / TorchScript)
  ↓
PRECISION REDUCTION (Quantization)
  ↓
SERVING (TensorRT, vLLM)
```

## Inference Optimization Techniques

| Technique | What it does | Trade-offs |
|---|---|---|
| **Quantization** | Converts FP32 (32-bit) weights to FP16, INT8, or INT4. | Saves massive memory/compute. Tiny accuracy drop (INT8) or moderate drop (INT4). |
| **Pruning** | Sets near-zero weights to exactly 0 (Sparsity). | Hard to get actual speedups on modern GPUs without structured pruning. |
| **Knowledge Distillation** | Train a small model (Student) to copy the outputs of a large model (Teacher). | Requires re-training. Very effective. |
| **KV Cache** | (LLMs only) Saves previously computed Attention Keys and Values. | Massive memory usage during long context generation, but 10x-100x speedup. |
| **Dynamic Batching** | (Serving) Combines incoming API requests on the fly into a single batch. | Slightly increases latency (wait time) but massively increases total throughput. |

---

# PART 37 — GPU / TRAINING SYSTEMS

```text
TRAINING BOTTLENECKS
│
├── COMPUTE BOUND (GPU is at 100%, but takes forever)
│   └── Model involves massive matrix multiplications (e.g., CNNs, dense MLPs).
│       → Fix: Use Mixed Precision (Tensor Cores), use a faster GPU (A100/H100).
│
├── MEMORY BOUND (GPU VRAM is full, throws CUDA OOM)
│   └── Model or Batch is too large.
│       → Fix: Gradient Accumulation, Activation Checkpointing, smaller batch.
│
└── I/O BOUND (GPU is at 30%, waiting for data)
    └── CPU or Disk is too slow to feed the GPU.
        → Fix: Fast NVMe drives, `num_workers=8` in DataLoader, offline augmentation.
```

## Distributed Training Map (When to use what)

| Framework | Scale | How it works |
|---|---|---|
| **DDP (Distributed Data Parallel)** | 2-8 GPUs (1 node) | Copies entire model to all GPUs. Splits *data* batch. Averages gradients. Default PyTorch. |
| **FSDP (Fully Sharded Data Parallel)** / **DeepSpeed ZeRO** | Multi-node, massive models (LLMs) | Model is too big for 1 GPU. Shards weights, optimizer states, and gradients across GPUs. |
| **Tensor Parallelism** | Massive LLMs (Megatron) | Splits the actual matrix multiplications across GPUs. |

---

# PART 38 — PYTORCH IMPLEMENTATION MAP

Where do the concepts actually live in code?

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 1. DATA (Dataset handles 1 item, DataLoader handles batching/multiprocessing)
class MyDataset(Dataset): ... 
dataloader = DataLoader(dataset, batch_size=32, num_workers=4)

# 2. ARCHITECTURE (Inherit from nn.Module, define layers in __init__, logic in forward)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(...)
        self.norm = nn.BatchNorm2d(...) # Architectural Normalization
        self.drop = nn.Dropout(0.2)     # Regularization
    def forward(self, x):
        return self.drop(self.norm(self.conv(x)))

model = MyModel().to('cuda')

# 3. OPTIMIZATION
criterion = nn.CrossEntropyLoss() # Objective
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# 4. MIXED PRECISION & SCALER
scaler = torch.cuda.amp.GradScaler()

# 5. THE LOOP
for epoch in range(10):
    model.train() # Set to train mode (enables dropout/batchnorm updates)
    for X, y in dataloader:
        X, y = X.to('cuda'), y.to('cuda')
        
        optimizer.zero_grad() # ALWAYS DO THIS FIRST
        
        with torch.cuda.amp.autocast(): # Mixed Precision
            preds = model(X)
            loss = criterion(preds, y)
            
        scaler.scale(loss).backward() # Backprop
        
        # Gradient Clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer) # Update weights
        scaler.update()
    
    scheduler.step() # Update LR per epoch
```

---

# PART 39 — CODE-LEVEL PIPELINE TEMPLATES (Pseudocode)

### Sklearn Regression/Classification
```python
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', XGBClassifier(learning_rate=0.1))
])
pipe.fit(X_train, y_train)
preds = pipe.predict(X_test)
```

### LoRA / QLoRA Fine-Tuning (Conceptual via HuggingFace PEFT)
```python
# 1. Load Base Model in 4-bit (Quantization saves VRAM)
model = AutoModelForCausalLM.from_pretrained("llama-7b", load_in_4bit=True)

# 2. Define LoRA Config (Train only 1% of parameters)
config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"])
peft_model = get_peft_model(model, config)

# 3. Train using standard loop/Trainer
trainer = Trainer(model=peft_model, train_dataset=data, args=TrainingArguments(...))
trainer.train()
```

---

# PART 40 — "WHERE DOES IT BELONG?" INDEX

| Concept | Pipeline Stage | Before/During/After | Why Used | Alternatives |
|---|---|---|---|---|
| **PCA** | Data Transformation | Before | Reduce dims, remove collinearity | UMAP, Autoencoder |
| **BatchNorm** | Architecture | Inside Loop | Stabilize CNN gradients | LayerNorm, GroupNorm |
| **Dropout** | Architecture | Inside Loop | Prevent overfitting | Weight decay, smaller model |
| **AdamW** | Optimization | Inside Loop | Update weights w/ decoupled L2 | SGD + Momentum |
| **SMOTE** | Augmentation | Before | Fix class imbalance | Class weights, Focal Loss |
| **LoRA** | Fine-Tuning | Inside Loop | PEFT (save memory) | Full fine-tuning, Adapters |
| **ONNX** | Inference | After | Fast deployment | TensorRT, TorchScript |

---

# PART 41 — HYPERPARAMETER CHEAT SHEET

| Model / Technique | Parameter | Typical Range | Good Starting Value |
|---|---|---|---|
| **Random Forest** | `n_estimators` | 100 - 1000 | 200 |
| | `max_depth` | 5 - None | None (let it expand) |
| **XGBoost / LGBM**| `learning_rate` | 0.01 - 0.2 | 0.1 |
| | `max_depth` | 3 - 10 | 6 |
| **CNN / ResNet** | `learning_rate` | 1e-4 - 1e-2 | 1e-3 |
| | `batch_size` | 16 - 256 | 32 (or max VRAM) |
| **Transformers** | `learning_rate` | 1e-5 - 5e-4 | 1e-4 |
| | `warmup_steps` | 1% - 10% of total | 5% |
| **LoRA** | `r` (rank) | 4 - 64 | 8 |
| | `lora_alpha` | 16 - 128 | 32 (often 2x rank) |

---

# PART 42 — "WHAT SHOULD I TRY FIRST?" TABLES

### Small Tabular Regression Dataset
1. **Baseline:** Mean of target.
2. **First Model:** Random Forest (Requires 0 scaling/tuning).
3. **Second Model:** Ridge Regression (Scale data first).
4. **Third Model:** XGBoost (Tune max_depth and LR).
5. **Next:** Inspect residuals. Fix outliers.

### Image Dataset
1. **Baseline:** Pretrained ResNet34.
2. **Training:** Freeze backbone, train a new Linear head for 5 epochs.
3. **Evaluation:** Check confusion matrix.
4. **Optimization:** Unfreeze backbone, train 10 epochs at LR=1e-5. Add RandomCrop/Flip.

### Text Classification
1. **Baseline:** TF-IDF + Naive Bayes or Logistic Regression.
2. **Deep Learning:** Fine-tune a small BERT (e.g., DistilBERT).
3. **LLM:** Zero-shot prompt a frontier model (GPT-4 / Claude) to establish an upper bound.

---

# PART 43 — EXPERIMENTATION LOOP

```text
QUESTION: "Why is the model struggling with small objects?"
  ↓
HYPOTHESIS: "The image resolution (224x224) is destroying small features."
  ↓
BASELINE: ResNet50 at 224x224 (mAP = 0.40)
  ↓
CONTROLLED CHANGE: Change resolution to 448x448. Change NOTHING else.
  ↓
TRAIN & EVALUATE
  ↓
COMPARE: New mAP = 0.45.
  ↓
CONCLUSION: Resolution matters. Keep 448x448. Log to Weights & Biases.
```

---

# PART 44 — FINAL MASTER DECISION TREE

```text
WHAT TYPE OF DATA DO I HAVE?
│
├── TABULAR
│   ├── TARGET IS CONTINUOUS? → Regression (XGBoost, Ridge)
│   ├── TARGET IS CATEGORY? → Classification (XGBoost, Logistic)
│   └── NO TARGET? → Clustering/PCA (KMeans, UMAP)
│
├── IMAGE
│   ├── ONE LABEL PER IMAGE? → Image Classification (ResNet, ConvNeXt, ViT)
│   ├── FIND OBJECTS + BOXES? → Object Detection (YOLO, DETR)
│   └── PIXEL-LEVEL MASKS? → Segmentation (U-Net, Mask R-CNN)
│
├── TEXT
│   ├── CLASSIFICATION? → Fine-tune BERT / Encoder
│   └── GENERATION? → Fine-tune/LoRA LLaMA/GPT / Decoder
│
└── MULTIMODAL
    ├── IMAGE + TEXT MATCHING? → Dual Encoder (CLIP)
    └── IMAGE IN, TEXT OUT? → Vision-Language Model (LLaVA-style connector + LLM)
```

---

# PART 45 — FINAL ONE-PAGE CHEAT SHEET

When starting a project, ask these exactly in order:

1. **What is my task?** (Classification, Regression, Generation?)
2. **What is my target?** (Are there missing values? Is it imbalanced?)
3. **What type of data?** (Tabular, Image, Text?)
4. **What can be wrong with the data?** (Leakage, Outliers, Duplicates?)
5. **What preprocessing is required?** (Imputation, Cleaning)
6. **What transformations?** (Log transform, One-hot, Tokenization)
7. **What split strategy?** (Random, Stratified, Time-based? NEVER leak train to test).
8. **What baseline?** (Mean, Majority class, Random Forest).
9. **What model family?** (Trees for Tabular, CNN for Image, Transformer for Text).
10. **What architecture?** (XGBoost, ResNet50, LLaMA-8B).
11. **What loss?** (MSE, Cross-Entropy, Focal Loss).
12. **What optimizer?** (AdamW, SGD).
13. **What LR?** (1e-3 to 1e-5).
14. **What scheduler?** (Cosine, ReduceLROnPlateau).
15. **What regularization?** (Weight Decay, Dropout, Early Stopping).
16. **What augmentation?** (Crop, Flip, MixUp - Train only).
17. **What batch size?** (Max that fits in VRAM).
18. **What normalization?** (StandardScaler for data, LayerNorm/BatchNorm for architecture).
19. **What should I freeze?** (Freeze backbone, train head. Then unfreeze).
20. **What metric?** (Don't use Accuracy for imbalanced data. Use F1/mAP/IoU).
21. **What failure mode?** (NaN loss? OOM? Overfitting?).
22. **What experiment next?** (Change ONE thing. Document it).
23. **How do I test?** (Untouched test set, confusion matrix, error analysis).
24. **How do I export?** (ONNX, weights file).
25. **How do I optimize inference?** (Quantization, Batching, vLLM).


---

