# MASTER ML/DL PIPELINE & DECISION MAP

*A decision-oriented engineering reference, not a textbook. Enter at any stage, find your choices, pick one, move to the next stage.*

---

## HOW TO USE THIS DOCUMENT

- Don't read linearly. Search for your current problem ("missing target", "overfitting", "CUDA OOM") and jump in.
- Every stage answers: **what happens here → what are my options → when do I pick each → what's next.**
- Tables give ranges/starting points where a genuine convention exists. Where none exists, it says so explicitly instead of inventing false precision.

---

# PART 0 — THE MASTER MAP: PROBLEM FAMILIES

Every ML/DL problem is a walk down this tree. The **backbone stages are shared**; what changes per family is representation, model class, loss, and evaluation.

```text
WHAT AM I BUILDING?
│
├── TABULAR / CLASSICAL SUPERVISED
│   ├── Regression
│   ├── Binary classification
│   ├── Multiclass classification
│   └── Multilabel classification
│
├── TABULAR / UNSUPERVISED
│   ├── Clustering
│   ├── Dimensionality reduction
│   └── Anomaly detection
│
├── TABULAR / NEURAL
│   └── NN-based tabular learning (MLP on structured data)
│
├── VISION
│   ├── Image classification
│   ├── Object detection
│   └── Image segmentation
│
├── SEQUENCE / TEXT
│   ├── RNN/LSTM/GRU sequence modelling
│   ├── NLP (classification, generation, retrieval)
│   └── Transformers
│
├── REPRESENTATION / FOUNDATION MODEL WORK
│   ├── Self-supervised learning
│   ├── Pretraining
│   ├── Transfer learning
│   ├── Fine-tuning (full, LoRA, QLoRA)
│   └── LLM training/fine-tuning
│
├── MULTIMODAL
│   ├── CLIP-style contrastive (dual encoder)
│   └── VLM training/fine-tuning
│
└── PRODUCTIONIZATION
    ├── Inference optimization (quantization, distillation, pruning)
    └── Deployment/serving
```

### What's shared across ALL of these

`DATA → CLEAN → SPLIT → BASELINE → MODEL → LOSS → OPTIMIZE → TRAIN → VALIDATE → ITERATE → TEST → EXPORT → INFER → MONITOR`

This 14-step skeleton is universal. Every family below is this skeleton with modality-specific substitutions at "clean/represent" and family-specific substitutions at "model/loss/evaluate."

### What differs, explicitly

| Dimension | Varies by |
|---|---|
| Representation step | modality (tabular encoding vs tokenization vs pixel tensor) |
| Model family | problem type (tree ensembles for tabular, CNN/ViT for images, Transformer for sequence/text) |
| Loss | task (MSE/regression, cross-entropy/classification, contrastive/multimodal) |
| Evaluation metric | task + business cost of error types |
| "Before training" steps | augmentation, tokenization, feature engineering — all modality-specific |
| "Inside training loop" steps | forward/backward/optimizer step — identical in shape across all NN families; classical ML has no loop (fit is closed-form or iterative internally) |
| "After training" steps | calibration, thresholding, export format — task-specific |
| "Inference-only" steps | KV-cache, batching strategy, quantization — mostly relevant to NN/LLM/VLM, less to classical ML |

### Mutually exclusive choices (pick one path, not both)

- Full fine-tuning **vs** LoRA/QLoRA/adapters (you don't do both on the same weights simultaneously)
- Freeze-then-fine-tune **vs** train-from-scratch (mutually exclusive strategies for the same backbone)
- Oversampling **vs** class-weighting as your *primary* imbalance fix (can combine cautiously, but pick a primary)
- One-hot encoding **vs** target/frequency encoding for the same high-cardinality categorical column

### Optional steps (skip when they don't apply)

Augmentation, dimensionality reduction, resampling, calibration, distributed training, quantization, pseudo-labeling, continued pretraining.

---

# PART 1 — THE UNIVERSAL ML LIFECYCLE

```text
RAW DATA
   ↓
DATA UNDERSTANDING        (inspect, don't touch)
   ↓
DATA VALIDATION           (schema, ranges, types)
   ↓
DATA CLEANING             (missing, dupes, invalid, outliers)
   ↓
DATA REFORMATION          (types, encodings, formats)
   ↓
DATA TRANSFORMATION       (scaling, log, tokenizing)
   ↓
FEATURE / LABEL PROCESSING(engineering, label encoding)
   ↓
SPLITTING                 (train/val/test — BEFORE fitting anything)
   ↓
AUGMENTATION              (train split only, if used)
   ↓
BASELINE                  (dumb model — sets the floor)
   ↓
MODEL SELECTION
   ↓
MODEL INITIALIZATION
   ↓
OBJECTIVE / LOSS
   ↓
OPTIMIZATION (algorithm choice)
   ↓
TRAINING (loop)
   ↓
VALIDATION
   ↓
HYPERPARAMETER OPTIMIZATION  ← loops back to TRAINING
   ↓
ERROR ANALYSIS
   ↓
ITERATION                 ← loops back to any earlier stage
   ↓
FINAL TEST                (touched exactly once)
   ↓
MODEL SELECTION (final)
   ↓
CHECKPOINT / EXPORT
   ↓
INFERENCE
   ↓
MONITORING                ← feeds back into "RAW DATA" for the next cycle
```

**Critical property of this loop:** everything above "FINAL TEST" can be repeated as many times as you want. The test set is touched once — the moment you tune anything based on test performance, it silently becomes a second validation set and your final number is no longer trustworthy.

## Stage-by-stage template

Every stage in this document follows this shape:

> **Stage** → **What happens here** → **Available techniques** → **When to use each** → **Alternatives** → **Trade-offs** → **Typical values/ranges** → **Common mistakes** → **Failure cases** → **What comes next**

---

# PART 2 — DATA UNDERSTANDING PIPELINE

**Goal at this stage: look, don't modify.** Every decision downstream depends on what you find here.

### Structural inspection

```python
df.shape                    # rows/columns
df.dtypes                   # inferred types — verify, don't trust
df.info()
df.head(); df.sample(10)    # eyeball actual rows, not just summary stats
```

### Per-column type audit

Classify every column into one of: **numerical (continuous/discrete), categorical (nominal/ordinal), binary, datetime, text, image, audio, ID/leakage-risk.**

```python
df.nunique()                 # cardinality — flags IDs (nunique ≈ len(df)) and near-constants (nunique == 1)
df.describe()                 # numerical: range, mean, std, quartiles → spot impossible values (age = -5, price = 0)
df.describe(include='object') # categorical: top category, frequency
```

### Missingness and quality

```python
df.isna().sum()               # missing per column
df.isna().mean()              # missing % — >40-50% missing changes strategy (see Part 3)
df.duplicated().sum()         # exact duplicate rows
df.duplicated(subset=[id_cols]).sum()   # duplicate entities under a business key
```

### Target inspection (do this before touching features)

```python
df[target].dtype
df[target].isna().sum()
df[target].value_counts(normalize=True)   # classification: class balance
df[target].describe()                      # regression: range, skew, outliers
df[target].hist()                          # regression: shape of distribution — informs log-transform decision
```

### Consistency checks

- Inconsistent formatting: `"NY"` vs `"New York"` vs `"ny "` — normalize casing/whitespace before treating as categorical.
- Impossible values: negative ages, dates in the future, percentages > 100.
- Corrupted samples: all-zero image, empty text field, truncated audio.
- Train/test distribution differences: compare feature distributions between splits (or across time) — a large shift signals leakage risk or a need for domain adaptation later.

### Leakage scan (early, cheap check)

Ask of every column: *"would this value be available at prediction time, in production, before the target is known?"* If no → drop or lag it now, before it contaminates anything downstream. (Full leakage taxonomy: Part 31.)

**What comes next:** Data Validation/Cleaning (Part 3) if issues were found; otherwise straight to Splitting (Part 6).

---

# PART 3 — MISSING-VALUE DECISION TREE

## Missing feature vs missing target — treat these completely differently

A missing **feature** value is a data-quality problem you impute or flag. A missing **target** value means you don't have a training example — this is a fundamentally different situation because you cannot compute supervised loss on it at all.

## A. Missing FEATURE — numerical

```text
MISSING NUMERICAL FEATURE
│
├── < 5% missing, MCAR-looking → mean/median impute, move on
├── Missingness correlates with other features/target → 
│       add a missing-indicator column + impute
│       (missingness itself may be informative)
├── Missing within a natural group (e.g. per-store, per-user) →
│       group-wise mean/median
├── Structure/order matters (time series) →
│       forward-fill / backward-fill / interpolation
├── Complex relationships between features →
│       KNN imputation or iterative imputation (MICE-style)
├── High missingness (>40-50%) in an otherwise informative column →
│       consider dropping the column, or keep only the missing-indicator
└── Using a model that handles missing values natively
        (many gradient boosting implementations: XGBoost, LightGBM, CatBoost) →
        leave as-is, let the model learn splits around missingness
```

| Method | When | Trade-off |
|---|---|---|
| Drop rows | missing is rare and MCAR | loses data, can bias if not MCAR |
| Mean/median | quick baseline, roughly symmetric/skewed distribution respectively | ignores relationships between features |
| Constant (e.g. 0, -999) | missingness is itself meaningful and model can learn a split on it | can distort scale-sensitive models (linear/NN) unless flagged |
| Forward/backward fill | ordered/time-series data | wrong if the sequence isn't continuous |
| Group-wise mean/median | clear natural grouping exists | breaks down for tiny groups |
| KNN imputation | moderate size data, features correlated | expensive at scale, sensitive to feature scaling |
| Iterative/model-based imputation | complex missingness patterns, enough data | slow, risk of overfitting the imputation itself |
| Missing-indicator feature | missingness correlates with target | doubles feature count if applied everywhere — apply selectively |
| Native handling (tree ensembles) | using XGBoost/LightGBM/CatBoost | not available in linear models, SVM, most NN layers |

## B. Missing FEATURE — categorical

```text
MISSING CATEGORICAL FEATURE
│
├── Missing is rare → mode imputation
├── "Missing" might be a real signal (e.g. field only filled for premium users) →
│       treat as its own explicit category ("Unknown"/"Missing")
├── Natural grouping exists → group-wise mode
└── Complex dependency on other features → predictive imputation
        (train a small classifier to predict the missing category)
```

Prefer **explicit "Unknown" category** over mode-imputation whenever missingness could be non-random — mode imputation silently destroys that signal.

## C. Missing TARGET — this is not the same problem as a missing feature

**Why:** you cannot compute a supervised loss without a target. Blindly filling it (e.g. with the mean) fabricates a label and trains the model on a lie you invented; this is qualitatively different from filling an input feature, where the model still has a real target to learn against.

### Worked example: target contains `?`

Do **not** default to "drop the rows." Work through this instead:

```text
TARGET HAS "?"
        ↓
What does "?" actually mean here? (check the data source / documentation / distribution)
        ↓
   ┌────────────┬─────────────────┬───────────────────────┐
   │            │                 │                        │
"?" = truly   "?" = a real     "?" = censored/       "?" = not yet
missing/       category         truncated value        measured (will
unmeasured    ("did not         (e.g. "income > $200k"    arrive later)
              disclose")         reported as "?")
   │            │                 │                        │
   ↓            ↓                 ↓                        ↓
proceed as   keep as its own   model as censored        treat as a
below         class if this    data (e.g. survival       forecasting/
              is a classifi-   analysis techniques,       streaming
              cation problem   or bucket into the         problem —
                               top open-ended bin)         not a
                                                            missing-value
                                                            problem at all
```

Once you've confirmed it is genuinely missing, decide by **how much is missing and whether you have a use for the unlabeled rows**:

```text
GENUINELY MISSING TARGET
│
├── Small % missing, plenty of labeled data left →
│       remove rows (safe default when loss is negligible)
│
├── Large % missing, unlabeled rows have real feature information →
│       ├── Semi-supervised learning: train on labeled subset,
│       │   use it to pseudo-label the unlabeled subset, retrain
│       │   (works when pseudo-labels are likely accurate — check
│       │    confidence before trusting them)
│       ├── Self-supervised pretraining: pretrain a representation
│       │   on ALL data (ignoring labels), then fine-tune on the
│       │   labeled subset — strong choice when unlabeled data is
│       │   abundant and labeled data is the scarce resource
│       └── Use only where the labeled subset alone is provably
│           insufficient — otherwise this adds complexity without
│           benefit
│
└── Missing is NOT random (e.g. only failures never got labeled) →
        this is a labeling-process bias, not a missing-data problem —
        fixing it requires addressing collection process, not imputation
```

**What comes next:** Data Reformation/Transformation (Part 4), or straight to Splitting if data is already clean.

# PART 4 — DATA REFORMATION / TRANSFORMATION

## Numerical transformations

| Technique | Formula/idea | When useful | When unnecessary | Outlier sensitivity |
|---|---|---|---|---|
| Standardization (z-score) | `(x - mean) / std` | linear models, SVM, NN, PCA, KNN — anything distance/gradient-scale sensitive | tree-based models (splits are scale-invariant) | sensitive — outliers distort mean/std |
| Normalization (min-max) | `(x - min) / (max - min)` | bounded-input needs (e.g. NN inputs into sigmoid), image pixels | when outliers exist (a single extreme value compresses everything else) | very sensitive |
| Robust scaling | `(x - median) / IQR` | data with outliers you want to keep but not let dominate scale | clean, roughly Gaussian data (no benefit over z-score) | robust by design |
| Log transform | `log(x)` or `log1p(x)` for zeros | right-skewed positive data (income, counts, prices) | already symmetric data, or data with negatives/zeros (use log1p or shift) | reduces influence of large values |
| Power transforms (Box-Cox, Yeo-Johnson) | learned power parameter to approach Gaussian | you need near-normal residuals for a linear model | tree-based/NN models that don't assume normality | Box-Cox requires strictly positive values; Yeo-Johnson handles zero/negative |
| Clipping / winsorization | cap values at a percentile (e.g. 1st/99th) | extreme values are believable-but-rare and you want to limit their leverage | when the extreme values are the signal (fraud detection, rare-event modelling) | directly reduces outlier influence |

**Model-dependent scaling requirement:**

```text
Requires scaling: Linear/Logistic Regression, Ridge/Lasso/ElasticNet, SVM, KNN, PCA, NNs
Does not require scaling: Decision Trees, Random Forest, Gradient Boosting (XGBoost/LightGBM/CatBoost)
```

## Categorical encodings

| Encoding | When | Danger |
|---|---|---|
| Label encoding | ordinal categories with a genuine order (e.g. "low/medium/high"), or as raw input to tree models that can split arbitrarily | **applying it to nominal categories implies a false ordering** — a linear model will treat category "3" as "greater than" category "1," which is meaningless if the categories have no order |
| Ordinal encoding | explicit, known ranking (e.g. education level, sizes S/M/L) | wrong to use when categories are not truly ordered |
| One-hot encoding | low-cardinality nominal categories, linear/NN models | cardinality explosion above ~15-50 unique values; sparse, memory-heavy |
| Target encoding | high-cardinality nominal categories (zip code, user ID) | leakage if not done with proper cross-fold/out-of-fold fitting; must be fit on train only |
| Frequency encoding | high-cardinality, when frequency itself is informative | collapses distinct categories with the same frequency into the same value |
| Embeddings (learned) | very high cardinality categories inside a neural network | needs enough data per category to learn a meaningful vector; overkill for small tabular data |
| Hashing | very high/unbounded cardinality, streaming settings | collisions between unrelated categories are possible and untraceable |

**When integer/label encoding accidentally implies ordering:** if you label-encode `{red:0, blue:1, green:2}` and feed it to linear regression, the model reads this as `green > blue > red` numerically — a completely fabricated relationship. Tree models are more forgiving (they just pick threshold splits) but still risk spurious "in-between" splits. Default to one-hot or target encoding for genuinely nominal data unless you have a specific reason (e.g., feeding a tree ensemble and cardinality is very high).

## Text

`raw text → tokenization → vocabulary → (subword: BPE / WordPiece / SentencePiece) → token IDs → embeddings → padding/truncation to fixed length → attention masks (mark real vs pad tokens)`

- Subword tokenization (BPE/WordPiece/SentencePiece) is the modern default — it handles unseen words gracefully by falling back to subword/character pieces, avoiding the huge out-of-vocabulary problem of word-level tokenization.
- Padding+masking is required whenever you batch variable-length sequences together; the mask stops the model from attending to / being penalized on pad tokens.

## Images

`raw image → decode → resize → crop (center for eval, random for train) → augmentation (train only) → tensor conversion → channel-wise normalization (dataset mean/std, e.g. ImageNet stats for transfer learning) → batch`

---

# PART 5 — TRAIN / VALIDATION / TEST SPLITTING

| Split | Purpose |
|---|---|
| Train | fit model parameters |
| Validation | tune hyperparameters, pick architecture, decide when to stop — model never directly trains on this |
| Test | one-shot, final, unbiased estimate of real-world performance |

## Splitting strategies

| Strategy | When |
|---|---|
| Random split | i.i.d. data, no groups/time structure |
| Stratified split | classification with class imbalance — preserves class ratios in each split |
| Group split | multiple rows belong to the same entity (same patient/user) — split by entity, not by row, or the model leaks entity-identity information across splits |
| Time-based / chronological split | anything with a time axis — always train on past, validate/test on future, never randomly shuffle time series |
| K-fold cross-validation | small datasets where a single split wastes data / is noisy |
| Stratified K-fold | small imbalanced classification datasets |
| Leave-one-out | very small datasets (expensive, high variance for larger ones) |
| Nested cross-validation | when you need an unbiased performance estimate *and* you're doing hyperparameter search — outer loop estimates performance, inner loop tunes |

## Leakage through preprocessing — the single most common beginner mistake

```text
WRONG:
  fit scaler/encoder/imputer on ENTIRE dataset
       ↓
  split into train/val/test
  (test statistics have now leaked into what the model "learned"
   about scale/distribution — validation and test scores are
   optimistic and will not hold up in production)

RIGHT:
  split into train/val/test FIRST
       ↓
  TRAIN → fit preprocessing (scaler, encoder, imputer, vocabulary) → transform train
       ↓
  VALIDATION → transform using TRAIN-fitted preprocessing only
       ↓
  TEST → transform using TRAIN-fitted preprocessing only
```

**Exceptions/special cases:**
- Target encoding needs *within-train* out-of-fold fitting even though it's "fit on train" — otherwise a row's own target leaks into its own encoded value.
- Global constants that don't depend on the sample distribution (e.g., fixed physical unit conversions) are safe to apply before splitting.
- In K-fold CV, this fit/transform discipline repeats *inside every fold* — refit preprocessing per fold, don't fit once outside the loop.

**What comes next:** Baseline → Model Selection (family-specific, Parts 7+).

---

# PART 6 — REGRESSION PIPELINE

```text
DATA → CLEANING → TRANSFORMATIONS → FEATURE ENGINEERING → SPLIT
  → BASELINE → MODEL → LOSS → OPTIMIZATION → VALIDATION
  → EVALUATION → ERROR ANALYSIS → ITERATE
```

**Baseline:** predict the mean (or median) of the training target for every row. Any model that can't beat this isn't learning anything.

## Model options

| Model | When to use | Strengths | Weaknesses | Scaling needed? | Interpretability |
|---|---|---|---|---|---|
| Linear Regression | linear relationship, interpretability matters, small-to-medium data | fast, interpretable, no hyperparameters | underfits nonlinear patterns, sensitive to outliers/multicollinearity | yes | high |
| Ridge | many correlated features, want to keep all of them but shrink | handles multicollinearity, stable | doesn't zero out features (no sparsity) | yes | high |
| Lasso | want automatic feature selection (sparse coefficients) | produces sparse, interpretable models | unstable selection under high correlation between features | yes | high |
| Elastic Net | correlated features AND you want sparsity | combines Ridge's stability with Lasso's sparsity | two hyperparameters to tune instead of one | yes | high |
| Polynomial Regression | known nonlinear-but-smooth relationship, few features | captures curvature simply | explodes in dimensionality with many features/high degree, easy to overfit | yes | medium |
| Decision Tree | nonlinear, interpretable-ish, mixed feature types | no scaling needed, handles interactions | overfits easily, unstable (high variance) | no | medium |
| Random Forest | want a strong nonlinear baseline with less tuning | robust to overfitting vs single tree, handles nonlinearity/interactions well | less interpretable, larger memory/compute | no | low-medium |
| Gradient Boosting | want top tabular accuracy, willing to tune | usually the strongest tabular baseline | needs careful tuning to avoid overfitting, slower to train sequentially | no | low |
| XGBoost / LightGBM | large tabular datasets, need speed + strong accuracy | fast, regularized, handles missing values natively | many hyperparameters, easy to overfit if untuned | no | low |
| Neural Networks | very large tabular data, or need to fuse with other modalities | flexible, scales with data | usually loses to boosting on small/medium tabular data, needs more tuning + scaling | yes | very low |

### Elastic Net worked example (pattern repeated throughout this document)

```python
ElasticNet(alpha=1.0, l1_ratio=0.5)
```

```text
alpha:      overall regularization strength.
            Typical search range: 1e-3 to 10 (log scale)
l1_ratio:   0   → pure Ridge (L2 only)
            1   → pure Lasso (L1 only)
            0 < l1_ratio < 1 → mixture of L1+L2
            Practical starting points to try: 0.1, 0.5, 0.9
```

## Common failure modes by model

- Linear/Ridge/Lasso: violated linearity assumption, multicollinearity, heteroscedastic residuals.
- Trees/Forests: overfitting on noisy features, poor extrapolation outside training range.
- Boosting: overfitting with too many rounds + high learning rate, leakage from target encoding done incorrectly.
- Neural nets: underfitting from too little data, unstable training without scaling.

---

# PART 7 — REGRESSION LOSS + EVALUATION

| Metric | Formula/idea | Use when |
|---|---|---|
| MAE | mean absolute error | outliers exist and you don't want them dominating the metric; errors should be penalized linearly |
| MSE | mean squared error | you want to punish large errors much more than small ones; used as the standard training loss for most regressors |
| RMSE | sqrt(MSE) | same as MSE but back in the original units — most commonly reported metric |
| R² | variance explained | communicating overall fit quality, comparing models on the same dataset — not comparable across different datasets |
| Huber loss | quadratic near zero, linear beyond a threshold `δ` | you want MSE's smoothness near zero but MAE's robustness to outliers |
| Log-based losses (e.g. MSLE) | target spans orders of magnitude, relative error matters more than absolute | price/count prediction where a $10 error on a $20 item matters more than a $10 error on a $2000 item |

**Decision rules:**
- Outliers matter (you don't want them to dominate) → MAE or Huber.
- Large errors should be punished strongly (e.g., safety-critical prediction) → MSE/RMSE.
- Interpretability to a non-technical stakeholder → MAE (same units, intuitive) or R² (in a business review).

**Training loss vs validation metric vs test metric — these need not match.** A common pattern: train with MSE (smooth gradients, easy to optimize) but report/validate on MAE or RMSE (more interpretable, more robust to a few bad predictions) if that's what stakeholders actually care about.

**Residual analysis** (do this before trusting any regression metric):
```python
residuals = y_true - y_pred
# plot residuals vs predicted → look for patterns (heteroscedasticity, curvature)
# histogram of residuals → should be roughly centered at 0, no heavy skew
# residuals vs each feature → reveals missed nonlinearity
```

**What comes next:** if metrics/residuals reveal systematic errors → Error Analysis / Iteration (Part 24 onward); if satisfactory → Final Test → Export (Part 36).

# PART 8 — CLASSIFICATION PIPELINES

## Shared backbone

`DATA → CLEANING → ENCODING (features + labels) → IMBALANCE CHECK → SPLIT (stratified) → BASELINE (majority-class predictor) → MODEL → LOSS → VALIDATION → THRESHOLD/CALIBRATION → EVALUATION`

### Binary vs Multiclass vs Multilabel — label representation

| Type | Label shape | Output activation | Loss |
|---|---|---|---|
| Binary | single 0/1 | sigmoid | binary cross-entropy |
| Multiclass (single label per sample) | one-hot / integer class | softmax | categorical cross-entropy |
| Multilabel (multiple labels per sample) | binary vector per class | sigmoid **per class independently** | binary cross-entropy per class, summed/averaged |

**Common mistake:** using softmax for multilabel problems — softmax forces probabilities to sum to 1 across classes, which is wrong when a sample can belong to several classes simultaneously. Multilabel needs independent sigmoids.

## Class imbalance decision tree

```text
IS THE CLASSIFICATION PROBLEM IMBALANCED?
│
├── Mild imbalance (e.g. 70/30) →
│       stratified split + class_weight='balanced' is often enough
│
├── Moderate-to-severe imbalance (e.g. 95/5) →
│       ├── class weighting in the loss (cheap, first thing to try)
│       ├── resampling:
│       │     oversampling minority (risk: overfitting to duplicated points)
│       │     undersampling majority (risk: discarding useful data)
│       │     SMOTE (synthetic minority oversampling — interpolates
│       │           between minority neighbors; works on numeric
│       │           features, not raw text/categorical without adaptation)
│       ├── threshold tuning (move decision threshold away from 0.5
│       │     toward the minority class instead of resampling at all)
│       └── focal loss (down-weights easy majority examples,
│             common in detection/segmentation with severe imbalance)
│
└── Extreme imbalance (e.g. fraud, 1000:1) →
        treat as anomaly detection instead of classification (Part 10),
        or use precision-recall-oriented metrics + heavy class weighting
        + collect more minority examples if at all possible
```

Probability calibration (Platt scaling / isotonic regression) matters when you need the *predicted probability itself* to be meaningful (e.g., risk scores), not just the ranking/decision — boosting and SVM outputs are often poorly calibrated by default.

## Model options

| Model | When | Preprocessing needs |
|---|---|---|
| Logistic Regression | linear decision boundary, interpretability, probability calibration is naturally good | scaling required |
| Naive Bayes | text classification, very fast baseline, small data | works on counts/frequencies directly, minimal preprocessing |
| Decision Tree | interpretable nonlinear boundary | no scaling |
| Random Forest | strong general-purpose baseline | no scaling |
| Gradient Boosting / XGBoost / LightGBM | best tabular accuracy, willing to tune | no scaling, handles missing values |
| SVM | clear-margin separable data, small-to-medium data, kernel trick for nonlinearity | scaling required, doesn't scale well to huge datasets |
| KNN | simple nonlinear baseline, low-dimensional data | scaling required, suffers from curse of dimensionality |
| Neural networks | large data, need to fuse modalities/embeddings | scaling required |
| CNN | image inputs | image-specific pipeline (Part 12) |
| Transformer | text/sequence inputs | tokenization pipeline |

---

# PART 9 — CLASSIFICATION LOSS + METRICS

```text
WHAT DO I OPTIMIZE FOR?
│
├── Classes roughly balanced, all errors equally costly →
│       Accuracy is meaningful; also check F1 as a sanity check
│
├── Classes heavily imbalanced →
│       Accuracy is MISLEADING (predicting the majority class always
│       scores high) → use Precision, Recall, F1, PR-AUC instead
│
├── False negatives are expensive (e.g. missed cancer diagnosis) →
│       optimize/report RECALL (sensitivity); consider lowering
│       the decision threshold to catch more positives
│
├── False positives are expensive (e.g. spam filter blocking real mail) →
│       optimize/report PRECISION; consider raising the threshold
│
├── Need to compare models across all possible thresholds →
│       ROC-AUC (balanced classes) or PR-AUC (imbalanced classes —
│       PR-AUC is far more informative than ROC-AUC when positives are rare)
│
└── Probability quality itself matters (risk scoring, downstream
    decision-making uses the probability, not just the class) →
        Log loss (penalizes confident wrong predictions heavily) +
        calibration curve, not just the discrete metrics above
```

| Metric | Formula/idea | Good for |
|---|---|---|
| Accuracy | correct / total | balanced classes only |
| Precision | TP / (TP + FP) | cost of false positives is high |
| Recall (sensitivity) | TP / (TP + FN) | cost of false negatives is high |
| F1 | harmonic mean of precision & recall | need one number balancing both |
| Specificity | TN / (TN + FP) | negative-class performance matters (e.g. healthy patients correctly cleared) |
| Balanced accuracy | average of recall per class | imbalanced multiclass, want equal weight per class regardless of size |
| ROC-AUC | ranking quality across thresholds | balanced-ish classes |
| PR-AUC | precision/recall trade-off across thresholds | imbalanced classes, positive class is the one of interest |
| Log loss | penalizes confident-and-wrong predictions | probability calibration matters |
| Confusion matrix | full breakdown of TP/FP/TN/FN | always worth inspecting, not just the summary number |

**Threshold tuning:** the default 0.5 cutoff is arbitrary. Sweep thresholds against precision/recall curves and pick the one matching your actual false-positive/false-negative cost trade-off — this is a post-training decision, independent of which loss you trained with.

**What comes next:** if metric is unsatisfactory, error analysis (Part 24) → back to feature engineering / model choice / imbalance handling.

---

# PART 10 — UNSUPERVISED PIPELINES

## Clustering

`DATA → SCALE (usually required — distance-based) → optional DIMENSIONALITY REDUCTION → CHOOSE ALGORITHM → CHOOSE K/PARAMS → FIT → EVALUATE (internal metrics + domain validation)`

| Algorithm | When | Needs K? | Distance metric | Notes |
|---|---|---|---|---|
| K-Means | spherical, similarly-sized clusters, large data | yes | Euclidean | sensitive to initialization (use k-means++), sensitive to scale |
| Hierarchical | small-medium data, want a dendrogram / don't know K upfront | no (cut tree at desired level) | configurable | expensive at scale (O(n²) or worse) |
| DBSCAN | arbitrary-shaped clusters, noise/outliers present | no (density-based) | configurable | struggles with varying-density clusters, needs `eps`/`min_samples` tuning |
| Gaussian Mixture Models | clusters may overlap, want soft/probabilistic assignment | yes (number of components) | — | can model elliptical clusters, more flexible than K-Means |

**Choosing K:** elbow method (inertia vs K), silhouette score, or domain knowledge. No universally "correct" K — validate against what's actionable for the use case.

**Evaluation metrics:**

| Metric | Idea | Limitation |
|---|---|---|
| Silhouette score | how similar a point is to its own cluster vs other clusters | can favor convex clusters, misleading on DBSCAN-style irregular shapes |
| Davies-Bouldin | average similarity between each cluster and its most-similar other cluster (lower is better) | same convexity bias as silhouette |
| Calinski-Harabasz | ratio of between-cluster to within-cluster dispersion | also biased toward convex, similarly-sized clusters |

**When these metrics are insufficient:** all three are purely geometric — they say nothing about whether the clusters are *meaningful* to the business problem. Always pair with domain validation: do the clusters correspond to something a stakeholder recognizes and can act on?

## Dimensionality reduction

| Technique | Belongs in | When | When NOT to use |
|---|---|---|---|
| PCA | preprocessing, feature reduction | linear structure, want variance-preserving compression, need speed | data has strongly nonlinear structure — PCA will miss it |
| t-SNE | visualization only | 2D/3D visual inspection of clusters/structure | as a preprocessing step for downstream modelling — distances in t-SNE space are not globally meaningful, and it has no consistent `transform` for new points |
| UMAP | visualization, sometimes preprocessing | faster and better global-structure preservation than t-SNE | when strict linear interpretability is required (use PCA instead) |

**What comes next:** clustering/reduction results usually feed into either reporting, further feature engineering, or as inputs to a downstream supervised model.

# PART 11 — NEURAL NETWORK MASTER PIPELINE

```text
DATA
  ↓
DATASET (wraps raw samples + labels, defines __getitem__)
  ↓
DATALOADER (batches, shuffles, parallelizes loading)
  ↓
BATCH
  ↓
FORWARD PASS
  ↓
PREDICTION
  ↓
LOSS
  ↓
BACKPROP (loss.backward())
  ↓
GRADIENTS
  ↓
OPTIMIZER (applies the update rule)
  ↓
UPDATE (optimizer.step())
  ↓
ZERO GRAD (clear for next batch)
  ↓
NEXT BATCH  ← loops until epoch ends
  ↓
VALIDATION (no gradient updates — model.eval(), no_grad())
  ↓
CHECKPOINT (save if validation improved)
```

## Everything that can modify this pipeline

| Lever | What it controls | Where it enters |
|---|---|---|
| Initialization | starting weight values | before training starts |
| Activations | nonlinearity between layers | inside forward pass |
| Loss | what "error" means | after forward pass |
| Optimizer | how weights move given gradients | at update step |
| Learning rate | step size of each update | inside optimizer |
| Scheduler | how LR changes over time | wraps the optimizer, called each step/epoch |
| Batch size | how many samples per gradient estimate | dataloader |
| Normalization (BatchNorm/LayerNorm) | stabilizes activation distributions | inside forward pass, between layers |
| Dropout | randomly zeroes activations during training | inside forward pass, training mode only |
| Weight decay | shrinks weights toward zero each step | inside optimizer (L2-like regularization) |
| Gradient clipping | caps gradient magnitude | between backward() and step() |
| Mixed precision | uses FP16/BF16 for speed/memory | wraps forward+backward |
| Gradient accumulation | simulates larger batch size on limited memory | accumulates gradients over N micro-batches before step() |
| Early stopping | halts training when validation stops improving | after validation, across epochs |
| Checkpointing | saves model state | after validation |
| EMA (exponential moving average of weights) | smooths weight trajectory for more stable eval | maintained alongside the live model, updated each step |
| Regularization (L1/L2/dropout/etc.) | fights overfitting | loss and/or architecture |
| Augmentation | expands effective training distribution | inside Dataset, train split only |

---

# PART 12 — NEURAL NETWORK ARCHITECTURE DECISION TREE

```text
WHAT STRUCTURE DOES MY INPUT HAVE?
│
├── Flat vectors, no spatial/sequential structure (tabular) →
│       MLP (fully-connected layers)
│
├── Spatial/local structure, translation-invariant patterns (images) →
│       CNN (or Vision Transformer at large data scale)
│
├── Sequential/temporal structure, order matters →
│       RNN/LSTM/GRU (short-medium sequences, limited compute)
│       or Transformer (long sequences, parallelizable, more data/compute)
│
└── Multiple structured modalities that must be related to each other →
        Multimodal architecture (dual encoder / cross-attention / VLM)
```

## CNN depth/width reasoning (not arbitrary rules)

> **When should I use one conv layer vs many?**
> One layer detects only simple local patterns (edges, colors). Real images need hierarchical composition — edges → textures → parts → objects. More layers = larger receptive field and more abstract features, at the cost of more parameters and harder optimization (vanishing gradients without residual connections).

> **When should I increase channels?**
> Increase channel count as spatial resolution shrinks (the classic pattern: halve spatial size, double channels) — this keeps computational cost roughly constant per layer while letting the network represent more distinct feature types once spatial detail has been discarded.

> **When should I downsample (pooling/stride)?**
> Downsample when you want to (a) grow the receptive field faster without more layers, (b) reduce compute for deeper layers, (c) build translation-invariance to small shifts. Don't downsample too early if fine spatial detail matters for the task (e.g., segmentation, small-object detection).

> **When should I use pooling vs stride for downsampling?**
> Pooling (max/avg) is parameter-free and provides a fixed downsampling rule; strided convolution learns the downsampling. Modern architectures (ResNet-style) increasingly prefer strided convolutions for more learnable capacity, reserving pooling mainly for a final global step (global average pooling before the classifier head).

> **When should I use residual connections?**
> Once depth grows past roughly a dozen or so layers, plain stacking causes vanishing/exploding gradients and degrades even training accuracy (not just generalization). Residual connections let gradients flow through an identity shortcut, making very deep networks trainable at all — use them whenever depth is more than modest, essentially the default for any CNN deeper than a simple few-layer baseline.

---

# PART 13 — CNN PIPELINE

```text
IMAGE
  → validate (corrupt/unreadable check)
  → decode
  → channels (grayscale→RGB or vice versa as required)
  → resize
  → crop (random for train, center for eval)
  → augmentation (train only)
  → tensor conversion
  → normalization (dataset-specific or pretrained-model-specific mean/std)
  → batch
  → CNN backbone
  → feature maps
  → pooling (often global average pooling before the head)
  → classifier head (fully connected → output)
  → loss
  → backprop
  → optimizer step
  → validation
  → evaluation
```

## Architectures

| Architecture | Problem it addressed | Idea | When to consider | Depth/params |
|---|---|---|---|---|
| LeNet | proof-of-concept for conv nets on small images | small conv+pool stack | teaching/toy problems only | very shallow, tiny |
| AlexNet | scaled CNNs to real image classification (ImageNet) | deeper convs + ReLU + dropout + GPU training | historical baseline, rarely used today | moderate depth, large FC layers |
| VGG | showed depth (uniform small 3x3 kernels) improves accuracy | stacks of 3x3 convs | when a simple, well-understood backbone is enough and compute isn't tightly constrained | deep, very parameter-heavy (large FC layers) |
| Inception/GoogLeNet | efficient multi-scale feature extraction | parallel conv branches of different kernel sizes per block | when you want multi-scale features without huge parameter cost | moderate depth, efficient per-parameter |
| ResNet | enabled very deep networks to train at all | residual/skip connections | strong general default backbone, especially for transfer learning | scalable depth (18 to 150+ layers) |
| DenseNet | maximize feature reuse, mitigate vanishing gradients | each layer receives all previous layers' feature maps | when parameter efficiency matters more than raw speed | fewer parameters per accuracy point than ResNet, memory-heavy activations |
| EfficientNet | principled scaling of depth/width/resolution together | compound scaling coefficient | when you need the best accuracy/compute trade-off, especially at inference-constrained budgets | scales smoothly across sizes (B0-B7) |
| MobileNet/ShuffleNet | run on-device / low compute | depthwise separable convolutions | edge/mobile deployment | small, fast |
| ConvNeXt | modernize CNNs using Transformer-era design choices | large kernels, LayerNorm, GELU, fewer activations | when you want CNN efficiency with Transformer-era training tricks | comparable to ResNet family, competitive with ViT |

---

# PART 14 — CNN ARCHITECTURE COMPONENT DECISION MAP

### Convolution

| Parameter | Effect | Typical starting point |
|---|---|---|
| Kernel size | larger = bigger receptive field per layer, more parameters | 3x3 is the modern default; 1x1 for channel mixing/bottlenecks; 5x5+ rare except early layers |
| Stride | >1 downsamples spatially | 1 for feature-preserving convs, 2 for downsampling convs |
| Padding | controls whether spatial size shrinks | "same" padding to preserve spatial size is the common default |
| Dilation | expands receptive field without more parameters/downsampling | used in segmentation (e.g. dilated/atrous convolutions) to keep resolution while growing context |
| Number of channels | representational capacity per layer | grows with depth (see reasoning in Part 12) |
| Receptive field | how much of the input a given neuron "sees" | grows with depth, kernel size, dilation, and stride — must exceed object size for the task to be solvable at that layer |

### Pooling

| Type | When |
|---|---|
| Max pooling | preserve strongest activation, common in classification backbones |
| Average pooling | smoother downsampling, often used for the final global step (global average pooling) before the classifier head |
| Adaptive pooling | need a fixed output size regardless of input resolution (handles variable input sizes) |
| No pooling (strided conv instead) | want the downsampling itself to be learnable |

### Normalization — WHERE does it go?

```text
Standard block order:
    Conv → BatchNorm → Activation

Some modern/ResNet-style variants use:
    BatchNorm → Activation → Conv   (pre-activation ResNet)
```

| Normalization | When |
|---|---|
| BatchNorm | CNNs with reasonably large batch sizes; normalizes across the batch dimension per channel |
| LayerNorm | Transformers, RNNs, small/variable batch sizes; normalizes across the feature dimension per sample (batch-size independent) |
| GroupNorm | small-batch training (e.g. detection/segmentation with large images, batch size 1-2) where BatchNorm's batch statistics become unreliable |

### Activation

| Activation | Where it belongs |
|---|---|
| ReLU | default hidden-layer activation in CNNs — fast, simple, works well |
| LeakyReLU | when "dying ReLU" (neurons stuck outputting zero) is observed |
| GELU | default in Transformer architectures |
| Sigmoid | binary output layer, or gates inside LSTM/GRU |
| Tanh | RNN hidden states historically, zero-centered outputs |
| Softmax | multiclass output layer (mutually exclusive classes) |

---

# PART 15 — "WHEN DO I FREEZE LAYERS?"

### Training from scratch

```text
freeze? usually no — the whole network needs to learn everything from
        random initialization; freezing anything just starves it of
        capacity to learn
```

### Transfer learning

```text
pretrained backbone
  → freeze backbone (all pretrained weights fixed)
  → train only the new head (randomly initialized, task-specific)
  → once head has converged reasonably →
  → optionally unfreeze some/all backbone layers
  → fine-tune with a SMALLER learning rate than used for the head
    (large LR on pretrained weights destroys the useful features
     already learned — this is "catastrophic forgetting")
```

**Decision factors:**

| Situation | Strategy |
|---|---|
| Small target dataset, similar domain to pretraining | freeze backbone, train head only — full fine-tuning would overfit fast |
| Small target dataset, different domain | freeze most of backbone, unfreeze only the last few layers, fine-tune gently |
| Large target dataset, similar domain | freeze briefly, then unfreeze everything and fine-tune fully — enough data to safely update all weights |
| Large target dataset, very different domain | consider training from scratch or long continued-pretraining before fine-tuning |

**Gradual unfreezing:** start with everything but the head frozen; unfreeze one block/layer group at a time, from the top (closest to output) toward the bottom, monitoring validation loss at each stage — this limits how much of the pretrained representation can be disturbed at once.

**Different learning rates for backbone vs head (discriminative learning rates):** a common pattern is head LR ~10x-100x higher than the unfrozen backbone LR, since the head starts from random weights and needs to move further/faster while the backbone only needs gentle adjustment.

```python
optimizer = torch.optim.AdamW([
    {"params": model.backbone.parameters(), "lr": 1e-5},
    {"params": model.head.parameters(), "lr": 1e-3},
])

# Freezing:
for param in model.backbone.parameters():
    param.requires_grad = False

# Later, unfreezing selectively:
for param in model.backbone.layer4.parameters():
    param.requires_grad = True
```

**Catastrophic forgetting:** aggressively fine-tuning a pretrained backbone on a small new dataset can overwrite the general features it learned, hurting performance on anything outside the narrow new task — mitigated by low LR, freezing early layers, and/or fewer fine-tuning epochs.

# PART 16 — RNN / LSTM / GRU PIPELINE

```text
SEQUENCE INPUT → embedding/encoding per timestep → recurrent unit
  (updates hidden state each step) → (optionally: sequence-to-sequence
  decoder with teacher forcing during training) → output per step
  or final-step output → loss → backprop through time → optimizer
```

| Component | Role |
|---|---|
| Hidden state | running summary of the sequence seen so far |
| Cell state (LSTM only) | separate long-term memory channel, protected from being overwritten every step |
| Forget/input/output gates (LSTM) | control what's discarded, added, and exposed from the cell state |
| Update/reset gates (GRU) | simplified gating (fewer parameters than LSTM, often similar performance) |
| Bidirectional | processes sequence forward AND backward, concatenates — useful when the full sequence is available at once (not for causal/streaming generation) |
| Teacher forcing | during training, feed the *true* previous token as decoder input rather than the model's own (possibly wrong) prediction — speeds up training but can cause train/inference mismatch ("exposure bias") |

## RNN vs LSTM vs GRU vs Transformer

```text
RNN vs LSTM vs GRU vs Transformer?
│
├── Very short sequences, limited compute, simplicity matters →
│       plain RNN (rarely chosen today outside teaching/toy tasks —
│       suffers badly from vanishing gradients on longer sequences)
│
├── Long-term dependencies matter, moderate sequence length,
│   limited compute/data →
│       LSTM (mitigates vanishing gradients via the cell state)
│
├── Similar to LSTM but want fewer parameters / faster training,
│   dataset is smaller →
│       GRU (often matches LSTM with less compute)
│
└── Long sequences, need parallelization across the whole sequence
    during training, have enough data/compute →
        Transformer (self-attention processes all positions at once
        instead of sequentially — dramatically more parallelizable,
        better at very long-range dependencies, but O(n²) attention
        cost in sequence length and needs more data to train well)
```

| Dimension | RNN/LSTM/GRU | Transformer |
|---|---|---|
| Sequence length handling | degrades on very long sequences (even LSTM) | handles long-range dependencies well, but compute grows quadratically with length |
| Parallelization | sequential — can't parallelize across timesteps during training | fully parallel across sequence positions during training |
| Memory | roughly linear in sequence length | quadratic in sequence length (attention matrix) unless using long-context tricks |
| Training stability | can be trickier (vanishing/exploding gradients) without careful gating/clipping | generally stabler with normalization + residuals, but sensitive to LR warmup |
| Data requirements | can work with smaller datasets | typically needs more data to reach its potential |

---

# PART 17 — TRANSFORMER PIPELINE

```text
INPUT
  → TOKENIZATION
  → TOKEN IDs
  → EMBEDDINGS (token embedding table lookup)
  → + POSITION INFORMATION (added or applied to embeddings)
  → Q/K/V projections
  → SELF-ATTENTION (scaled dot-product over Q/K/V)
  → MULTI-HEAD ATTENTION (multiple attention "views" in parallel, concatenated)
  → RESIDUAL (add input back)
  → NORMALIZATION (LayerNorm)
  → FEED-FORWARD NETWORK (position-wise MLP)
  → RESIDUAL
  → NORMALIZATION
  → OUTPUT (repeat block N times, then task-specific head)
```

## Architecture decisions at each point

| Decision | Options | When |
|---|---|---|
| Encoder / Decoder / Encoder-Decoder | encoder-only, decoder-only, both | encoder-only for understanding tasks (classification, embeddings, search); decoder-only for open-ended generation; encoder-decoder for input→output transformation tasks (translation, summarization) |
| Attention masking | causal (each position sees only earlier positions) vs bidirectional (sees the whole sequence) | causal for autoregressive generation; bidirectional for understanding tasks where the full context is available upfront |
| Cross-attention | decoder attends to encoder outputs | needed whenever an encoder-decoder architecture must condition generation on a separately-encoded input |
| Positional encoding | sinusoidal (fixed), learned absolute, RoPE (rotary), ALiBi | RoPE and similar relative schemes are now the common modern default for long-context/LLM models — they generalize better to sequence lengths beyond what was seen in training than fixed absolute encodings |
| Context length | how many tokens the model attends to at once | driven by application needs and quadratic compute cost; longer context = more compute/memory per token processed |
| Number of layers / hidden dim / attention heads / FFN dim | model capacity | scale together; larger values increase capacity and compute cost roughly proportionally — no universal "right" values, driven by available data/compute and target model size |

---

# PART 18 — TRANSFORMER TRAINING

## Objectives

| Objective | Task shape |
|---|---|
| Masked language modelling (MLM) | predict randomly-masked tokens from bidirectional context — encoder-style pretraining (BERT-style) |
| Causal language modelling (CLM) | predict the next token given only preceding tokens — decoder-style pretraining (GPT-style) |
| Sequence-to-sequence objectives | map an input sequence to a target output sequence (translation, summarization) |
| Contrastive objectives | pull matching pairs (e.g. image-text) together in embedding space, push non-matching pairs apart |
| Task-specific supervised objectives | standard cross-entropy fine-tuning on a labeled downstream task |

```text
LOSS → BACKPROP → OPTIMIZER (AdamW is the standard choice)
  → LR SCHEDULE (warmup → decay) → GRADIENT MANAGEMENT (clipping)
  → CHECKPOINTING → VALIDATION
```

| Technique | Purpose | Practical range/note |
|---|---|---|
| AdamW | decouples weight decay from the gradient update (fixes a subtle bug in plain Adam+L2) — the standard Transformer optimizer | — |
| LR warmup | linearly ramp LR from ~0 up to peak over the first portion of training | commonly the first 1-10% of total training steps |
| Cosine decay | smoothly decays LR to near-zero following a cosine curve after warmup | very common schedule for pretraining and fine-tuning |
| Linear decay | decays LR linearly to zero after warmup | simpler alternative to cosine, also widely used |
| Gradient clipping | caps gradient norm to prevent destabilizing spikes | typical global-norm clip value: 1.0 |
| Mixed precision (FP16/BF16) | speeds up training, halves memory | BF16 preferred over FP16 on hardware that supports it — wider dynamic range avoids overflow issues that plague FP16 |
| Gradient accumulation | simulates a larger effective batch size by summing gradients over several micro-batches before stepping | used whenever the desired batch size doesn't fit in memory at once |

---

# PART 19 — PRETRAINING VS FINE-TUNING

| Term | Meaning |
|---|---|
| Pretraining | training from scratch on a large, general (often unlabeled/self-supervised) dataset to learn broad representations |
| Continued pretraining | further pretraining an already-pretrained model on new domain data, still using the original self-supervised objective |
| Transfer learning | reusing a pretrained model's learned representations for a new (often different) task |
| Fine-tuning | adapting a pretrained model's weights to a specific downstream task, usually with labeled data |
| Supervised fine-tuning (SFT) | fine-tuning specifically with labeled input-output pairs (common terminology in LLM post-training) |
| Instruction tuning | SFT specifically on instruction-following examples, so the model generalizes to following novel instructions rather than a single fixed task |
| Parameter-efficient fine-tuning (PEFT) | updating only a small number of new/added parameters instead of the whole model |
| LoRA | PEFT technique: freeze original weights, learn small low-rank update matrices added to selected layers |
| QLoRA | LoRA on top of a quantized (e.g. 4-bit) base model — combines memory savings of quantization with LoRA's parameter efficiency |

## Decision tree

```text
Do I have a pretrained model relevant to my task?
│
├── NO → pretrain from scratch (needs large data + large compute;
│         rarely justified unless building a foundation model)
│
└── YES
     │
     Is my target DOMAIN very different from the pretraining domain?
     │
     ├── YES, and I have a lot of target-domain data →
     │       continued pretraining (same self-supervised objective,
     │       new domain data) → then fine-tune
     │
     ├── YES, but I have limited target-domain data →
     │       fine-tune directly, but conservatively (low LR,
     │       consider PEFT to avoid destroying general knowledge)
     │
     └── NO (domain is similar) →
             fine-tune directly
             │
             How much labeled data + compute/memory do I have?
             │
             ├── Plenty of both → full fine-tuning (updates all weights,
             │       best ceiling on performance, most expensive)
             │
             ├── Limited compute/memory, model is large →
             │       LoRA (small trainable parameter count, much less
             │       memory for optimizer states)
             │
             ├── Very limited memory (can't even fit the base model
             │   in fp16) →
             │       QLoRA (quantize base model to 4-bit, train LoRA
             │       adapters on top)
             │
             └── Want to keep multiple task-specific variants of one
                 base model cheaply →
                     adapters/LoRA — swap small adapter weights per
                     task instead of storing full model copies per task
```

**When to just freeze + train a small head instead of any of the above:** when the downstream task is simple (e.g. linear-probe classification) and preserving the pretrained representation exactly is more valuable than adapting it.

---

# PART 20 — MULTIMODAL / CLIP PIPELINE

```text
IMAGE                                   TEXT
  → vision encoder                        → tokenizer
  → image embedding                       → text encoder
  → projection head                       → text embedding
       ↘                                 ↗
         → normalize both embeddings (L2 norm)
         → cosine similarity between all image-text pairs in the batch
         → scale by learned/fixed temperature
         → contrastive loss (e.g. InfoNCE: pull matching pairs together,
           push all other in-batch pairs apart)
         → training
         → retrieval evaluation (image→text and text→image top-k accuracy)
```

| Component | Role |
|---|---|
| Dual encoders | separate networks per modality, mapped into a **shared** embedding space |
| Contrastive learning | the training signal — no explicit labels beyond "this image and this text are a matching pair" |
| Projection head | small learned layer mapping each encoder's native output dimension into the shared embedding dimension |
| Temperature | scales similarity scores before the softmax in the contrastive loss — controls how sharply the model must separate matches from non-matches |
| Modality alignment | the emergent property that semantically similar concepts land near each other in the shared space regardless of modality |

Larger batch sizes generally help contrastive training — more negative pairs per batch gives a harder, more informative training signal.

---

# PART 21 — VLM PIPELINE

```text
IMAGE
  → vision encoder (often a pretrained ViT/CLIP-style encoder)
  → visual features / patch tokens
  → connector / projector (maps visual features into the language
    model's embedding space — e.g. a small MLP or cross-attention module)
  → multimodal token sequence (visual tokens + text tokens combined)
  → language model (processes the combined sequence)
  → output (text generation, grounded in the image)
```

| Component | Notes |
|---|---|
| Vision encoder | usually frozen or lightly fine-tuned pretrained encoder |
| Image resolution / patch tokens | higher resolution → more visual tokens → more compute; must balance detail vs context budget |
| Connector/projector | the critical adaptation layer — often the main thing trained in early VLM fine-tuning stages |
| Cross-attention vs concatenation | cross-attention lets the LM attend into visual features at each layer; token-concatenation instead treats visual tokens as just more sequence tokens for a decoder-only LM — a simpler, increasingly common approach |
| Visual instruction tuning | SFT stage specifically on (image, instruction, response) triples so the model follows multimodal instructions |
| Packing | concatenating multiple training examples into one sequence to fill context length efficiently, with masking to prevent cross-example attention |

### VLM training pipeline

```text
DATA → cleaning → deduplication → filtering → quality scoring
  → formatting (image-text pairs / instruction format)
  → train/validation split → preprocessing (tokenize text, encode images)
  → model (connector + LM, vision encoder often frozen initially)
  → objective (usually next-token prediction over the response tokens only)
  → optimization (often staged: connector-only → then LoRA/full fine-tune
    of the LM → optionally unfreeze vision encoder late)
  → evaluation → failure analysis
```

---

# PART 22 — DATA CURATION FOR LARGE-SCALE TRAINING

```text
RAW DATA
  → ingestion
  → validation (schema/format checks at scale)
  → filtering (quality heuristics, language ID, toxicity/safety filters)
  → deduplication (exact + near-duplicate, e.g. via hashing/embeddings)
  → quality scoring (heuristic or model-based scoring of each example)
  → leakage checks (ensure eval sets aren't present in training data)
  → split
  → sharding (breaking the dataset into manageable file chunks)
  → storage
  → streaming (reading shards on-the-fly rather than loading all into memory)
  → training
```

| Concept | Where it matters |
|---|---|
| Synthetic data | augmenting scarce domains/languages/tasks; risk of amplifying model biases/errors if not filtered |
| Quality filtering | happens before training — removing low-quality/duplicate/unsafe content improves downstream model quality more reliably than just adding more raw data |
| Provenance | tracking where each example came from, for licensing/audit/debugging |
| Dataset versioning | reproducibility — know exactly which data version produced which model |
| Multilingual considerations | balancing language representation to avoid a handful of high-resource languages dominating training |
| Sharding/streaming | necessary once a dataset no longer fits in memory or even on a single disk — training reads sequentially from a stream of shards |

# PART 23 — OPTIMIZATION MASTER MAP

Optimization is not one thing — it happens at several independent levels simultaneously.

## A. Model/architecture optimization
Depth, width, number of heads, hidden dimensions, receptive field, overall model size. Larger/deeper = more capacity but more compute, more overfitting risk on small data, harder optimization without residuals/normalization.

## B. Objective optimization
Loss function choice, class weighting, label smoothing (softens hard 0/1 targets, e.g. to 0.9/0.1 — reduces overconfidence, helps calibration and generalization), contrastive temperature, auxiliary losses (extra loss terms on side-tasks to regularize/guide representation learning).

## C. Optimizer

| Optimizer | Idea | When |
|---|---|---|
| SGD | plain gradient step | simple, well-understood, generalizes well when tuned but converges slowly |
| Momentum | accumulates a velocity term to smooth/accelerate SGD | almost always better than plain SGD, essentially free improvement |
| Adam | adaptive per-parameter learning rates + momentum | good default, fast convergence, less LR-tuning sensitive |
| AdamW | Adam with properly decoupled weight decay | standard choice for Transformers/LLMs |

## D. Learning-rate optimization

| Technique | Purpose | Typical note |
|---|---|---|
| Initial LR | starting step size | no universal range — order of magnitude differs hugely by optimizer/architecture (e.g. ~1e-3 for Adam on a fresh small NN, ~1e-4 to 1e-5 for fine-tuning a pretrained Transformer) |
| Warmup | avoid destabilizing large updates on freshly-initialized/adaptive-optimizer statistics early in training | common for Transformers, less critical for small CNNs trained with SGD+momentum |
| Cosine decay | smooth decay to ~0 | common modern default |
| Linear decay | simpler decay | also common |
| Step decay | drop LR by a factor at fixed milestones | classic CNN training recipe |
| ReduceLROnPlateau | drop LR when validation metric stops improving | useful when you don't want to hand-pick a schedule in advance |

## E. Regularization
L1, L2/weight decay, dropout, augmentation, early stopping, label smoothing — see Part 25 for the full decision tree.

## F. Training stability
Gradient clipping, careful initialization, normalization layers, mixed precision, explicit NaN/Inf handling (e.g. skip a step or reduce LR if a NaN loss is detected).

## G. Memory optimization

| Technique | Saves |
|---|---|
| Smaller batch size | activation memory |
| Gradient accumulation | lets you use a small physical batch while getting a large effective batch |
| Activation checkpointing | trades compute for memory — recomputes activations during backward instead of storing them all |
| Mixed precision | roughly halves memory for weights/activations |
| PEFT (LoRA/QLoRA) | drastically reduces optimizer-state memory by only training a small parameter subset |

## H. Hyperparameter optimization

| Method | When |
|---|---|
| Manual tuning | small number of hyperparameters, strong prior intuition, early exploration |
| Grid search | few hyperparameters, cheap to evaluate each combination |
| Random search | more hyperparameters than grid search can afford — generally more efficient than grid search for the same budget |
| Bayesian optimization | expensive-to-evaluate models, want to use past trial results to guide the next trial intelligently |
| Optuna-style / other automated frameworks | need pruning of bad trials early, parallel search, or a repeatable/production HPO workflow |

## Hyperparameter template (used throughout this document)

```text
Parameter:              <name>
Purpose:                <what it controls>
Typical range:          <range or "no universal range">
Good starting value:    <default to try first>
When to increase:       <symptom that suggests increasing>
When to decrease:       <symptom that suggests decreasing>
Symptoms if wrong:      <what breaks>
```

---

# PART 24 — NORMALIZATION MASTER MAP

**Two entirely different things share the word "normalization" — keep them separate:**

### Data preprocessing normalization
Happens once, on the dataset, before training. Fitted on train, applied to val/test. Examples: StandardScaler, MinMaxScaler, RobustScaler (Part 4).

### Neural-network architectural normalization
Happens every forward pass, inside the model, as a layer. Learnable (has its own parameters). Examples:

| Layer | Normalizes across | When |
|---|---|---|
| BatchNorm | the batch dimension, per channel | CNNs with reasonably large, consistent batch sizes |
| LayerNorm | the feature dimension, per sample | Transformers, RNNs, variable/small batch sizes — batch-independent |
| GroupNorm | groups of channels, per sample | small-batch CNN training (detection/segmentation with batch size 1-4) |

The two categories solve different problems: preprocessing normalization fixes the *input data's* scale; architectural normalization fixes *internal activation* distributions during training, and its statistics shift as the network learns.

---

# PART 25 — REGULARIZATION MASTER MAP

```text
OVERFITTING? (train metric good, validation metric much worse)
        ↓
Possible causes:
  - model has too much capacity for the amount of data
  - insufficient/no regularization
  - training too long past the point of generalization
  - noisy/leaky features the model is memorizing
        ↓
Possible interventions:
  ├── more data (best fix if obtainable — addresses the root cause)
  ├── augmentation (synthetic data diversity)
  ├── dropout (NN — randomly zero activations during training)
  ├── L1 (drives some weights to exactly zero — implicit feature selection)
  ├── L2 / weight decay (shrinks all weights smoothly toward zero)
  ├── early stopping (stop at the validation-optimal epoch)
  ├── smaller model (reduce capacity directly)
  ├── data cleaning (remove noise/leakage the model is exploiting)
  ├── transfer learning (start from a representation that already
  │     generalizes, rather than learning everything from this
  │     small dataset)
  ├── freezing layers (limits how much can be overfit to the new data)
  ├── label smoothing (prevents overconfident predictions on noisy labels)
  └── mixup / cutmix (blend training examples/labels together —
        forces smoother decision boundaries, common in image classification)
```

| Technique | When to use | Trade-off | Interacts with |
|---|---|---|---|
| More data | always, if feasible | cost/time to acquire | reduces need for all others |
| Augmentation | vision/audio/some NLP tasks | can hurt if augmentations are unrealistic for the task | works alongside dropout/weight decay |
| Dropout | NN with fully-connected layers especially | too much dropout underfits, slows convergence | often reduced when weight decay is also strong |
| L1 | want sparse, more interpretable features | unstable selection under correlated features | can combine with L2 (Elastic Net) |
| L2/weight decay | almost always applied at some small value in NN training | too much causes underfitting | standard alongside dropout |
| Early stopping | any iterative training | needs a validation set to monitor, adds a decision point about patience | complements all other regularization |
| Smaller model | small datasets, tabular problems | may underfit if too small | alternative rather than complement to dropout/weight decay |
| Label smoothing | noisy labels, or want better calibration | slightly softens the model's confident correct predictions too | used in classification cross-entropy losses, e.g. Transformer training |
| Mixup/CutMix | image classification with enough data to still generalize | can slow early convergence, weird interaction with some architectures (e.g. BatchNorm statistics) | usually combined with standard augmentation |

---

# PART 26 — DATA AUGMENTATION MASTER MAP

**Augmentation belongs to the DATA PIPELINE, not the model architecture** — it changes what the model *sees*, not how the model is structured, and is applied only to the training split (never validation/test, which must reflect real, unaugmented data).

### Image
Crop, flip (horizontal usually safe, vertical often not — check if it's semantically valid for the task), rotation, color jitter, blur, noise, cutout (mask out a random patch), mixup, cutmix.

### Text
Synonym replacement, back-translation, random insertion/deletion/swap of words, paraphrasing (e.g. via another model). Generally used more cautiously than image augmentation — text is less tolerant of noise before meaning breaks.

### Multimodal
Must keep image and text augmentations *consistent* with each other — e.g., don't crop an image in a way that removes an object still described in the paired caption.

### When augmentation can be harmful
- Augmentations that violate the task's actual invariances (e.g. flipping digit images horizontally turns some digits into different digits, or flipping an X-ray the wrong way).
- Too-aggressive augmentation on already-scarce data can make the *effective* task harder than the real one, hurting rather than helping.
- Applying augmentation inconsistently between train and eval pipelines silently shifts the distribution the model is evaluated on.

---

# PART 27 — OUTLIER DECISION TREE

```text
OUTLIERS DETECTED
        ↓
Is this a data ERROR or a legitimate RARE observation?
   (check: is the value physically/logically possible? does it
    correspond to a real, verifiable event?)
        │
   ┌────┴────┐
 ERROR      LEGITIMATE
   │           │
   ↓           ↓
 fix/remove  keep, but consider how much LEVERAGE to give it:
 the bad     │
 record      ├── remove (only if it would dominate/distort the
             │     model AND is not representative of what
             │     you need to predict on in production)
             ├── cap/winsorize (limit its influence while
             │     keeping the record)
             ├── transform (log/power transform reduces its
             │     relative influence on scale-sensitive models)
             ├── robust scaler instead of standard scaler
             ├── robust loss (Huber, etc.) instead of MSE
             └── keep as-is (tree-based models are naturally
                   robust to outliers in feature values; and if
                   the outlier IS the phenomenon of interest —
                   e.g. fraud, rare disease — removing it removes
                   the signal you're trying to model)
```

**Model-specific sensitivity:** linear models, KNN, SVM, and NN (with MSE-type losses) are all sensitive to outliers via distance/gradient magnitude. Tree-based models (splits based on order, not magnitude) are comparatively robust to outliers in *features*, though not immune to outliers in the *target* for regression trees.

---

# PART 28 — LABEL QUALITY DECISION TREE

```text
LABELS ARE...
│
├── Missing → see Part 3-C (missing target)
│
├── Inconsistent (different annotators, different conventions) →
│       standardize labeling guidelines retroactively where possible;
│       consider a label-review pass on a sample to estimate the
│       inconsistency rate before deciding it's tolerable
│
├── Corrupted (clearly wrong, e.g. flipped 0/1) →
│       identify systematically if possible (e.g. via a
│       confident-learning / noise-detection pass comparing
│       model predictions to given labels), fix or remove
│       identified cases — don't blanket-remove without evidence
│
├── Noisy (some fraction are wrong but you don't know which) →
│       label smoothing, robust loss functions, or training with
│       noise-robust techniques rather than trying to hand-fix
│       every example
│
├── Ambiguous (genuinely unclear even to a human) →
│       consider whether the task definition itself needs
│       refining, or use soft/multi-annotator labels instead of
│       forcing a single hard label
│
└── Imbalanced → see Part 8 (class imbalance decision tree)
```

Do not automatically recommend removing data for any of the above without first estimating *how much* is affected and *whether the noise is random or systematic* — systematic label noise (e.g., one class is mislabeled far more than another) biases the model in a specific direction and needs targeted correction, not just "more data" or "more regularization."

---

# PART 29 — LEAKAGE DECISION TREE

| Type | How it happens | How to detect | How to fix |
|---|---|---|---|
| Target leakage | a feature is a proxy for (or directly derived from) the target, only knowable after the fact | suspiciously high validation performance from a single feature; check each feature's plausible availability at prediction time | drop or lag the feature to when it would actually be known |
| Train/test contamination | same records (or near-duplicates) appear in both splits | check for duplicate/near-duplicate rows across splits, check for shared entity IDs | deduplicate before splitting, use group-based splitting |
| Preprocessing leakage | scaler/encoder/imputer fit on the full dataset before splitting | review the pipeline order explicitly | always fit preprocessing on train only, transform val/test with it (Part 5) |
| Temporal leakage | using future information to predict the past (random split on time-series data) | check whether any feature could only be known after the target's timestamp | use chronological splitting, lag time-dependent features properly |
| Duplicate leakage | exact/near-duplicate rows split across train and test | hash rows / near-duplicate detection | deduplicate globally before any split |
| Entity leakage | multiple rows from the same entity (patient, user) end up in different splits | check ID overlap across splits | group-based splitting keyed on entity ID |
| Augmentation leakage | augmented versions of a training image end up in the validation set | check if augmentation is applied before or after splitting | augment only after splitting, only within the train partition |
| Feature engineering leakage | an engineered feature (e.g. a rolling aggregate) is computed using future rows or across split boundaries | audit exactly which rows contribute to each engineered feature's computation | compute time-respecting/train-only aggregates |

---

# PART 30 — TRAINING FAILURE DEBUGGING TREE

### Loss = NaN
Check: learning rate too high, invalid/unnormalized input data, missing/incorrect normalization, exploding gradients (add clipping), mixed-precision underflow/overflow (try BF16 or disable AMP to isolate), bad/corrupted labels (e.g. a label index out of range), log(0) or division-by-zero inside a custom loss.

### Loss does not decrease at all
Check: labels are correctly aligned with inputs (a shuffling bug is extremely common), output shape matches what the loss function expects, loss function matches the task (e.g. accidentally using MSE on classification logits), learning rate far too low, optimizer not actually receiving/updating the right parameters, gradients aren't flowing (check for accidentally frozen parameters or a broken computation graph), model capacity is too small for the task, data pipeline is feeding garbage (visualize a batch directly).

### Training good, validation bad
Check: overfitting (see Part 25), leakage (see Part 29 — validation is "too good," or gap appears suddenly), distribution shift between train and validation, augmentation is train-only and creating too large a distribution gap, insufficient regularization, split wasn't representative (re-check stratification/grouping).

### Both training and validation bad
Check: underfitting (model too small, or trained too briefly), bad preprocessing (features/labels genuinely broken, not just leaky), wrong architecture for the data structure, wrong loss for the task, optimization not converging (LR/optimizer misconfigured), insufficient data for the task's complexity.

### GPU utilization low
Check: DataLoader isn't providing batches fast enough (increase `num_workers`, use pinned memory), CPU-bound preprocessing/augmentation bottleneck, batch size too small to saturate the GPU, slow storage I/O, unnecessary CPU-GPU synchronization points in the code (e.g. calling `.item()` or printing every step).

### CUDA OOM
Check: reduce batch size, reduce image resolution/sequence length, reduce model size, enable mixed precision, use gradient accumulation to keep effective batch size while lowering the physical one, use activation checkpointing, clear unused cached tensors/gradients between runs.

# PART 31 — EVALUATION PIPELINE

Evaluation is not "calculate accuracy" — it's a workflow with its own stages, and it differs by task family.

| Task family | Primary metrics | Notes |
|---|---|---|
| Regression | MAE, RMSE, R² (Part 7) | pair with residual analysis |
| Classification | Precision/Recall/F1/ROC-AUC/PR-AUC (Part 9) | pick based on class balance + error cost |
| Clustering | Silhouette, Davies-Bouldin, Calinski-Harabasz (Part 10) | pair with domain validation |
| Object detection | mAP, IoU (Part 32) | threshold-dependent (NMS, IoU cutoff) |
| Segmentation | IoU, Dice, pixel accuracy (Part 32) | class-imbalanced pixels need per-class reporting |
| NLP | task-specific (accuracy/F1 for classification; BLEU/ROUGE for generation) | automated generation metrics are weak proxies (Part 33) |
| Generative models | FID, Perplexity, human eval | automated metrics rarely sufficient alone |
| VLMs | task-specific (VQA accuracy, captioning metrics) + human/model-graded eval | grounding/hallucination checks matter as much as fluency |

### Training metric vs validation metric vs test metric vs business/domain metric

These four can legitimately differ:
- **Training metric/loss**: what's actually optimized (often chosen for smooth gradients — e.g. cross-entropy, MSE).
- **Validation metric**: what you use to select hyperparameters/checkpoints — often more interpretable (e.g. F1, RMSE).
- **Test metric**: the same as the validation metric, reported once, unbiased.
- **Business/domain metric**: what actually matters to the stakeholder (e.g. revenue impact, cost saved) — may only loosely correlate with the model metric, and is the ultimate arbiter of whether the model is "good enough."

### Beyond the headline number

- **Confidence intervals**: a single point metric hides sampling variance — especially important on small test sets.
- **Calibration**: are predicted probabilities trustworthy, not just rank-ordered correctly?
- **Subgroup evaluation**: does performance hold up across relevant slices (demographics, regions, rare classes), or does the headline metric hide a subgroup failure?
- **Robustness**: does performance degrade under small input perturbations, distribution shift, or adversarial-ish inputs?
- **Distribution shift**: is the deployment-time data distribution still close to what the model was evaluated on?
- **Failure analysis**: look directly at the worst mispredictions — this is often more informative than any aggregate metric for deciding what to fix next.

---

# PART 32 — COMPUTER VISION EVALUATION

### Classification
Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC — same definitions as Part 9, applied per-class for multiclass problems and often macro/micro-averaged.

### Detection

| Metric | Meaning |
|---|---|
| IoU (Intersection over Union) | overlap between predicted and ground-truth bounding box, `area(intersection) / area(union)` — the basic building block for "is this detection correct" |
| NMS (Non-Max Suppression) | post-processing step, not a metric — removes duplicate overlapping boxes for the same object before scoring/using predictions |
| mAP (mean Average Precision) | precision-recall curve area, averaged across classes (and often across IoU thresholds, e.g. mAP@[.5:.95]) — the standard detection benchmark metric |
| Precision/Recall | at a chosen IoU/confidence threshold, same TP/FP/FN logic as classification but defined via box-overlap matching |

### Segmentation

| Metric | Meaning |
|---|---|
| IoU | same overlap idea, applied per-pixel-region instead of per-box |
| Dice coefficient | `2·|A∩B| / (|A|+|B|)` — similar to IoU, more sensitive to overlap in typically-imbalanced segmentation masks (e.g. small lesions), common in medical imaging |
| Pixel accuracy | fraction of correctly classified pixels — can be misleading with severe class imbalance (e.g. background dominates), report per-class alongside |

---

# PART 33 — GENERATIVE / FOUNDATION MODEL EVALUATION

| Metric | Task | What it captures | Limitation |
|---|---|---|---|
| BLEU | translation/generation | n-gram overlap with reference text | penalizes valid paraphrases, insensitive to meaning |
| ROUGE | summarization | recall-oriented n-gram/longest-common-subsequence overlap | same n-gram-overlap blind spots as BLEU |
| METEOR | translation | overlap with synonym/stem matching, better than BLEU at capturing meaning | still a surface-level proxy |
| Perplexity | language modelling | how "surprised" the model is by held-out text (lower = better fit) | doesn't directly measure downstream task usefulness or factuality |
| FID (Fréchet Inception Distance) | image generation | distributional similarity between generated and real images in a feature space | doesn't capture per-image faithfulness to a specific prompt/condition |

**Critical caveat:** metric choice depends entirely on the task, and automated metrics are frequently *not sufficient* for generative or VLM systems — a caption can score well on BLEU/ROUGE while being factually wrong, and an image can have good FID while misrepresenting the prompt. Human evaluation, or model-graded evaluation against explicit rubrics, is usually necessary alongside automated metrics for any generation system going to production.

---

# PART 34 — INFERENCE + MODEL OPTIMIZATION

```text
TRAINED MODEL
  → export (to a serving format)
  → precision choice (FP32 / FP16 / BF16 / INT8 / lower)
  → quantization (reduce numeric precision of weights/activations)
  → pruning (remove unnecessary weights/structures)
  → distillation (train a smaller "student" to mimic a larger "teacher")
  → batching (group requests together for throughput)
  → dynamic batching (batch requests arriving asynchronously, on the fly)
  → KV cache (for autoregressive Transformers — cache past keys/values
    so each new token doesn't require recomputing attention over the
    whole prefix)
  → serving
  → monitoring
```

### Precision options

| Precision | Trade-off |
|---|---|
| FP32 | full precision, most accurate, largest memory/slowest |
| FP16 | half memory/faster, narrower dynamic range (risk of overflow/underflow) |
| BF16 | same memory as FP16, wider dynamic range (fewer stability issues), needs hardware support |
| INT8 | further memory/speed reduction, requires calibration to preserve accuracy |
| Lower (INT4, etc.) | maximal compression, typically needs quantization-aware techniques (e.g. as in QLoRA) to avoid large accuracy loss |

### Compression techniques

| Technique | Idea | Trade-off |
|---|---|---|
| Quantization | reduce numeric precision | speed/memory win, potential accuracy loss if not calibrated |
| Pruning | remove low-importance weights/structures | can maintain accuracy at moderate sparsity, harder to get hardware speedups without structured pruning |
| Distillation | train a smaller model to match a larger one's outputs | needs the teacher's compute during distillation training, but yields a genuinely smaller deployed model |

### Serving concepts

ONNX (a standard model interchange format for cross-framework/cross-runtime deployment), TensorRT (NVIDIA-specific inference optimization/compilation), vLLM-style serving concepts (continuous/dynamic batching + efficient KV-cache management specifically for high-throughput LLM serving).

### The core trade-off space

```text
latency  ←→  throughput  ←→  memory  ←→  accuracy
```
Every optimization above moves you along these axes — there is no free win; the question is always which axis matters most for the deployment target (a real-time single-user app cares about latency; a batch-processing pipeline cares about throughput; an edge device cares about memory).

---

# PART 35 — GPU / TRAINING SYSTEMS

```text
CPU vs GPU:
  CPU → few, powerful cores, good at sequential/branchy logic
  GPU → many, simpler cores, good at massively parallel tensor operations
        (which is essentially all of deep learning's core math)
```

| Concept | Meaning |
|---|---|
| VRAM | GPU memory — holds model weights, activations, gradients, optimizer states |
| Compute bound | the GPU's arithmetic throughput is the bottleneck (large matmuls) |
| Memory bound | data movement (not arithmetic) is the bottleneck — common with small ops or poor batching |
| Tensor operations | the fundamental unit of GPU work in deep learning (matmuls, convolutions) |
| CUDA | NVIDIA's parallel-computing platform underlying most deep learning GPU execution |
| Utilization | how much of the GPU's compute capacity is actually being used — low utilization signals a bottleneck elsewhere (see Part 30, "GPU utilization low") |
| Fragmentation | memory allocated/freed in a pattern that leaves unusable gaps, causing OOM even when total free memory would technically suffice |
| Throughput | amount of work done per unit time (e.g. samples/sec) |
| Latency | time for one unit of work (e.g. one inference request) to complete |

### Distributed training — at what scale does each become relevant?

| Technique | Idea | Relevant when |
|---|---|---|
| DDP (Distributed Data Parallel) | each GPU holds a full model copy, processes a different data shard, gradients synced across GPUs | model fits comfortably on one GPU, you just want to speed up training with more data throughput |
| FSDP (Fully Sharded Data Parallel) | shards model parameters/gradients/optimizer states across GPUs, not just data | model is too large to fit (with its optimizer states) on a single GPU |
| DeepSpeed | a library implementing sharding (ZeRO stages similar in spirit to FSDP) plus additional memory/throughput optimizations | very large model training, want a mature ecosystem of memory-optimization features |
| Data parallelism | replicate model, split data across devices | the default first choice when a single device already fits the model |
| Tensor parallelism | split individual layers' matrix operations across devices | a single layer/weight matrix is too large for one device |
| Pipeline parallelism | split the model's layers across devices, pass activations between stages | the whole model is too large for one device even after tensor parallelism, and you want to keep multiple devices' compute busy across pipeline stages |
| Sequence parallelism | split the sequence dimension across devices | extremely long context lengths where even activation memory for one sequence exceeds a single device |

Do not reach for anything beyond plain DDP until a single GPU genuinely cannot hold the model/batch you need — every additional parallelism strategy adds real engineering and communication overhead.

---

# PART 36 — PYTORCH IMPLEMENTATION MAP

```text
Dataset          → defines how to access one sample: __getitem__, __len__
  ↓
DataLoader       → batches, shuffles, parallelizes (num_workers)
  ↓
nn.Module        → defines the model's layers and parameters
  ↓
forward()        → defines how input becomes output (the computation graph)
  ↓
loss_fn()        → compares output to target, produces a scalar loss
  ↓
loss.backward()  → autograd computes gradients w.r.t. all parameters
  ↓
optimizer.step() → applies the update rule using those gradients
  ↓
optimizer.zero_grad() → clears gradients before the next batch
```

| Concept | Role |
|---|---|
| `autograd` | PyTorch's automatic differentiation engine — tracks operations on tensors with `requires_grad=True` to compute gradients automatically |
| Scheduler | wraps the optimizer, adjusts LR each step/epoch (`scheduler.step()`) |
| `state_dict` | a dict of all learnable parameters (and optionally optimizer state) — the actual thing you save/load for checkpointing |
| Checkpoint | a saved `state_dict` (model, and often optimizer + epoch + metrics) at a point in training |
| AMP (Automatic Mixed Precision) | `torch.autocast` context manager runs eligible ops in FP16/BF16 automatically |
| `GradScaler` | scales loss up before backward (and unscales gradients before the optimizer step) to prevent FP16 gradient underflow — not needed with BF16 |
| Gradient clipping | `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)` called between `backward()` and `step()` |
| Hooks | functions that fire on forward/backward pass of a specific module — used for debugging, feature extraction, gradient inspection |
| Profiling | `torch.profiler` — measures where time/memory is actually spent, used to diagnose the GPU-utilization/OOM issues in Part 30 |

```python
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(batch["input"])
            loss = loss_fn(outputs, batch["target"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        val_loss = sum(loss_fn(model(b["input"]), b["target"]) for b in val_loader)
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), "checkpoint.pt")
```

# PART 37 — CODE-LEVEL PIPELINE TEMPLATES

The point of each template is **correct order**, not completeness — fill in specifics per your problem.

### 1. sklearn regression
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)          # fit on train only
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

model = ElasticNet(alpha=1.0, l1_ratio=0.5).fit(X_train_s, y_train)
preds = model.predict(X_test_s)
print(mean_squared_error(y_test, preds, squared=False))  # RMSE
```

### 2. sklearn classification
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

model = RandomForestClassifier(n_estimators=300, class_weight="balanced").fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))
```

### 3. sklearn clustering
```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X_s = StandardScaler().fit_transform(X)
model = KMeans(n_clusters=5, n_init="auto", random_state=42).fit(X_s)
print(silhouette_score(X_s, model.labels_))
```

### 4. PyTorch MLP
```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)
```

### 5. PyTorch CNN
```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, num_classes)
    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.head(x)
```

### 6. PyTorch transfer learning
```python
import torchvision.models as models

backbone = models.resnet50(weights="IMAGENET1K_V2")
for p in backbone.parameters():
    p.requires_grad = False              # freeze backbone
backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)  # new head, trainable by default

optimizer = torch.optim.AdamW(backbone.fc.parameters(), lr=1e-3)
# ... train head only, then later:
for p in backbone.layer4.parameters():
    p.requires_grad = True               # unfreeze last block
optimizer = torch.optim.AdamW([
    {"params": backbone.layer4.parameters(), "lr": 1e-5},
    {"params": backbone.fc.parameters(), "lr": 1e-3},
])
```

### 7. PyTorch RNN/LSTM
```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden=256, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden * 2, num_classes)
    def forward(self, x):
        emb = self.embedding(x)
        _, (h_n, _) = self.lstm(emb)
        h_cat = torch.cat([h_n[-2], h_n[-1]], dim=1)   # forward+backward final states
        return self.head(h_cat)
```

### 8. PyTorch Transformer (using a pretrained model)
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
outputs = model(**inputs, labels=labels)
loss = outputs.loss
loss.backward()
```

### 9. Contrastive image-text model (CLIP-style, conceptual)
```python
image_emb = normalize(vision_encoder(images))       # (B, D)
text_emb  = normalize(text_encoder(input_ids))       # (B, D)
logits = image_emb @ text_emb.T / temperature        # (B, B) similarity matrix
labels = torch.arange(len(images))                   # diagonal = matching pairs
loss = (cross_entropy(logits, labels) + cross_entropy(logits.T, labels)) / 2
```

### 10. Fine-tuning a pretrained model (full)
```python
model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=n)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
# standard training loop (Part 36) with a small LR appropriate for fine-tuning
```

### 11. LoRA/QLoRA conceptual pipeline
```python
from peft import LoraConfig, get_peft_model

base_model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True)  # QLoRA base
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"], lora_dropout=0.05)
model = get_peft_model(base_model, lora_config)   # only LoRA params are trainable
model.print_trainable_parameters()
# then train normally — only a small fraction of parameters actually update
```

### 12. Evaluation pipeline (generic)
```python
model.eval()
all_preds, all_targets = [], []
with torch.no_grad():
    for batch in test_loader:
        preds = model(batch["input"])
        all_preds.append(preds); all_targets.append(batch["target"])
# then: compute task-specific metrics (Parts 7, 9, 31-33) on concatenated preds/targets
```

---

# PART 38 — "WHERE DOES THIS TECHNIQUE BELONG?" INDEX

| Concept | Pipeline stage | Before/During/After training | Why used | Alternatives |
|---|---|---|---|---|
| PCA | preprocessing / feature reduction | before | reduce dimensionality, remove correlated redundancy | UMAP, autoencoders |
| L1 | objective/regularization | during (in the loss) | sparsity, feature selection | L2, Elastic Net |
| L2 / weight decay | objective/regularization | during | shrink weights, reduce overfitting | L1, dropout |
| Elastic Net | model/regularization | during | combine L1 sparsity + L2 stability | plain L1 or L2 |
| BatchNorm | architecture | during (forward pass) | stabilize activation distributions, allow higher LR | LayerNorm, GroupNorm |
| LayerNorm | architecture | during | batch-independent normalization | BatchNorm, GroupNorm |
| Dropout | architecture/regularization | during (training mode only) | prevent co-adaptation, reduce overfitting | weight decay, augmentation |
| Adam | optimizer | during | adaptive per-parameter LR, fast convergence | SGD+momentum, AdamW |
| AdamW | optimizer | during | Adam + correctly decoupled weight decay | Adam+L2 (flawed), Lion |
| SGD | optimizer | during | simple, generalizes well when tuned | Adam family |
| LR scheduler | optimizer wrapper | during | adjust step size over training | fixed LR (rarely ideal) |
| Gradient clipping | training stability | during | prevent exploding-gradient spikes | lower LR, better init |
| Augmentation | data pipeline | before (per-batch, train split only) | expand effective data diversity | more real data, synthetic data |
| LoRA | fine-tuning | during | parameter-efficient adaptation | full fine-tuning, adapters |
| QLoRA | fine-tuning | during | LoRA + quantized base for memory savings | LoRA on full-precision base |
| Quantization | inference optimization | after training | reduce memory/latency at inference | pruning, distillation |
| Pruning | inference optimization | after training (usually) | remove redundant weights/structures | quantization, distillation |
| Distillation | inference optimization | after (separate training run) | compress into a smaller student model | quantization, pruning |
| RAG | inference-time architecture | during inference | ground generation in retrieved documents, reduce hallucination | long-context, fine-tuning knowledge in |
| MoE (Mixture of Experts) | architecture | both (training + inference) | scale model capacity while keeping per-token compute lower | dense scaling |
| ViT | architecture | both | apply Transformer self-attention directly to image patches | CNN backbones |
| ResNet | architecture | both | enable very deep CNN training via residual connections | plain CNN, DenseNet |
| Swin Transformer | architecture | both | windowed attention for efficient high-resolution vision Transformers | plain ViT |
| DETR | architecture | both | end-to-end object detection via Transformer + set prediction | R-CNN family, YOLO |
| U-Net | architecture | both | encoder-decoder with skip connections for dense prediction (segmentation) | FCN, DeepLab |
| BERT | architecture/pretrained model | both | bidirectional encoder pretraining for understanding tasks | RoBERTa, DeBERTa |
| Transformer | architecture | both | attention-based sequence modelling, parallelizable | RNN/LSTM/GRU |
| CLIP | architecture/objective | both | contrastive dual-encoder image-text alignment | ALIGN, BLIP |
| VLM | architecture | both | connect a vision encoder to a language model for grounded generation | separate captioning + LLM pipeline |
| Diffusion | generative architecture | both | iterative denoising generative process | GAN, VAE |
| GAN | generative architecture | both | adversarial generator/discriminator training | diffusion, VAE |
| VAE | generative architecture | both | probabilistic encoder-decoder with a regularized latent space | GAN, diffusion |
| Knowledge distillation | inference optimization | after (separate run) | transfer a large model's behavior into a small one | quantization, pruning |

*(Cross-reference the relevant Part above for full detail on any row — this index is a locator, not a duplicate explanation.)*

---

# PART 39 — HYPERPARAMETER CHEAT SHEETS

| Model/Technique | Parameter | Typical range | Starting point | Increase when | Decrease when |
|---|---|---|---|---|---|
| Ridge/Lasso | `alpha` | 1e-3 – 100 (log scale) | 1.0 | underfitting is not the issue, still overfitting | model underfits, coefficients too shrunk |
| Elastic Net | `l1_ratio` | 0–1 | 0.5 | want more sparsity | want more Ridge-like stability |
| Logistic Regression | `C` (inverse reg. strength) | 1e-3 – 100 | 1.0 | model underfits | model overfits |
| SVM | `C` | 1e-2 – 100 | 1.0 | underfitting (too much margin violation tolerance) | overfitting |
| SVM (RBF) | `gamma` | 1e-4 – 10 | "scale" (sklearn default) | decision boundary too smooth/underfit | boundary too complex/overfit |
| KNN | `n_neighbors` | 3–50 | 5–15 | boundary too noisy (overfitting to individual points) | boundary too smooth (underfitting) |
| Decision Tree | `max_depth` | 3–30 | tune via CV, no universal value | underfitting | overfitting |
| Random Forest | `n_estimators` | 100–1000 | 300–500 | more stable estimate needed, compute allows | compute-constrained (returns diminish after a point) |
| Random Forest | `max_depth` | None (full) or 5–30 | None, then constrain if overfitting | underfitting | overfitting |
| XGBoost/LightGBM | `learning_rate` | 0.01–0.3 | 0.05–0.1 | training too slow (with more `n_estimators` to compensate) | overfitting, unstable training |
| XGBoost/LightGBM | `n_estimators` | 100–5000 | tune with early stopping | underfitting | overfitting / compute-constrained |
| XGBoost/LightGBM | `max_depth` | 3–10 | 6 | underfitting | overfitting |
| K-Means | `n_clusters` (K) | problem-dependent | use elbow/silhouette | underlying structure has more groups | fewer, more separable groups suffice |
| MLP | hidden units per layer | no universal range | 64–256 for tabular | underfitting | overfitting, small data |
| MLP/CNN/Transformer | `dropout` | 0.1–0.5 | 0.2 (dense layers), lower for conv layers | overfitting | underfitting, slow convergence |
| CNN/Transformer | `weight_decay` | 1e-5 – 1e-2 | 1e-4 (CNN), 0.01 (Transformer/AdamW) | overfitting | underfitting |
| NN (general) | `batch_size` | 16–512+ | 32–128, hardware-dependent | need smoother gradient estimates, have memory | memory-constrained, want regularizing noise from smaller batches |
| NN (general, Adam/AdamW) | `learning_rate` | 1e-5 – 1e-2 | 1e-3 (from scratch), 1e-5–2e-5 (fine-tuning pretrained) | training too slow to converge | loss unstable/diverging |
| Transformer fine-tuning | `learning_rate` | 1e-6 – 5e-5 | 2e-5 | underfitting the new task | catastrophic forgetting / instability |
| LoRA | `r` (rank) | 4–64 | 8–16 | task needs more adaptation capacity | want maximal parameter efficiency |
| LoRA | `lora_alpha` | typically 2x `r` | 16 (with r=8) | — | — |
| Gradient clipping | `max_norm` | 0.5–5 | 1.0 | gradients aren't actually exploding (clipping rarely triggers) | still seeing instability |
| LR warmup | warmup steps/ratio | 0–10% of total steps | ~5% | training unstable at the start | training already stable without it |

For anything not listed with a specific range: **"No universal range; depends strongly on dataset/model — tune via validation."** This applies especially to model-size-related parameters (hidden dims, number of layers/heads) and to `n_clusters`/component counts, which are inherently dataset-dependent.

---

# PART 40 — "WHAT SHOULD I TRY FIRST?" TABLES

### Small tabular regression dataset
1. Baseline (predict mean/median).
2. Linear/Ridge regression — cheap sanity check on linear signal.
3. A single tree-based model (Random Forest) — captures nonlinearity/interactions with minimal tuning.
4. Gradient boosting (XGBoost/LightGBM) — usually the strongest option for small-medium tabular data.
5. Compare all above on the validation metric.
6. Inspect residuals of the best model.
7. Tune the winning model's key hyperparameters (Part 39).

### Small tabular classification dataset
Same order as above, swapping in Logistic Regression and classification metrics; check class balance *before* step 1 and stratify the split accordingly.

### Small image dataset
1. Pretrained CNN backbone (e.g. ResNet50) — don't train from scratch with limited data.
2. Freeze backbone, train a new head.
3. Add augmentation appropriate to the domain.
4. Once head converges, unfreeze the last block(s).
5. Fine-tune with a much smaller LR on unfrozen layers.
6. Evaluate, check for overfitting given the small dataset size, add regularization (Part 25) if needed.

### Large image dataset
1. Decide pretrained-vs-scratch based on how much data/compute you actually have — pretrained is still usually the safer default even at scale.
2. Strong augmentation pipeline.
3. AdamW or SGD+momentum with a proper LR schedule (warmup + decay).
4. Mixed precision to speed up training.
5. Checkpointing on every validation improvement.
6. Distributed training (Part 35) only once a single GPU's throughput is the actual bottleneck.

### The reasoning behind this ordering, generally
Always establish a floor (baseline) before anything else — it tells you whether your "sophisticated" model is doing any real work. Always try the cheapest reasonable model before the most complex one — added complexity should be justified by a measured improvement, not assumed. Always prefer transfer learning over training from scratch unless data/compute genuinely favor scratch training (rare in practice below very large dataset sizes).

---

# PART 41 — EXPERIMENTATION LOOP

```text
QUESTION            (what do I actually want to know?)
   ↓
HYPOTHESIS          (what change do I believe will help, and why?)
   ↓
BASELINE            (what am I comparing against?)
   ↓
ONE CONTROLLED CHANGE  (change exactly one variable — otherwise you
                        can't attribute the result to anything)
   ↓
TRAIN
   ↓
EVALUATE
   ↓
COMPARE             (against baseline, on the SAME validation set/protocol)
   ↓
ERROR ANALYSIS      (look at actual failure cases, not just the metric)
   ↓
ABLATION            (if the change helped, verify which part of it mattered)
   ↓
CONCLUSION          (keep, discard, or refine the hypothesis → loop again)
```

### Experiment tracking — record for every run

`configuration (all hyperparameters), dataset version, random seed, model architecture/checkpoint used, resulting metric(s), saved checkpoint path, hardware used, wall-clock runtime.`

Without this, you cannot reliably reproduce your best result or explain why two runs differ.

---

# PART 42 — FINAL MASTER DECISION TREE

```text
WHAT TYPE OF DATA DO I HAVE?
│
├── TABULAR
│   └── TARGET?
│       ├── CONTINUOUS         → REGRESSION            (Part 6-7)
│       ├── CATEGORY           → CLASSIFICATION         (Part 8-9)
│       └── NONE                → UNSUPERVISED
│                                  ├── grouping needed   → CLUSTERING (Part 10)
│                                  ├── compress/visualize → DIM. REDUCTION (Part 10)
│                                  └── find rare/bad points → ANOMALY DETECTION (Part 8 imbalance tree, extreme case)
│
├── IMAGE
│   ├── one label per image     → IMAGE CLASSIFICATION  (Part 13)
│   ├── locate + label objects  → OBJECT DETECTION       (Part 32)
│   └── per-pixel labeling      → SEGMENTATION            (Part 32)
│
├── TEXT / SEQUENCE
│   ├── short sequence, limited compute → RNN/LSTM/GRU  (Part 16)
│   ├── classification/understanding    → Transformer encoder (Part 17)
│   ├── open-ended generation           → Transformer decoder (Part 17-19)
│   └── search/retrieval                → embeddings + RAG (index in Part 38)
│
└── MULTIMODAL
    ├── align two modalities in a shared space → DUAL ENCODER / CLIP (Part 20)
    ├── condition generation on another modality → CROSS-ATTENTION / VLM (Part 21)
    └── image + text → open-ended text output   → VLM (Part 21)
```

Every leaf then re-enters the universal skeleton:

`DATA → PREPROCESSING → REPRESENTATION → MODEL → LOSS → OPTIMIZER → TRAINING → VALIDATION → EVALUATION → OPTIMIZATION (inference) → INFERENCE`

---

# PART 43 — FINAL ONE-PAGE CHEAT SHEET

Work through these in order for any new problem:

```text
 1. What is my task?                     → locate it in Part 0/42
 2. What is my target?                    → continuous / category / none / generative
 3. What type of data?                    → tabular / image / text / sequence / multimodal
 4. What can be wrong with the data?      → Parts 2-3, 27-29 (missing, outliers, leakage)
 5. What preprocessing is required?       → Part 4 (scaling/encoding), model-dependent
 6. What transformations?                 → log/power transforms, tokenization, image ops
 7. What split strategy?                  → Part 5 (random/stratified/group/time)
 8. What baseline?                        → dumb model — mean/median/majority class
 9. What model family?                    → Part 6/8/10 classical, Part 11+ neural
10. What architecture?                    → Part 12 decision tree (MLP/CNN/RNN/Transformer)
11. What representation?                  → embeddings/tokenization/pixel tensors
12. What loss?                            → matches the task (Parts 7, 9, 18)
13. What optimizer?                       → Part 23-C (Adam/AdamW/SGD+momentum)
14. What LR?                              → Part 23-D + Part 39 table
15. What scheduler?                       → warmup+decay for Transformers, step/plateau for CNNs
16. What regularization?                  → Part 25, driven by overfitting evidence
17. What augmentation?                    → Part 26, train split only
18. What batch size?                      → Part 39, memory-constrained
19. What normalization?                   → Part 24 — data-level AND/OR architectural
20. What should I freeze?                 → Part 15, transfer-learning decision tree
21. What should I tune?                   → Part 23-H HPO method, Part 39 ranges
22. What metric?                          → Part 31, matched to task + error costs
23. What failure mode?                    → Part 30 debugging tree
24. What experiment next?                 → Part 41 loop, one controlled change
25. How do I test?                        → touch the test set exactly once (Part 5)
26. How do I export?                      → Part 34 (checkpoint/precision/format)
27. How do I optimize inference?          → Part 34 (quantize/prune/distill/batch/serve)
```

---

*End of document. Every numbered Part above is independently searchable — jump straight to the Part matching your current problem rather than reading front-to-back.*
