# MASTER ML/DL FIELD MANUAL & DECISION PIPELINE

This document provides a rigorous engineering framework for ML/DL implementation, training, evaluation, and deployment. It prioritizes conditional decision-making over universal defaults, focusing on diagnostics, measurement, and validating assumptions.

---

## 1. DATA UNDERSTANDING & PREPROCESSING CONTRACT

**Condition:** Are you extracting parameters or decisions from the data (e.g., means, clipping bounds, vocabulary)?
**Action:** Any such operation must be fitted ONLY on the training distribution.
**Validation:** Ask, "Could this code execute on a single new user in production?" If it requires the entire dataset to compute a mean before processing a single row, it must be fitted strictly inside the cross-validation or training fold.

### 1.1 Safe Schema and Integrity Inspection
*   **Schema & Types:** Validate expected types, ranges, and formats.
*   **Corruption:** Detect unreadable images, null bytes, or broken encodings.
*   **Provenance:** Audit the source system, extraction timestamp, and lineage to ensure the data matches the intended prediction scenario.

### 1.2 Outlier Diagnosis and Intervention
Outliers are not a monolith, nor are they strictly "errors."
*   **Corrupted Measurement / Data-Entry Error:** E.g., Age = 250.
    *   *Action:* Treat as missing data, correct via source if possible, or domain-exclude.
*   **Valid Extreme (Tail Event):** E.g., Salary = $5M.
    *   *Action:* Retain if deployment requires scoring these entities.
*   **Covariate Shift / Leverage Point:** A valid sample that exerts undue influence on a linear model.
    *   *Action:* Use a robust transformation (e.g., QuantileTransformer) fitted on training data, or capping/clipping.
*   **Target Noise:** An anomalous target value for a standard input.
    *   *Action:* Consider robust regression objectives (Huber, SmoothL1).

*Note on Models:* Do not assume tree models are perfectly immune to outliers, or that outliers always explode NN gradients. Measure the sensitivity by comparing standard vs. robust transformations under leakage-safe CV.

---

## 2. MISSING VALUES DIAGNOSIS

**Principle:** MCAR, MAR, and MNAR are assumptions about the data-generating process, not properties that can be conclusively proven from the dataset alone. 

### 2.1 Analyzing the Missingness Process
*   **MCAR (Missing Completely At Random):** Missingness is entirely independent of any data. (Rare in practice).
*   **MAR (Missing At Random):** Missingness is fully explained by *other observed* covariates (e.g., men reporting depression less). 
*   **MNAR (Missing Not At Random):** Missingness depends on the *unobserved* value itself (e.g., wealthy people refusing to report income). 

### 2.2 Decision Framework for Missing Data
1.  **Understand Operational Collection:** Why is it missing? Is it structurally absent (e.g., "Days since last purchase" for a user who never purchased)? If so, use a domain-specific representation (e.g., a categorical "Never" or sentinel value).
2.  **Establish Inference Availability:** Will this feature be missing at deployment? If yes, dropping rows with missing values during training will result in a model that cannot score deployment data.
3.  **Evaluate Indicators:** Adding a binary `is_missing` indicator can capture the predictive signal of the missingness itself. 
    *   *Hypothesis Check:* If an indicator improves performance, it means missingness carries information. It does *not* prove MNAR; it could mean structural missingness, operational routing, site effects, or leakage.
4.  **Compare Interventions via CV:**
    *   *Simple Imputation:* May be an acceptable predictive baseline under plausible assumptions and successful validation, but it can alter variance/correlation and does not automatically preserve inferential uncertainty. Ensure it is fitted only on training folds and tested across subgroups.
    *   *Model-based (MICE/KNN):* Conditional on the imputation model being correct. Does not guarantee distribution recovery.
    *   *Native Tree Routing:* Algorithms like XGBoost/LightGBM learn a default path for NaNs. This is a strong predictive implementation choice, but it does *not* remove selection bias or fix MNAR.
    *   *Separate Availability Modeling:* Model the probability of missingness separately.

---

## 3. COMPOUND SPLITS & EVALUATION REGIMES

**Principle:** The validation split must mirror the deployment sampling constraints. Real-world tasks frequently combine multiple dependency axes.

### 3.1 Deployment Unit (Entity Constraints)
**First Question: AT DEPLOYMENT, ARE PREDICTIONS MADE FOR: A) EXISTING ENTITIES WITH HISTORY, B) PREVIOUSLY UNSEEN ENTITIES, C) BOTH?**
*   *Existing entities:* (e.g., future prediction for a known patient). Historical patient data may legitimately occur in training while later observations are test.
*   *Unseen entities:* (e.g., generalization to new patients). Patient grouping must prevent cross-patient leakage (e.g., GroupKFold).
*   *Both:* Evaluate both regimes separately.

### 3.2 Dependency Axes
Check independently for:
*   **Entity/Group:** Multiple measurements from the same patient, session, or device. (Requires `GroupKFold`).
*   **Time:** Predicting the future from the past. (Requires Walk-forward/Rolling split).
*   **Site/Domain:** Deploying to new hospitals or geographical regions. (Requires Leave-One-Group-Out).
*   **Duplicate Family:** Near-duplicate images or documents. (Must be kept in the same fold).

### 3.2 Composing Constraints
Do not force a choice between group and time if both matter.
*   *Grouped + Temporal:* E.g., predicting patient outcomes next week. All historical data for Patient A must remain in the train set, and the test set must be strictly in the future.
*   *Site + Temporal:* Train on Hospital X data from 2022, evaluate on Hospital Y data from 2023.

### 3.3 Multiple Evaluation Regimes
A single test set is often insufficient. Maintain:
1.  **In-domain test set:** Baseline generalization.
2.  **Future-time test set:** Evaluates temporal decay.
3.  **Cross-site test set:** Evaluates geographic/domain robustness.
4.  **OOD / Subgroup test set:** Evaluates fairness and extreme distribution shift.


## 4. CATEGORICAL REPRESENTATION & FEATURE ENGINEERING

**Condition:** You have discrete categorical variables.
**Action:** Do not arbitrarily apply integer/label encoding to linear or tree models when category order is not meaningful.

### 4.1 Representation Matrix
1.  **One-Hot Encoding (OHE):** Creates binary columns. Use for low cardinality. 
    *   *Limitation:* Explodes memory and dimensionality for high cardinality. Use sparse matrices if supported.
2.  **Ordinal Encoding:** Use ONLY when a strict, monotonically meaningful order exists (e.g., Low, Medium, High). 
3.  **Frequency / Count Encoding:** Replaces categories with their training frequency. Useful for tree models to isolate rare vs. common groups.
4.  **Target Encoding:** Replaces categories with the mean target value. 
    *   *Limitation:* Extreme leakage risk. MUST be fitted using cross-fitting (calculating the mean using out-of-fold data) and smoothed/regularized for rare categories.
5.  **Embeddings (Neural Networks):** Learns a dense, low-dimensional vector for each category. Excellent for high cardinality (e.g., Zip codes).
6.  **Hashing:** Hashes categories to a fixed number of bins. Use when cardinality is practically infinite or the category universe is unknown at train time (accepting collisions).
7.  **Native Support:** Models like LightGBM/CatBoost partition categories mathematically without OHE. Highly efficient, but memory limits apply.

*Unseen Categories:* Always define a policy (e.g., map to "Unknown", use a default embedding, or handle via OHE zero-vectors) for categories that appear in deployment but not in training.

---

## 5. CLASSICAL MODEL-FAMILY SELECTION

**Principle:** A strong classical baseline is valuable, even if deep learning is the ultimate goal, to establish the lower bound of acceptable performance.

### 5.1 Selection Matrix
*   **Linear / Logistic Regression:**
    *   *When to test:* High dimensionality, sparse data (e.g., TF-IDF), strict latency constraints, or when linear interpretability is mandated.
    *   *Generally benefits from:* Feature scaling, interaction terms, missing value imputation, and explicit nonlinear transformations.
*   **Decision Trees:**
    *   *When to test:* Extremely simple baseline, strong non-linear interactions, low latency. Very prone to overfitting.
*   **Random Forests:**
    *   *When to test:* Small to medium tabular data, resistant to overfitting, easily parallelized. Less tuning required than GBMs.
*   **Gradient Boosting (XGBoost, LightGBM, CatBoost):**
    *   *When to test:* Highly effective for structured/tabular data. Model choice depends on dataset size, sparsity, cardinality, nonlinear structure, missingness, latency, interpretability, monotonicity, and calibration requirements. Handles missing data natively.
    *   *Generally benefits from:* Careful HPO (depth, learning rate, regularization) to prevent overfitting.
*   **SVM (Support Vector Machines):**
    *   *When to test:* Small sample sizes with wide margins, text classification.
    *   *Limitation:* Scales poorly to large data ($O(N^2)$ to $O(N^3)$).
*   **Neural Tabular Models (TabNet, FT-Transformer):**
    *   *When to test:* Massive tabular datasets ($>1M$ rows) or when joint multimodal training (tabular + text/image) is required.

---

## 6. HYPERPARAMETER OPTIMIZATION (HPO)

**Principle:** HPO must not repeatedly overfit a single validation set without acknowledgement.

### 6.1 Search Strategies
1.  **Manual Baseline:** Always establish a performance floor with default or intuition-based parameters before initiating automated search.
2.  **Grid Search:** Exhaustive. Only use for 1-2 parameters with small domains.
3.  **Random Search:** Statistically dominates Grid Search for high-dimensional spaces because it explores more unique values of the most important (but unknown) hyperparameters.
4.  **Bayesian Optimization:** Fits a surrogate model (e.g., Gaussian Process, TPE) to predict performance and sample promising regions. Highly efficient for expensive models.
5.  **Hyperband / Successive Halving:** Allocates budget to many random configurations, pruning the worst half early (e.g., at 10 epochs) and training the survivors longer. Ideal for Deep Learning.

### 6.2 Validation Overfitting
If you test 1,000 configurations on Validation Set A, you will select the configuration that optimally exploits the noise in Set A. 
*   *Validation:* Use **Nested CV** if you need an unbiased estimate of the final model's performance, or maintain an absolute hold-out test set that is NEVER touched during HPO.

---

## 7. DATA DISTRIBUTION SHIFT

**Principle:** Understand how the data generator changes between training and deployment.

*   **Covariate Shift:** The input distribution $P(X)$ changes, but the relationship $P(Y|X)$ remains constant (e.g., training on young patients, deploying on old patients). 
    *   *Action:* Importance weighting (consider support overlap and weighting variance), domain adaptation.
*   **Prior / Label Shift:** The target distribution $P(Y)$ changes, but $P(X|Y)$ remains constant (e.g., a disease becomes an epidemic).
    *   *Action:* Adjusting the threshold based on prevalence does *not* automatically correct probabilities. Use dedicated probability recalibration methods if probability quality is needed.
*   **Concept Shift:** The relationship $P(Y|X)$ changes (e.g., macroeconomic shifts change purchasing behavior).
    *   *Action:* Retraining on recent data is a common response, but not always mandatory. Alternatives include recalibration, policy adjustment, abstention, model replacement, or feature investigation. The correct action depends on diagnosis and operational constraints.
*   **Open Set / OOD (Out of Distribution):** Deployment contains inputs entirely outside the training support (e.g., an unseen disease class).
    *   *Action:* Define an abstention/rejection policy. Do not assume a softmax confidence score equals uncertainty (models can be highly confident about OOD data).


## 8. LABEL QUALITY & IMBALANCE

### 8.1 Label Disagreement and Ambiguity
When annotators disagree (e.g., Annotator A says "Tumor", Annotator B says "Healthy"):
1.  **Preserve Annotator Labels:** Do not immediately flatten to a majority vote.
2.  **Is there a single objectively correct label?**
    *   *Yes:* Expert adjudication or annotator-noise models (Weak Supervision) may be appropriate to recover the "true" label. Majority vote is only reasonable under explicit assumptions about equal annotator skill.
3.  **Is the ambiguity genuine or part of the target?**
    *   *Yes:* Soft/probabilistic labels (e.g., `[0.6, 0.4]`) or distributional targets may be appropriate. Disagreement can represent genuine ambiguity, poor instructions, or systemic confusion.

### 8.2 Imbalance & Metric Logic
Accuracy is often insufficient under severe imbalance. Ask: **What is the evaluation objective?**
1.  **Ranking:** Use ROC-AUC or PR-AUC.
2.  **Probability quality:** Use Brier Score or Log Loss (Calibration metrics).
3.  **Operating-point decision:** Use F1-Score, Precision, Recall, or Specificity.
4.  **Cost-sensitive utility:** Use explicit Cost-weighted utility.
5.  **General Quality:** Accuracy can be valid when aligned with the objective.

**Intervention Hierarchy:**
Interventions can be composed.
1.  **Class Weighting:** Modifies the loss function.
2.  **Resampling / Augmentation:** Adjusts the data.
3.  **Focal Loss:** Down-weights easy, high-confidence examples.
4.  **Threshold Selection:** Changes the operating point on the PR/ROC curve. *Note:* Changing the threshold alters Precision/Recall, but it does *not* improve the model's fundamental ranking ability or calibrate its probabilities.

---

## 9. OUTPUT CONTRACTS, UNCERTAINTY & REGRESSION

**Principle:** Ensure explicit alignment between output shape, activation, target shape, and loss.

### 9.1 Regression Transformations
*   **Log-Target Transformations:** Taking $\log(Y)$ stabilizes variance (heteroscedasticity) for strictly positive targets with multiplicative noise.
    *   *Limitation:* Retransforming via $\exp(\hat{Y})$ predicts the *geometric mean*, not the arithmetic mean. If business logic requires the arithmetic mean, you must apply a variance correction (Duan's smearing estimator).

### 9.2 Uncertainty, Intervals, and Quantiles
Do not conflate confidence scores with uncertainty.
*   **Aleatoric Uncertainty:** Inherent noise in the data (e.g., sensor noise).
*   **Epistemic Uncertainty:** Model ignorance due to lack of training data. (Can be reduced by more data).
*   **Quantile Regression:** Predicts conditional quantiles (e.g., 10th and 90th percentile). It estimates the bounds of the target distribution given $X$. *It does NOT automatically provide guaranteed worst-case bounds or epistemic uncertainty. (Note: Quantile crossing is a common implementation issue).*
*   **Conformal Prediction:** A separate, post-hoc approach to generate prediction intervals that provide mathematically calibrated coverage guarantees (e.g., "The true value is within this interval 90% of the time"), assuming exchangeability. Standard split conformal provides marginal coverage under this assumption, but time dependence, group dependence, covariate shift, or distribution shift can invalidate naive guarantees. Always require empirical coverage and interval width evaluation.

---

## 10. EVALUATION & CALIBRATION FRAMEWORK

### 10.1 Calibration
**Condition:** Do you need probability quality for risk decisions, cost allocation, or downstream triage?
*   *If No:* Calibration may be a lower priority than ranking (ROC-AUC).
*   *If Yes:* 
    1. Measure calibration (Reliability diagrams, Brier score).
    2. Apply Platt Scaling, Isotonic Regression, or Temperature Scaling.
    3. *Constraint:* Calibration MUST be fitted on a disjoint validation set or via cross-fitting to avoid reusing the exact outcomes used to fit the base model.
    4. *Note:* For multiclass/multilabel, simple binary Platt scaling is often insufficient; use matrix/temperature scaling.

### 10.2 Statistical Comparison of Models
**Condition:** Model B scores 0.90, Model A scores 0.89. Is B better?
*   *Invalid Check:* Do not assume that overlapping Confidence Intervals (CIs) for the two models means there is no statistical difference. Two individual CIs can overlap even if the paired difference is highly significant.
*   *Valid Check:* Base the statistical procedure on the metric and design conditions:
    *   **Classification:** Paired binary predictions/errors (McNemar), ROC-AUC (DeLong/Bootstrap), Calibration/Thresholded metrics.
    *   **Regression:** MAE, MSE, RMSE, Quantile loss.
    *   **Design dependence:** Distinguish fixed test sets (paired tests), stochastic training seeds (matched-seed comparison), grouped/temporal data (block/group bootstrap), and clustered observations.
*   *Focus:* Emphasize effect size, uncertainty of the difference, practical significance, and dependence structure rather than merely seeking "statistical significance."

### 10.3 Clustering Evaluation
Internal metrics (Silhouette) are imperfect because they assume cluster geometries (like spheres) that may not align with density-based reality.
*   *Validation Protocol:* 
    1. Test stability under data resampling (does the clustering fundamentally change?).
    2. Check density/reachability diagnostics.
    3. Measure domain coherence (do the clusters make subject-matter sense?).
    4. Measure downstream predictive utility or business ROI.


## 11. NEURAL NETWORK TRAINING DIAGNOSIS & REGULARIZATION

**Principle:** Do not apply interventions blindly. Use staged diagnosis: Hypothesis $\rightarrow$ Test $\rightarrow$ Observation $\rightarrow$ Action.

### 11.1 Staged Diagnosis Flow
If a model fails to train or generalize:
1.  **Data & Labels:** Are inputs scaled? Are labels corrupted?
2.  **Output Contract:** Does the loss function match the final layer activation?
3.  **Numerical Stability (NaNs detected):** Check finite inputs/targets $
ightarrow$ check first non-finite tensor/op $
ightarrow$ check loss-domain constraints $
ightarrow$ run one-batch FP32 diagnostic $
ightarrow$ inspect activations/gradients/normalization $
ightarrow$ then test LR / precision / clipping hypotheses. *Make clear that gradient clipping can hide a root cause.*
4.  **Small-Batch Overfit Test:** Can the model memorize a single batch (Loss $
ightarrow 0$)? If not, the architecture, capacity, or optimization routing is fundamentally broken.

### 11.2 Regularization & Optimization
Avoid declaring one optimizer (e.g., AdamW) universally best.
*   **Initialization:** Kaiming/Xavier based on activation functions.
*   **Weight Decay:** L2 penalty. *Crucial:* Exclude biases and LayerNorm/BatchNorm affine parameters from weight decay as a commonly tested optimizer recipe whose usefulness depends on architecture and optimizer.
*   **EMA / SWA (Exponential Moving Average / Stochastic Weight Averaging):** Maintains a moving average of weights for smoother, flatter optima. Describe as optional techniques to evaluate, with costs and interaction considerations.
*   **Dropout:** Place strategically (e.g., before dense layers, not typically before conv layers).
*   **Schedules:** Cosine decay with warmup is standard, but adjust warmup length based on batch size changes and initialization stability.

### 11.3 Normalization Axes
Normalization depends on batch size, spatial dimensions, and distributed setups.
*   **BatchNorm:** Calculates statistics across the batch. 
    *   *Limitations:* Becomes noisy with small effective batch sizes. Requires synchronization (`SyncBN`) across GPUs in distributed setups. 
    *   *Evaluation:* Running statistics must be carefully managed. When freezing a backbone, either freeze the BN running stats or force `.eval()` mode.
*   **LayerNorm:** Calculates across channels for a single sequence/sample.
*   **GroupNorm:** Divides channels into groups. Excellent alternative to BN when VRAM constraints force micro-batches.

---

## 12. DATA AUGMENTATION

**Condition:** You need to expand dataset support or enforce invariances.
**Decision Framework:**
1.  *Is it label-preserving?* (E.g., flipping a "6" vertically creates a "9" - not label preserving for OCR).
2.  *Does it match deployment variation?* (Adding Gaussian noise is useless if deployment noise is JPEG compression artifacts).
3.  *Does geometry affect labels?* (In Object Detection, if you crop/rotate the image, you MUST mathematically update the bounding box targets).
4.  *Is it applied train-only?* (Validation data must remain un-augmented unless using Test-Time Augmentation for ensembling).
5.  *Does it alter class balance or harm rare subgroups?* 
*   *Validation:* Ablate augmentations independently to provide evidence they help validation performance under the specific experimental design.

---

## 13. DISTRIBUTED TRAINING & HARDWARE PROFILING

### 13.1 GPU Profiling & Bottlenecks
Do not assume $<50\%$ GPU utilization automatically means an I/O bottleneck, or that $100\%$ means you are compute-bound. **Profile.**
*   *Diagnostic Chain:* Measure $\rightarrow$ Identify wait/compute/comm $\rightarrow$ Test bottleneck $\rightarrow$ Optimize.
*   *Dataloader Timing:* Measure time spent fetching batches.
*   *H2D (Host-to-Device) Transfer:* Are you moving data to the GPU unnecessarily?
*   *Memory Bandwidth vs Compute:* Transformers are often memory-bandwidth bound during generation (fetching weights), not compute-bound.
*   *Communication:* Are GPUs stalling waiting for All-Reduce synchronization?

### 13.2 Memory and Topology Decision Framework
Parallelism is a composable engineering decision. Consider these dimensions: parameter memory, gradient memory, optimizer-state memory, activation memory, KV-cache memory, microbatch size, global batch, communication volume, interconnect bandwidth, latency, topology, model architecture, and framework support.

*   **DDP (Distributed Data Parallel):** Replicates model, shards data.
*   **FSDP (Fully Sharded Data Parallel) / ZeRO:** Shards model, gradients, and optimizer states.
*   **Tensor Parallelism (TP):** Shards matrix multiplications. NVLink/high-bandwidth interconnect makes TP substantially more attractive, but is not technically required.
*   **Pipeline / Context / Sequence Parallelism:** Shards layers or context windows across GPUs.
*   **Memory Budgeting Techniques:** Activation Checkpointing, CPU/NVMe Offload, and Quantization. Make clear that techniques may be combined.

**Practical Memory Worksheet:**
At minimum, distinguish:
1. Weights
2. Gradients
3. Optimizer states
4. Activations
5. Temporary/workspace buffers
6. KV cache
7. Distributed replication/sharding
8. Fragmentation/allocator overhead
*Actual fit depends on precision, sequence length, microbatch, architecture, optimizer, checkpointing, sharding, offload, and quantization.*

### 13.3 Precision & Accumulation
*   **Mixed Precision:** Detect device support. Use `torch.amp.autocast`. Use `GradScaler` primarily for FP16. Do not unnecessarily scale BF16. Disable AMP when numerically unsafe (e.g., certain loss calculations).
*   **Gradient Accumulation:** Accumulate gradients over micro-batches. 
    *   *Implementation requirement:* You must explicitly handle the final partial-window (either normalize by the actual number of micro-batches or drop the remainder).
    *   *Scheduler Cadence:* Do not blindly call `scheduler.step()` per epoch. Ensure it matches the optimizer update cadence.

### 13.4 Exact vs Partial Resumable Checkpoints
Do not claim a checkpoint provides "exact resume" unless it guarantees bitwise replay.
*   **Partial Resumable Checkpoint (Standard):** Saves `model`, `optimizer`, `scheduler`, `scaler`, `epoch`.
*   **High-Fidelity Deterministic Resume:** Requires saving RNG states (Python, NumPy, PyTorch CPU, CUDA), dataloader/sampler states, streaming data position, EMA state, and worker RNG states. Even matching seeds does not guarantee bitwise reproducibility across different hardware/driver versions.


---

## 14. ARCHITECTURE SELECTION & TRANSFORMER DYNAMICS

### 14.1 Computer Vision Constraints
Do not blindly default to ConvNeXt, ViT, or DETR as "universal SOTA." Route through task, data, and compute constraints.
*   *Classification:* CNNs (ResNet) for limited data/compute; ViT for massive scale and parameter efficiency at scale.
*   *Detection:* One-stage (YOLO variants) for low latency/edge; Two-stage (Faster R-CNN) for high mAP; Transformer (DETR) if NMS heuristics fail, but requires long training.

### 14.2 Transformer & KV Cache Dynamics
*   **Norm Ordering:** Pre-Norm is widely used for stability in deep networks without warmup, but Post-Norm remains valid depending on architecture and initialization scheme.
*   **KV Cache Complexity:** 
    *   *Misconception:* "KV cache gives O(1) generation per token."
    *   *Reality:* While KV cache avoids recomputing historical K/V projections, the new query still attends over the entire cached context. Therefore, attention compute per generated token remains dependent on context length. Furthermore, cache memory grows linearly with context length, severely bottlenecking batch size (memory-bound inference).

### 14.3 Foundation Model Adaptation
Adaptation is an empirical investment decision.
1.  **RAG (Retrieval-Augmented Generation):** 
    *   *Use:* Grounding answers in external/dynamic data.
    *   *Limitations:* Introduces retrieval quality bottlenecks, stale documents, prompt injection risks, and citation/grounding failures.
2.  **SFT (Supervised Fine-Tuning):** 
    *   *Use:* Formatting behavior, tone, instruction following.
3.  **PEFT (LoRA/QLoRA):** 
    *   *Hardware/Memory Budgeting:* Feasibility depends on parameter precision, gradients, sequence length, microbatch size, architecture, and activation checkpointing. Use a memory worksheet rather than relying on claims like "fits on a 24GB GPU."
4.  **DPO / RLHF:**
    *   *Use:* Preference optimization against a dataset/objective. 
    *   *Limitation:* Optimizes for the preference model; it does NOT guarantee safety. Requires rigorous red-teaming.
5.  **Continued Pretraining (Domain Adaptation):** 
    *   *Decision:* Not the default. Measure the baseline, diagnose the domain gap, estimate the corpus-token budget, evaluate catastrophic forgetting, and compare general vs. domain held-out performance before committing.

---

## 15. MULTIMODAL (VLMs) & CONTRASTIVE LEARNING

### 15.1 CLIP and Contrastive Dynamics
*   *Misconception:* "CLIP relies entirely on hard negatives and small batches cause collapse."
*   *Reality:* Standard CLIP uses *in-batch* negatives. Large batches increase negative diversity, but they are a representation/retrieval-quality tradeoff, not a universal requirement. Consider in-batch negatives, false negatives, multi-positive situations, hard-negative mining, cross-device negative gathering (weighing communication/memory costs), and the overall quality of the negative pool.

### 15.2 VLM (Vision-Language Model) Taxonomy
*   **Architectures:** Connector-based, Cross-Attention, Resampler, and Unified Multimodal Transformers.
*   **Document Understanding Routing Branch:** Do not prescribe VLM as the universal document solution. Ask:
    1. Is the task OCR?
    2. Is it field extraction?
    3. Is layout/structure required?
    4. Is visual grounding required?
    5. What image resolution is required?
    6. What is the token budget?
    7. What is the latency budget?
    8. Can a dedicated OCR/layout model solve it more efficiently?
    9. Is a VLM actually needed?
    10. What is the fallback if latency or memory fails?
*   **VLM Evaluation Matrix:** Include measured metrics such as OCR accuracy, field extraction accuracy, grounding accuracy (where relevant), latency (p50/p95/p99), throughput, and memory. Dynamic tiling improves OCR but consumes massive context/compute and introduces stitching tradeoffs.

---

## 16. DATA CURATION & TOKENIZATION

### 16.1 Tokenization and OOV
*   *Misconception:* "OOV is inevitable." Many subword tokenizers (BPE, SentencePiece) can represent unfamiliar text using byte-fallback or character-level tokens without classical OOV.
*   *Decisions:* Focus on provenance boundaries, tokenizer fragmentation (does the domain use too many tokens?), and evaluating whether to extend the vocabulary vs. training a new tokenizer.

### 16.2 Benchmark Contamination Policy
*   *Misconception:* "You MUST strip all benchmark datasets." (Perfect deletion is a fictional guarantee at web-scale).
*   *Reality:* Contamination is a risk management process. 
*   *Protocol:* Establish a provenance policy. Use exact matching and near-duplicate (fuzzy) matching. Track source documents, paraphrases, answer keys, and derived datasets. Disclose residual uncertainty.

---

## 17. INFERENCE ECOSYSTEM
*   *Ecosystem Suitability:* 
    *   ONNX/TensorRT: Excellent for static graphs (CNNs, encoders) where dimensions are known.
    *   vLLM/TGI: Designed for dynamic autoregressive generation, handling PagedAttention and continuous batching.
    *   Do not treat these as impossible/possible absolutes, but as ecosystem suitability decisions based on latency, throughput, memory, and batch concurrency.


## 18. PRODUCTION MONITORING & MODEL CARDS

### 18.1 Production Monitoring Lifecycle
Training $\rightarrow$ Deployment $\rightarrow$ Monitoring $\rightarrow$ Incident Detection $\rightarrow$ Diagnosis $\rightarrow$ Rollback/Retraining.
*   **Drift Detection:** Monitor schema drift (missing columns), feature drift (distribution changes), class-prior drift, and missingness drift.
*   **Performance Decay:** Track calibration drift, threshold drift, and subgroup degradation. Handle delayed labels systematically.
*   **System Metrics:** Monitor OOD (Out of Distribution) rates, latency, throughput, error rates, and resource usage.
*   **Retraining Policy:** Define explicit triggers for retraining and rollback. Maintain model/data lineage for every deployed artifact.

### 18.2 Model/Data Cards & Versioning
Operational guidance requires explicit documentation:
*   Dataset version (hash/timestamp).
*   Code version and configuration.
*   Hardware/Software environment.
*   Random seeds used.
*   Evaluation datasets and known failure modes/limitations.

---

## 19. EXPERIMENTATION & RESEARCH ENGINEERING

### 19.1 Experimental Design
*   *Misconception:* "5 seeds are required" or "ablations prove causality."
*   *Protocol:* Start with an explicit hypothesis. Use matched seeds and paired comparisons. 
*   *Uncertainty:* Report the uncertainty of differences. Establish a practical effect threshold (resource cost vs. compute budget).
*   *Ablations:* Ablations provide evidence about component effects under the specific experimental design; they do not automatically prove causality (due to interaction effects). Consider factorial designs where relevant.

---

## 20. REUSABLE CODE TEMPLATES

### 20.1 Leakage-Safe Sklearn Pipeline (IID binary-classification example)
*Note: Pipeline becomes fold-safe only when supplied to a CV procedure that fits the entire pipeline separately per fold. StratifiedKFold is appropriate only for the demonstrated IID classification case. For groups, time, sites, or repeated entities, refer to the compound-split framework.*

`python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Data definitions (Placeholders)
num_cols = ['age', 'income']
cat_cols = ['city', 'status']

num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median'))])

# Categorical: Impute 'missing', OHE (Prefer sparse output where appropriate, unless densification is explicitly justified)
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols), 
    ('cat', cat_pipe, cat_cols)
])

model = Pipeline([
    ('preprocessor', preprocessor), 
    ('clf', XGBClassifier(random_state=42))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
```

### 20.2 PyTorch Training Loop (Partial Template)
*Note: This is a partial template demonstrating coherent device-aware precision, accumulation remainder handling, and scheduler cadence.*

`python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Explicitly expose precision choice as configuration
precision_config = "bf16" # or "fp16", "fp32"

if precision_config == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    amp_dtype = torch.bfloat16
    use_scaler = False
elif precision_config == "fp16" and torch.cuda.is_available():
    amp_dtype = torch.float16
    use_scaler = True
else:
    amp_dtype = torch.float32
    use_scaler = False

scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
accumulation_steps = 4

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    for i, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
            logits = model(X)
            loss = criterion(logits, y) 
            
            # Handle final partial window for accumulation
            is_final_step = (i + 1) == len(dataloader)
            current_accum_steps = len(dataloader) % accumulation_steps if is_final_step and len(dataloader) % accumulation_steps != 0 else accumulation_steps
            loss = loss / current_accum_steps
            
        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0 or is_final_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            # Scheduler cadence: typically stepped per optimizer update, not per epoch
            scheduler.step() 
```

### 20.3 Conceptual PEFT / LoRA Workflow
*Note: Explicitly conceptual. Target modules, 4-bit config, and evaluation logic are heavily model/version dependent.*

```python
# 1. Inspect architecture & formulate LoRA vs QLoRA plan
# 2. Load model with device mapping & precision
# model = AutoModelForCausalLM.from_pretrained(..., device_map="auto")
# (If QLoRA: use BitsAndBytesConfig for 4-bit loading, then prepare_model_for_kbit_training)

# 3. Configure architecture-specific target modules
# config = LoraConfig(r=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")

# 4. Apply adapters & assert trainable parameter count
# peft_model = get_peft_model(model, config)
# peft_model.print_trainable_parameters() 

# 5. Train -> Save Adapter -> Merge (if needed) -> Evaluate
```


---

## 21. THE MASTER DECISION PIPELINE (ROUTING MAP)

This routing system maps the dependencies of a full ML/DL project.

### STAGE 0: Deployment Objective
*   *What is the task?* (Classification, regression, retrieval, generation, multimodal).
*   *What are the operational costs?* (False positives vs negatives).

### STAGE 1: Inference Feature Availability
*   *What information is available at inference time?* 
*   *Action:* Remove unavailable features.

### STAGE 2: Deployment Unit
*   *Are predictions for:* A) Existing entities with history? B) Previously unseen entities? C) Both?

### STAGE 3: Compound Split Constraints
*   Check independently and compose all applicable constraints: Entity/Group, Time/Horizon, Site/Domain, Duplicate family, Repeated measurements.

### STAGE 4: Leakage Controls
*   Identify target proxies, post-outcome features, and future information. Ensure EDA/preprocessing fits only on training folds.

### STAGE 5: Simplest Credible Baseline
*   Majority/mean baseline, linear/logistic, or tree/boosting depending on tabular vs unstructured data.

### STAGE 6: Metric Objective
*   Is the evaluation objective ranking, probability quality, operating-point decision, cost-sensitive utility, retrieval, or coverage? Select metrics accordingly (e.g., ROC-AUC for ranking, Brier for probability).

### STAGE 7: Probability, Threshold, and Uncertainty Needs
*   Determine if the deployment requires calibrated probabilities, a tuned operating threshold, or formal uncertainty estimation (aleatoric/epistemic/conformal).

### STAGE 8: Candidate Model Families
*   Use data, task, and compute constraints. (e.g., Sequence data might route to an RNN for strict latency, or a Transformer for context reasoning). Consider OCR/layout fallback before committing to a massive VLM.

### STAGE 9: Resource Constraints (Budget)
*   Evaluate parameter memory, gradient memory, optimizer-state memory, activation memory, KV-cache, microbatch size, global batch, and latency.

### STAGE 10: Training & Distributed Strategy
*   Choose based on memory budget: DDP, FSDP, TP, PP, Offload, or Quantization. Compose as needed.

### STAGE 11: Training Diagnosis
*   If performance is poor: Check inputs -> first NaN op -> loss domain -> 1-batch FP32 -> activations -> then LR/clipping. 

### STAGE 12: Experiment Design
*   Evaluate based on matched seeds, paired comparisons, block bootstrap, and practical effect sizes.

### STAGE 13: Calibration & Uncertainty Estimation
*   Fit calibrators on a disjoint split. Evaluate empirical coverage and interval width for conformal prediction.

### STAGE 14: Robustness & Shift
*   Evaluate subgroup performance, covariate shift, label shift, concept shift, temporal decay, OOD, and rare cases. 

### STAGE 15: Inference Benchmarking
*   Benchmark: Latency, throughput, memory, batch size, concurrency, tail latency. Choose serving stack based on benchmarks.

### STAGE 16: Deploy & Monitor
*   Monitor: Inputs, drift, calibration, performance, latency, failures.

### STAGE 17: Retraining Policy
*   Establish rollback triggers, recalibration policies, and data lineage updates.


---

## PHASE 5 REPAIR TRACEABILITY

| Terra Finding | Section Repaired | What Changed |
| :--- | :--- | :--- |
| **P0-1** (MCAR/MAR claims) | 2. Missing Values | Replaced deterministic rules with assumptions; added operational checking and CV-based comparisons. |
| **P0-2** (Indicator $\neq$ MNAR) | 2. Missing Values | Explained alternative causes (structural, site, etc.) for missing indicator utility. |
| **P0-3** (Native tree $\neq$ MNAR fix) | 2. Missing Values | Clarified that native routing is a strong implementation but doesn't fix selection bias. |
| **P0-5** (Label disagreement) | 8. Label Quality | Added decision flow for expert adjudication vs soft labels depending on genuine ambiguity. |
| **P0-6** (Accuracy/Thresholds) | 8. Label Quality | Replaced "NEVER use Accuracy." Explained thresholding alters operating point, not ranking. |
| **P0-7** (Quantiles/Uncertainty) | 9. Regression | Removed "worst-case guarantee". Distinguished aleatoric/epistemic uncertainty and conformal prediction. |
| **P0-9** (CI overlap) | 10. Evaluation | Removed "overlap = no difference." Added paired differences and matched seed comparisons. |
| **P0-10** (Clustering evaluation) | 10. Evaluation | Added stability, internal metrics, and domain coherence. Removed "downstream only" claim. |
| **P0-12** (BatchNorm rules) | 11. Normalization | Removed <8 batch rule. Detailed effective stats, spatial dimensions, and freezing dynamics. |
| **P0-13** (DDP/FSDP topology) | 13. Distributed | Replaced GPU count rules with a memory-component budgeting framework. |
| **P0-14** (Exact resume) | 13. Distributed | Reclassified as "Partial Resumable Checkpoint"; listed exact deterministic resume requirements. |
| **P0-15** (KV Cache complexity) | 14. Transformers | Clarified attention work still scales with context length; memory scales linearly. |
| **P0-16** (Pretraining rationale) | 14. Foundation Models | Replaced tokenizer rationale with a full evaluation of domain gap, budget, and forgetting. |
| **P0-17** (QLoRA hardware promises) | 14. Foundation Models | Replaced "24GB" promises with a memory budgeting requirement (gradients, precision, etc.). |
| **P0-18** (CLIP negatives) | 15. Multimodal | Removed "relies entirely on hard negatives/collapse". Added memory banks and tradeoffs. |
| **P0-20** (Benchmark contamination) | 16. Data Curation | Replaced "MUST strip" with a contamination risk management and provenance policy. |
| **P0-22** (PyTorch AMP safety) | 20. Code Templates | Rebuilt PyTorch loop: device-aware, BF16 coherence, and accumulation remainder handling. |
| **P0-26** (Final decision tree) | 21. Master Pipeline | Completely rebuilt as a 17-stage routing map composing constraints rather than a flat lookup. |
| **P1-4** (Outlier taxonomy) | 1. Preprocessing | Added taxonomy (valid, corrupted, leverage) and fit-time constraints. |
| **P1-8** (Calibration limits) | 10. Evaluation | Made calibration conditional on business need. Emphasized disjoint sets. |
| **P1-11** (Training diagnosis) | 11. Diagnosis | Changed from "bugs" to Hypothesis $\rightarrow$ Test $\rightarrow$ Action flow. |
| **P1-19** (VLM architectures) | 15. Multimodal | Added unified transformers, dynamic tiling, and a detailed evaluation matrix. |
| **P1-21** (GPU Utilization) | 13. Distributed | Replaced absolute percentages with a profiling/diagnostic chain. |
| **P1-23** (Gradient accumulation) | 20. Code Templates | Addressed final window remainder handling and correct scheduler cadence. |
| **P1-24** (Sklearn CV) | 20. Code Templates | Showed an actual `cross_val_score` split proving how pipelines prevent leakage. |
| **P1-25** (LoRA template) | 20. Code Templates | Labeled strictly conceptual; added architecture inspection gates. |
| **P1-27** (RAG/DPO limits) | 14. Foundation Models | Detailed RAG retrieval bottlenecks and DPO safety limitations. |
| **P1-28** (Tokenizer OOV) | 16. Data Curation | Corrected OOV claims; focused on provenance and fragmentation. |
| **P1-29** (Compound splits) | 3. Compound Splits | Added explicit composition of group, time, and site constraints. |
| **P1-30** (Experimental design) | 19. Experimentation | Removed "5 seeds required" and "ablations prove causality." Emphasized matched seeds. |
| **P2-31, 32, 33, 34, 35** | Various Sections | Qualified log-targets, activation memory, SOTA claims, pre/post norm, and ONNX ecosystem suitability. |
| **PART D** (Missing high-value) | Various Sections | Added categorical representations, classical model HPO, Data Augmentation, Shift taxonomy, Production Monitoring, and Model Cards. |


