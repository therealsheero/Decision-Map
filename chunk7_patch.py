import re

content = open(r'C:\Users\acer\.gemini\antigravity\brain\00a254c6-3eee-43e1-8d76-b154f1269a46\scratch\chunk7.md', 'r', encoding='utf-8').read()

map_new = '''## 21. THE MASTER DECISION PIPELINE (ROUTING MAP)

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
*   Establish rollback triggers, recalibration policies, and data lineage updates.'''

content = re.sub(r'## 21\. THE MASTER DECISION PIPELINE \(ROUTING MAP\).*?data lineage updates\.', map_new, content, flags=re.DOTALL)

open(r'C:\Users\acer\.gemini\antigravity\brain\00a254c6-3eee-43e1-8d76-b154f1269a46\scratch\chunk7.md', 'w', encoding='utf-8').write(content)
print('Chunk 7 patched')
