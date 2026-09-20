import re

content = open(r'C:\Users\acer\.gemini\antigravity\brain\00a254c6-3eee-43e1-8d76-b154f1269a46\scratch\chunk3.md', 'r', encoding='utf-8').read()

# Fix 1: Imbalance Logic
imbalance_new = '''### 8.2 Imbalance & Metric Logic
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
4.  **Threshold Selection:** Changes the operating point on the PR/ROC curve. *Note:* Changing the threshold alters Precision/Recall, but it does *not* improve the model's fundamental ranking ability or calibrate its probabilities.'''
content = re.sub(r'### 8.2 Imbalance Interventions.*?probabilities\.', imbalance_new, content, flags=re.DOTALL)

# Fix 2: Quantiles/Conformal
content = content.replace('*It does NOT automatically provide guaranteed worst-case bounds or epistemic uncertainty.*', '*It does NOT automatically provide guaranteed worst-case bounds or epistemic uncertainty. (Note: Quantile crossing is a common implementation issue).*')
content = content.replace('assuming exchangeability between calibration and test data.', 'assuming exchangeability. Standard split conformal provides marginal coverage under this assumption, but time dependence, group dependence, covariate shift, or distribution shift can invalidate naive guarantees. Always require empirical coverage and interval width evaluation.')

# Fix 3: Statistical Comparison
stat_new = '''### 10.2 Statistical Comparison of Models
**Condition:** Model B scores 0.90, Model A scores 0.89. Is B better?
*   *Invalid Check:* Do not assume that overlapping Confidence Intervals (CIs) for the two models means there is no statistical difference. Two individual CIs can overlap even if the paired difference is highly significant.
*   *Valid Check:* Base the statistical procedure on the metric and design conditions:
    *   **Classification:** Paired binary predictions/errors (McNemar), ROC-AUC (DeLong/Bootstrap), Calibration/Thresholded metrics.
    *   **Regression:** MAE, MSE, RMSE, Quantile loss.
    *   **Design dependence:** Distinguish fixed test sets (paired tests), stochastic training seeds (matched-seed comparison), grouped/temporal data (block/group bootstrap), and clustered observations.
*   *Focus:* Emphasize effect size, uncertainty of the difference, practical significance, and dependence structure rather than merely seeking "statistical significance."'''
content = re.sub(r'### 10.2 Statistical Comparison of Models.*?compute costs\.', stat_new, content, flags=re.DOTALL)

open(r'C:\Users\acer\.gemini\antigravity\brain\00a254c6-3eee-43e1-8d76-b154f1269a46\scratch\chunk3.md', 'w', encoding='utf-8').write(content)
print('Chunk 3 patched')
