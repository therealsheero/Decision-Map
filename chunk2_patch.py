import re

content = open(r'C:\Users\acer\.gemini\antigravity\brain\00a254c6-3eee-43e1-8d76-b154f1269a46\scratch\chunk2.md', 'r', encoding='utf-8').read()

# Fix 1: Classical Models
content = content.replace('**Principle:** A strong classical baseline is mandatory', '**Principle:** A strong classical baseline is valuable')
content = content.replace('*   *When to test:* The default SOTA for structured/tabular data.', '*   *When to test:* Highly effective for structured/tabular data. Model choice depends on dataset size, sparsity, cardinality, nonlinear structure, missingness, latency, interpretability, monotonicity, and calibration requirements.')
content = content.replace('*   *Requires:* Feature scaling', '*   *Generally benefits from:* Feature scaling')
content = content.replace('*   *Requires:* Careful HPO', '*   *Generally benefits from:* Careful HPO')

# Fix 2: Shift
content = content.replace(
    '*   *Action:* Adjust the decision threshold or output probabilities using the deployment base rate.',
    '*   *Action:* Adjusting the threshold based on prevalence does *not* automatically correct probabilities. Use dedicated probability recalibration methods if probability quality is needed.'
)
content = content.replace(
    '*   *Action:* Importance weighting, domain adaptation.',
    '*   *Action:* Importance weighting (consider support overlap and weighting variance), domain adaptation.'
)
content = content.replace(
    '*   *Action:* Retraining on recent data is usually mandatory.',
    '*   *Action:* Retraining on recent data is a common response, but not always mandatory. Alternatives include recalibration, policy adjustment, abstention, model replacement, or feature investigation. The correct action depends on diagnosis and operational constraints.'
)

open(r'C:\Users\acer\.gemini\antigravity\brain\00a254c6-3eee-43e1-8d76-b154f1269a46\scratch\chunk2.md', 'w', encoding='utf-8').write(content)
print('Chunk 2 patched')
