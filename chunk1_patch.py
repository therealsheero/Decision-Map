import re

content = open(r'C:\Users\acer\.gemini\antigravity\brain\00a254c6-3eee-43e1-8d76-b154f1269a46\scratch\chunk1.md', 'r', encoding='utf-8').read()

# Fix 1: Imputation
content = content.replace(
    '*   *Simple Imputation:* Safe only under strong MCAR/MAR assumptions.',
    '*   *Simple Imputation:* May be an acceptable predictive baseline under plausible assumptions and successful validation, but it can alter variance/correlation and does not automatically preserve inferential uncertainty. Ensure it is fitted only on training folds and tested across subgroups.'
)

# Fix 2: Compound Splits (Deployment Unit)
deployment_unit = '''### 3.1 Deployment Unit (Entity Constraints)
**First Question: AT DEPLOYMENT, ARE PREDICTIONS MADE FOR: A) EXISTING ENTITIES WITH HISTORY, B) PREVIOUSLY UNSEEN ENTITIES, C) BOTH?**
*   *Existing entities:* (e.g., future prediction for a known patient). Historical patient data may legitimately occur in training while later observations are test.
*   *Unseen entities:* (e.g., generalization to new patients). Patient grouping must prevent cross-patient leakage (e.g., GroupKFold).
*   *Both:* Evaluate both regimes separately.

### 3.2 Dependency Axes
'''
content = content.replace('### 3.1 Dependency Axes\n', deployment_unit)

open(r'C:\Users\acer\.gemini\antigravity\brain\00a254c6-3eee-43e1-8d76-b154f1269a46\scratch\chunk1.md', 'w', encoding='utf-8').write(content)
print('Chunk 1 patched')
