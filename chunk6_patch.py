import re

content = open(r'C:\Users\acer\.gemini\antigravity\brain\00a254c6-3eee-43e1-8d76-b154f1269a46\scratch\chunk6.md', 'r', encoding='utf-8').read()

# Fix 1: Sklearn
skl_new = '''### 20.1 Leakage-Safe Sklearn Pipeline (IID binary-classification example)
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
`'''
content = re.sub(r'### 20.1 Leakage-Safe Sklearn Pipeline \(Runnable Concept\).*?scoring=\'roc_auc\'\)\n`', skl_new, content, flags=re.DOTALL)

# Fix 2: PyTorch AMP
pt_new = '''### 20.2 PyTorch Training Loop (Partial Template)
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
`'''
content = re.sub(r'### 20.2 PyTorch Training Loop \(Partial Template\).*?scheduler\.step\(\) \n`', pt_new, content, flags=re.DOTALL)

open(r'C:\Users\acer\.gemini\antigravity\brain\00a254c6-3eee-43e1-8d76-b154f1269a46\scratch\chunk6.md', 'w', encoding='utf-8').write(content)
print('Chunk 6 patched')
