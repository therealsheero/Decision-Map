import re

content = open(r'C:\Users\acer\.gemini\antigravity\brain\00a254c6-3eee-43e1-8d76-b154f1269a46\scratch\chunk4.md', 'r', encoding='utf-8').read()

# Fix 1: NaN Diagnosis Order
nan_new = '''### 11.1 Staged Diagnosis Flow
If a model fails to train or generalize:
1.  **Data & Labels:** Are inputs scaled? Are labels corrupted?
2.  **Output Contract:** Does the loss function match the final layer activation?
3.  **Numerical Stability (NaNs detected):** Check finite inputs/targets $\\rightarrow$ check first non-finite tensor/op $\\rightarrow$ check loss-domain constraints $\\rightarrow$ run one-batch FP32 diagnostic $\\rightarrow$ inspect activations/gradients/normalization $\\rightarrow$ then test LR / precision / clipping hypotheses. *Make clear that gradient clipping can hide a root cause.*
4.  **Small-Batch Overfit Test:** Can the model memorize a single batch (Loss $\\rightarrow 0$)? If not, the architecture, capacity, or optimization routing is fundamentally broken.'''
content = re.sub(r'### 11.1 Staged Diagnosis Flow.*?Are ReLUs dead\?', nan_new, content, flags=re.DOTALL)

# Fix 2: Regularization language
content = content.replace('to prevent collapsing representation scales.', 'as a commonly tested optimizer recipe whose usefulness depends on architecture and optimizer.')
content = content.replace('Highly recommended for robust generalization.', 'Describe as optional techniques to evaluate, with costs and interaction considerations.')
content = content.replace('Requires synchronization (SyncBN) across GPUs in distributed setups.', 'If per-rank statistics are adequate, keep BN. If statistics are too noisy and synchronization is acceptable, compare SyncBN. For small batches or unsuitable BN stats, compare GroupNorm/LayerNorm or frozen BN depending on architecture and transfer setting.')

# Fix 3: Augmentation
content = content.replace('to prove they help validation performance.', 'to provide evidence they help validation performance under the specific experimental design.')

# Fix 4: Distributed topology
dist_new = '''### 13.2 Memory and Topology Decision Framework
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
*Actual fit depends on precision, sequence length, microbatch, architecture, optimizer, checkpointing, sharding, offload, and quantization.*'''
content = re.sub(r'### 13.2 Memory and Topology Decision Framework.*?Pipeline Parallelism \(PP\):.*?[^\n]*\n', dist_new + '\n', content, flags=re.DOTALL)

open(r'C:\Users\acer\.gemini\antigravity\brain\00a254c6-3eee-43e1-8d76-b154f1269a46\scratch\chunk4.md', 'w', encoding='utf-8').write(content)
print('Chunk 4 patched')
