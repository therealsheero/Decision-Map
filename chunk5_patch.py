import re

content = open(r'C:\Users\acer\.gemini\antigravity\brain\00a254c6-3eee-43e1-8d76-b154f1269a46\scratch\chunk5.md', 'r', encoding='utf-8').read()

# Fix 1: CLIP
clip_new = '''### 15.1 CLIP and Contrastive Dynamics
*   *Misconception:* "CLIP relies entirely on hard negatives and small batches cause collapse."
*   *Reality:* Standard CLIP uses *in-batch* negatives. Large batches increase negative diversity, but they are a representation/retrieval-quality tradeoff, not a universal requirement. Consider in-batch negatives, false negatives, multi-positive situations, hard-negative mining, cross-device negative gathering (weighing communication/memory costs), and the overall quality of the negative pool.'''
content = re.sub(r'### 15.1 CLIP and Contrastive Dynamics.*?alternative objective designs\.', clip_new, content, flags=re.DOTALL)

# Fix 2: VLM
vlm_new = '''### 15.2 VLM (Vision-Language Model) Taxonomy
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
*   **VLM Evaluation Matrix:** Include measured metrics such as OCR accuracy, field extraction accuracy, grounding accuracy (where relevant), latency (p50/p95/p99), throughput, and memory. Dynamic tiling improves OCR but consumes massive context/compute and introduces stitching tradeoffs.'''
content = re.sub(r'### 15.2 VLM \(Vision-Language Model\) Taxonomy.*?stitching tradeoffs\.', vlm_new, content, flags=re.DOTALL)

open(r'C:\Users\acer\.gemini\antigravity\brain\00a254c6-3eee-43e1-8d76-b154f1269a46\scratch\chunk5.md', 'w', encoding='utf-8').write(content)
print('Chunk 5 patched')
