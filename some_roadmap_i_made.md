
# ML Career Transition Plan --- SDE → Model-Focused ML Engineer → VLM / Research Engineer

> **Goal:** Move from my current SDE role into serious, model-building
> ML work, with **Sarvam Vision as the north-star target** while
> becoming qualified for strong ML/DL/CV/NLP/VLM roles anywhere.
>
> **Important:** This is a robust plan, not a guarantee of a job. Hiring
> depends on openings, interview performance, competition, location,
> timing, and demonstrated ability. The purpose of this plan is to make
> my skill profile strong enough that I can credibly compete.

---

# 0. My Career Target

## Primary target

**Model-focused Machine Learning Engineer / Deep Learning Engineer**

with a specialization path:

**Computer Vision → Transformers → Multimodal / Vision-Language Models**

## Secondary targets

Apply to roles such as:

1. Machine Learning Engineer
2. Deep Learning Engineer
3. Computer Vision Engineer
4. ML Engineer --- Vision
5. ML Engineer --- Multimodal / VLM
6. ML Engineer --- Foundation Models
7. Research Engineer --- ML/CV/NLP
8. ML Research Engineer
9. Applied Scientist --- ML
10. Model Optimization / Inference Engineer
11. ML Engineer --- Training / Post-training
12. AI Engineer roles **only when the role is genuinely
    model-development-heavy**

## Roles I do NOT want as my primary direction

- Pure LLM application developer
- RAG-only developer
- Agent orchestration developer
- Prompt-engineering-only roles
- Backend engineer with an "AI" label
- Roles where the main work is integrating third-party model APIs
- Generic software roles disguised as AI roles

I can still use these technologies when needed, but they are not my
specialization.

---

# 1. Why Sarvam Is the North Star

Sarvam's current Vision ML Engineer role is unusually aligned with my
goal: VLM training/fine-tuning, multimodal data pipelines, architecture
experimentation, evaluation, inference optimization, GPU clusters,
PyTorch/model internals, and production ML.

The current Researcher, Vision role goes further into VLM architecture,
pretraining objectives, scaling, training methods, data mixtures,
evaluation, failure analysis, robustness, and rigorous experimental
design.

Sarvam's current careers page also shows adjacent opportunities in
training infrastructure, foundational-model data, inference, kernels, ML
Ops, and other AI infrastructure roles.

Official references:

- Sarvam Careers: https://www.sarvam.ai/careers
- Machine Learning Engineer, Vision:
  https://www.sarvam.ai/careers/jobs/282019f4-f86b-4fe0-a643-bbac0cd8a751
- Researcher, Vision:
  https://www.sarvam.ai/careers/jobs/255816b4-c44d-481c-a093-73ae114d1952
- ML Engineer, Training Infra:
  https://www.sarvam.ai/careers/jobs/e9fb1269-fc4b-4879-aa35-0b75a8c83611
- ML Engineer, Foundational Models --- Data:
  https://www.sarvam.ai/careers/jobs/2d72a541-c805-4bd9-81cd-ac2db98611fa

## Sarvam is the target, NOT the only target

I will use Sarvam's JD as a benchmark for skill depth.

The objective is not:

> "Learn exactly enough to pass one Sarvam interview."

The objective is:

> "Become the kind of ML engineer who could credibly interview at
> Sarvam, Google/DeepMind-style teams, NVIDIA/AMD-style model
> optimization teams, strong AI startups, research engineering teams,
> and serious ML organizations."

---

# 2. My Starting Point

## Existing advantages

I already have:

- BTech in CSE with AIML specialization
- Broad academic exposure to statistical ML
- Deep learning coursework
- CNNs
- RNN/LSTM/GRU
- Transformers
- BERT
- ViT
- Diffusion
- NLP
- Computer Vision
- Statistical learning theory
- Optimization
- Bayesian methods
- EM/GMM
- Kernel methods
- Sparsity
- Generalization theory
- Experience teaching ML topics to friends/juniors
- \~8 months of professional SDE experience
- Production software engineering exposure

## Current weaknesses

My main problem is NOT "I never learned ML."

My problem is:

- ML knowledge is rusty after 8 months of SDE
- PyTorch/model-building fluency needs rebuilding
- I need serious hands-on training experience
- I need GPU experience
- I need VLM/multimodal depth
- I need research-style experimentation
- I need model optimization knowledge
- I need distributed training exposure
- I need a public portfolio proving these skills
- I need to convert academic knowledge into interview-ready depth

## Core strategy

**Do NOT restart BTech from zero.**

Use my existing notes as a reference and reactivate knowledge through
implementation.

My learning loop:

> **Recall → Review → Implement → Train → Break → Debug → Experiment →
> Explain → Document**

---

# 3. The Skill Pyramid

I will build skills in this order.

```text
                         Research / Novel Experiments
                                  ▲
                                  │
                     VLM / Multimodal / Foundation Models
                                  ▲
                                  │
                       Transformers + Modern DL
                                  ▲
                                  │
                     Computer Vision + NLP
                                  ▲
                                  │
                     PyTorch + Training Systems
                                  ▲
                                  │
                 ML Fundamentals + Optimization + Math
                                  ▲
                                  │
                   Python + Software Engineering
```

The foundation must be strong, but I will spend most of my time on the
upper layers once the fundamentals are reactivated.

---

# 4. The 12-Month Master Plan

## Phase 0 --- Setup and Baseline

**Duration:** 3--5 days

### Objectives

- Set up development environment
- Create ML portfolio repository
- Establish baseline of current knowledge
- Start tracking progress

### Setup

- Python
- PyTorch
- NumPy
- pandas
- scikit-learn
- matplotlib
- Jupyter
- Git/GitHub
- Docker
- Weights & Biases or MLflow
- Linux basics
- CUDA-capable environment when available

### Repository structure

```text
ml-transition/
│
├── 00_foundations/
├── 01_classical_ml/
├── 02_neural_networks/
├── 03_computer_vision/
├── 04_transformers/
├── 05_multimodal/
├── 06_vlm/
├── 07_training/
├── 08_optimization/
├── 09_research/
├── 10_systems/
├── projects/
├── papers/
├── experiments/
└── README.md
```

### Baseline test

Without notes, explain:

- Gradient descent
- Backpropagation
- Bias vs variance
- Cross-entropy
- Batch normalization
- CNN
- Attention
- Transformer
- ViT
- Fine-tuning
- LoRA
- Mixed precision
- Quantization

Then mark each:

- Green = can explain and implement
- Yellow = understand but rusty
- Red = cannot currently explain

Do not spend days making the baseline perfect. It only tells me where to
start.

---

# PHASE 1 --- Reactivate ML + Deep Learning

**Duration:** Weeks 1--4

## Goal

Recover the knowledge I already had and make it executable again.

## Week 1 --- Classical ML + Optimization

Review only what is necessary:

### ML

- Linear regression
- Logistic regression
- Loss functions
- Maximum likelihood
- Regularization
- Bias/variance
- Underfitting/overfitting
- Decision trees
- Random forests
- Gradient boosting
- SVM
- PCA
- Clustering
- Evaluation metrics

### Optimization

- Gradient descent
- SGD
- Momentum
- Adam
- AdamW
- Learning-rate schedules
- Convex vs non-convex optimization
- L1/L2 regularization

### Implement

From scratch with NumPy:

1. Linear regression
2. Logistic regression
3. Gradient descent
4. One simple optimizer

### Exit criterion

I can explain:

> What is being optimized, why the gradient exists, how parameters
> change, why regularization helps, and why a model generalizes or
> overfits.

---

## Week 2 --- Neural Networks + PyTorch

### Review

- Forward propagation
- Backpropagation
- Chain rule
- Activation functions
- Initialization
- Loss functions
- Optimizers
- Dropout
- BatchNorm
- LayerNorm
- Residual connections

### Implement

1. MLP from NumPy
2. Same MLP in PyTorch
3. Custom training loop
4. Validation loop
5. Checkpointing
6. Experiment tracking

### Must understand in PyTorch

```python
nn.Module
forward()
Dataset
DataLoader
autograd
optimizer
scheduler
loss
state_dict
checkpointing
AMP / autocast
```

### Exit criterion

Given a PyTorch training script, I can read it, modify it, debug it, and
explain tensor shapes.

---

## Week 3 --- CNNs + ResNet

### Review

- Convolution
- Stride
- Padding
- Receptive field
- Pooling
- Feature maps
- BatchNorm
- Residual blocks
- ResNet

### Build

- CNN from scratch
- ResNet-style model
- Train on CIFAR-10 or another manageable dataset

### Experiments

At least five:

1. Learning rate
2. Batch size
3. Optimizer
4. Data augmentation
5. Regularization

### Record

For every experiment:

- Hypothesis
- Change
- Baseline
- Metric
- Result
- Interpretation

### Exit criterion

I can diagnose obvious training problems instead of blindly changing
hyperparameters.

---

## Week 4 --- Transformers + ViT

### Review

- Tokenization
- Embeddings
- Positional encoding
- Q/K/V
- Scaled dot-product attention
- Multi-head attention
- Feed-forward network
- Residual connections
- LayerNorm
- Encoder/decoder
- Masking
- ViT patch embeddings

### Implement

1. Self-attention
2. Multi-head attention
3. Transformer block
4. Small Transformer
5. Small ViT

### Exit criterion

I can draw a Transformer and explain every major tensor transformation.

---

# PHASE 2 --- Become a Strong PyTorch Model Engineer

**Duration:** Weeks 5--8

This phase is critical.

The goal is to move from:

> "I know PyTorch."

to:

> "I can modify model internals and run serious experiments."

## Learn deeply

### PyTorch

- Modules
- Parameter registration
- Autograd
- Custom losses
- Custom layers
- Hooks
- Gradient inspection
- Gradient clipping
- Mixed precision
- Gradient accumulation
- Checkpointing
- Resume training
- Reproducibility
- Profiling
- Memory management

### Data

- Dataset design
- DataLoader
- multiprocessing
- augmentation
- shuffling
- caching
- serialization
- train/validation/test splits
- leakage
- class imbalance

### Debugging

Learn to identify:

- exploding gradients
- vanishing gradients
- NaNs
- bad labels
- data leakage
- overfitting
- underfitting
- wrong normalization
- learning-rate problems
- memory problems
- slow data loaders
- GPU under-utilization

## Required mini-project

Create:

**"Training a Vision Model From Scratch: Engineering and Experimentation
Report"**

It must include:

- architecture
- dataset
- training loop
- baseline
- 8--10 experiments
- failure cases
- plots
- final model
- lessons learned

---

# PHASE 3 --- Deep Computer Vision

**Duration:** Months 3--4

## Why Vision first?

Vision is the initial specialization because it naturally leads to:

> CNN → ViT → multimodal → VLM

and is closely aligned with Sarvam Vision.

## Topics

### Core vision

- Image representations
- Augmentation
- Classification
- Detection
- Segmentation
- Object tracking
- Feature extraction
- Transfer learning

### Architectures

- AlexNet --- historical context
- VGG --- historical context
- ResNet
- EfficientNet --- conceptually
- U-Net
- DETR
- Vision Transformer
- Swin Transformer --- conceptual depth

### Detection

Understand:

- bounding boxes
- IoU
- NMS
- mAP
- anchor-based vs anchor-free
- one-stage vs two-stage detection
- DETR-style detection

### Segmentation

Understand:

- semantic segmentation
- instance segmentation
- Dice
- IoU
- U-Net
- transformer-based segmentation concepts

## Main Project A

### Vision Research Project

Choose one:

- Object detection
- Document layout detection
- OCR/document understanding
- Image classification
- Segmentation

Recommended for Sarvam alignment:

**Document understanding / document vision**

### Project requirement

Do not simply train a model.

Ask a research question.

Examples:

- How does image resolution affect accuracy vs latency?
- How does augmentation affect small-object detection?
- How does backbone size affect accuracy/throughput?
- Which normalization strategy works best?
- What happens when training data is reduced?
- What failure modes appear on low-quality documents?

### Deliverables

- GitHub repository
- Reproducible training code
- Experiment logs
- Results table
- Failure analysis
- Technical report

---

# PHASE 4 --- Transformers + Multimodal Foundations

**Duration:** Month 5

## Goal

Understand the common architecture connecting NLP and Vision.

### Transformers

Deep understanding of:

- Attention
- Multi-head attention
- positional representations
- encoder
- decoder
- causal masking
- cross-attention
- normalization
- scaling
- training objectives

### NLP foundations

Refresh:

- tokenization
- embeddings
- Word2Vec conceptually
- contextual representations
- BERT
- autoregressive language modeling
- sequence-to-sequence
- fine-tuning

Do NOT spend months re-learning old NLP.

The purpose is to understand the language side of VLMs.

---

# PHASE 5 --- Multimodal Learning

**Duration:** Month 6

This is the bridge into the target specialization.

## Learn

### Image-text representation

- dual encoders
- contrastive learning
- CLIP-style training
- image embeddings
- text embeddings
- cosine similarity
- temperature scaling

### Fusion

Understand:

- early fusion
- late fusion
- cross-attention
- projection layers
- modality alignment
- vision encoder + language model

### VLM anatomy

```text
Image
  ↓
Vision Encoder
  ↓
Visual Features / Tokens
  ↓
Projection / Connector
  ↓
Language Model
  ↓
Text
```

Know what each component does and why it exists.

## Main Project B

### Build a Mini CLIP-Style System

Implement:

- image encoder
- text encoder
- projection
- contrastive loss
- training loop
- retrieval evaluation

Then perform experiments.

Questions:

- What happens when embedding size changes?
- What happens when batch size changes?
- What happens when temperature changes?
- How does data quality affect retrieval?
- How does hard-negative sampling affect results?

---

# PHASE 6 --- VLM Engineering

**Duration:** Months 7--8

This is the most Sarvam-specific phase.

## Learn VLM architecture deeply

Understand:

- vision encoders
- patch/token representations
- connectors/projectors
- cross-attention
- multimodal tokenization
- image resolution
- context length
- multimodal positional information
- language model integration
- visual instruction tuning

## Training

Learn:

- pretraining
- continued pretraining
- supervised fine-tuning
- instruction tuning
- LoRA
- QLoRA
- PEFT
- data formatting
- batching
- padding
- sequence packing
- checkpointing
- evaluation

## Fine-tune an open VLM

Use a model small enough to train within available compute.

Do not chase parameter count for prestige.

The goal is to demonstrate:

> I understand the full training loop.

### Project C

**VLM Fine-Tuning + Evaluation**

Build:

```text
Dataset
 ↓
Cleaning
 ↓
Filtering
 ↓
Train/Validation split
 ↓
Formatting
 ↓
Fine-tuning
 ↓
Evaluation
 ↓
Error analysis
 ↓
Improvement
```

### Required experiments

At least:

- baseline
- LoRA vs full/alternative fine-tuning where feasible
- learning-rate comparison
- data-quality experiment
- prompt/template experiment
- at least one architecture/data experiment

---

# PHASE 7 --- Data Engineering for ML

**Duration:** Month 8

This is easy to underestimate.

Sarvam's VLM role explicitly cares about multimodal data pipelines.

## Learn

- ingestion
- validation
- deduplication
- filtering
- dataset statistics
- synthetic data
- quality scoring
- train/validation leakage
- sharding
- streaming
- storage formats
- dataset versioning
- provenance
- multilingual data issues

## Build

A multimodal data pipeline:

```text
Raw Data
   ↓
Validation
   ↓
Filtering
   ↓
Deduplication
   ↓
Quality Scoring
   ↓
Train/Val Split
   ↓
Sharding
   ↓
Training Dataset
```

### Demonstrate

- before/after dataset size
- filtering rules
- duplicate rate
- quality distribution
- examples removed
- impact on model performance

---

# PHASE 8 --- Training Systems + GPU

**Duration:** Months 8--9

You do NOT need to become a CUDA kernel expert immediately.

You do need to understand how serious model training works.

## GPU fundamentals

Learn:

- CPU vs GPU
- GPU memory
- VRAM
- compute vs memory bound
- tensor operations
- CUDA basics
- utilization
- memory fragmentation
- throughput
- latency

## Training optimization

Learn:

- mixed precision
- BF16
- FP16
- gradient accumulation
- gradient checkpointing
- activation memory
- optimizer memory
- data-loader bottlenecks
- checkpointing

## Distributed training

Understand conceptually and practically:

- DDP
- FSDP
- DeepSpeed
- data parallelism
- tensor parallelism
- pipeline parallelism
- sequence parallelism
- communication overhead

### Mini-project

Take a model and measure:

- single GPU throughput
- mixed precision throughput
- batch-size effects
- memory usage
- gradient accumulation
- data-loader performance

Produce a report.

---

# PHASE 9 --- Inference Optimization

**Duration:** Month 9

Sarvam's Vision JD explicitly calls out inference optimization.

## Learn

- batching
- dynamic batching
- quantization
- FP16/BF16/INT8 concepts
- distillation
- pruning concepts
- KV cache
- latency
- throughput
- memory
- model serving
- ONNX/TensorRT concepts
- vLLM concepts for LLM-style serving

## Project

Take one model.

Measure:

```text
FP32
FP16/BF16
Quantized
```

Compare:

- model size
- latency
- throughput
- memory
- accuracy

Then explain the trade-offs.

---

# PHASE 10 --- Research Engineering

**Duration:** Months 10--11

This phase is what separates:

> "I can use ML."

from:

> "I can do ML research/engineering."

## Research workflow

For every serious project:

```text
Question
 ↓
Hypothesis
 ↓
Baseline
 ↓
Controlled change
 ↓
Experiment
 ↓
Evaluation
 ↓
Ablation
 ↓
Failure analysis
 ↓
Conclusion
```

## Read papers properly

For each paper, answer:

1. What problem?
2. Why existing methods fail?
3. What is the core idea?
4. What changed architecturally?
5. What changed in the objective?
6. What data was used?
7. What experiments prove the claim?
8. What are the ablations?
9. What are the limitations?
10. Could I reproduce it?

## Required paper sequence

Start with foundational papers relevant to your path:

1. ResNet
2. Attention Is All You Need
3. BERT
4. Vision Transformer
5. CLIP
6. A modern VLM paper
7. A VLM instruction-tuning paper
8. A paper on efficient fine-tuning
9. A paper on multimodal evaluation
10. A paper on document understanding

Do not try to read 100 papers.

Read \~10 deeply.

---

# PHASE 11 --- Capstone: Sarvam-Level VLM Project

**Duration:** Month 11--12

This is the most important portfolio project.

## Project D

### "Training and Evaluating a Vision-Language Model for Document Understanding"

Possible task:

- document question answering
- form understanding
- table understanding
- chart understanding
- OCR + structured extraction
- multilingual document understanding

## Pipeline

```text
Document Dataset
       ↓
Data Cleaning
       ↓
Deduplication
       ↓
Quality Filtering
       ↓
Synthetic / Curated Data
       ↓
Train/Val/Test
       ↓
VLM Fine-tuning
       ↓
Evaluation Harness
       ↓
Failure Analysis
       ↓
Optimization
       ↓
Inference Benchmark
       ↓
Technical Report
```

## Evaluation

Create an evaluation harness.

Track:

- accuracy
- exact match where applicable
- structured extraction accuracy
- hallucination/error categories
- latency
- throughput
- memory
- performance by document type
- performance by language
- performance by difficulty

## Failure analysis

Categorize failures:

- OCR failure
- visual reasoning failure
- language understanding failure
- hallucination
- formatting failure
- spatial reasoning failure
- data issue
- evaluation issue

This project should look like a small research-engineering project, not
a college demo.

---

# 5. The Four Portfolio Projects I Need

I do NOT need 15 projects.

I need four strong pieces.

## Project 1 --- Training Engineering

**Deep Learning Training Lab**

Shows:

- PyTorch
- training loops
- optimization
- debugging
- experiments
- profiling

## Project 2 --- Vision

**Vision Research Project**

Shows:

- CNN/ViT
- dataset
- training
- evaluation
- ablation
- failure analysis

## Project 3 --- Multimodal

**Mini CLIP / Image-Text Retrieval**

Shows:

- multimodal learning
- dual encoders
- contrastive learning
- embeddings
- evaluation

## Project 4 --- VLM

**Document VLM Fine-Tuning + Evaluation**

Shows:

- VLM
- fine-tuning
- data pipeline
- evaluation
- GPU training
- model optimization
- production thinking
- research methodology

Project 4 is the flagship.

---

# 6. What "Good Enough" Means Before Applying

I should NOT wait until I know everything.

I can start applying when I satisfy approximately:

## Must-have

- [ ] Strong Python
- [ ] Strong PyTorch
- [ ] Can implement/debug training loops
- [ ] Can explain backpropagation
- [ ] Can explain Transformers
- [ ] Can explain CNN/ViT
- [ ] Can train models myself
- [ ] Can fine-tune a model
- [ ] At least one serious vision project
- [ ] At least one multimodal/VLM project
- [ ] GitHub with clean code
- [ ] Can discuss experiments quantitatively
- [ ] Comfortable with Linux/Git
- [ ] Strong software engineering fundamentals

## Strong differentiators

- [ ] GPU training
- [ ] mixed precision
- [ ] profiling
- [ ] LoRA/QLoRA
- [ ] quantization
- [ ] distributed training
- [ ] data curation
- [ ] evaluation harness
- [ ] paper reproduction
- [ ] technical report
- [ ] open-source contribution

## Research track differentiators

- [ ] Independent research project
- [ ] Strong experimental design
- [ ] Paper reproduction
- [ ] Novel ablation
- [ ] Technical report
- [ ] Open-source contribution
- [ ] Publication, if it happens naturally

---

# 7. Sarvam JD → My Preparation Matrix

  Sarvam requirement             How I will prove it

---

  Python                         Professional SDE + ML repos
  PyTorch                        All major projects
  Model internals                Transformer/ViT/VLM implementation
  Training/fine-tuning           VLM project
  Debug broken runs              Experiment logs + failure analysis
  Multimodal data pipelines      VLM data project
  Filtering/deduplication        Dataset pipeline
  Synthetic data                 VLM project
  Evaluation harness             Capstone
  Regression tracking            W&B/MLflow + automated eval
  Architecture experimentation   Ablation studies
  Quantization                   Inference project
  Batching                       Inference benchmark
  Serving                        Model serving mini-project
  Transformers                   Transformer implementation
  Vision                         Vision project
  VLM                            VLM capstone
  FSDP/DeepSpeed                 Training systems phase
  RLHF/DPO                       Learn after core VLM competence
  Open source                    Target contribution
  Document AI                    Capstone
  Production systems             SDE experience + deployed ML project
  Secure/reliable coding         SDE experience
  Ambiguity                      Research-style projects

---

# 8. Skills That Transfer to Almost Any ML Role

These are the skills I should prioritize because they survive changes in
tooling and job titles.

## Tier 1 --- Non-negotiable

### Python

- clean code
- OOP
- typing
- testing
- debugging
- profiling
- multiprocessing/concurrency basics

### ML fundamentals

- supervised learning
- unsupervised learning
- optimization
- generalization
- evaluation
- statistics

### PyTorch

- model internals
- autograd
- training loops
- datasets
- optimization
- checkpointing
- mixed precision
- profiling

### Deep learning

- CNNs
- Transformers
- attention
- normalization
- regularization
- optimization

---

# 9. Tier 2 --- High-value specialization

- Computer Vision
- NLP fundamentals
- Transformers
- ViTs
- Multimodal learning
- VLMs
- fine-tuning
- PEFT
- LoRA/QLoRA
- evaluation

---

# 10. Tier 3 --- ML Systems

- GPU fundamentals
- CUDA concepts
- distributed training
- DDP
- FSDP
- DeepSpeed
- data pipelines
- inference optimization
- quantization
- batching
- serving
- profiling

---

# 11. Tier 4 --- Research

- paper reading
- reproduction
- hypothesis formation
- ablations
- statistical significance where appropriate
- failure analysis
- experimental design
- technical writing
- open-source contribution

---

# 12. What NOT to Prioritize Early

Do not allow these to derail the roadmap:

- collecting certificates
- learning every AI framework
- building dozens of CRUD + chatbot projects
- learning LangChain deeply
- learning every agent framework
- memorizing every new LLM release
- trying to train a 7B/70B model just for the parameter count
- reading papers without implementing anything
- endlessly watching courses
- learning CUDA kernels before understanding PyTorch/GPU profiling
- learning RLHF/DPO before becoming strong at ordinary supervised
  fine-tuning

---

# 13. The Learning Rule

For every topic:

## Level 1 --- Explain

Can I explain it without notes?

## Level 2 --- Implement

Can I implement a small version?

## Level 3 --- Debug

Can I diagnose when it breaks?

## Level 4 --- Experiment

Can I change one variable and explain the result?

## Level 5 --- Optimize

Can I improve quality, speed, memory, or cost?

## Level 6 --- Research

Can I form a hypothesis that is not directly answered by the tutorial?

I should not claim mastery after Level 1.

---

# 14. Weekly Schedule Alongside a Full-Time SDE Job

Assume approximately **15--18 focused hours/week**.

## Monday

**1.5--2 hours**

Theory refresh + recall

## Tuesday

**2 hours**

Implementation

## Wednesday

**1.5--2 hours**

Implementation/debugging

## Thursday

**2 hours**

Experiment

## Friday

**1--1.5 hours**

Paper / technical reading

## Saturday

**5--6 hours**

Deep project work

## Sunday

**3--4 hours**

Project + documentation + weekly review

Total:

**\~16--19 hours/week**

If work becomes intense:

Minimum viable schedule:

**10 hours/week**

Do not stop completely.

Consistency beats heroic weekends.

---

# 15. Daily Study Rule

Each session should have a concrete output.

Bad:

> "Study Transformers."

Good:

> "Implement multi-head attention and verify tensor shapes."

Bad:

> "Learn VLMs."

Good:

> "Draw the architecture of a VLM and explain how image tokens reach the
> language model."

Bad:

> "Learn quantization."

Good:

> "Quantize one model and compare memory, latency, and accuracy."

---

# 16. Notes Strategy

I already have BTech notes.

Use them.

Do not rewrite everything.

For each topic create a one-page "active recall sheet":

```text
Topic:
Why does it exist?
Core intuition:
Mathematical formulation:
Important equations:
Implementation:
Common bugs:
Important hyperparameters:
Failure modes:
Interview questions:
Research questions:
```

Example:

```text
Topic: Attention

Why:
Intuition:
Q/K/V:
Formula:
Scaling:
Multi-head:
Masking:
PyTorch implementation:
Tensor shapes:
Failure modes:
Experiments:
Interview questions:
Papers:
```

This converts old notes into working knowledge.

---

# 17. Paper Reading Strategy

For each paper:

## Pass 1 --- 10 minutes

- Abstract
- Introduction
- Figures
- Conclusion

## Pass 2 --- 30--60 minutes

- Method
- Architecture
- Objective
- Experiments

## Pass 3 --- implementation

Recreate the core idea.

## Pass 4 --- experiment

Change one thing.

The goal is not to say:

> "I read 50 papers."

The goal is:

> "I deeply understood and implemented 8--10 papers."

---

# 18. Interview Preparation

Start serious interview preparation around Month 4, not at the end.

## ML theory

Prepare:

- bias/variance
- regularization
- loss functions
- optimization
- calibration
- class imbalance
- metrics
- overfitting
- generalization
- data leakage

## Deep learning

Prepare:

- backpropagation
- initialization
- normalization
- dropout
- residuals
- optimizers
- learning-rate schedules
- attention
- Transformer
- CNN
- ViT

## PyTorch

Be able to code:

- Dataset
- DataLoader
- training loop
- validation loop
- custom loss
- attention
- small Transformer
- fine-tuning loop

## Vision

Know:

- CNN
- ResNet
- ViT
- IoU
- mAP
- NMS
- segmentation metrics
- augmentations

## VLM

Know:

- vision encoder
- connector
- LLM
- cross-attention
- contrastive learning
- VLM pretraining
- SFT
- LoRA
- evaluation
- hallucination
- multimodal failure modes

## ML systems

Know:

- GPU memory
- batching
- throughput
- latency
- quantization
- mixed precision
- distributed training concepts

---

# 19. DSA and SDE Preparation

Do NOT abandon software engineering.

I already have a professional SDE background.

Maintain:

- DSA
- Python
- Linux
- Git
- APIs
- databases
- concurrency basics
- system design
- testing
- clean code

Suggested maintenance:

**2--3 hours/week**

This protects my existing career while I transition.

---

# 20. Application Strategy

Do not wait until Month 12.

## Month 3+

Start monitoring jobs.

## Month 4+

Start low-pressure applications to:

- junior ML Engineer
- ML Engineer
- Computer Vision Engineer
- Applied ML Engineer

## Month 6+

Increase applications.

## Month 8+

Target:

- VLM
- multimodal
- foundation-model
- model optimization
- research engineering

## Month 10+

Aggressively target Sarvam and comparable companies if the skill matrix
is strong.

---

# 21. Application Buckets

I should maintain three buckets.

## Bucket A --- Dream

Examples:

- Sarvam
- frontier-model teams
- elite AI startups
- research-heavy ML teams

Apply even if I am slightly underqualified.

## Bucket B --- Strong Match

Companies where I satisfy \~70--85% of the JD.

This should be my main application pool.

## Bucket C --- Bridge

Roles that move me closer to model work:

- Computer Vision Engineer
- ML Engineer
- ML Platform Engineer
- Model Evaluation Engineer
- ML Infrastructure Engineer
- Applied ML Engineer

A bridge role is NOT failure.

It is strategic positioning.

---

# 22. Resume Strategy

My resume should NOT say:

> "Aspiring AI/ML Engineer."

Instead:

> **Software Engineer with CSE/AIML background specializing in deep
> learning, computer vision, multimodal models and PyTorch-based model
> development.**

Once the portfolio is strong enough, emphasize:

- training
- fine-tuning
- experiments
- model architecture
- evaluation
- optimization
- GPU
- VLM
- production engineering

My SDE experience remains prominent.

---

# 23. GitHub Standard

A recruiter/interviewer should be able to open my GitHub and immediately
see:

```text
1. Training Engineering
2. Vision Research
3. Multimodal Learning
4. VLM Capstone
```

Each serious repository must contain:

```text
README
Architecture diagram
Problem statement
Dataset
Training instructions
Evaluation
Experiments
Results
Failure analysis
Limitations
Future work
```

Do not upload messy notebooks as the final portfolio.

---

# 24. Research Portfolio Standard

At least one project should have a report structured like a paper:

```text
Abstract
Introduction
Problem
Related Work
Method
Dataset
Experimental Setup
Baselines
Experiments
Ablations
Results
Failure Analysis
Limitations
Conclusion
```

This is valuable for both research-engineering and ML-engineering
interviews.

---

# 25. Open Source Strategy

Do not randomly make typo fixes just to say "open source."

After Month 6:

Look for meaningful contributions around:

- PyTorch
- Hugging Face
- Transformers
- datasets
- evaluation frameworks
- VLM libraries
- inference frameworks
- multimodal tooling

Start small.

Then attempt substantive contributions.

A good contribution can be:

- bug fix
- performance improvement
- evaluation addition
- dataset tooling
- documentation with technical depth
- model implementation
- benchmark

---

# 26. When to Consider a Master's

Do NOT decide now.

Re-evaluate after 9--12 months.

## Master's becomes attractive if:

- I love research
- I want research-heavy roles
- I enjoy papers more than product work
- I want access to research labs
- I want to build a publication record
- I find myself wanting to invent methods rather than primarily
  implement them

## Master's is NOT mandatory if:

- I enjoy model engineering
- I enjoy training and optimization
- I want industry ML engineering
- I can build strong evidence through projects/work
- I get an ML Engineer role directly

The decision should come from experience, not fear.

---

# 27. The Sarvam Researcher Track

Long-term target:

**ML Engineer, Vision → Research Engineer → Researcher**

To become competitive for the research side, add:

- deeper VLM architecture knowledge
- rigorous experimental design
- paper reproduction
- original experiments
- technical reports
- open-source work
- multilingual/Indic ML
- document understanding
- evaluation
- eventually publications if useful

Do not obsess over publishing immediately.

First become excellent at experimentation.

---

# 28. The Sarvam-Specific Bonus Track

After the core plan is working, add:

## Indic / multilingual

Learn:

- multilingual tokenization
- low-resource language challenges
- OCR for Indic scripts
- code-mixed text
- multilingual evaluation
- document understanding

## Document AI

Build with:

- OCR
- layout analysis
- tables
- forms
- charts
- structured extraction
- document VQA

This creates direct relevance to Sarvam Vision.

---

# 29. The "Never Get Stuck Again" Decision Tree

When I don't know what to learn next:

### If I cannot explain the fundamentals:

Go back to theory.

### If I understand theory but cannot code it:

Implement it.

### If I can code but cannot train it:

Debug the training pipeline.

### If I can train but cannot improve it:

Run controlled experiments.

### If I can improve it but cannot explain why:

Study the theory/paper.

### If I can explain it and reproduce it:

Try a novel modification.

### If I have a novel modification:

Write a technical report.

### If the project is strong:

Publish/open-source/apply.

---

# 30. The Anti-Tutorial Rule

Never spend more than approximately **2--3 days passively consuming
material on a topic** without writing code.

For every 1 hour of lecture/video:

Target roughly:

**2--4 hours of implementation/experimentation.**

The goal is not content completion.

The goal is skill acquisition.

---

# 31. What Success Looks Like After 3 Months

I should be able to:

- write PyTorch training loops from memory
- implement attention
- implement a small Transformer
- implement a CNN/ViT
- train models
- debug training
- use experiment tracking
- explain optimization
- run controlled experiments
- discuss model failures

At this point I should already begin applying to suitable ML roles.

---

# 32. What Success Looks Like After 6 Months

I should have:

- strong PyTorch
- strong Vision
- Transformer depth
- multimodal foundations
- a serious vision project
- a CLIP-style project
- paper reproduction experience
- GPU training exposure
- fine-tuning experience
- a clean GitHub

At this point I should be competitive for a growing set of ML Engineer /
CV / multimodal roles.

---

# 33. What Success Looks Like After 9 Months

I should have:

- VLM fine-tuning
- data pipeline experience
- evaluation harness
- inference optimization
- quantization
- distributed training concepts
- technical report
- serious capstone
- strong ML interview preparation

This is the point where Sarvam Vision and similar roles should become
serious targets.

---

# 34. What Success Looks Like After 12 Months

Target profile:

```text
BTech CSE + AIML
        +
Professional SDE Experience
        +
Strong Python/PyTorch
        +
Deep Learning
        +
Computer Vision
        +
Transformers
        +
Multimodal/VLM
        +
Model Training
        +
GPU
        +
Fine-tuning
        +
Evaluation
        +
Optimization
        +
Research-style experimentation
        +
Strong GitHub
```

This is the profile I want to take into interviews.

---

# 35. The Most Important Technical Projects

If I become overwhelmed, prioritize in this exact order:

## Priority 1

**PyTorch training + debugging**

## Priority 2

**CNN/ViT vision project**

## Priority 3

**Transformer implementation**

## Priority 4

**CLIP-style multimodal project**

## Priority 5

**VLM fine-tuning**

## Priority 6

**VLM evaluation + data pipeline**

## Priority 7

**GPU/distributed training**

## Priority 8

**Inference optimization**

## Priority 9

**Research reproduction**

## Priority 10

**Open source**

Do not reverse this order.

---

# 36. What I Should Learn Later, Not Now

After I am strong at the core:

- RLHF
- DPO
- preference optimization
- advanced alignment
- CUDA kernels
- Triton
- Megatron-LM
- advanced inference systems
- scaling laws
- mixture-of-experts
- advanced distributed training
- advanced interpretability

These are important, but they should not become distractions before I
can train and debug models properly.

---

# 37. My Core Weekly Metrics

Every Sunday, record:

```text
Hours studied:
Hours coding:
Hours experimenting:
Papers deeply read:
Models trained:
Experiments completed:
Bugs/debugging solved:
GitHub commits:
Interview questions:
Applications:
```

More important than hours:

### Evidence created this week

Examples:

- implemented attention
- trained ViT
- fixed NaN loss
- reproduced baseline
- ran 5 ablations
- profiled GPU
- improved latency by 20%
- built dataset filter
- wrote technical report

---

# 38. Monthly Review

At the end of every month answer:

1. What can I implement now that I couldn't last month?
2. What model can I train?
3. What failure can I debug?
4. What experiment did I run?
5. What did I learn from the result?
6. What can I explain without notes?
7. What is on GitHub?
8. What interview questions can I answer?
9. What JD requirements did I close?
10. What is the biggest remaining gap?

---

# 39. The Golden Rule

I am NOT trying to become:

> "Someone who knows AI."

I am trying to become:

> **Someone who can take an ML problem, choose/formulate the task, build
> the data pipeline, implement or modify the model, train it, debug it,
> evaluate it, analyze failures, optimize it, and explain why the
> results happened.**

That is the skill that transfers across ML roles.

---

# 40. My Final Target

The long-term direction is:

```text
                 SOFTWARE ENGINEERING
                         +
                 MACHINE LEARNING
                         +
                  DEEP LEARNING
                         +
                  COMPUTER VISION
                         +
                   TRANSFORMERS
                         +
                 MULTIMODAL ML
                         +
                     VLMs
                         +
               TRAINING / OPTIMIZATION
                         +
                 RESEARCH MINDSET
                         ↓
        MODEL-FOCUSED ML ENGINEER
                         ↓
                  RESEARCH ENGINEER
                         ↓
              RESEARCH / FRONTIER ML
```

I do not need to force the final step today.

My immediate objective is:

# Become excellent at building and training models.

Then:

# Become excellent at multimodal/VLM systems.

Then:

# Become excellent at experimentation and optimization.

Then:

# Decide whether I want ML engineering, research engineering, or research science.

---

# 41. The One-Sentence Rule for My Next Year

> **Every month, I should become demonstrably better at building,
> training, debugging, evaluating, experimenting with, or optimizing ML
> models.**

If an activity does not contribute to that goal, it is probably lower
priority.

---

# 42. Immediate Next 7 Days

Do NOT start a giant course.

## Day 1

- Create repository
- Install environment
- Baseline test
- Organize BTech notes

## Day 2

- Linear regression from scratch
- Gradient descent

## Day 3

- Logistic regression
- Loss functions
- Regularization

## Day 4

- Backpropagation
- MLP from NumPy

## Day 5

- Same MLP in PyTorch
- Training loop

## Day 6

- Start CNN
- Train on CIFAR-10

## Day 7

- Review
- Write what I remember
- Write what I forgot
- Plan Week 2

No perfection.

Just restart.

---

# 43. Final Mindset

I am not starting over.

I am not "an SDE trying to become an AI person."

I am:

> **A CSE/AIML graduate who already built a strong ML foundation, gained
> professional software-engineering experience, and is now deliberately
> specializing that engineering ability toward model development.**

The 8 months of SDE did not erase my AIML background.

They gave me engineering experience.

Now I need to reconnect the two.

## North Star

**Sarvam Vision / VLM**

## Broad target

**Model-focused ML Engineering**

## Specialization

**Computer Vision + Transformers + Multimodal/VLM**

## Long-term option

**Research Engineering / Research**

## Core strategy

**Learn less. Build more. Train more. Experiment more. Document
everything. Apply before perfect.**

---
