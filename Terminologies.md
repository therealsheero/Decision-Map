# Complete Machine Learning & AI Architecture Map

## Table of Contents

- Machine Learning Paradigms
- Classical Machine Learning Models
- Deep Learning Architectures
- CNNs and Computer Vision
- RNNs and Sequence Models
- Attention and Transformers
- LLM Concepts
- Generative AI Architectures
- Multimodal AI
- Graph Neural Networks
- Recommendation Systems
- Time Series Models
- RAG and AI Agents
- Model Compression
- Evaluation Metrics
- Current Frontier Architectures

---


# 1. Machine Learning Paradigms

## 1.1 Supervised Learning

Learning from labeled data.

**Tasks**

- Classification
- Regression
- Ranking
- Structured prediction

**Models**

- Linear Regression
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- Gradient Boosting Machines
- Neural Networks

---

## 1.2 Unsupervised Learning

Learning patterns from unlabeled data.

**Tasks**

- Clustering
- Dimensionality reduction
- Representation learning
- Density estimation

**Models**

- K-Means
- DBSCAN
- Gaussian Mixture Models (GMM)
- Hierarchical Clustering
- PCA
- Autoencoders
- Self-Organizing Maps

---

## 1.3 Semi-Supervised Learning

Combination of labeled + unlabeled data.

Methods:

- Pseudo-labeling
- Consistency regularization
- Label propagation
- Mean Teacher
- FixMatch
- MixMatch

---

## 1.4 Self-Supervised Learning

Models create their own training signals.

Examples:

- Masked Language Modeling
- Contrastive Learning
- Autoencoding
- Next-token prediction

Models:

- BERT
- GPT
- SimCLR
- BYOL
- MAE

---

## 1.5 Reinforcement Learning (RL)

Agent learns by interacting with environments.

Components:

- Agent
- Environment
- State
- Action
- Reward
- Policy
- Value function

Algorithms:

- Q-Learning
- SARSA
- DQN
- PPO
- A2C
- A3C
- SAC
- TD3

---

# 2. Classical Machine Learning Models

## Linear Models

### Linear Regression

Predict continuous values.

Variants:

- Ridge Regression
- Lasso Regression
- Elastic Net
- Bayesian Regression

### Logistic Regression

Classification using probability.

Variants:

- Multinomial Logistic Regression
- Ordinal Regression

---

# Decision Tree Family

## Decision Tree

Splitting based on:

- Gini impurity
- Information gain
- Entropy

## Random Forest

Ensemble of decision trees.

Concepts:

- Bagging
- Bootstrap sampling
- Feature randomness

## Gradient Boosting

Sequential tree improvement.

Models:

- AdaBoost
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

---

# Instance-Based Learning

## K-Nearest Neighbors (KNN)

Concepts:

- Distance metrics
- Neighborhood search
- Curse of dimensionality

## Kernel Methods

Models:

- SVM
- Kernel Ridge Regression

Kernels:

- Linear kernel
- Polynomial kernel
- RBF kernel
- Sigmoid kernel

---

# 3. Deep Learning Architectures

# Artificial Neural Networks (ANN)

Basic components:

- Neuron
- Weight
- Bias
- Activation function
- Layer
- Forward propagation
- Backpropagation

Architecture types:

- Feed Forward Neural Network
- Multilayer Perceptron (MLP)

---

# Activation Functions

## Classical

- Sigmoid
- Tanh
- ReLU

## Modern

- Leaky ReLU
- ELU
- GELU
- Swish
- Mish
- Softmax

---

# Optimization Algorithms

- Gradient Descent
- SGD
- Momentum
- Nesterov Momentum
- AdaGrad
- RMSProp
- Adam
- AdamW
- LAMB
- Lion

---

# 4. Convolutional Neural Networks (CNN)

Used for:

- Images
- Video
- Spatial data

Architectures:

## Early CNNs

- LeNet
- AlexNet
- ZFNet

## Modern CNNs

- VGG
- GoogLeNet
- Inception
- ResNet
- DenseNet
- EfficientNet
- MobileNet
- ShuffleNet
- ConvNeXt

CNN concepts:

- Convolution
- Kernel/filter
- Stride
- Padding
- Pooling
- Feature map
- Receptive field
- Batch normalization

---

# 5. Recurrent Neural Networks (RNN)

Used for sequential data.

Architectures:

## Basic RNN

Problems:

- Vanishing gradients
- Exploding gradients

## LSTM

Components:

- Forget gate
- Input gate
- Output gate
- Cell state

## GRU

Components:

- Update gate
- Reset gate

Other RNN models:

- Bidirectional RNN
- Seq2Seq
- Echo State Network

---

# 6. Attention-Based Architectures

## Attention Mechanism

Concepts:

- Query
- Key
- Value
- Attention score
- Context vector

Types:

- Additive Attention
- Dot Product Attention
- Self Attention
- Cross Attention
- Multi-head Attention

---

# 7. Transformer Architecture

The foundation of modern AI.

Introduced in: "Attention Is All You Need"

Components:

## Encoder

Contains:

- Self-attention
- Feed-forward network
- Layer normalization
- Residual connections

## Decoder

Contains:

- Masked self-attention
- Cross attention
- Feed-forward layers

Key concepts:

- Tokenization
- Embeddings
- Positional encoding
- Context window
- Attention heads
- Transformer blocks

---

# Transformer Model Families

## Encoder-only Models

Used for understanding.

Examples:

- BERT
- RoBERTa
- ALBERT
- ELECTRA
- DeBERTa
- DistilBERT

Tasks:

- Classification
- Search
- Embeddings

---

## Decoder-only Models

Used for generation.

Examples:

- GPT series
- LLaMA
- Mistral
- Falcon
- Claude-style models
- PaLM
- Gemini

Tasks:

- Text generation
- Coding
- Reasoning

---

## Encoder-Decoder Models

Examples:

- T5
- BART
- FLAN-T5
- mT5

Tasks:

- Translation
- Summarization
- Text transformation

---

# 8. Large Language Model (LLM) Terminology

## Model Components

### Token

Small unit of text.

Examples:

- word
- subword
- character

### Vocabulary

Collection of tokens.

### Embedding

Conversion of tokens into vectors.

### Parameter

Learned value inside a model.

### Weight

Numerical value learned during training.

### Bias

Additional learned offset.

### Context Window

Maximum tokens model can process.

### Inference

Using a trained model.

### Training

Updating model parameters.

---

# 9. LLM Training Concepts

## Pretraining

Learning general patterns.

Examples:

- GPT pretraining
- BERT pretraining

## Fine-tuning

Adapting model for a task.

Types:

- Full fine-tuning
- Parameter Efficient Fine Tuning (PEFT)

PEFT methods:

- LoRA
- QLoRA
- Adapter Layers
- Prefix tuning
- Prompt tuning

---

# 10. Generative AI Architectures

## Autoencoders

Structure:

Encoder → Latent Space → Decoder

Types:

- Vanilla Autoencoder
- Sparse Autoencoder
- Denoising Autoencoder
- Variational Autoencoder (VAE)

---

# Generative Adversarial Networks (GANs)

Two networks:

Generator:

- Creates samples

Discriminator:

- Judges samples

GAN variants:

- DCGAN
- CycleGAN
- StyleGAN
- BigGAN
- Pix2Pix

---

# Diffusion Models

Used for image/video generation.

Concepts:

- Forward diffusion
- Reverse diffusion
- Noise schedule
- Denoising
- Score function

Models:

- DDPM
- DDIM
- Stable Diffusion
- Imagen
- DALL-E

---

# 11. Computer Vision Architectures

## Object Detection

Models:

Two-stage:

- R-CNN
- Fast R-CNN
- Faster R-CNN

One-stage:

- YOLO
- SSD
- RetinaNet

Concepts:

- Bounding box
- IoU
- Non-Max Suppression
- Anchor boxes

---

## Image Segmentation

Models:

- FCN
- U-Net
- DeepLab
- Mask R-CNN
- Segment Anything Model (SAM)

Types:

- Semantic segmentation
- Instance segmentation
- Panoptic segmentation

---

# Vision Transformers

Architectures:

- ViT
- DeiT
- Swin Transformer
- BEiT

Concepts:

- Image patches
- Patch embeddings
- Window attention

---

# 12. Multimodal AI Models

Models handling multiple data types.

Modalities:

- Text
- Image
- Audio
- Video
- Sensor data

Architectures:

- CLIP
- ALIGN
- Flamingo
- BLIP
- GPT-4V style models
- Gemini-style models

Concepts:

- Multimodal embeddings
- Vision encoder
- Cross-modal attention

---

# 13. Speech AI Models

## Speech Recognition

Models:

- HMM-GMM
- DeepSpeech
- wav2vec
- Whisper

## Text-to-Speech

Models:

- Tacotron
- WaveNet
- FastSpeech
- VITS

---

# 14. Graph Neural Networks (GNN)

For graph data.

Types:

- GCN
- GraphSAGE
- GAT
- GGNN
- ChebNet

Concepts:

- Nodes
- Edges
- Message passing
- Graph embedding

Applications:

- Molecules
- Social networks
- Recommendation systems

---

# 15. Recommendation Models

Classical:

- Collaborative Filtering
- Matrix Factorization
- SVD

Deep learning:

- Neural Collaborative Filtering
- Wide & Deep
- DeepFM
- Two Tower Networks
- Transformer Recommenders

---

# 16. Time Series Models

Classical:

- ARIMA
- SARIMA
- Exponential Smoothing

ML:

- Random Forest
- XGBoost

Deep learning:

- LSTM
- Temporal CNN
- N-BEATS
- Temporal Fusion Transformer
- Informer
- Autoformer

---

# 17. Retrieval-Augmented Generation (RAG)

Architecture:

User Query ↓ Retriever ↓ Vector Database ↓ Relevant Documents ↓ LLM ↓
Answer

Components:

- Embedding model
- Vector database
- Chunking
- Retrieval
- Reranking
- Context injection

---

# 18. AI Agent Architectures

Components:

- LLM
- Memory
- Tools
- Planning
- Reflection
- Execution

Framework concepts:

- Tool calling
- Function calling
- ReAct
- Chain-of-thought
- Planning agents
- Multi-agent systems

---

# 19. Model Compression

Techniques:

## Quantization

Reduce precision.

Examples:

- FP32
- FP16
- BF16
- INT8
- INT4

## Pruning

Remove parameters.

## Knowledge Distillation

Teacher model → Student model

---

# 20. Machine Learning Evaluation Terminology

## Classification Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

## Regression Metrics

- MAE
- MSE
- RMSE
- R²

## Generative Metrics

- BLEU
- ROUGE
- METEOR
- Perplexity
- FID

---

# 21. Common ML Terms

  TermMeaning

---

  Epoch                 One complete training pass
  Batch                 Group of training examples
  Learning rate         Step size for optimization
  Loss function         Training error measure
  Gradient              Direction of parameter update
  Overfitting           Memorizing training data
  Underfitting          Too simple model
  Regularization        Prevent overfitting
  Dropout               Random neuron removal
  Normalization         Scaling data
  Generalization        Performance on unseen data
  Hyperparameter        Configuration chosen before training
  Checkpoint            Saved model state
  Fine-tuning           Additional training
  Transfer learning     Reusing learned knowledge
  Inference             Model prediction
  Latent space          Hidden representation
  Feature engineering   Creating useful inputs
  Data augmentation     Creating more training data
  Ensemble              Combining models

---

# 22. Current Frontier Architectures (2024--2026)

Emerging areas:

- Mixture of Experts (MoE)
- Sparse Transformers
- Retrieval-Augmented Models
- Long-context Transformers
- State Space Models (SSM)
- Mamba architecture
- Neural ODEs
- World Models
- Diffusion Transformers (DiT)
- Multimodal Foundation Models
- Agentic AI architectures

---

# High-Level AI Architecture Map

    Artificial Intelligence
    │
    ├── Machine Learning
    │   ├── Supervised Learning
    │   ├── Unsupervised Learning
    │   ├── Reinforcement Learning
    │
    ├── Deep Learning
    │   ├── CNN
    │   ├── RNN
    │   ├── Transformer
    │   ├── GNN
    │
    ├── Generative AI
    │   ├── GAN
    │   ├── VAE
    │   ├── Diffusion
    │   └── LLM
    │
    ├── Multimodal AI
    │
    ├── AI Agents
    │
    └── Foundation Models
