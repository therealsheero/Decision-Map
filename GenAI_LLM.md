# GENAI / LLM ENGINEERING ROADMAP

**Goal:** ML/DL  practical GenAI/LLM engineering .

---

## LAYER 1  LLM FUNDAMENTALS

**Learn:**

* Tokenization
* Tokens & token IDs
* Context window
* Embeddings
* Transformer  LLM connection
* Next-token prediction
* Logits & softmax
* Temperature
* Top-K / Top-P
* LLM inference
* Prompt vs system instruction

** Video:**

* Andrej Karpathy  **Let's build GPT from scratch**
* https://www.youtube.com/watch?v=kCc8FmEb1nY

** Additional FREE Videos:**

* Andrej Karpathy  **Let's build the GPT Tokenizer**
* https://www.youtube.com/watch?v=zduSFxRajkE
* Excellent for:

  * Tokenization
  * BPE
  * Token IDs
  * Encoding / decoding
  * Vocabulary
  * Special tokens
* 3Blue1Brown  **Attention in transformers, step-by-step**
* https://www.youtube.com/watch?v=eMlx5fFNoYc
* Excellent for:

  * Self-attention
  * Query / Key / Value
  * Attention mechanism
  * Multi-head attention
  * Context
  * Transformers
* 3Blue1Brown  **But what is a GPT?**
* Search YouTube:
  3Blue1Brown But what is a GPT
* Excellent conceptual explanation of:

  * Transformer architecture
  * Next-token prediction
  * Embeddings
  * Attention
  * Neural-network layers
* Andrej Karpathy  **Neural Networks: Zero to Hero**
* Search YouTube:
  Andrej Karpathy Neural Networks Zero to Hero
* Useful for understanding the neural-network foundations underneath LLMs.

** FREE Reading:**

* Jay Alammar  **The Illustrated Transformer**
* https://jalammar.github.io/illustrated-transformer/
* Very useful visual explanation of:

  * Attention
  * Transformers
  * Encoder/decoder
  * Embeddings
  * Query/Key/Value

**Priority:**

---

## LAYER 2  EMBEDDINGS

**Learn:**

* What is an embedding?
* Text  vector
* Semantic similarity
* Embedding dimensions
* Embedding models
* Dense representations
* Domain-specific embeddings

**Practice:**

* Generate embeddings for 10100 sentences
* Compare similar vs unrelated sentences
* Calculate cosine similarity

** Resource:**

* DeepLearning.AI  **Vector Databases: from Embeddings to Applications**
* https://www.deeplearning.ai/courses/vector-databases-embeddings-applications

Focus especially on:

* Vector representations
* Similar-vector search
* Embedding generation

** Additional FREE Videos:**

* Sentence Transformers  search YouTube:
  Sentence Transformers embeddings tutorial
* Hugging Face  search YouTube:
  Hugging Face embeddings sentence transformers
* Pinecone  search YouTube:
  Pinecone embeddings vector search

** FREE Resource:**

* Sentence Transformers documentation
* https://www.sbert.net/

Focus on:

* Sentence embeddings
* Semantic similarity
* Cosine similarity
* Pre-trained embedding models

**Practice resource:**

* Hugging Face  search models for sentence embeddings
* https://huggingface.co/models

**Priority:**

---

## LAYER 3  VECTOR SEARCH + VECTOR DATABASES

**Learn:**

* Cosine similarity
* Euclidean distance
* Nearest-neighbor search
* Top-K retrieval
* ANN (Approximate Nearest Neighbors)
* FAISS
* ChromaDB
* Metadata filtering
* Dense vs sparse search
* Hybrid search

** Main resource:**

* DeepLearning.AI  **Vector Databases: from Embeddings to Applications**
* https://www.deeplearning.ai/courses/vector-databases-embeddings-applications

This is a ~55-minute course with lessons on embeddings, similarity search, ANN, vector databases, and sparse/dense/hybrid search.

**Optional:**

* DeepLearning.AI  **Building Applications with Vector Databases**
* https://www.deeplearning.ai/courses/building-applications-vector-databases

** Additional FREE Videos:**

* FAISS tutorial  search YouTube:
  FAISS vector database tutorial Python
* ChromaDB tutorial  search YouTube:
  ChromaDB tutorial Python RAG
* Vector databases explained  search YouTube:
  vector database embeddings similarity search

** FREE Official Resources:**

* FAISS documentation
* https://faiss.ai/
* Chroma documentation
* https://docs.trychroma.com/
* Pinecone  **Vector Database Learning Center**
* https://www.pinecone.io/learn/

**Priority:**

---

## LAYER 4  RAG FUNDAMENTALS

**Learn:**

* What is RAG?
* Why RAG is needed
* Document loading
* Chunking
* Chunk size
* Chunk overlap
* Embeddings
* Vector database
* Retrieval
* Context
* Prompt construction
* Generation

**Core pipeline:**

Documents
 Chunk
 Embeddings
 Vector DB

Question
 Embedding
 Similarity Search
 Top-K Chunks
 Prompt + Context
 LLM
 Answer

** Video/course:**

* freeCodeCamp  **Building RAG Applications**

**If you can't find the exact freeCodeCamp video:** search YouTube for
freeCodeCamp Building RAG Applications

**Alternative verified resource:**

* DeepLearning.AI  **Building Applications with Vector Databases**
* https://www.deeplearning.ai/courses/building-applications-vector-databases

It includes a dedicated RAG lesson and covers semantic search, RAG and hybrid search.

** BEST FREE VIDEO ADDITION:**

* freeCodeCamp  **Learn RAG From Scratch**
* https://www.youtube.com/watch?v=sVcwVQRHIc8

This is especially useful because it goes beyond basic RAG and includes:

* Indexing
* Retrieval
* Generation
* Multi-query
* RAG Fusion
* Query decomposition
* Step-back prompting
* HyDE
* Routing
* Query construction
* Multi-representation indexing
* ColBERT
* CRAG
* Adaptive RAG

**Priority:**

---

## LAYER 5  RAG FROM SCRATCH (NO LANGCHAIN)

### MOST IMPORTANT PRACTICE

Build:

PDF
 Text extraction
 Your own chunking
 HuggingFace embeddings
 FAISS
 Cosine similarity / Top-K
 Context creation
 Prompt
 Ollama/local LLM
 Answer

**DO NOT use LangChain initially.**

**Goal:**

Understand what LangChain is doing underneath.

** FREE VIDEOS:**

* freeCodeCamp  **Learn RAG From Scratch**
* https://www.youtube.com/watch?v=sVcwVQRHIc8
* Search YouTube:
  RAG from scratch Python FAISS HuggingFace Ollama
* Search YouTube:
  build RAG from scratch without LangChain

** FREE TOOLS / DOCUMENTATION:**

* Hugging Face
* https://huggingface.co/
* FAISS
* https://faiss.ai/
* Ollama
* https://ollama.com/
* PyMuPDF
* https://pymupdf.readthedocs.io/

**Important practice:**

Do NOT copy a complete tutorial.

Build each component yourself:

text
PDF loader

Text cleaner

Chunker

Embedding generator

FAISS index

Retriever

Prompt builder

LLM

Answer

**Priority:**

---

## LAYER 6  LANGCHAIN

**Learn:**

* Document loaders
* Text splitters
* Embeddings
* Vector stores
* Retrievers
* Prompt templates
* Chains
* LLM interfaces

Understand:

Loader
 Splitter
 Embeddings
 Vector Store
 Retriever
 Prompt
 LLM

**Important:**

> LangChain ≠ RAG.

LangChain is an abstraction/framework for building LLM applications and workflows.

** Official resource:**

* LangChain Learn
* https://docs.langchain.com/oss/python/learn

The official tutorials currently include semantic search over PDFs and RAG agents.

** FREE VIDEOS:**

* LangChain official YouTube
* Search:
  LangChain official RAG tutorial
* freeCodeCamp  **LangChain / RAG tutorials**
* Search:
  freeCodeCamp LangChain RAG
* Search YouTube:
  LangChain complete tutorial Python RAG

** Additional FREE Resource:**

* LangChain documentation
* https://docs.langchain.com/

Focus on understanding the abstraction rather than memorizing APIs.

**Priority:**

---

## LAYER 7  HALLUCINATION + GROUNDED GENERATION

**Learn:**

* Why LLMs hallucinate
* Grounding
* Faithfulness
* Context relevance
* Answer relevance
* Source attribution
* Citations
* "I don't know" behavior
* Why RAG reduces but does NOT eliminate hallucination

**Understand:**

Bad retrieval
 Bad context
 Bad answer

Good retrieval
 Good context
 Potentially good answer

** FREE VIDEOS:**

* Search YouTube:
  LLM hallucination explained RAG
* Search YouTube:
  RAG hallucination grounding explained
* Search YouTube:
  why RAG does not eliminate hallucinations

** Recommended:**

* DeepLearning.AI  search YouTube:
  DeepLearning.AI hallucination RAG

** FREE Reading:**

* Google  **Introduction to Large Language Models**
* Search:
  Google Machine Learning Crash Course large language models
* OpenAI  search documentation for:
  hallucinations grounding evaluations

**Practice:**

Create a RAG system where the answer is deliberately absent from the documents.

Your system should answer:

> "I don't have enough information in the provided documents to answer this."

instead of inventing an answer.

**Priority:**

**Practice question:**

> If my RAG system is hallucinating, how do I determine whether the problem is retrieval or generation?

---

## LAYER 8  RAG EVALUATION

### Retrieval metrics

* Recall@K
* Precision@K
* MRR
* NDCG

### Generation metrics

* Faithfulness
* Answer relevance
* Context relevance
* Groundedness

**Key question:**

Is the problem caused by:

1. Bad retrieval?
2. Good retrieval but bad generation?

** Official resource:**

* LangSmith  **Evaluate a RAG application**
* https://docs.langchain.com/langsmith/evaluate-rag-tutorial

It covers creating evaluation datasets, running the RAG application against them, and measuring performance with evaluators.

** FREE VIDEOS:**

* Search YouTube:
  RAG evaluation Recall@K Precision@K MRR
* Search YouTube:
  RAG evaluation LangSmith tutorial
* Search YouTube:
  RAGAS RAG evaluation tutorial

** FREE Additional Resources:**

* Ragas
* https://docs.ragas.io/

Focus on:

* Faithfulness
* Answer relevance
* Context precision
* Context recall

**Practice:**

Create a dataset:

text
Question
Expected answer
Expected source/chunk

Then measure:

text
Retrieval quality
+
Generation quality

**Priority:**

---

## LAYER 9  ADVANCED RAG

**Learn:**

* Query rewriting
* Query transformation
* Metadata filtering
* Hybrid search
* BM25
* Reranking
* Cross-encoder rerankers
* RRF
* HyDE
* Multi-query retrieval
* Conversational RAG

### BEST VIDEO

**Venelin Valkov  Advanced Retrieval Pipeline for RAG**

https://www.youtube.com/watch?v=_ZHM4wsUwPs

Covers:

* Why vector search isn't enough
* Hybrid search
* RRF
* HyDE
* Reranking
* FlashRank
* Complete retrieval pipeline

The video is ~15 minutes and specifically demonstrates the advanced retrieval architecture.

### Optional

**Venelin Valkov  Build Production-Ready Retrieval RAG Pipeline in LangChain**

https://www.youtube.com/watch?v=YNcoFoRwoc8

Covers:

* BM25
* Hybrid search
* Reranking
* HyDE
* Citations

** Additional FREE Video:**

* freeCodeCamp  **Learn RAG From Scratch**
* https://www.youtube.com/watch?v=sVcwVQRHIc8

Pay particular attention to:

* Multi-query
* RAG Fusion
* Query decomposition
* Step-back prompting
* HyDE
* Routing
* ColBERT
* CRAG
* Adaptive RAG

** FREE Resources:**

* Elasticsearch  search YouTube:
  BM25 explained Elasticsearch
* Sentence Transformers  search:
  cross encoder reranking sentence transformers
* Cohere  search:
  reranking explained RAG

**Practice:**

Build:

text
User Query

Query rewriting

Dense Search + BM25

RRF

Top 20

Reranker

Top 5

LLM

**Priority:**

---

## LAYER 10  AGENTS + TOOL CALLING

**Learn AFTER RAG**

* Function calling
* Tool calling
* Agents
* Agent loops
* State
* Memory
* Tool selection
* LangGraph basics

** Official LangChain Academy:**

* https://academy.langchain.com/

### Start here:

**Introduction to LangGraph**

* https://academy.langchain.com/courses/intro-to-langgraph

It covers:

* Chains
* Routers
* Agents
* Agent memory
* State
* Human-in-the-loop
* Deployment basics

** FREE VIDEOS:**

* Search YouTube:
  LangGraph full course
* Search YouTube:
  LangGraph agents tutorial Python
* Search YouTube:
  function calling LLM tools tutorial
* Search YouTube:
  AI agents explained LangGraph

** FREE Official Resources:**

* LangGraph documentation
* https://docs.langchain.com/oss/python/langgraph
* LangChain Academy
* https://academy.langchain.com/

**Practice project:**

Build an agent with:

text
User

Agent

Tool selection
 Calculator
 Search
 RAG
 Database

Observation

Agent

Final answer

**Priority:**

---

## LAYER 11  PRODUCTION GENAI

**Learn:**

* FastAPI
* Async APIs
* Streaming
* Docker
* Authentication
* Rate limiting
* Caching
* Logging
* Monitoring
* Observability
* Token usage & cost
* Latency optimization
* Model serving
* Prompt injection
* Security
* PII protection
* Evaluation pipelines

**Build:**

Client
 FastAPI
 Auth
 Query processing
 Retrieval
 Reranking
 Prompt
 LLM
 Evaluation / Guardrails
 Response

** BEST FREE VIDEO ADDITION:**

* freeCodeCamp  **Production RAG with LangChain & Vector Databases**
* https://www.youtube.com/watch?v=mHxLXzYjQRE

This is particularly valuable for this layer because it covers:

* RAG architecture
* Chroma
* Similarity search
* Debugging
* Hybrid search
* Token budgeting
* Observability
* LangSmith
* RAG optimization
* Scaling
* Production hosting
* pgvector
* Security
* LangGraph
* FastAPI
* Testing
* Advanced RAG
* Contextual retrieval
* Late chunking
* Agentic RAG
* GraphRAG
* Multimodal RAG

** Additional FREE Videos:**

### FastAPI

Search YouTube:

FastAPI full course Python

FastAPI async API tutorial

FastAPI production deployment Docker

### Docker

Search YouTube:

Docker full course freeCodeCamp

Docker for Python FastAPI

### LLM Production

Search YouTube:

LLM production deployment FastAPI Docker

production RAG architecture

LLM application observability

### Prompt Injection / Security

Search YouTube:

prompt injection explained LLM security

LLM security prompt injection tutorial

OWASP LLM Top 10

** FREE Security Resource:**

* OWASP  **Top 10 for LLM Applications**
* https://owasp.org/www-project-top-10-for-large-language-model-applications/

Learn especially:

* Prompt injection
* Insecure output handling
* Sensitive information disclosure
* Excessive agency
* Overreliance
* Model denial of service

** FREE Production Resources:**

* FastAPI
* https://fastapi.tiangolo.com/
* Docker
* https://docs.docker.com/
* LangSmith
* https://smith.langchain.com/
* OpenTelemetry
* https://opentelemetry.io/

**Priority:**

---

# PRIORITY ORDER

If time is limited:

**1. LLM Fundamentals**

**2. Embeddings**

**3. Vector Search / Vector DB**

**4. RAG**

**5. RAG from Scratch**

**6. LangChain**

**7. Hallucination / Grounding**

**8. RAG Evaluation**

**9. Advanced RAG**

**10. Agents**

**11. Production GenAI**

---

# MUST-KNOW BEFORE

Be able to explain WITHOUT NOTES:

1. What is an LLM?
2. What is tokenization?
3. What is an embedding?
4. Why do we need embeddings?
5. What is cosine similarity?
6. What is a vector database?
7. FAISS vs ChromaDB
8. What is chunking?
9. Why chunk documents?
10. What is chunk overlap?
11. What is RAG?
12. Explain the complete RAG pipeline.
13. Why does RAG reduce hallucination?
14. Can RAG still hallucinate?
15. What is Top-K retrieval?
16. What is a retriever?
17. What does LangChain actually do?
18. Dense vs keyword search
19. What is hybrid search?
20. What is reranking?
21. What is Recall@K?
22. How do you evaluate a RAG system?
23. What is prompt injection?
24. What is an AI agent?
25. How would you deploy a RAG system in production?

---

# ONE PROJECT TO BUILD

## Production-Grade RAG API

PDFs
 Chunking
 Embeddings
 FAISS/Chroma
 Retrieval
 Reranking
 LLM
 Citations
 Evaluation
 FastAPI
 Docker
 Tests
 Logging

### Build it TWICE:

**Version 1:** Without LangChain

**Version 2:** With LangChain

### Final goal:

Be able to explain **every component and every design decision** rather than saying:

> "I used LangChain and ChromaDB because the tutorial used them."

Instead be able to say:

> "I chose this embedding model because..., chunked documents this way because..., used vector retrieval because..., added reranking because..., and evaluated retrieval using Recall@K because..."

That is the level you want before .
