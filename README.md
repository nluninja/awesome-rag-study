# Awesome RAG Study [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A curated collection of papers, frameworks, tools, and resources on Retrieval-Augmented Generation (RAG).

Maintained for students of the *Text Mining and Data Visualization* course as a starting point for thesis research.

## What is RAG?

Retrieval-Augmented Generation is a technique that enhances Large Language Models (LLMs) by grounding their responses in external knowledge retrieved at inference time, reducing hallucinations and enabling domain-specific answers without fine-tuning.

---

## Contents

- [Foundational Papers](#foundational-papers)
- [Survey Papers](#survey-papers)
- [Advanced Techniques](#advanced-techniques)
  - [Chunking and Indexing](#chunking-and-indexing)
  - [Retrieval Strategies](#retrieval-strategies)
  - [Reranking](#reranking)
  - [Query Transformation](#query-transformation)
  - [Evaluation](#evaluation)
- [Frameworks and Libraries](#frameworks-and-libraries)
- [Vector Databases](#vector-databases)
- [Embedding Models](#embedding-models)
- [Tutorials and Guides](#tutorials-and-guides)
- [Videos and Talks](#videos-and-talks)
- [Datasets and Benchmarks](#datasets-and-benchmarks)
- [Contributing](#contributing)
- [License](#license)

---

## Foundational Papers

- **[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)** (2020) - The original RAG paper by Lewis et al. (Meta AI). Introduces the RAG architecture combining a pre-trained seq2seq model with a dense retriever (DPR).
- **[Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)** (2020) - DPR — the dense retrieval method that underpins many RAG systems.
- **[REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909)** (2020) - Pre-trains a language model jointly with a knowledge retriever.
- **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** (2017) - The Transformer architecture — foundational to all modern LLMs used in RAG.

## Survey Papers

- **[Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997)** (2023) - Comprehensive survey covering Naive RAG, Advanced RAG, and Modular RAG paradigms. Excellent starting point.
- **[A Survey on RAG Meeting LLMs: Towards Retrieval-Augmented Large Language Models](https://arxiv.org/abs/2405.06211)** (2024) - Covers the evolution of RA-LLMs, taxonomies, and training strategies.
- **[Seven Failure Points When Engineering a RAG System](https://arxiv.org/abs/2401.05856)** (2024) - Practical guide to what can go wrong in RAG pipelines — highly recommended for thesis work.

## Advanced Techniques

### Chunking and Indexing

- **[Unstructured](https://github.com/Unstructured-IO/unstructured)** - Pre-processing library for parsing PDFs, HTML, Word docs into clean chunks.
- **[Semantic Chunking](https://arxiv.org/abs/2312.06648)** - Splitting documents based on semantic similarity rather than fixed token windows.
- **Hierarchical Indexing** - Using summaries at different granularity levels (document → section → paragraph) to improve retrieval precision.
- **Parent-Child Chunking** - Retrieve small chunks for precision, but pass the parent (larger) chunk to the LLM for context.

### Retrieval Strategies

| Strategy | Description |
|----------|-------------|
| **Dense Retrieval** | Encode queries and documents into vector embeddings, retrieve by cosine similarity. |
| **Sparse Retrieval (BM25)** | Traditional keyword-based retrieval. Still competitive and often used as a baseline. |
| **Hybrid Search** | Combine dense + sparse retrieval (e.g., via Reciprocal Rank Fusion). Often outperforms either alone. |
| **Multi-Query Retrieval** | Generate multiple query variations with an LLM and retrieve for each, then merge results. |
| **HyDE** | [Hypothetical Document Embeddings](https://arxiv.org/abs/2212.10496) - Generate a hypothetical answer first, then use it as the retrieval query. |
| **Contextual Retrieval** | [Anthropic's approach](https://www.anthropic.com/news/contextual-retrieval) - Prepend chunk-specific context before embedding to reduce retrieval failures. |

### Reranking

- **[Cohere Rerank](https://docs.cohere.com/docs/reranking)** - Cross-encoder reranking API.
- **[ColBERT](https://arxiv.org/abs/2004.12832)** - Late interaction model for efficient and effective reranking.
- **[bge-reranker](https://huggingface.co/BAAI/bge-reranker-v2-m3)** - Open-source cross-encoder reranker by BAAI.
- **[RankLLM](https://arxiv.org/abs/2309.15088)** - Using LLMs themselves as rerankers via listwise prompting.

### Query Transformation

- **Query Rewriting** - Use an LLM to reformulate the user query for better retrieval.
- **[Step-Back Prompting](https://arxiv.org/abs/2310.06117)** - Ask a more abstract question first to retrieve broader context.
- **Query Decomposition** - Break complex questions into sub-questions, retrieve for each, then synthesize.

### Evaluation

| Framework | Description |
|-----------|-------------|
| [RAGAS](https://github.com/explodinggradients/ragas) | Reference-free evaluation framework. Metrics: faithfulness, answer relevancy, context precision/recall. |
| [TruLens](https://github.com/truera/trulens) | Evaluation and tracking for LLM apps, including RAG-specific metrics. |
| [DeepEval](https://github.com/confident-ai/deepeval) | Unit testing framework for LLM outputs with RAG-aware metrics. |
| [ARES](https://arxiv.org/abs/2311.09476) | Automated RAG Evaluation System — uses LLM judges with statistical confidence. |

---

## Frameworks and Libraries

| Framework | Language | Description |
|-----------|----------|-------------|
| [LangChain](https://github.com/langchain-ai/langchain) | Python/JS | The most widely adopted framework for building RAG pipelines. Large ecosystem, many integrations. |
| [LlamaIndex](https://github.com/run-llama/llama_index) | Python | Data framework specifically designed for RAG. Strong focus on indexing and retrieval abstractions. |
| [Haystack](https://github.com/deepset-ai/haystack) | Python | Production-ready NLP framework by deepset. Pipeline-based architecture. |
| [RAGFlow](https://github.com/infiniflow/ragflow) | Python | Open-source RAG engine with deep document understanding and chunk visualization. |
| [Verba](https://github.com/weaviate/Verba) | Python | Open-source RAG chatbot powered by Weaviate. Good for quick prototyping. |
| [Cognita](https://github.com/truefoundry/cognita) | Python | Open-source modular RAG framework for production use. |

## Vector Databases

| Database | Type | Notes |
|----------|------|-------|
| [Chroma](https://github.com/chroma-core/chroma) | Embedded | Lightweight, easy to start with. Good for prototyping and smaller projects. |
| [Weaviate](https://github.com/weaviate/weaviate) | Self-hosted / Cloud | Supports hybrid search natively. GraphQL API. |
| [Qdrant](https://github.com/qdrant/qdrant) | Self-hosted / Cloud | Written in Rust. Excellent filtering and payload support. |
| [Milvus](https://github.com/milvus-io/milvus) | Self-hosted / Cloud | Highly scalable. Used in many production deployments. |
| [Pinecone](https://www.pinecone.io/) | Cloud-only | Fully managed. Simple API. Popular in industry. |
| [FAISS](https://github.com/facebookresearch/faiss) | Library | Meta's similarity search library. Not a database, but extremely fast for local use. |
| [pgvector](https://github.com/pgvector/pgvector) | PostgreSQL Extension | Add vector search to your existing PostgreSQL database. Great if you already use Postgres. |

## Embedding Models

| Model | Provider | Notes |
|-------|----------|-------|
| [text-embedding-3-small/large](https://platform.openai.com/docs/guides/embeddings) | OpenAI | Strong general-purpose embeddings. `large` variant has 3072 dimensions. |
| [Cohere Embed v3](https://docs.cohere.com/docs/cohere-embed) | Cohere | Supports multiple input types (search_document, search_query). |
| [BGE (BAAI)](https://huggingface.co/BAAI/bge-large-en-v1.5) | Open-source | Top-performing open-source embeddings. Available in multiple sizes. |
| [E5-Mistral](https://huggingface.co/intfloat/e5-mistral-7b-instruct) | Open-source | LLM-based embedding model. Strong performance on MTEB benchmark. |
| [Nomic Embed](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) | Open-source | Long-context (8192 tokens), fully open-source with open training data. |
| [Jina Embeddings](https://huggingface.co/jinaai/jina-embeddings-v3) | Open-source | Multilingual, supports 8192 token context. Good for non-English corpora. |

> **Tip:** Check the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) for up-to-date embedding model benchmarks.

---

## Tutorials and Guides

- **[RAG From Scratch (LangChain)](https://github.com/langchain-ai/rag-from-scratch)** - Series of notebooks covering RAG concepts from basics to advanced patterns.
- **[Building RAG Applications with LlamaIndex](https://docs.llamaindex.ai/en/stable/getting_started/concepts/)** - Official LlamaIndex documentation and conceptual guide.
- **[Pinecone RAG Learning Center](https://www.pinecone.io/learn/retrieval-augmented-generation/)** - Well-written introduction to RAG with practical examples.
- **[Contextual Retrieval Guide](https://www.anthropic.com/news/contextual-retrieval)** - Practical improvements to standard RAG with contextual embeddings and BM25.
- **[Full Stack RAG App Tutorial (freeCodeCamp)](https://www.youtube.com/watch?v=sVcwVQRHIc8)** - Video walkthrough of building a complete RAG application.

## Videos and Talks

- **[But what is RAG? (3Blue1Brown-style explainer)](https://www.youtube.com/watch?v=T-D1OfcDW1M)** - Visual, intuitive explanation of how RAG works.
- **[RAG is Dead? Long Live RAG! (Keynote)](https://www.youtube.com/watch?v=JGpmQvlYRdQ)** - Discussion on the future of RAG vs. long-context models.
- **[Building Production RAG (AI Engineer Summit)](https://www.youtube.com/watch?v=jENqvjpkwmw)** - Practical lessons from deploying RAG at scale.
- **[Advanced RAG Techniques (DeepLearning.AI)](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/)** - Short course by Andrew Ng's platform.

## Datasets and Benchmarks

| Dataset/Benchmark | Description |
|-------------------|-------------|
| [Natural Questions (NQ)](https://ai.google.com/research/NaturalQuestions) | Google's open-domain QA dataset. Standard benchmark for retrieval systems. |
| [HotpotQA](https://hotpotqa.github.io/) | Multi-hop QA requiring reasoning over multiple documents. |
| [MS MARCO](https://microsoft.github.io/msmarco/) | Large-scale passage retrieval and QA benchmark. |
| [BEIR](https://github.com/beir-cellar/beir) | Heterogeneous benchmark for zero-shot evaluation of retrieval models across diverse tasks. |
| [RGB (Retrieval-Augmented Generation Benchmark)](https://arxiv.org/abs/2309.01431) | Specifically designed to evaluate RAG systems on noise robustness, negative rejection, information integration, and counterfactual robustness. |

---

## Contributing

Contributions are welcome! This is a collaborative resource for students and researchers.

Please follow these guidelines:
1. Fork this repository
2. Add your resource in the appropriate section
3. Follow the existing format: `**[Resource Name](link)** - Brief description.`
4. Ensure all links are working and resources are relevant to RAG
5. Submit a pull request

## License

[![CC0](https://licensebuttons.net/p/zero/1.0/88x31.png)](https://creativecommons.org/publicdomain/zero/1.0/)

This work is dedicated to the public domain under [CC0 1.0 Universal](LICENSE).
