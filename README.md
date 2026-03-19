# DeepDocs AI: RAG Research Playground

DeepDocs AI is a system for experimenting with the internals of Retrieval-Augmented Generation (RAG) using LlamaIndex and Gemini 2.0 Flash at its core.

## 🚀 Key Features to Explore

- **Multiple Indexing Strategies**: Fixed vs Semantic chunking, metadata tagging.
- **Advanced Retrieval**: Top-K vs Top-N, Hybrid Search, Multi-query retrieval.
- **Query Rewriting**: Transforming user queries for better retrieval.
- **Reranking**: Using advanced rerankers to improve precision.
- **Evaluation System**: Measuring Faithfulness, Relevance, and Context Match.
- **LlamaIndex Deep Dive**: Custom retrievers, node parsers, and synthesizers.

## ⚙️ Setup

1. Initialize environment (if not already done):
   ```bash
   uv init
   uv add llama-index-llms-gemini llama-index-embeddings-gemini llama-index python-dotenv
   ```
2. Configure `.env`:
   Copy `.env.example` to `.env` and add your `GEMINI_API_KEY`.

3. Prepare data:
   Place your PDFs/Docs in the `data/` directory.

## 🧪 Experiments

Check the `src/` modules for specific implementations of RAG components.
