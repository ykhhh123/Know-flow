# RAG Retrieval

The document retrieval path now supports hybrid recall, RRF fusion, optional reranking, and metadata filters.

## Retrieval Modes

- `hybrid`: vector recall plus PostgreSQL full-text recall, fused with Reciprocal Rank Fusion.
- `vector`: pgvector cosine similarity only.
- `keyword`: PostgreSQL full-text search only, ranked with `ts_rank_cd`.

## Document Metadata

Document upload accepts optional metadata:

- `knowledge_base`: defaults to `default`.
- `tags`: comma-separated tag list.

Example:

```text
POST /api/documents/upload?knowledge_base=research&tags=rag,eval
```

## Search Parameters

`POST /api/documents/search` accepts:

- `retrieval_mode`: `hybrid`, `vector`, or `keyword`.
- `use_rerank`: enables reranking when a `rerank_provider` is supplied.
- `rerank_provider`: `bge`, `cross_encoder`, `qwen`, `dashscope`, or `alibaba`.
- `rerank_model`: optional provider model name or local model path.
- `knowledge_bases`: comma-separated filter.
- `document_types`: comma-separated filter, for example `pdf,md`.
- `created_after` / `created_before`: datetime filters.
- `tags`: comma-separated tag filter.
- `document_ids`: comma-separated UUID filter.

Local cross-encoder reranking uses `sentence-transformers` and defaults to `BAAI/bge-reranker-base`. Qwen/DashScope reranking uses the existing API key and defaults to `gte-rerank-v2`.
