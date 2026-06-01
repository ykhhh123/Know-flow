from pydantic import AliasChoices, BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = None
    conversationId: Optional[str] = None
    apiKey: str
    model: str = "gpt-5-mini"
    provider: Optional[str] = None
    baseUrl: Optional[str] = None
    providerApiKeys: Optional[Dict[str, str]] = None
    providerBaseUrls: Optional[Dict[str, str]] = None
    fallbackChain: Optional[str] = None


class DocumentResponse(BaseModel):
    id: str
    title: str
    file_type: str
    knowledge_base: Optional[str] = None
    tags: Optional[List[str]] = None
    status: str
    created_at: datetime
    chunks_count: Optional[int] = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]


class SearchResult(BaseModel):
    id: str
    content: str
    document_id: str
    chunk_index: int
    retrieval_score: Optional[float] = None
    vector_rank: Optional[int] = None
    keyword_rank: Optional[int] = None
    rerank_score: Optional[float] = None


class RAGSearchFilters(BaseModel):
    knowledge_bases: Optional[List[str]] = Field(
        default=None,
        validation_alias=AliasChoices("knowledge_bases", "knowledgeBases"),
    )
    document_types: Optional[List[str]] = Field(
        default=None,
        validation_alias=AliasChoices("document_types", "documentTypes"),
    )
    created_after: Optional[datetime] = Field(
        default=None,
        validation_alias=AliasChoices("created_after", "createdAfter"),
    )
    created_before: Optional[datetime] = Field(
        default=None,
        validation_alias=AliasChoices("created_before", "createdBefore"),
    )
    tags: Optional[List[str]] = None
    document_ids: Optional[List[str]] = Field(
        default=None,
        validation_alias=AliasChoices("document_ids", "documentIds"),
    )


class RAGChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = None
    use_rag: bool = True
    use_memory: bool = True
    use_tools: bool = False
    conversationId: Optional[str] = None
    apiKey: str
    model: str = "gpt-5-mini"
    provider: Optional[str] = None
    baseUrl: Optional[str] = None
    providerApiKeys: Optional[Dict[str, str]] = None
    providerBaseUrls: Optional[Dict[str, str]] = None
    fallbackChain: Optional[str] = None
    use_local_embedding: bool = Field(
        default=False,
        validation_alias=AliasChoices("use_local_embedding", "useLocalEmbedding"),
    )
    retrieval_mode: str = Field(
        default="hybrid",
        validation_alias=AliasChoices("retrieval_mode", "retrievalMode"),
    )
    filters: Optional[RAGSearchFilters] = None
    use_rerank: bool = Field(
        default=True,
        validation_alias=AliasChoices("use_rerank", "useRerank"),
    )
    rerank_provider: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("rerank_provider", "rerankProvider"),
    )
    rerank_model: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("rerank_model", "rerankModel"),
    )


class RAGEvaluationExpectedSource(BaseModel):
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None
    chunk_index: Optional[int] = None


class RAGEvaluationCase(BaseModel):
    query: str
    expected_sources: List[RAGEvaluationExpectedSource]


class RAGEvaluationRequest(BaseModel):
    cases: List[RAGEvaluationCase]
    apiKey: str = ""
    provider: str = "openai"
    baseUrl: Optional[str] = None
    topK: int = 5
    useLocalEmbedding: bool = False
    retrievalMode: str = "hybrid"
    filters: Optional[RAGSearchFilters] = None
    useRerank: bool = True
    rerankProvider: Optional[str] = None
    rerankModel: Optional[str] = None


class RAGEvaluationRetrievedSource(BaseModel):
    rank: int
    chunk_id: str
    document_id: str
    chunk_index: int
    matched: bool
    content_preview: str
    retrieval_score: Optional[float] = None
    vector_rank: Optional[int] = None
    keyword_rank: Optional[int] = None
    rerank_score: Optional[float] = None


class RAGEvaluationCaseResult(BaseModel):
    query: str
    hit: bool
    rank: Optional[int]
    reciprocal_rank: float
    recall: float
    latency_ms: float
    expected_sources: List[RAGEvaluationExpectedSource]
    retrieved_sources: List[RAGEvaluationRetrievedSource]


class RAGEvaluationResponse(BaseModel):
    total_cases: int
    topK: int
    recall_at_k: float
    mrr: float
    avg_latency_ms: float
    results: List[RAGEvaluationCaseResult]


# Memory schemas
class MemoryCreate(BaseModel):
    content: str
    category: str = "fact"  # preference, fact, goal, important
    importance: int = 5  # 1-10


class MemoryUpdate(BaseModel):
    content: Optional[str] = None
    importance: Optional[int] = None


class MemoryResponse(BaseModel):
    id: str
    content: str
    category: str
    importance: int
    source: str
    created_at: datetime
    access_count: int
