import asyncio
import json
import traceback
import uuid

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from models.schemas import ChatRequest, RAGChatRequest
from models.database import Message
from services.llm_service import llm_service
from services.conversation_service import conversation_service
from services.citation_service import citation_service
from services.memory_service import memory_service
from services.tools_service import tools_service
from services.rag_service import RetrievalFilters
from core.database import get_session, async_session_maker

router = APIRouter(prefix="/api", tags=["chat"])

STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _sse_frame(event_type: str, payload: dict | None = None) -> str:
    data = {"type": event_type}
    if payload:
        data.update(payload)
    return (
        f"event: {event_type}\n"
        f"data: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"
    )


def _to_retrieval_filters(filters) -> RetrievalFilters | None:
    if filters is None:
        return None

    return RetrievalFilters(
        knowledge_bases=filters.knowledge_bases,
        document_types=filters.document_types,
        created_after=filters.created_after,
        created_before=filters.created_before,
        tags=filters.tags,
        document_ids=filters.document_ids,
    )


def _is_rewrite_followup(message: str) -> bool:
    normalized = " ".join(message.strip().lower().split())
    if not normalized:
        return False

    rewrite_patterns = [
        "用中文回答",
        "中文回答",
        "翻译成中文",
        "翻译为中文",
        "转成中文",
        "改成中文",
        "用中文",
        "translate to chinese",
        "translate into chinese",
        "in chinese",
        "answer in chinese",
        "rewrite in chinese",
    ]
    return any(pattern in normalized for pattern in rewrite_patterns)


def _has_prior_assistant_message(history) -> bool:
    return any(message.get("role") == "assistant" for message in history)


def _build_rewrite_message(message: str) -> str:
    return (
        "请只基于上一条助手回答完成用户的改写/翻译请求。"
        "如果用户要求中文，就把上一条助手回答翻译成中文。"
        "必须保留原回答中的引用编号（例如 [1]、[2]），不要新增事实，不要重新检索或扩展内容。\n\n"
        f"用户请求：{message}"
    )


async def _get_latest_assistant_citation_payload(
    session: AsyncSession,
    conversation_id: str,
) -> dict | None:
    result = await session.execute(
        select(Message)
        .options(selectinload(Message.citations))
        .where(
            Message.conversation_id == uuid.UUID(conversation_id),
            Message.role == "assistant",
        )
        .order_by(desc(Message.created_at))
        .limit(1)
    )
    message = result.scalar_one_or_none()
    if not message or not message.citations:
        return None

    return {
        "sources": [
            citation_service.citation_to_dict(citation)
            for citation in message.citations
        ]
    }


async def _extract_memories_background(
    conversation_id: str,
    api_key: str,
    provider: str = "openai",
    base_url: str = None
):
    """Background task to extract memories from conversation"""
    try:
        async with async_session_maker() as session:
            # Get recent conversation messages
            messages = await conversation_service.get_recent_messages(
                session=session,
                conversation_id=conversation_id,
                limit=10  # Last 10 messages
            )

            if len(messages) < 2:  # Need at least user + assistant
                return

            # Format conversation text
            conversation_text = "\n".join([
                f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}"
                for m in messages
            ])

            # Extract memories
            extracted = await memory_service.extract_memories_from_conversation(
                conversation_text=conversation_text,
                api_key=api_key,
                session=session,
                provider=provider,
                base_url=base_url
            )

            if extracted:
                print(f"[Auto Memory] Extracted {len(extracted)} memories from conversation {conversation_id}")

    except Exception as e:
        print(f"[Auto Memory Error] {type(e).__name__}: {e}")


@router.post("/chat")
async def chat(
    request: ChatRequest,
    http_request: Request,
    session: AsyncSession = Depends(get_session)
):
    """Basic chat with conversation persistence"""
    # 如果指定了 conversation_id，使用持久化对话的上下文
    optimized_context = []
    if request.conversationId:
        optimized_context = await conversation_service.get_optimized_context(
            session=session,
            conversation_id=request.conversationId,
            current_model=request.model
        )
        # 保存用户消息
        await conversation_service.add_message(
            session=session,
            conversation_id=request.conversationId,
            role="user",
            content=request.message,
            model=request.model
        )

    async def generate():
        assistant_content = ""
        completed = False
        try:
            yield _sse_frame("start")
            # 使用优化的上下文（包含摘要+最近消息）
            async for event in llm_service.stream_chat_events(
                message=request.message,
                api_key=request.apiKey,
                model=request.model,
                use_rag=False,
                use_memory=False,
                provider=request.provider,
                base_url=request.baseUrl,
                provider_api_keys=request.providerApiKeys,
                provider_base_urls=request.providerBaseUrls,
                fallback_chain=request.fallbackChain,
                session=session,
                history=optimized_context
            ):
                if await http_request.is_disconnected():
                    return

                event_type = event["type"]
                if event_type == "content":
                    content = event.get("content", "")
                    assistant_content += content
                    yield _sse_frame("content", {"content": content})
                elif event_type == "reasoning_content":
                    yield _sse_frame(
                        "reasoning_content",
                        {"content": event.get("content", "")},
                    )
                elif event_type == "gateway":
                    yield _sse_frame(
                        "gateway",
                        {"gateway": event.get("payload", {})},
                    )

            completed = True
            yield _sse_frame("done")

            # 保存助手回复
            if completed and request.conversationId:
                assistant_content, _ = citation_service.split_trailer(
                    assistant_content
                )
                await conversation_service.add_message(
                    session=session,
                    conversation_id=request.conversationId,
                    role="assistant",
                    content=assistant_content,
                    model=request.model
                )

                # 检查是否需要生成摘要
                should_summary = await conversation_service.should_generate_summary(
                    session=session,
                    conversation_id=request.conversationId
                )
                if should_summary:
                    # 异步生成摘要（不阻塞响应）
                    import asyncio
                    asyncio.create_task(
                        conversation_service.generate_summary(
                            session=session,
                            conversation_id=request.conversationId,
                            api_key=request.apiKey
                        )
                    )

        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[Chat Error] {type(e).__name__}: {e}")
            traceback.print_exc()
            yield _sse_frame("error", {"message": str(e)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=STREAM_HEADERS,
    )


@router.post("/chat/rag")
async def chat_with_rag(
    request: RAGChatRequest,
    http_request: Request,
    session: AsyncSession = Depends(get_session)
):
    """Chat with RAG, memory and conversation persistence"""
    # 获取优化的上下文
    optimized_context = []
    if request.conversationId:
        optimized_context = await conversation_service.get_optimized_context(
            session=session,
            conversation_id=request.conversationId,
            current_model=request.model
        )
        # 保存用户消息
        await conversation_service.add_message(
            session=session,
            conversation_id=request.conversationId,
            role="user",
            content=request.message,
            model=request.model
        )

    async def generate():
        assistant_content = ""
        citation_payload = None
        completed = False
        try:
            yield _sse_frame("start")
            is_rewrite_followup = (
                _is_rewrite_followup(request.message)
                and _has_prior_assistant_message(optimized_context)
            )
            inherited_citation_payload = (
                await _get_latest_assistant_citation_payload(
                    session=session,
                    conversation_id=request.conversationId,
                )
                if is_rewrite_followup and request.conversationId
                else None
            )
            effective_message = (
                _build_rewrite_message(request.message)
                if is_rewrite_followup
                else request.message
            )
            async for event in llm_service.stream_chat_events(
                message=effective_message,
                api_key=request.apiKey,
                model=request.model,
                use_rag=request.use_rag and not is_rewrite_followup,
                use_memory=request.use_memory and not is_rewrite_followup,
                use_tools=request.use_tools,
                provider=request.provider,
                base_url=request.baseUrl,
                provider_api_keys=request.providerApiKeys,
                provider_base_urls=request.providerBaseUrls,
                fallback_chain=request.fallbackChain,
                session=session,
                history=optimized_context,
                use_local_embedding=request.use_local_embedding,
                retrieval_mode=request.retrieval_mode,
                retrieval_filters=_to_retrieval_filters(request.filters),
                use_rerank=request.use_rerank,
                rerank_provider=request.rerank_provider,
                rerank_model=request.rerank_model,
            ):
                if await http_request.is_disconnected():
                    return

                event_type = event["type"]
                if event_type == "content":
                    content = event.get("content", "")
                    assistant_content += content
                    yield _sse_frame("content", {"content": content})
                elif event_type == "reasoning_content":
                    yield _sse_frame(
                        "reasoning_content",
                        {"content": event.get("content", "")},
                    )
                elif event_type == "gateway":
                    yield _sse_frame(
                        "gateway",
                        {"gateway": event.get("payload", {})},
                    )
                elif event_type == "citation":
                    citation_payload = event.get("payload")
                    yield _sse_frame(
                        "citation",
                        {"citations": (citation_payload or {}).get("sources", [])},
                    )

            completed = True
            yield _sse_frame("done")

            # 保存助手回复
            if completed and request.conversationId:
                assistant_content, trailer_payload = citation_service.split_trailer(
                    assistant_content
                )
                citation_payload = (
                    citation_payload
                    or trailer_payload
                    or inherited_citation_payload
                )
                assistant_message = await conversation_service.add_message(
                    session=session,
                    conversation_id=request.conversationId,
                    role="assistant",
                    content=assistant_content,
                    model=request.model
                )
                await citation_service.save_message_citations(
                    session=session,
                    message_id=assistant_message.id,
                    payload=citation_payload,
                )

                # 检查是否需要生成摘要
                should_summary = await conversation_service.should_generate_summary(
                    session=session,
                    conversation_id=request.conversationId
                )
                if should_summary:
                    asyncio.create_task(
                        conversation_service.generate_summary(
                            session=session,
                            conversation_id=request.conversationId,
                            api_key=request.apiKey
                        )
                    )

                # 自动提取记忆（当开启记忆功能时）
                if request.use_memory:
                    asyncio.create_task(
                        _extract_memories_background(
                            conversation_id=request.conversationId,
                            api_key=request.apiKey,
                            provider=request.model.split("-")[0] if "-" in request.model else "openai",
                            base_url=request.baseUrl
                        )
                    )

        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[RAG Chat Error] {type(e).__name__}: {e}")
            traceback.print_exc()
            yield _sse_frame("error", {"message": str(e)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=STREAM_HEADERS,
    )


@router.get("/tools")
async def list_tools():
    """List available tools for agent"""
    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
            for tool in tools_service.tools.values()
        ]
    }
