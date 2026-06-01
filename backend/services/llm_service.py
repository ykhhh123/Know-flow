from inspect import isawaitable
import re
from typing import Any, AsyncIterable, Optional, List

from sqlalchemy.ext.asyncio import AsyncSession

from services.citation_service import REFUSAL_MESSAGE, citation_service
from services.rag_service import rag_service
from services.memory_service import memory_service
from services.model_gateway_service import model_gateway_service


def _to_text(value: Any) -> str:
    if isinstance(value, str):
        return value

    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    return ""


def _get_delta_value(delta: Any, name: str) -> Any:
    if isinstance(delta, dict):
        return delta.get(name)
    return getattr(delta, name, None)


def _normalize_history(history: Optional[List[Any]]) -> list[dict[str, str]]:
    """兼容 pydantic 对象和 dict 类型的历史消息"""
    normalized = []

    for msg in history or []:
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content")
        else:
            role = getattr(msg, "role", None)
            content = getattr(msg, "content", None)

        if role and content:
            normalized.append({"role": role, "content": str(content)})

    return normalized


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text))


def _extract_response_text(response: Any) -> str:
    choices = response.get("choices", []) if isinstance(response, dict) else getattr(response, "choices", [])
    if not choices:
        return ""

    first_choice = choices[0]
    message = first_choice.get("message", {}) if isinstance(first_choice, dict) else getattr(first_choice, "message", None)
    if isinstance(message, dict):
        return _to_text(message.get("content")).strip()

    return _to_text(getattr(message, "content", None)).strip()


class LLMService:
    async def _close_stream_response(self, response: Any) -> None:
        for method_name in ("aclose", "close"):
            close = getattr(response, method_name, None)
            if close is None:
                continue

            result = close()
            if isawaitable(result):
                await result
            return

    async def _rewrite_query_for_retrieval(
        self,
        message: str,
        api_key: str,
        model: str,
        provider: Optional[str],
        base_url: Optional[str],
        provider_api_keys: Optional[dict[str, str]] = None,
        provider_base_urls: Optional[dict[str, str]] = None,
        fallback_chain: Optional[str] = None,
    ) -> str:
        if not _contains_cjk(message):
            return message

        rewrite_messages = [
            {
                "role": "system",
                "content": (
                    "Rewrite the user's question as a concise English search query "
                    "for retrieving evidence from English academic papers. Preserve key "
                    "technical terms, model names, acronyms and method names. Return only "
                    "the rewritten query, with no explanation."
                ),
            },
            {"role": "user", "content": message},
        ]
        try:
            response = await model_gateway_service.completion(
                messages=rewrite_messages,
                api_key=api_key,
                model=model,
                provider=provider,
                base_url=base_url,
                provider_api_keys=provider_api_keys,
                provider_base_urls=provider_base_urls,
                fallback_chain=fallback_chain,
                use_rag=True,
            )
        except Exception as e:
            print(f"[RAG Query Rewrite Warning] {type(e).__name__}: {e}")
            return message

        rewritten = _extract_response_text(response)
        return rewritten or message

    async def stream_chat(
        self,
        message: str,
        api_key: str,
        model: str = "gpt-5-mini",
        use_rag: bool = False,
        use_memory: bool = False,
        use_tools: bool = False,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        provider_api_keys: Optional[dict[str, str]] = None,
        provider_base_urls: Optional[dict[str, str]] = None,
        fallback_chain: Optional[str] = None,
        session: Optional[AsyncSession] = None,
        history: Optional[List[Any]] = None,
        use_local_embedding: bool = False,
        retrieval_mode: str = "hybrid",
        retrieval_filters: Optional[Any] = None,
        use_rerank: bool = True,
        rerank_provider: Optional[str] = None,
        rerank_model: Optional[str] = None,
    ) -> AsyncIterable[str]:
        async for event in self.stream_chat_events(
            message=message,
            api_key=api_key,
            model=model,
            use_rag=use_rag,
            use_memory=use_memory,
            use_tools=use_tools,
            provider=provider,
            base_url=base_url,
            provider_api_keys=provider_api_keys,
            provider_base_urls=provider_base_urls,
            fallback_chain=fallback_chain,
            session=session,
            history=history,
            use_local_embedding=use_local_embedding,
            retrieval_mode=retrieval_mode,
            retrieval_filters=retrieval_filters,
            use_rerank=use_rerank,
            rerank_provider=rerank_provider,
            rerank_model=rerank_model,
        ):
            if event["type"] in {"content", "citation"}:
                yield event.get("content", "")

    async def stream_chat_events(
        self,
        message: str,
        api_key: str,
        model: str = "gpt-5-mini",
        use_rag: bool = False,
        use_memory: bool = False,
        use_tools: bool = False,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        provider_api_keys: Optional[dict[str, str]] = None,
        provider_base_urls: Optional[dict[str, str]] = None,
        fallback_chain: Optional[str] = None,
        session: Optional[AsyncSession] = None,
        history: Optional[List[Any]] = None,
        use_local_embedding: bool = False,
        retrieval_mode: str = "hybrid",
        retrieval_filters: Optional[Any] = None,
        use_rerank: bool = True,
        rerank_provider: Optional[str] = None,
        rerank_model: Optional[str] = None,
    ) -> AsyncIterable[dict[str, Any]]:
        system_parts = [
            "You are KnowFlow, a helpful AI assistant for knowledge work. Answer clearly and accurately."
        ]
        citation_sources: list[dict[str, Any]] = []

        if use_rag and session is not None:
            try:
                retrieval_query = await self._rewrite_query_for_retrieval(
                    message=message,
                    api_key=api_key,
                    model=model,
                    provider=provider,
                    base_url=base_url,
                    provider_api_keys=provider_api_keys,
                    provider_base_urls=provider_base_urls,
                    fallback_chain=fallback_chain,
                )
                chunks = await rag_service.search_similar(
                    query=retrieval_query,
                    api_key=api_key,
                    session=session,
                    limit=5,
                    provider=provider or "openai",
                    base_url=base_url,
                    use_local_embedding=use_local_embedding,
                    retrieval_mode=retrieval_mode,
                    filters=retrieval_filters,
                    use_rerank=use_rerank,
                    rerank_provider=rerank_provider,
                    rerank_model=rerank_model,
                )
                if not chunks:
                    yield {"type": "content", "content": REFUSAL_MESSAGE}
                    return

                citation_sources = citation_service.build_sources(chunks)
                context = citation_service.render_context(citation_sources, chunks)
                if context:
                    system_parts.append(
                        "You are answering from a trusted knowledge base.\n"
                        "Rules:\n"
                        "1. Answer only with facts supported by the retrieved document context.\n"
                        "2. Every factual sentence or bullet must include citation markers like [1] or [2].\n"
                        "3. Use only the source numbers shown in the retrieved context.\n"
                        f"4. If the context is insufficient, answer exactly: {REFUSAL_MESSAGE}\n"
                        "5. Do not use long-term memory as evidence for factual claims.\n\n"
                        "6. Answer in the same language as the user's question unless the user asks otherwise.\n\n"
                        f"Original user question: {message}\n"
                        f"Retrieval query: {retrieval_query}\n\n"
                        "Retrieved document context:\n"
                        + context
                    )
            except Exception as e:
                print(f"[RAG Warning] {type(e).__name__}: {e}")
                yield {"type": "content", "content": REFUSAL_MESSAGE}
                return

        if use_memory and session is not None:
            try:
                memory_context = await memory_service.get_memory_context(
                    query=message,
                    api_key=api_key,
                    session=session,
                    provider="openai",
                    base_url=base_url,
                )
                if memory_context:
                    system_parts.append(
                        "Use the following long-term memory when relevant:\n"
                        + memory_context
                    )
            except Exception as e:
                print(f"[Memory Warning] {type(e).__name__}: {e}")

        messages = [{"role": "system", "content": "\n\n".join(system_parts)}]
        messages.extend(_normalize_history(history))
        messages.append({"role": "user", "content": message})

        gateway_stream = await model_gateway_service.open_stream(
            messages=messages,
            api_key=api_key,
            model=model,
            provider=provider,
            base_url=base_url,
            stream=True,
            provider_api_keys=provider_api_keys,
            provider_base_urls=provider_base_urls,
            fallback_chain=fallback_chain,
            use_rag=use_rag,
        )

        response = gateway_stream.response
        yield {
            "type": "gateway",
            "content": "",
            "payload": {
                "provider": gateway_stream.selection.provider,
                "model": gateway_stream.selection.model,
                "protocol": gateway_stream.selection.protocol,
                "chain": gateway_stream.selection.chain,
                "attempts": list(gateway_stream.selection.attempts),
            },
        }

        assistant_text = ""
        try:
            async for chunk in response:
                try:
                    if isinstance(chunk, dict):
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                    else:
                        choices = getattr(chunk, "choices", [])
                        if not choices:
                            continue
                        delta = getattr(choices[0], "delta", None)

                    reasoning = _to_text(
                        _get_delta_value(delta, "reasoning_content")
                        or _get_delta_value(delta, "reasoning")
                    )
                    if reasoning:
                        yield {
                            "type": "reasoning_content",
                            "content": reasoning,
                        }

                    text = _to_text(_get_delta_value(delta, "content"))
                    if text:
                        assistant_text += text
                        yield {"type": "content", "content": text}

                except Exception as e:
                    print(f"[Stream Parse Warning] {type(e).__name__}: {e}")
        finally:
            await self._close_stream_response(response)

        if citation_sources and assistant_text.strip() != REFUSAL_MESSAGE:
            used_numbers = citation_service.extract_used_numbers(
                assistant_text,
                len(citation_sources),
            )
            if not used_numbers:
                fallback_marker = " [1]"
                assistant_text += fallback_marker
                yield {"type": "content", "content": fallback_marker}

            trailer = citation_service.build_trailer(assistant_text, citation_sources)
            _, payload = citation_service.split_trailer(trailer)
            yield {
                "type": "citation",
                "content": trailer,
                "payload": payload or {"sources": []},
            }


llm_service = LLMService()
