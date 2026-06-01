import json
import time
from dataclasses import dataclass
from importlib import import_module
from typing import Any, AsyncIterator, Optional

import httpx

litellm = import_module("litellm")
acompletion = getattr(litellm, "acompletion")


@dataclass(frozen=True)
class ProviderConfig:
    id: str
    protocol: str
    default_base_url: str
    litellm_prefix: Optional[str] = None


@dataclass(frozen=True)
class GatewayRoute:
    provider: str
    model: str


@dataclass(frozen=True)
class GatewaySelection:
    provider: str
    model: str
    protocol: str
    base_url: Optional[str]
    chain: str
    attempts: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class GatewayStream:
    response: Any
    selection: GatewaySelection


class ModelGatewayUnavailable(Exception):
    pass


PROVIDER_CONFIGS: dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        id="openai",
        protocol="openai_compat",
        default_base_url="https://api.openai.com/v1",
        litellm_prefix="openai",
    ),
    "anthropic": ProviderConfig(
        id="anthropic",
        protocol="anthropic_compat",
        default_base_url="https://api.anthropic.com",
        litellm_prefix="anthropic",
    ),
    "google": ProviderConfig(
        id="google",
        protocol="gemini_native",
        default_base_url="https://generativelanguage.googleapis.com/v1beta",
    ),
    "zhipu": ProviderConfig(
        id="zhipu",
        protocol="openai_compat",
        default_base_url="https://open.bigmodel.cn/api/paas/v4",
        litellm_prefix="openai",
    ),
    "minimax": ProviderConfig(
        id="minimax",
        protocol="openai_compat",
        default_base_url="https://api.minimax.io/v1",
        litellm_prefix="openai",
    ),
    # Existing non-target providers continue to work through the same gateway.
    "moonshot": ProviderConfig(
        id="moonshot",
        protocol="openai_compat",
        default_base_url="https://api.moonshot.cn/v1",
        litellm_prefix="openai",
    ),
    "deepseek": ProviderConfig(
        id="deepseek",
        protocol="openai_compat",
        default_base_url="https://api.deepseek.com/v1",
        litellm_prefix="openai",
    ),
    "alibaba": ProviderConfig(
        id="alibaba",
        protocol="openai_compat",
        default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        litellm_prefix="openai",
    ),
    "cohere": ProviderConfig(
        id="cohere",
        protocol="openai_compat",
        default_base_url="https://api.cohere.ai/v1",
        litellm_prefix="openai",
    ),
    "mistral": ProviderConfig(
        id="mistral",
        protocol="openai_compat",
        default_base_url="https://api.mistral.ai/v1",
        litellm_prefix="openai",
    ),
}


FALLBACK_CHAINS: dict[str, tuple[GatewayRoute, ...]] = {
    "chat-openai-premium": (
        GatewayRoute("openai", "gpt-5.4"),
        GatewayRoute("anthropic", "claude-opus-4-6"),
        GatewayRoute("google", "gemini-3.1-pro-preview"),
        GatewayRoute("zhipu", "glm-5"),
        GatewayRoute("minimax", "MiniMax-Text-01"),
    ),
    "chat-openai-balanced": (
        GatewayRoute("openai", "gpt-5-mini"),
        GatewayRoute("anthropic", "claude-sonnet-4-6"),
        GatewayRoute("google", "gemini-3-flash-preview"),
        GatewayRoute("zhipu", "glm-4.7"),
        GatewayRoute("minimax", "MiniMax-Text-01"),
    ),
    "chat-openai-fast": (
        GatewayRoute("openai", "gpt-5-nano"),
        GatewayRoute("google", "gemini-3.1-flash-lite-preview"),
        GatewayRoute("zhipu", "glm-4.7-flash"),
        GatewayRoute("anthropic", "claude-haiku-4-5"),
        GatewayRoute("minimax", "MiniMax-Text-01"),
    ),
    "chat-anthropic-premium": (
        GatewayRoute("anthropic", "claude-opus-4-6"),
        GatewayRoute("openai", "gpt-5.4"),
        GatewayRoute("google", "gemini-3.1-pro-preview"),
        GatewayRoute("zhipu", "glm-5"),
        GatewayRoute("minimax", "MiniMax-Text-01"),
    ),
    "chat-anthropic-balanced": (
        GatewayRoute("anthropic", "claude-sonnet-4-6"),
        GatewayRoute("openai", "gpt-5-mini"),
        GatewayRoute("google", "gemini-3-flash-preview"),
        GatewayRoute("zhipu", "glm-4.7"),
        GatewayRoute("minimax", "MiniMax-Text-01"),
    ),
    "chat-anthropic-fast": (
        GatewayRoute("anthropic", "claude-haiku-4-5"),
        GatewayRoute("google", "gemini-3.1-flash-lite-preview"),
        GatewayRoute("openai", "gpt-5-nano"),
        GatewayRoute("zhipu", "glm-4.7-flash"),
        GatewayRoute("minimax", "MiniMax-Text-01"),
    ),
    "chat-gemini-premium": (
        GatewayRoute("google", "gemini-3.1-pro-preview"),
        GatewayRoute("openai", "gpt-5.4"),
        GatewayRoute("anthropic", "claude-opus-4-6"),
        GatewayRoute("zhipu", "glm-5"),
        GatewayRoute("minimax", "MiniMax-Text-01"),
    ),
    "chat-gemini-balanced": (
        GatewayRoute("google", "gemini-3-flash-preview"),
        GatewayRoute("openai", "gpt-5-mini"),
        GatewayRoute("anthropic", "claude-sonnet-4-6"),
        GatewayRoute("zhipu", "glm-4.7"),
        GatewayRoute("minimax", "MiniMax-Text-01"),
    ),
    "chat-gemini-fast": (
        GatewayRoute("google", "gemini-3.1-flash-lite-preview"),
        GatewayRoute("openai", "gpt-5-nano"),
        GatewayRoute("anthropic", "claude-haiku-4-5"),
        GatewayRoute("zhipu", "glm-4.7-flash"),
        GatewayRoute("minimax", "MiniMax-Text-01"),
    ),
    "chat-zhipu-premium": (
        GatewayRoute("zhipu", "glm-5"),
        GatewayRoute("openai", "gpt-5.4"),
        GatewayRoute("anthropic", "claude-opus-4-6"),
        GatewayRoute("google", "gemini-3.1-pro-preview"),
        GatewayRoute("minimax", "MiniMax-Text-01"),
    ),
    "chat-zhipu-balanced": (
        GatewayRoute("zhipu", "glm-4.7"),
        GatewayRoute("openai", "gpt-5-mini"),
        GatewayRoute("anthropic", "claude-sonnet-4-6"),
        GatewayRoute("google", "gemini-3-flash-preview"),
        GatewayRoute("minimax", "MiniMax-Text-01"),
    ),
    "chat-zhipu-fast": (
        GatewayRoute("zhipu", "glm-4.7-flash"),
        GatewayRoute("google", "gemini-3.1-flash-lite-preview"),
        GatewayRoute("openai", "gpt-5-nano"),
        GatewayRoute("anthropic", "claude-haiku-4-5"),
        GatewayRoute("minimax", "MiniMax-Text-01"),
    ),
    "chat-minimax-premium": (
        GatewayRoute("minimax", "MiniMax-Text-01"),
        GatewayRoute("openai", "gpt-5.4"),
        GatewayRoute("anthropic", "claude-opus-4-6"),
        GatewayRoute("google", "gemini-3.1-pro-preview"),
        GatewayRoute("zhipu", "glm-5"),
    ),
    "chat-minimax-balanced": (
        GatewayRoute("minimax", "MiniMax-Text-01"),
        GatewayRoute("openai", "gpt-5-mini"),
        GatewayRoute("anthropic", "claude-sonnet-4-6"),
        GatewayRoute("google", "gemini-3-flash-preview"),
        GatewayRoute("zhipu", "glm-4.7"),
    ),
    "chat-minimax-fast": (
        GatewayRoute("minimax", "MiniMax-Text-01"),
        GatewayRoute("google", "gemini-3.1-flash-lite-preview"),
        GatewayRoute("openai", "gpt-5-nano"),
        GatewayRoute("anthropic", "claude-haiku-4-5"),
        GatewayRoute("zhipu", "glm-4.7-flash"),
    ),
    "rag-long-context": (
        GatewayRoute("google", "gemini-3.1-pro-preview"),
        GatewayRoute("openai", "gpt-5-mini"),
        GatewayRoute("anthropic", "claude-sonnet-4-6"),
        GatewayRoute("zhipu", "glm-4.7"),
        GatewayRoute("minimax", "MiniMax-Text-01"),
    ),
    "low-latency": (
        GatewayRoute("openai", "gpt-5-nano"),
        GatewayRoute("google", "gemini-3.1-flash-lite-preview"),
        GatewayRoute("zhipu", "glm-4.7-flash"),
        GatewayRoute("anthropic", "claude-haiku-4-5"),
        GatewayRoute("minimax", "MiniMax-Text-01"),
    ),
}


class ModelGatewayService:
    circuit_ttl_seconds = 60

    def __init__(self) -> None:
        self._circuit_open_until: dict[str, float] = {}

    def _select_chain(self, provider: Optional[str], model: str, use_rag: bool) -> str:
        normalized_provider = (provider or self._infer_provider(model)).lower()
        if use_rag:
            return "rag-long-context"

        lowered_model = model.lower()
        if any(token in lowered_model for token in ("nano", "flash", "haiku")):
            tier = "fast"
        elif any(token in lowered_model for token in ("pro", "opus", "max", "5.4")):
            tier = "premium"
        else:
            tier = "balanced"

        chain_name = f"chat-{normalized_provider}-{tier}"
        if chain_name in FALLBACK_CHAINS:
            return chain_name
        return "chat-openai-balanced"

    def _infer_provider(self, model: str) -> str:
        lowered = model.lower()
        if lowered.startswith("claude"):
            return "anthropic"
        if lowered.startswith("gemini"):
            return "google"
        if lowered.startswith("glm"):
            return "zhipu"
        if lowered.startswith("minimax"):
            return "minimax"
        return "openai"

    def _build_routes(
        self,
        provider: Optional[str],
        model: str,
        use_rag: bool,
        fallback_chain: Optional[str],
    ) -> tuple[str, list[GatewayRoute]]:
        chain_name = fallback_chain or self._select_chain(provider, model, use_rag)
        selected_route = GatewayRoute((provider or self._infer_provider(model)).lower(), model)
        routes = [selected_route]
        seen = {(selected_route.provider, selected_route.model)}

        for route in FALLBACK_CHAINS.get(chain_name, FALLBACK_CHAINS["chat-openai-balanced"]):
            key = (route.provider, route.model)
            if key not in seen:
                routes.append(route)
                seen.add(key)

        return chain_name, routes

    def _clean_map(self, value: Optional[dict[str, str]]) -> dict[str, str]:
        if not value:
            return {}
        return {
            str(key).strip().lower(): str(item).strip()
            for key, item in value.items()
            if str(key).strip() and str(item).strip()
        }

    def _credential_maps(
        self,
        api_key: str,
        provider: Optional[str],
        provider_api_keys: Optional[dict[str, str]],
        provider_base_urls: Optional[dict[str, str]],
        base_url: Optional[str],
    ) -> tuple[dict[str, str], dict[str, str]]:
        keys = self._clean_map(provider_api_keys)
        urls = self._clean_map(provider_base_urls)

        normalized_provider = (provider or "").strip().lower()
        if api_key and normalized_provider and normalized_provider not in keys:
            keys[normalized_provider] = api_key.strip()
        if base_url and normalized_provider and normalized_provider not in urls:
            urls[normalized_provider] = base_url.strip()

        return keys, urls

    def _is_circuit_open(self, provider: str) -> bool:
        until = self._circuit_open_until.get(provider, 0)
        if until <= time.monotonic():
            self._circuit_open_until.pop(provider, None)
            return False
        return True

    def _mark_failure(self, provider: str) -> None:
        self._circuit_open_until[provider] = time.monotonic() + self.circuit_ttl_seconds

    def _mark_success(self, provider: str) -> None:
        self._circuit_open_until.pop(provider, None)

    def _model_for_litellm(self, route: GatewayRoute, config: ProviderConfig) -> str:
        if "/" in route.model:
            return route.model
        if config.litellm_prefix:
            return f"{config.litellm_prefix}/{route.model}"
        return route.model

    def _kwargs_for_litellm(
        self,
        route: GatewayRoute,
        config: ProviderConfig,
        messages: list[dict[str, str]],
        api_key: str,
        base_url: Optional[str],
        stream: bool,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model_for_litellm(route, config),
            "messages": messages,
            "api_key": api_key,
            "stream": stream,
        }
        if base_url:
            kwargs["base_url"] = base_url
        return kwargs

    def _gemini_url(self, base_url: str, model: str, stream: bool, api_key: str) -> str:
        normalized = base_url.rstrip("/")
        if normalized.endswith("/v1") or normalized.endswith("/v1beta"):
            api_base = normalized
        else:
            api_base = f"{normalized}/v1beta"
        method = "streamGenerateContent" if stream else "generateContent"
        suffix = "?alt=sse" if stream else ""
        return f"{api_base}/models/{model}:{method}{suffix}&key={api_key}" if suffix else f"{api_base}/models/{model}:{method}?key={api_key}"

    def _messages_to_gemini_contents(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[Optional[str], list[dict[str, Any]]]:
        system_parts: list[str] = []
        contents: list[dict[str, Any]] = []

        for message in messages:
            role = message.get("role", "user")
            content = str(message.get("content", ""))
            if not content:
                continue
            if role == "system":
                system_parts.append(content)
                continue
            contents.append(
                {
                    "role": "model" if role == "assistant" else "user",
                    "parts": [{"text": content}],
                }
            )

        return "\n\n".join(system_parts) if system_parts else None, contents

    async def _gemini_completion(
        self,
        route: GatewayRoute,
        messages: list[dict[str, str]],
        api_key: str,
        base_url: str,
        stream: bool,
    ) -> Any:
        system_instruction, contents = self._messages_to_gemini_contents(messages)
        payload: dict[str, Any] = {"contents": contents}
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        url = self._gemini_url(base_url, route.model, stream, api_key)
        if stream:
            return _GeminiStream(url=url, payload=payload)

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        text = self._extract_gemini_text(data)
        return {"choices": [{"message": {"content": text}}]}

    def _extract_gemini_text(self, data: dict[str, Any]) -> str:
        parts = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [])
        )
        return "".join(part.get("text", "") for part in parts if isinstance(part, dict))

    async def completion(
        self,
        *,
        messages: list[dict[str, str]],
        api_key: str,
        model: str,
        provider: Optional[str],
        base_url: Optional[str],
        provider_api_keys: Optional[dict[str, str]] = None,
        provider_base_urls: Optional[dict[str, str]] = None,
        fallback_chain: Optional[str] = None,
        use_rag: bool = False,
    ) -> Any:
        stream = await self.open_stream(
            messages=messages,
            api_key=api_key,
            model=model,
            provider=provider,
            base_url=base_url,
            stream=False,
            provider_api_keys=provider_api_keys,
            provider_base_urls=provider_base_urls,
            fallback_chain=fallback_chain,
            use_rag=use_rag,
        )
        return stream.response

    async def open_stream(
        self,
        *,
        messages: list[dict[str, str]],
        api_key: str,
        model: str,
        provider: Optional[str],
        base_url: Optional[str],
        stream: bool,
        provider_api_keys: Optional[dict[str, str]] = None,
        provider_base_urls: Optional[dict[str, str]] = None,
        fallback_chain: Optional[str] = None,
        use_rag: bool = False,
    ) -> GatewayStream:
        keys, urls = self._credential_maps(
            api_key=api_key,
            provider=provider,
            provider_api_keys=provider_api_keys,
            provider_base_urls=provider_base_urls,
            base_url=base_url,
        )
        chain_name, routes = self._build_routes(
            provider=provider,
            model=model,
            use_rag=use_rag,
            fallback_chain=fallback_chain,
        )
        attempts: list[dict[str, str]] = []

        for route in routes:
            config = PROVIDER_CONFIGS.get(route.provider)
            if not config:
                attempts.append(
                    {
                        "provider": route.provider,
                        "model": route.model,
                        "status": "unsupported_provider",
                    }
                )
                continue

            if self._is_circuit_open(route.provider):
                attempts.append(
                    {
                        "provider": route.provider,
                        "model": route.model,
                        "status": "circuit_open",
                    }
                )
                continue

            route_api_key = keys.get(route.provider)
            if not route_api_key:
                attempts.append(
                    {
                        "provider": route.provider,
                        "model": route.model,
                        "status": "missing_api_key",
                    }
                )
                continue

            route_base_url = urls.get(route.provider) or config.default_base_url

            try:
                if config.protocol == "gemini_native":
                    response = await self._gemini_completion(
                        route=route,
                        messages=messages,
                        api_key=route_api_key,
                        base_url=route_base_url,
                        stream=stream,
                    )
                    if stream and isinstance(response, _GeminiStream):
                        await response.start()
                else:
                    kwargs = self._kwargs_for_litellm(
                        route=route,
                        config=config,
                        messages=messages,
                        api_key=route_api_key,
                        base_url=route_base_url,
                        stream=stream,
                    )
                    response = await acompletion(**kwargs)

                self._mark_success(route.provider)
                attempts.append(
                    {
                        "provider": route.provider,
                        "model": route.model,
                        "status": "selected",
                    }
                )
                return GatewayStream(
                    response=response,
                    selection=GatewaySelection(
                        provider=route.provider,
                        model=route.model,
                        protocol=config.protocol,
                        base_url=route_base_url,
                        chain=chain_name,
                        attempts=tuple(attempts),
                    ),
                )
            except Exception as exc:
                self._mark_failure(route.provider)
                attempts.append(
                    {
                        "provider": route.provider,
                        "model": route.model,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        raise ModelGatewayUnavailable(
            "All model gateway routes failed or were unavailable: "
            + json.dumps(attempts, ensure_ascii=False)
        )


class _GeminiStream:
    def __init__(self, *, url: str, payload: dict[str, Any]) -> None:
        self.url = url
        self.payload = payload
        self._client: Optional[httpx.AsyncClient] = None
        self._response: Optional[httpx.Response] = None
        self._aiter: Optional[AsyncIterator[str]] = None

    def __aiter__(self) -> "_GeminiStream":
        return self

    async def start(self) -> None:
        if self._aiter is not None:
            return
        self._client = httpx.AsyncClient(timeout=None)
        request = self._client.build_request("POST", self.url, json=self.payload)
        self._response = await self._client.send(request, stream=True)
        self._response.raise_for_status()
        self._aiter = self._response.aiter_lines()

    async def __anext__(self) -> dict[str, Any]:
        if self._aiter is None:
            await self.start()

        assert self._aiter is not None
        async for line in self._aiter:
            if not line.startswith("data:"):
                continue
            raw = line.removeprefix("data:").strip()
            if not raw or raw == "[DONE]":
                continue
            data = json.loads(raw)
            text = ModelGatewayService()._extract_gemini_text(data)
            if text:
                return {"choices": [{"delta": {"content": text}}]}

        await self.aclose()
        raise StopAsyncIteration

    async def aclose(self) -> None:
        if self._response is not None:
            await self._response.aclose()
            self._response = None
        if self._client is not None:
            await self._client.aclose()
            self._client = None


model_gateway_service = ModelGatewayService()
