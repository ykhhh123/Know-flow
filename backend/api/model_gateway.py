from fastapi import APIRouter

from services.model_gateway_service import FALLBACK_CHAINS, PROVIDER_CONFIGS

router = APIRouter(prefix="/api/model-gateway", tags=["model-gateway"])


@router.get("/chains")
async def list_fallback_chains():
    return {
        "providers": {
            provider_id: {
                "protocol": config.protocol,
                "defaultBaseUrl": config.default_base_url,
            }
            for provider_id, config in PROVIDER_CONFIGS.items()
        },
        "chains": {
            name: [
                {"provider": route.provider, "model": route.model}
                for route in routes
            ]
            for name, routes in FALLBACK_CHAINS.items()
        },
    }
