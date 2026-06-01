from services.model_gateway_service import FALLBACK_CHAINS, PROVIDER_CONFIGS


def test_model_gateway_has_target_provider_protocols():
    assert len(FALLBACK_CHAINS) == 17
    assert PROVIDER_CONFIGS["openai"].protocol == "openai_compat"
    assert PROVIDER_CONFIGS["anthropic"].protocol == "anthropic_compat"
    assert PROVIDER_CONFIGS["google"].protocol == "gemini_native"
    assert PROVIDER_CONFIGS["zhipu"].protocol == "openai_compat"
    assert PROVIDER_CONFIGS["minimax"].protocol == "openai_compat"
