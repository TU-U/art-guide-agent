from services.vlm_service import VLMService


class LLMService(VLMService):
    """Compatibility wrapper. The runtime implementation is now the configured VLM provider."""


__all__ = ["LLMService", "VLMService"]
