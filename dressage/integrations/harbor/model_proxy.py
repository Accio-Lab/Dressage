from typing import Any

from blackbox_server.proxy.rollout_llm_proxy import RolloutLLMProxy


class GatewaySeedProxy(RolloutLLMProxy):
    """Carry the gateway's request seed through conversion to Chat Completions."""

    def _openai_responses_to_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        chat = super()._openai_responses_to_chat_completion(payload)
        if "seed" in payload:
            chat["seed"] = payload["seed"]
        return chat

    def _anthropic_messages_to_openai_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        chat = super()._anthropic_messages_to_openai_chat_completion(payload)
        if "seed" in payload:
            chat["seed"] = payload["seed"]
        return chat
