# src/handlers/llm_handler.py
from typing import Generator, Dict, List
from src.models.schemas import AgentConfig
import os, json


class LLMHandler:
    def __init__(self):
        import os
        self.client = None
        self._provider_configs = {
            "deepinfra": {
                "base_url": "https://api.deepinfra.com/v1/openai",
                "api_key_env": "DEEP_INFRA_API_KEY"
            },
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "api_key_env": "OPENAI_API_KEY"
            },
            "togetherai": {
                "base_url": "https://api.together.ai/v1",
                "api_key_env": "TOGETHERAI_API_KEY"
            }
        }

    def initialize_client(self, provider: str):
        from together import Together
        """Initialize OpenAI client with provider-specific configuration."""
        provider = provider.strip()
        config = self._provider_configs.get(provider, self._provider_configs["openai"])
        if config["api_key_env"] not in os.environ:
            raise ValueError(f"Missing API key for provider {provider}")
        return Together(
            api_key=os.environ[config["api_key_env"]],
            base_url=config["base_url"]
        )

    def generate(self,
                 messages: List[Dict],
                 agent_config: AgentConfig,
                 temperature: float = None,
                 model: str = "",
                 provider: str = "",
                 stop_words: List[str] = None,
                 max_tokens: int = None,
                 frequency_penalty: float = None,
                 presence_penalty: float = None,
                 repetition_penalty: float = None,
                 top_p: float = None,
                 top_k: int = None,
                 min_p: float = None) -> Generator[str, None, None]:
        """Generate text using the configured LLM provider."""
        if not agent_config.llm_config.provider:
            raise ValueError("LLM provider not specified in config")

        together_ai_models = ['meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo',
                              'NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO',
                              'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo', 'mistralai/Mixtral-8x22B-Instruct-v0.1',
                              'Gryphe/MythoMax-L2-13b']
        openai_models = ['gpt4-o-mini', 'gpt4-o']
        deepinfra_models = ['NousResearch/Hermes-3-Llama-3.1-405B', 'Sao10K/L3.3-70B-Euryale-v2.3',
                            'Sao10K/L3.1-70B-Euryale-v2.2', 'mistralai/Mistral-Small-24B-Instruct-2501',
                            'nvidia/Llama-3.1-Nemotron-70B-Instruct', 'meta-llama/Llama-3.3-70B-Instruct-Turbo',
                            'meta-llama/Meta-Llama-3.1-405B-Instruct', 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo']

        extra_body = {}

        if agent_config.llm_config.model in together_ai_models:
            provider = "togetherai"
        if agent_config.llm_config.model in openai_models:
            provider = "openai"
        if agent_config.llm_config.model in deepinfra_models:
            provider = "deepinfra"

        cleaned_messages = []
        for msg in messages:
            if 'content' in msg:
                cleaned_messages.append({'role': msg['role'], 'content': msg['content']})

        payload = {
            "model": model or agent_config.llm_config.model,
            "messages": cleaned_messages,
            "stream": True,
            "temperature": temperature or agent_config.llm_config.temperature,
            "max_tokens": max_tokens or agent_config.llm_config.max_tokens,
            "stop": stop_words if stop_words is not None else agent_config.llm_config.stop,
            "frequency_penalty": frequency_penalty or agent_config.llm_config.frequency_penalty,
            "presence_penalty": presence_penalty or agent_config.llm_config.presence_penalty,
            "top_p": top_p or agent_config.llm_config.top_p,
        }

        if provider == 'deepinfra' or provider == 'togetherai':
            # Use None check instead of logical OR to handle zero values correctly
            payload['min_p'] = min_p if min_p is not None else agent_config.llm_config.min_p
            payload[
                'repetition_penalty'] = repetition_penalty if repetition_penalty is not None else agent_config.llm_config.repetition_penalty

        provider_name = provider or agent_config.llm_config.provider
        # print(f"Calling chat.completions.create with provider: {provider_name}, model: {payload.get('model', 'unknown')}")

        try:
            self.client = self.initialize_client(provider_name)
            # print(f"Client initialized: {type(self.client).__name__}")

            # Deep debug of payload
            # print("Payload keys:", list(payload.keys()))

            response = self.client.chat.completions.create(**payload)
            for chunk in response:
                if len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            error_message = f"Error in run method: {type(e).__name__}: {str(e)}"
            print(error_message)

            # Print stack trace for debugging
            import traceback
            print("Stack trace:")
            traceback.print_exc()

            yield f"Error: {error_message}"

    def _get_provider_config(self, provider: str) -> tuple[str, str]:
        """Get provider configuration (base URL and API key)."""
        config = self._provider_configs.get(provider, self._provider_configs["openai"])
        # if config["api_key_env"] not in os.environ:
        # raise ValueError(f"Missing API key for provider {provider}")
        return config["base_url"], os.environ[config["api_key_env"]]