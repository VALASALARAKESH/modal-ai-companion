# src/handlers/llm_handler.py
from typing import Generator, Dict, List
from src.models.schemas import AgentConfig
import os,json



class LLMHandler:
    def __init__(self):
        from openai import OpenAI
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
        from openai import OpenAI
        """Initialize OpenAI client with provider-specific configuration."""
        config = self._provider_configs.get(provider, self._provider_configs["openai"])
        #print("*** LLM CONFIG ***: ",config)
        if config["api_key_env"] not in os.environ:
            raise ValueError(f"Missing API key for provider {provider}")

        return OpenAI(
            base_url=config["base_url"],
            api_key=os.environ[config["api_key_env"]]
        )

    def generate(self, 
                messages: List[Dict], 
                agent_config: AgentConfig,temperature=None,model=None,provider=None,stop_words=None) -> Generator[str, None, None]:
        """Generate text using the configured LLM provider."""
        if not agent_config.llm_config.provider:
            raise ValueError("LLM provider not specified in config")
        
        together_ai_models = ['meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo','NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO','meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo','mistralai/Mixtral-8x22B-Instruct-v0.1','Gryphe/MythoMax-L2-13b']
        

        if agent_config.llm_config.model in together_ai_models:
            provider = "togetherai"
            
        if provider and provider in together_ai_models:
            provider = "togetherai"
            
        self.client = self.initialize_client(provider or agent_config.llm_config.provider)
        
        try:
            response = self.client.chat.completions.create(
                model=model or agent_config.llm_config.model,
                messages=messages,
                stream=True,
                temperature=temperature or agent_config.llm_config.temperature,
                max_tokens=agent_config.llm_config.max_tokens,
                stop= stop_words or agent_config.llm_config.stop
            )

            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            print(f"Error during generation: {str(e)}")
            yield f"Error: {str(e)}"

    def _get_provider_config(self, provider: str) -> tuple[str, str]:
        """Get provider configuration (base URL and API key)."""
        config = self._provider_configs.get(provider, self._provider_configs["openai"])
        #if config["api_key_env"] not in os.environ:
            #raise ValueError(f"Missing API key for provider {provider}")
        return config["base_url"], os.environ[config["api_key_env"]]