# src/handlers/agent_config_handler.py
from typing import Optional, Dict, Union
from src.models.schemas import AgentConfig, PromptConfig
from src.handlers.index_handler import IndexHandler
from src.services.file_service import FileService
from src.services.cache_service import CacheService


class AgentConfigHandler:
    def __init__(self):
        self.file_service = FileService('/data')
        self.cache_service = CacheService()
        self.index_handler = IndexHandler()

    def get_or_create_config(self, agent_config: Union[AgentConfig, PromptConfig], update_config: bool = False) -> \
    Union[AgentConfig, PromptConfig]:
        """
        Get existing config or create new one if it doesn't exist
        """

        if not agent_config:
            print(f"No agent config provided, create defaults")
            if not agent_config.workspace_id:
                print(f"No workspace ID provided, create defaults")
            agent_config = AgentConfig()

        # Try to get from Modal Dict cache
        if not update_config:
            cached_config = self.cache_service.get(agent_config.workspace_id, agent_config.agent_id)
            if cached_config:
                print("Returning cached config")
                if isinstance(agent_config, PromptConfig):
                    base_dict = cached_config.model_dump()
                    if 'prompt' in base_dict:
                        del base_dict['prompt']
                    return PromptConfig(**base_dict, prompt=agent_config.prompt)
                return cached_config

        # Try to get from file cache
        config_path = f"{agent_config.agent_id}_config.json"
        if not update_config:
            existing_config = self.file_service.load_json(
                agent_config.workspace_id,
                config_path
            )
            if existing_config:
                print("Return config from file")
                existing_config = AgentConfig(**existing_config)
                self.cache_service.set(agent_config.workspace_id, agent_config.agent_id, existing_config)
                return existing_config

        # Create embedding index if background text exists and is long enough
        if (agent_config.character and
                agent_config.character.backstory and
                len(agent_config.character.backstory) > 10000):
            print("Creating embedding index for background text")
            success = self.index_handler.create_and_save_index(
                agent_config.character.backstory,
                agent_config,
            )

        print("Saving config to cache and file")
        # Save to both cache and file
        self.cache_service.set(agent_config.workspace_id, agent_config.agent_id, agent_config)
        self.file_service.save_json(
            agent_config.model_dump(),
            agent_config.workspace_id,
            config_path
        )
        return agent_config

    def update_config(self, agent_config: AgentConfig) -> AgentConfig:
        """Update existing configuration"""
        return self.get_or_create_config(agent_config, update_config=True)

    def get_config(self, workspace_id: str, agent_id: str) -> Optional[AgentConfig]:
        """Retrieve configuration by workspace and agent IDs"""
        # Try Modal Dict cache first
        cached_config = self.cache_service.get(workspace_id, agent_id)
        if cached_config:
            return cached_config

        # Try file cache
        config_data = self.file_service.load_json(
            workspace_id,
            f"{agent_id}_config.json"
        )
        if config_data:
            config = AgentConfig(**config_data)
            self.cache_service.set(workspace_id, agent_id, config)
            return config
        return None

    def delete_config(self, workspace_id: str, agent_id: str) -> bool:
        """Delete configuration"""
        # Remove from Modal Dict cache
        self.cache_service.delete(workspace_id, agent_id)
        # Remove from file storage
        return self.file_service.delete_file(
            workspace_id,
            f"{agent_id}_config.json"
        )

    def clear_cache(self, workspace_id: str, agent_id: str) -> bool:
        """
        Clear the configuration cache
        """
        self.cache_service.clear(workspace_id, agent_id)