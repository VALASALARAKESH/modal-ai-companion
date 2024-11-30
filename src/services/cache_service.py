# src/services/cache_service.py
from typing import Any, Optional
import modal
import hashlib

class CacheService:
    
    def get_dict_name(self, workspace_id: str, agent_id: str) -> str:
        """
        Create a deterministic, shortened name for the Modal Dict
        using first 16 chars of SHA-256 hash of workspace_id + agent_id
        """
        combined = f"{workspace_id}_{agent_id}"
        hash_object = hashlib.sha256(combined.encode())
        truncated_hash = hash_object.hexdigest()[:16]
        return f"agent-config-{truncated_hash}"
        
    def get(self, workspace_id: str, agent_id: str) -> Optional[Any]:
        dict_name = self.get_dict_name(workspace_id, agent_id)
        try:
            agent_dict = modal.Dict.from_name(dict_name,create_if_missing=True)
            return agent_dict.get("config") if agent_dict else None
        except KeyError:
            print(f"KeyError: {dict_name}")
            return None

    def set(self, workspace_id: str, agent_id: str, value: Any):
        dict_name = self.get_dict_name(workspace_id, agent_id)
        agent_dict = modal.Dict.from_name(dict_name, create_if_missing=True)
        agent_dict["config"] = value
        
    def delete(self, workspace_id: str, agent_id: str):
        """Delete the entire Dict for an agent"""
        dict_name = self.get_dict_name(workspace_id, agent_id)
        modal.Dict.delete(dict_name)
        
    def clear(self, workspace_id: str, agent_id: str):
        """Clear all entries in an agent's Dict"""
        dict_name = self.get_dict_name(workspace_id, agent_id)
        try:
            agent_dict = modal.Dict.lookup(dict_name)
            if agent_dict:
                agent_dict.clear()
               
        except KeyError:
            pass