# src/handlers/chat_handler.py
from typing import List, Dict, Optional
from src.models.schemas import AgentConfig
from src.services.file_service import FileService
from src.handlers.index_handler import IndexHandler

models_4k_context = ["lizpreciatior/lzlv_70b_fp16_hf","Gryphe/MythoMax-L2-13b"]

class ChatHandler:

    def __init__(self):
        import shortuuid
        self.file_service = FileService('/bucket-mount')
        self.index_handler = IndexHandler()
        
    def prepare_messages(self, prompt: str, agent_config: AgentConfig) -> List[Dict]:
        """Prepare messages for the LLM including system prompt and relevant background."""
        messages = []
        relevant_backstory = None
        # Get relevant background if available
        if (agent_config.character and 
            agent_config.character.backstory and 
            len(agent_config.character.backstory) > 1000):

            similar_chunks = self.index_handler.search(prompt, agent_config)
            if similar_chunks:
                relevant_backstory = "\n".join(similar_chunks)
                #print("\nRelevant backstory: "+relevant_backstory)
                
        # Get the formatted system prompt
        system_prompt = self._format_system_prompt(agent_config,updated_backstory=relevant_backstory)
        #print(f"System prompt fromatted: {system_prompt}")
        messages.append({"tag":"initial_system_prompt","role": "system", "content": system_prompt})
        # Get chat history (excluding system messages)
        history = self.get_chat_history(agent_config)  # This returns List[Dict]
        messages.extend([msg for msg in history if msg.get('role') != 'system'])

        # Add the user's prompt
        messages.append({"tag":"text","role": "user", "content": prompt})
        # Filter messages to fit context window
        max_context_size = agent_config.llm_config.context_size
        
        if agent_config.llm_config.model in models_4k_context:
            max_context_size = 4096
            
        return self.filter_messages_for_context(
            messages, 
            max_context_size=max_context_size
        )
        
    def _format_system_prompt(self, agent_config: AgentConfig,updated_backstory = None) -> Optional[str]:
        """Format system prompt with character details."""
        try:
            return agent_config.llm_config.system_prompt.format(
                char_name=agent_config.character.name,
                char_description=agent_config.character.description,
                char_appearance=agent_config.character.appearance,
                char_personality=agent_config.character.personality,
                char_backstory=updated_backstory or agent_config.character.backstory,
                tags=agent_config.character.tags,
                char_seed=agent_config.character.seed_message,
                char=agent_config.character.name
            )
        except (AttributeError, KeyError) as e:
            print(f"Error formatting system prompt: {str(e)}")
            return agent_config.llm_config.system_prompt

    def get_chat_history(self, agent_config: AgentConfig,full_history:bool = False) -> List[Dict]:
        """Load chat history from bucket."""
        filepath = f"/bucket-mount/{agent_config.workspace_id}/{agent_config.context_id}_chat.json"
        history = self.file_service.load_json(agent_config.workspace_id,f'{agent_config.context_id}_chat.json') or []
        # Ensure history is a list
        if isinstance(history, dict):
            history = [history]  # Convert single dictionary to a list
            
        # Filter out invalid messages
        valid_history = [
            msg for msg in history
            if isinstance(msg, dict) and 'role' in msg and 'content' in msg
        ]
        if full_history:
            valid_history = self.filter_messages_for_context(valid_history, agent_config.llm_config.context_size)

        return valid_history

    def save_chat_history(self, messages: List[Dict], agent_config: AgentConfig):
        import shortuuid
        """Save chat history to bucket."""
        filename = f"{agent_config.context_id}_chat.json"
        filepath = f"/bucket-mount/{agent_config.workspace_id}/{agent_config.context_id}_chat.json"
        self.file_service.save_json(messages,agent_config.workspace_id,filename)
        
    def append_chat_history(self, messages: List[Dict], agent_config: AgentConfig):
        """
        Append messages to chat history
        messages: List of message dictionaries with 'role' and 'content'
        """
        if not messages:
            return

        # Load existing messages
        existing_messages = self.file_service.load_json(agent_config.workspace_id, "default_chat.json") or []

        # Ensure existing_messages is a list
        if not isinstance(existing_messages, list):
            existing_messages = []

        # Filter out any invalid messages and ensure each message has required fields
        valid_messages = [
            msg for msg in messages 
            if isinstance(msg, dict) and 'role' in msg and 'content' in msg
        ]

        # Append only valid messages
        existing_messages.extend(valid_messages)

        # Clean up any stray characters or invalid entries
        cleaned_messages = [
            msg for msg in existing_messages
            if isinstance(msg, dict) and 'role' in msg and 'content' in msg
        ]

        # Save cleaned messages
        self.file_service.save_json(cleaned_messages, agent_config.workspace_id, "default_chat.json")
        

    def filter_messages_for_context(self, messages: List[Dict], max_context_size: int = 4096) -> List[Dict]:
        """
        Filter messages to fit within context size while preserving system message and chronological order.
        """
        def estimate_tokens(message: Dict) -> int:
            """Estimate tokens in a message using 4 chars/token ratio."""
            content = message.get('content', '')
            return len(content) // 4
        if not messages:
            return []

        # Extract system message
        system_message = next((msg for msg in messages if msg.get('role') == 'system'), None)
        filtered_messages = [system_message] if system_message else []
        # Get non-system messages
        conversation = [msg for msg in messages if msg.get('role') != 'system']
        # Calculate current token count starting with system message
        current_tokens = estimate_tokens(system_message) if system_message else 0
        # Add messages from newest to oldest until we hit token limit
        for message in reversed(conversation):
            message_tokens = estimate_tokens(message)
            if current_tokens + message_tokens <= max_context_size:
                filtered_messages.append(message)
                current_tokens += message_tokens
            else:
                break
        # Restore message order: system first, then chronological
        return ([msg for msg in filtered_messages if msg.get('role') == 'system'] + 
                list(reversed([msg for msg in filtered_messages if msg.get('role') != 'system'])))

    
    def delete_chat_history(self, agent_config: AgentConfig) -> bool:
        """Delete chat history file for the given agent config."""
        filename = f"{agent_config.context_id}_chat.json"
        print(f"Deleting file {filename}")
        return self.file_service.delete_file(agent_config.workspace_id, filename)

    def remove_image_messages(self, messages: List[dict]) -> List[dict]:
        """Remove messages that contain image tag from chat history."""

        return [
            msg for msg in messages 
            if msg.get('tag') != 'image'
        ]

    def keep_last_image_message(self, messages: List[dict]) -> List[dict]:
        """Keep the last image in chat history but remove all prior ones."""
        
        last_image_index = next((i for i in reversed(range(len(messages))) if messages[i].get('tag') == 'image'), None)
        
        return [
            msg for i, msg in enumerate(messages)
            if msg.get('tag') != 'image' or i == last_image_index
        ]