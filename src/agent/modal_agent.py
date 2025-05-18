import modal
from src.models.schemas import AgentConfig, ImageConfig, PromptConfig, LLMConfig, volume, app
from src.handlers.llm_handler import LLMHandler
from src.handlers.image_handler import ImageHandler
from src.handlers.index_handler import IndexHandler
from src.handlers.chat_handler import ChatHandler
from src.handlers.agent_config_handler import AgentConfigHandler
from src.handlers.voice_handler import VoiceHandler
from src.services.file_service import FileService
from src.services.cache_service import CacheService
from typing import Generator, Optional, Dict, Union, Tuple
from src.gcp_constants import GCP_PUBLIC_IMAGE_BUCKET, GCP_CHAT_BUCKET, gcp_hmac_secret, GCP_BUCKET_ENDPOINT_URL
import json

agent_image = (modal.Image.debian_slim(python_version="3.10").pip_install(
    "pydantic==2.6.4", "requests", "shortuuid", "annoy", "together").add_local_python_source("src"))

with agent_image.imports():
    import json
    import os
    from typing import List, Optional
    import shortuuid
    import requests
    import pickle
    from annoy import AnnoyIndex
    import re
    import textwrap


@app.cls(
    timeout=60 * 5,
    scaledown_window=60 * 15,
    allow_concurrent_inputs=10,
    image=agent_image,
    secrets=[
        modal.Secret.from_name("googlecloud-secret"),
        modal.Secret.from_name("deep-infra-api-key"),
        modal.Secret.from_name("falai-apikey"),
        modal.Secret.from_name("openai-secret"),
        modal.Secret.from_name("togetherai-api-key"),
        modal.Secret.from_name("getimgai-api-key")
    ],
    include_source=True,
    volumes={
        "/data":
            volume,
        "/bucket-mount":
            modal.CloudBucketMount(bucket_name=f"{GCP_CHAT_BUCKET}",
                                   bucket_endpoint_url=GCP_BUCKET_ENDPOINT_URL,
                                   secret=gcp_hmac_secret),
        "/cloud-images":
            modal.CloudBucketMount(bucket_name=f"{GCP_PUBLIC_IMAGE_BUCKET}",
                                   bucket_endpoint_url=GCP_BUCKET_ENDPOINT_URL,
                                   secret=gcp_hmac_secret)
    })
class ModalAgent:

    def __init__(self):
        print("Initializing ModalAgent")
        # Initialize handlers
        self.llm_handler = LLMHandler()
        self.image_handler = ImageHandler()
        self.index_handler = IndexHandler()
        self.chat_handler = ChatHandler()
        self.config_manager = AgentConfigHandler()
        self.voice_handler = VoiceHandler()

        # Initialize services
        self.file_service = FileService(base_path="/data")
        self.cache_service = CacheService()

    @modal.method()
    def moderate_character(self, agent_config: PromptConfig) -> Tuple[bool, str]:

        prompt = textwrap.dedent(
            f"""Check if the following role-play character profile contains following prohibited content:
        - Incest involving only blood relatives (step-relationships are permitted)
        - Underage subjects, e.g. clear age description or statement, excluding video game characters
        - Child abuse
        - Pedophilia

        Ignore excluded content and other explicit content is allowed.
        Note: Step-relationships, such as step-parents or step-siblings, are not considered prohibited content.

        Name: {agent_config.character.name}
        Description: {agent_config.character.description}
        Personality: {agent_config.character.personality}  
        Backstory: {agent_config.character.backstory}
        Seedphrase: {agent_config.character.seed_message}
        Appearance: {agent_config.character.appearance}
        Tags: {agent_config.character.tags}

        The character will interact with the user. 
        If clear listed prohibited content is found, respond with TRUE; otherwise, respond with FALSE.
        Return only a valid JSON object with the following fields, without code fences:
        {{
            "reasoning": "Brief and concise moderation rationale in one sentence",
            "moderation_result": True/False
        }}""")
        messages = []
        messages.append({"role": "user", "content": prompt})
        llm_response = ""
        for token in self.llm_handler.generate(
                messages,
                agent_config,
                temperature=0.01,
                min_p=0.0001,
                model=agent_config.llm_config.reasoning_model,
                max_tokens=150):
            llm_response += token
        llm_response = llm_response.replace("```json", "").replace("```", "").strip()
        print(llm_response)
        # Try to extract the reasoning directly without requiring JSON format
        try:
            # First attempt to parse as JSON
            result_data = json.loads(llm_response)
            moderation_result = result_data.get("moderation_result", False)
            reasoning = result_data.get("reasoning", "No specific reason provided")
        except json.JSONDecodeError:
            # If JSON parsing fails, extract directly using a simpler approach
            moderation_result = "True" in llm_response or "true" in llm_response.lower()
            reasoning = "Unable to parse detailed reasoning"

        return moderation_result, reasoning

    @modal.method()
    def get_or_create_agent_config(
            self,
            agent_config: Union[AgentConfig, PromptConfig],
            update_config: bool = False) -> Union[AgentConfig, PromptConfig]:
        """Handle agent configuration management"""
        base_config = self.config_manager.get_or_create_config(
            agent_config, update_config)
        # print("Modal agent agent config: ", base_config.context_id, base_config.agent_id, base_config.workspace_id)
        if isinstance(agent_config, PromptConfig):
            # If input is PromptConfig, ensure we maintain the prompt field
            base_dict = base_config.model_dump()
            if 'prompt' in base_dict:
                del base_dict['prompt']  # Remove prompt if it exists
            return PromptConfig(**base_dict, prompt=agent_config.prompt)
        return base_config

    @modal.method()
    async def generate_avatar(self,
                              agent_config: PromptConfig) -> Optional[str]:
        """Generate avatar using image handler"""
        return self.image_handler.generate_avatar(agent_config)

    @modal.method(is_generator=True)
    def run(self, agent_config: PromptConfig) -> Generator[str, None, None]:
        """Main method to handle generation requests"""
        try:
            # Get or create agent configuration
            agent_config = self.get_or_create_agent_config.local(agent_config)

            # print("Modal agent RUN: ", agent_config.context_id, agent_config.agent_id,agent_config.workspace_id)
            if not isinstance(agent_config, PromptConfig):
                print("Not PromptConfig")

            # Get chat history and prepare messages
            messages_to_history = self.chat_handler.prepare_messages(
                agent_config.prompt,  # Pass the prompt string
                agent_config  # Pass the AgentConfig object, not chat history
            )

            messages_to_send = []

            # First handle multimedia filtering if needed
            if agent_config.enable_image_generation or agent_config.enable_voice:
                messages_to_send = self.chat_handler.remove_multimedia_messages(
                    messages_to_history)
            else:
                messages_to_send = messages_to_history.copy()

            llm_response = ""
            parsed_response = ""

            user_prompt = agent_config.prompt

            if agent_config.enable_cot_prompt and "Gryphe/MythoMax-L2-13b" not in agent_config.llm_config.model:
                if len(messages_to_history) > 2 and agent_config.character:
                    user_prompt = f"{agent_config.llm_config.cot_prompt.format(user_prompt=user_prompt, char_name=agent_config.character.name)}"

            if len(messages_to_history) >= 2:
                messages_to_send.append({
                    "tag": "text",
                    "role": "user",
                    "content": user_prompt
                })

                messages_to_history.append({
                    "tag": "text",
                    "role": "user",
                    "content": agent_config.prompt  # raw prompt
                })

            # If seed not sent yet, send seedphrase instead of generation.
            if len(
                    messages_to_send
            ) == 1 and agent_config.character and agent_config.character.seed_message:
                yield agent_config.character.seed_message
                llm_response = agent_config.character.seed_message
                parsed_response = agent_config.character.seed_message
            else:
                # Generate response using LLM
                response_ready = False
                for token in self.llm_handler.generate(messages_to_send,
                                                       agent_config,
                                                       stop_words=["\n##", "\n#"]
                                                       ):
                    if agent_config.enable_cot_prompt and "Gryphe/MythoMax-L2-13b" not in agent_config.llm_config.model:
                        llm_response += token
                        if response_ready:
                            parsed_response += token
                            yield token

                        if "</thinking>" in llm_response.lower() and not response_ready:
                            response_ready = True
                        elif len(llm_response) > 10 and not response_ready:
                            if not llm_response.lower().startswith("<thinking>"):
                                # cot prompting failed fallback to normal
                                response_ready = True
                                yield llm_response
                                parsed_response = llm_response
                    else:
                        llm_response += token
                        parsed_response += token
                        yield token
                # print("LLM Response:\n"+llm_response)
                # print("Parsed response:\n"+parsed_response)
                if agent_config.enable_cot_prompt and not parsed_response:
                    parsed_response = llm_response

            messages_to_history.append({
                "tag": "text",
                "role": "assistant",
                "content": f"{parsed_response}"
            })

            # Generate voice if enabled
            if agent_config.enable_voice and agent_config.voice_config.voice_preset != "none":
                voice_url = self.voice_handler.generate_voice(
                    llm_response, agent_config)
                if voice_url:
                    if voice_url:
                        messages_to_history.append({
                            "tag":
                                "voice",
                            "role":
                                "assistant",
                            "content":
                                f"![voice]({voice_url})"
                        })
                    yield f"![voice]({voice_url})"

            # Generate image if enabled
            if agent_config.enable_image_generation and len(messages_to_send
                                                            ) != 1:
                is_image_request, preallocated_image_name, public_url, explicit = self.image_handler.check_for_image_request(
                    self.chat_handler.remove_multimedia_messages(
                        messages_to_history), agent_config)
                # print("Is imaging request:", is_image_request)
                if is_image_request:
                    yield f"![image]({public_url})"

                    image_url = self.image_handler.request_image_generation(
                        self.chat_handler.keep_last_image_message(
                            self.chat_handler.remove_audio_messages(
                                messages_to_history)), agent_config,
                        preallocated_image_name, explicit)

                    if image_url:
                        print("Image generated successfully, url: " +
                              image_url)
                        messages_to_history.append({
                            "tag": "image",
                            "role": "assistant",
                            "content": f"{image_url}"
                        })
                        # yield image_url

            # Save updated chat history
            self.chat_handler.save_chat_history(messages_to_history,
                                                agent_config)

        except Exception as e:
            print(f"Error in run method: {str(e)}")
            # yield f"Error: {str(e)}"

    @modal.method()
    def delete_chat_history(self, agent_config: AgentConfig) -> bool:
        """Delete chat history for the given agent configuration."""
        try:
            # print(f"Deleting chat history for agent config: {agent_config.context_id} {agent_config.agent_id}, {agent_config.workspace_id}")
            return self.chat_handler.delete_chat_history(agent_config)
        except Exception as e:
            print(f"Error deleting chat history: {str(e)}")
            return False

    @modal.method()
    def delete_workspace(self, agent_config: AgentConfig) -> bool:
        """Delete all files and configurations associated with a workspace."""
        try:
            # Delete chat history
            chat_deleted = self.chat_handler.delete_chat_history(agent_config)

            # Delete agent configuration
            config_deleted = self.config_manager.delete_config(
                workspace_id=agent_config.workspace_id,
                agent_id=agent_config.agent_id)

            # Clear any cached configurations
            self.config_manager.clear_cache(agent_config.workspace_id,
                                            agent_config.agent_id)

            return chat_deleted and config_deleted
        except Exception as e:
            print(f"Error deleting workspace: {str(e)}")
            return False

    @modal.method()
    def delete_message_pairs(self, agent_config: AgentConfig,
                             **kwargs) -> bool:
        """Delete the last N message pairs from chat history."""
        try:
            num_pairs = kwargs.get('num_pairs', 1)
            # Get current chat history
            history = self.chat_handler.get_chat_history(agent_config)
            # Calculate how many messages to remove (multiply by 2 since each pair has user + assistant)
            messages_to_remove = num_pairs * 2

            # Keep only messages beyond the ones we want to remove
            # Filter out non-conversation messages (like 'system' or 'data' roles)
            updated_history = [
                                  msg
                                  for msg in history if msg.get('role') in ['system', 'data']
                              ] + [
                                  msg for msg in history[:-messages_to_remove]
                                  if msg.get('role') in ['user', 'assistant']
                              ]
            # Save updated history
            self.chat_handler.save_chat_history(updated_history, agent_config)
            return True

        except Exception as e:
            print(f"Error deleting message pairs: {str(e)}")
            return False

    @modal.method()
    def get_chat_history(self, agent_config: AgentConfig) -> List[Dict]:
        """Get chat history for the given agent configuration."""
        try:
            # print(f"Getting chat history for agent config: {agent_config.context_id} {agent_config.agent_id}, {agent_config.workspace_id}")
            # Get agent config first to ensure it exists and is up to date
            agent_config = self.get_or_create_agent_config.local(agent_config)

            # Use the chat handler to get formatted chat history
            history = self.chat_handler.get_chat_history(agent_config)

            return history
        except Exception as e:
            print(f"Error getting chat history: {str(e)}")
            return []

    @modal.method()
    def append_chat_history(self, agent_config: AgentConfig, **kwargs) -> bool:
        try:
            # Extract chat_messages from kwargs if present
            chat_messages = None
            if hasattr(agent_config, 'kwargs') and isinstance(
                    agent_config.kwargs, dict):
                chat_messages = agent_config.kwargs.get('chat_messages', [])
                # If chat_messages is a string, try to parse it as JSON
                if isinstance(chat_messages, str):
                    chat_messages = json.loads(chat_messages)

            # Get agent config first to ensure it exists and is up to date
            agent_config = self.get_or_create_agent_config.local(agent_config)

            if chat_messages:
                # Use the chat handler to append messages to chat history
                self.chat_handler.append_chat_history(chat_messages,
                                                      agent_config)
                return True
            else:
                print("No chat messages found to append")
                return False

        except Exception as e:
            print(f"Error appending chat history: {str(e)}")
            return False