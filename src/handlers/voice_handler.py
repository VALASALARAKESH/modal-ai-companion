import os
import requests
import base64
from src.models.schemas import AgentConfig
from src.services.file_service import FileService


class VoiceHandler:
    def __init__(self):
        self.file_service = FileService('/cloud-images')
        self.api_url = "https://api.deepinfra.com/v1/inference/hexgrad/Kokoro-82M"
        self.api_key = os.environ.get("DEEP_INFRA_API_KEY")

    def generate_voice(self, text: str, agent_config: AgentConfig) -> str:
        """Generate voice from text using DeepInfra Kokoro model"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        # Remove all *action* text between asterisks
        import re
        clean_text = re.sub(r"\*([^*]+)\*", "", text)
        # remove text between <think> tags
        clean_text = re.sub(r"<think>(.*?)</think>", "", clean_text)
        if "</think>" in clean_text:
            clean_text = clean_text.split("</think>")[1]
        clean_text = clean_text.strip()

        # check if the text is empty and restore original text
        if not clean_text:
            clean_text = text

        data = {
            "text": clean_text,
            "preset_voice": agent_config.voice_config.voice_preset
        }

        response = requests.post(self.api_url, headers=headers, json=data)
        response.raise_for_status()

        result = response.json()
        if "audio" not in result:
            raise ValueError("No audio data in response")

        # Extract base64 data after the data:audio/wav;base64, prefix
        base64_audio = result["audio"].split("base64,")[1]
        audio_data = base64.b64decode(base64_audio)
        preallocated_audio_name, public_url = self.file_service.generate_preallocated_url(agent_config, "audio",
                                                                                          file_format="wav")
        voice_url = self.file_service.save_binary_to_bucket(
            audio_data,
            agent_config,
            "voice",
            preallocated_name=preallocated_audio_name
        )

        return voice_url