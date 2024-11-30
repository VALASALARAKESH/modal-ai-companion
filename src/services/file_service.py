# src/services/file_service.py
import modal
import json
import pathlib
from modal.volume import Volume
import requests
import shortuuid
from typing import Any, Optional
from src.gcp_constants import GCP_PUBLIC_IMAGE_BUCKET
from src.models.schemas import AgentConfig,volume

class FileService:
    def __init__(self,  base_path = "/data"):
        import shortuuid

        self.volume = volume
        self.base_path = pathlib.Path(base_path)
        self.image_base_path = pathlib.Path("/cloud-images")
        self.public_url_base = f"https://storage.googleapis.com/{GCP_PUBLIC_IMAGE_BUCKET}"

    def get_path(self, workspace_id: str, filename: str) -> pathlib.Path:
        path = pathlib.Path(f"{self.base_path}/{workspace_id}/{filename}")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save_json(self, data: Any, workspace_id: str, filename: str):
        path = self.get_path(workspace_id, filename)
        print(f"Saving JSON to {path}\n")
        with path.open('w') as f:
            json.dump(data, f)
        self.volume.commit()

    def load_json(self, workspace_id: str, filename: str) -> Optional[dict]:
        self.volume.reload()
        path = self.get_path(workspace_id, filename)
        print(f"Loading JSON from {path}\n")
        if path.exists():
            with path.open('r') as f:
                return json.load(f)
        return None

    def save_image_to_bucket(self, image_url: str, agent_config: AgentConfig, sub_folder: str = "",preallocated_image_name:str ="") -> str:
        """Save image to cloud bucket and return public URL."""
        filename = ""
        if preallocated_image_name:
            filename = preallocated_image_name
        else:
            filename = f"{shortuuid.uuid()}.png"

        if sub_folder and not sub_folder.endswith('/'):
            sub_folder += "/"

        image_path = pathlib.Path(f"{self.image_base_path}/{sub_folder}{agent_config.workspace_id}/{filename}")
        public_url = f"{self.public_url_base}/{sub_folder}{agent_config.workspace_id}/{filename}"

        # Create directory if needed
        image_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving image to bucket {image_path}")
        # Download and save image
        response = requests.get(image_url)
        response.raise_for_status()
        image_path.write_bytes(response.content)

        return public_url
        
    def save_binary_to_bucket(self, binary_data: bytes, agent_config: AgentConfig, sub_folder: str = "", preallocated_image_name: str = "") -> str:
        """Save binary data directly to bucket and return public URL."""
        filename = preallocated_image_name if preallocated_image_name else f"{shortuuid.uuid()}.png"

        if sub_folder and not sub_folder.endswith('/'):
            sub_folder += "/"

        image_path = pathlib.Path(f"{self.image_base_path}/{sub_folder}{agent_config.workspace_id}/{filename}")
        public_url = f"{self.public_url_base}/{sub_folder}{agent_config.workspace_id}/{filename}"
        # Create directory if needed
        image_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving image to bucket {image_path}")

        # Save binary data directly
        image_path.write_bytes(binary_data)

        return public_url
            
    def generate_preallocated_url(self, agent_config: AgentConfig, sub_folder: str = "") -> tuple[str, str]:
        """Generate preallocated filename and public URL."""
        filename = f"{shortuuid.uuid()}.png"
        if sub_folder and not sub_folder.endswith('/'):
            sub_folder += "/"

        public_url = f"{self.public_url_base}/{sub_folder}{agent_config.workspace_id}/{filename}"
        return filename, public_url
    
    def delete_file(self, workspace_id: str, filename: str) -> bool:
        """Delete a file from the workspace."""
        try:
            path = self.get_path(workspace_id, filename)
            print(f"Deleting file {path}")
            if path.exists():
                print("Deleting file:", path)
                path.unlink()
                volume.commit()
                return True
            return False
        except Exception as e:
            print(f"Error deleting file: {str(e)}")
            return False