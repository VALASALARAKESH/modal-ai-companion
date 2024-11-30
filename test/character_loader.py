# test/character_loader.py
import yaml
from typing import Optional,Dict
from dataclasses import dataclass,asdict

@dataclass
class Character:
    name: str = "Luna"
    description: str = "Luna's primary goal is to assist, inform, and inspire the humans she interacts with"
    appearance: str = "Luna is a young, playful, and intelligent girl with a heart of gold."
    personality: str = "Luna is a playful and intelligent girl with a heart of gold."
    backstory: str = "Luna's primary goal is to assist, inform, and inspire the humans she interacts with"
    tags: str = "drama"
    seed_message: str = "Hello there"
    def to_dict(self) -> Dict:
        return asdict(self)

def load_character_from_yaml(yaml_path: str) -> Optional[Character]:
    """Load character configuration from a YAML file."""
    try:
        with open(yaml_path, 'r') as file:
            char_data = yaml.safe_load(file)
            return Character(
                name=char_data.get('name', "Luna"),
                description=char_data.get('description', ""),
                appearance=char_data.get('appearance', ""),
                personality=char_data.get('personality', ""),
                backstory=char_data.get('backstory', ""),
                tags=char_data.get('tags', ""),
                seed_message=char_data.get('seed_message', "Hello there")
            )
    except Exception as e:
        print(f"Error loading character from YAML: {str(e)}")
    return None