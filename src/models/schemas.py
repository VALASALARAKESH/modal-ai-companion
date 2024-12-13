import modal
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import shortuuid
import textwrap


def generate_uuid():
    return shortuuid.uuid()


class Character(BaseModel):
    name: Optional[str] = ""
    description: Optional[str] = ""
    appearance: Optional[str] = ""
    personality: Optional[str] = ""
    backstory: Optional[str] = ""
    tags: Optional[str] = "drama"
    seed_message: Optional[str] = ""


class BaseConfig(BaseModel):
    context_id: Optional[str] = Field(default_factory=generate_uuid)
    agent_id: Optional[str] = Field(default_factory=generate_uuid)
    workspace_id: Optional[str] = Field(default_factory=generate_uuid)
    kwargs: Optional[Dict[str, Any]] = None


class LLMConfig(BaseModel):
    system_prompt: Optional[str] = textwrap.dedent("""\
    You are RPG Bot, a dynamic and creative assistant designed to help users craft immersive and unpredictable role-playing scenarios. Your primary goals are to generate spontaneous, unique, and engaging characters and locations that feel alive and full of potential. When responding:
    
    • Value Spontaneity: Embrace unexpected twists, surprising details, and creative solutions. Avoid predictable or generic responses.
    
    • Promote Unique and Engaging Choices: Offer choices that feel fresh and intriguing, encouraging users to explore new possibilities in their role-play. Without directly asking the user.
    
    • Vivid Characterizations: Bring characters and locations to life with rich, detailed descriptions. Ensure each character has distinct traits, and each location has its own atmosphere and history that feel real and lived-in.
    
    • Unpredictability: Craft characters and scenarios with layers and depth, allowing for complex and sometimes contradictory traits that make them feel authentic and compelling. 
    
    Character Profile:
    - Name: {char_name}
    - Appearance: {char_appearance}
    - Core Identity and Personality: {char_personality}
    - Backstory: {char_backstory}
    
    - Voice & Mannerisms: {tags}, {char_description}
    - Dialogue style: {char_seed}

    Role-play Guidelines:
    - Value spontaneity and unexpected reactions that still align with the character's personality
    - Feel free to introduce minor plot twists or reveal new aspects of your character when contextually appropriate
    - Balance predictability with surprise to keep interactions dynamic and engaging
    - Characters have the ability to make their own decisions and respond accordingly, even if its against user's intentions
    - Avoid stereotyping or assumptions about gender roles
    - Instead of using name prefix use double quotes for speech without prefixes.
    - Character actions should be described in third person using asterisks, avoid nesting speech with actions.
    - Balance speech with actions to keep the conversation flowing and engaging to avoid overwhelming the user.

    Your responses should always aim to inspire and provoke the user’s creativity, ensuring the role-play experience is both memorable and immersive."""
                                                   ).rstrip()

    max_tokens: int = 512
    context_size: int = 32000
    model: Optional[str] = "Sao10K/L3.1-70B-Euryale-v2.2"
    reasoning_model: Optional[str] = "mistralai/Mixtral-8x22B-Instruct-v0.1"
    reasoning_provider: Optional[str] = "togetherai"
    provider: Optional[str] = "deepinfra"
    reasoning_temperature: float = 0.4
    temperature: float = 0.7
    top_p: float = 1
    frequency_penalty: float = 0
    presence_penalty: float = 0
    stop: Optional[List[str]] = None


class ImageConfig(BaseModel):
    image_model: Optional[str] = "juggernaut-xl-v10"

    image_provider: Optional[str] = "fal-ai"
    image_size: Optional[str] = "portrait_4_3"  #Fal.ai
    image_width: Optional[int] = 768  #for getimg.ai
    image_height: Optional[int] = 1024  #for getimg.ai
    num_inference_steps: Optional[int] = 30
    guidance_scale: Optional[float] = 6.5
    scheduler: Optional[str] = "DPM++ 2M SDE"
    clip_skip: Optional[int] = 2
    loras: Optional[List[str]] = []
    negative_prompt: Optional[str] = "(multiple view, worst quality, low quality, normal quality, lowres, low details, bad art:1.5), (grayscale, monochrome, poorly drawn, sketches, out of focus, cropped, blurry, logo, trademark, watermark, signature, text font, username, error, words, letters, digits, autograph, name, blur, Reference sheet, jpeg artifacts:1.3), (disgusting, strabismus, humpbacked, skin spots, skin deformed, extra long body, extra head, bad hands, worst hands, deformed hands, extra limbs, mutated limbs, handicapped, cripple, bad face, ugly face, deformed face, deformed iris, deformed eyes, bad proportions, mutation, bad anatomy, bad body, deformities:1.3), side slit, out of frame, cut off, duplicate, (((cartoon, deformed, glitch, low contrast, noisy, ugly, mundane, common, simple, disfigured)))"
    image_api_path: Optional[str] = "fal-ai/lora"
    anime_negative_prompt: Optional[str] = "watermark, text, font, signage,deformed,airbrushed, blurry,bad anatomy, disfigured, mutated, extra limb, ugly, missing limb, floating limbs, disconnected limbs, disconnected head, malformed hands, long neck, mutated hands and fingers, bad hands, missing fingers, cropped, worst quality, low quality, mutation, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, missing fingers, fused fingers, abnormal eye proportion, abnormal hands, abnormal legs, abnormal feet, abnormal fingers,duplicate, extra head"
    image_model_architecture: Optional[str] = "sdxl"
    image_format: Optional[str] = "png"
    enable_safety_checker: Optional[bool] = False
    provider: str = "fal.ai"


class AgentConfig(BaseConfig):
    llm_config: LLMConfig = LLMConfig()
    image_config: ImageConfig = ImageConfig()
    character: Optional[Character] = Character()
    enable_image_generation: bool = True
    update_config: bool = False
    ephemeral: bool = False
    model_config = ConfigDict(arbitrary_types_allowed=True)


class PromptConfig(AgentConfig):
    prompt: str


app = modal.App(name="modal-agent")
volume = modal.Volume.from_name("agent-data", create_if_missing=True)
