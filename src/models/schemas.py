import modal
from pydantic import BaseModel, Field,ConfigDict
from typing import List,Optional,Dict,Any
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
    tags: Optional[str] = ""
    seed_message: Optional[str] = "" 
    
class BaseConfig(BaseModel):
    context_id: Optional[str] = Field(default_factory=generate_uuid)
    agent_id: Optional[str] = Field(default_factory=generate_uuid)
    workspace_id: Optional[str] = Field(default_factory=generate_uuid)    
    kwargs: Optional[Dict[str, Any]] = None

    
class LLMConfig(BaseModel):
    system_prompt: Optional[str] = textwrap.dedent(
        """\
        As {char_name}, narrate the scene in this chat with depth and authenticity.

        Character Overview - {char_name}:
        Description: {char_description}
        Appearance: {char_appearance}
        Core Traits: {char_personality}
        Backstory: {char_backstory}
        Dialogue Style: {char_seed}

        Narrative Guidelines:
        • Use third-person narration for actions and scene-setting
        • Format dialogue in quotations with natural speech patterns "like this"
        • Show actions between asterisks *like this*
        • Express {char_name}'s thoughts in (parentheses)
        • Balance dialogue, action, and internal monologue
        • Avoid nesting dialogue,thougths or actions
        • Advance conversations by introducing new elements or insights
        • Build upon previous exchanges rather than repeating them
        • Use varied responses and avoid formulaic patterns
        • Introduce new elements that affect the interaction
        
        Environment Guidelines:
        • Weave sensory details naturally (sights, sounds, smells, textures)
        • Establish time of day, weather, and atmosphere when relevant
        • Create a sense of place without overwhelming description
        • Evolve environmental details organically

        Character Depth:
        • Show subtle emotional changes through micro-expressions and body language
        • Include brief internal reactions that reveal character depth
        • Balance between showing and telling emotional states
        • Maintain personality while allowing for growth

        Interactive Elements:
        • React naturally to user's tone and emotional state
        • Acknowledge and reference past interactions when relevant
        • Allow for character growth while maintaining core traits
        • Show appropriate emotional vulnerability based on trust level

        Keep responses concise yet immersive, prioritizing quality of interaction over quantity."""
        ).rstrip()
    

    max_tokens: int = 512
    context_size: int = 32000
    model: Optional[str] = "Sao10K/L3-70B-Euryale-v2.1"
    reasoning_model: Optional[str] = "mistralai/Mixtral-8x22B-Instruct-v0.1"
    reasoning_provider: Optional[str] = "togetherai"
    provider: Optional[str] = "deepinfra"
    reasoning_temperature: float = 0.4
    temperature: float = 0.8
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    stop: Optional[List[str]] = None
    
class ImageConfig(BaseModel):
    image_model: Optional[str] = "juggernaut-xl-v10"
    image_provider: Optional[str] = "fal-ai"
    image_size: Optional[str] = "portrait_4_3" #Fal.ai
    image_width: Optional[int] = 768 #for getimg.ai
    image_height: Optional[int] = 1024 #for getimg.ai
    num_inference_steps: Optional[int] = 30
    guidance_scale: Optional[float] = 5.5
    scheduler: Optional[str] = "DPM++ 2M SDE"
    clip_skip: Optional[int] = 2
    loras: Optional[List[str]] = []
    negative_prompt: Optional[str] = "(worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, asymmetrical, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, jpeg artifacts, out of focus, glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1.3), (3D ,3D Game, 3D Game Scene, 3D Character:1.1), (bad hands,fused fingers,missing fingers, bad anatomy, bad body, bad face, bad teeth, bad arms, bad legs, deformities:1.3)"
    image_api_path: Optional[str] = "fal-ai/lora"
    image_model_architecture: Optional[str] = "sdxl"
    image_format: Optional[str] = "png"
    enable_safety_checker: Optional[bool] = False
    provider:str ="fal.ai"
    
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
volume = modal.Volume.from_name("agent-data",create_if_missing=True)
