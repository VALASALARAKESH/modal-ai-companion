# src/handlers/image_handler.py
from typing import Optional, List, Dict, Any,Literal
import os
import json
import time
import requests
from src.models.schemas import AgentConfig, PromptConfig
from src.handlers.llm_handler import LLMHandler
from src.services.file_service import FileService
import textwrap
import shortuuid
import base64
import re

euler_a_models = [ "156375","286821","303526","378499","293564","384264" ]
getimg_anime_style_models = ["reproduction-v3-31","real-cartoon-xl-v6","sdvn7-niji-style-xl-v1","counterfeit-xl-v2-5","animagine-xl-v-3-1"]

class ImageHandler:
    def __init__(self):
        self.file_service = FileService('/cloud-images')
        self.base_url = "https://queue.fal.run"
        self.getimg_base_url = "https://api.getimg.ai/v1/stable-diffusion/text-to-image"
        self.api_key = os.environ["FALAI_API_KEY"]
        self.getimg_api_key = os.environ["GETIMGAI_API_KEY"]
        self.llm_handler = LLMHandler()

    def _generate_with_getimg(self, prompt: str, agent_config: AgentConfig) -> Optional[bytes]:
        """Generate image using GetImg API and return binary data"""
        headers = {
            "Authorization": f"Bearer {self.getimg_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": agent_config.image_config.image_model,
            "prompt":prompt,
            "negative_prompt": agent_config.image_config.negative_prompt,
            "width": agent_config.image_config.image_width,
            "height": agent_config.image_config.image_height,
            "steps": agent_config.image_config.num_inference_steps,
            "guidance": agent_config.image_config.guidance_scale,
            "scheduler": "dpmsolver++",
            "output_format": agent_config.image_config.image_format
        }

        get_img_sdxl_models = [ "juggernaut-xl-v10","realvis-xl-v4","reproduction-v3-31","real-cartoon-xl-v6","sdvn7-niji-style-xl-v1","counterfeit-xl-v2-5","animagine-xl-v-3-1"]
        
        if payload.get("model") in get_img_sdxl_models:
            self.getimg_base_url = "https://api.getimg.ai/v1/stable-diffusion-xl/text-to-image"
            payload['scheduler'] = 'euler'
            
        if payload.get("model") in getimg_anime_style_models:
            payload['negative_prompt'] = agent_config.image_config.anime_negative_prompt
            
        #print(f"Generating image with GetImg API: {payload}")
        try:
            response = requests.post(self.getimg_base_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()

            if "image" in result:
                # Decode base64 image to binary
                return base64.b64decode(result["image"])
            return None
        except Exception as e:
            print(f"Error generating image with GetImg: {str(e)}")
            return None
            
    def generate_image(self, prompt: str, agent_config: AgentConfig, folder: str = "images",preallocated_image_name:str = "") -> Optional[str]:
        """Generate an image and save it to the specified folder."""
        try:
            
            #print("Image model: ", agent_config.image_config.image_model)
            if (agent_config.image_config.provider == "getimg" or "https" not in agent_config.image_config.image_model)\
            and "flux-general-with-lora" not in agent_config.image_config.image_model \
            and "SG161222/Realistic_Vision_V6.0_B1_noVAE" not in agent_config.image_config.image_model \
            and "fal-ai/flux/dev" not in agent_config.image_config.image_model \
            and "fal-ai/stable-diffusion-v35-medium" not in agent_config.image_config.image_model:
                
                if agent_config.image_config.image_model in getimg_anime_style_models:
                    prompt = "masterpiece, "+prompt
                    
                #print(f"Generating image with GetImg")
                image_data = self._generate_with_getimg(prompt, agent_config)
                if image_data:
                    return self.file_service.save_binary_to_bucket(
                        image_data,
                        agent_config,
                        folder,
                        preallocated_image_name=preallocated_image_name
                    )
                return None
            else:
                # Existing FAL.ai implementation
                status_url = self._submit_job(prompt, agent_config)
                if not status_url:
                    print("Failed to submit image generation job")
                    return None
                image_url = self._check_status_and_download_image(status_url)
                if not image_url:
                    print("Failed to generate image")
                    return None
                print(image_url)
                return self.file_service.save_image_to_bucket(
                    image_url,
                    agent_config,
                    folder,
                    preallocated_image_name=preallocated_image_name
                )
        except Exception as e:
            print(f"Error generating image: {str(e)}")
            return None

    def generate_avatar(self, agent_config: PromptConfig,folder="") -> Optional[str]:
        """Generate an avatar image."""
        
        return self.generate_image(agent_config.prompt,agent_config,folder= "")
        
    def format_image_size(self,image_size):
        """Helper to format image size from string or dict"""
        
        if isinstance(image_size, str):
            try:
                # Check if string is JSON formatted
                parsed_size = json.loads(image_size)
                if isinstance(parsed_size, dict):
                    return parsed_size
                return image_size
            except json.JSONDecodeError:
                # If not valid JSON, return original string (like "portrait_4_3")
                return image_size
        return image_size
        
    def _submit_job(self, prompt: str, agent_config: AgentConfig) -> Optional[str]:
        """Submit image generation job to FAL AI."""
        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json"
        }



        scheluder_config = agent_config.image_config.scheduler
        negative_prompt = agent_config.image_config.negative_prompt
        num_inference_steps = agent_config.image_config.num_inference_steps
        model_architecture = agent_config.image_config.image_model_architecture
        api_path = agent_config.image_config.image_api_path
        prompt_prefix ="Photo of "
        if any(euler_model in agent_config.image_config.image_model for euler_model in euler_a_models):
            prompt_prefix = "masterpiece, "
            scheluder_config = "Euler A"
            num_inference_steps = "30"
            negative_prompt = agent_config.image_config.anime_negative_prompt

        if agent_config.image_config.image_model == "SG161222/Realistic_Vision_V6.0_B1_noVAE":
            agent_config.image_config.image_api_path = "fal-ai/realistic-vision"
            
        image_size = self.format_image_size(agent_config.image_config.image_size)
                

        # Prepare job details based on agent config
        job_details = {
            "prompt": prompt_prefix+prompt,
            "model_name": agent_config.image_config.image_model,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": agent_config.image_config.guidance_scale,
            "image_size": image_size,
            "scheduler": scheluder_config,
            "clip_skip": agent_config.image_config.clip_skip,
            "enable_safety_checker": agent_config.image_config.enable_safety_checker,
            "model_architecture":agent_config.image_config.image_model_architecture,
            "image_format": agent_config.image_config.image_format,
            "prompt_weighting": True,
            "num_images": 1
            
        }
        
        formatted_job_details = json.dumps(job_details, indent=4)
        #print(f"Submitting job with details: {formatted_job_details}")
        # Add optional loras if configured
        if agent_config.image_config.loras:
            job_details["loras"] = agent_config.image_config.loras

        if agent_config.image_config.image_model == "fal-ai/stable-diffusion-v35-medium":
            agent_config.image_config.image_api_path = "/fal-ai/stable-diffusion-v35-medium"
            
        if len(prompt) > 10:
            try:
                response = requests.post(
                    f"{self.base_url}/{agent_config.image_config.image_api_path}",
                    headers=headers,
                    json=job_details
                )
                response.raise_for_status()
                return response.json().get("status_url")
    
            except requests.exceptions.RequestException as e:
                print(f"Error submitting job: {str(e)}")
                return None
            except json.JSONDecodeError as e:
                print(f"Error decoding response: {str(e)}")
                return None
        return None

    def _check_status_and_download_image(self, status_url: str) -> Optional[str]:
        """Check job status and return image URL when complete."""
        headers = {"Authorization": f"Key {self.api_key}"}
        max_retries = 250
        retry_count = 0

        while retry_count < max_retries:
            try:
                response = requests.get(status_url, headers=headers)
                response.raise_for_status()
                status_data = response.json()

                status = status_data.get("status")
                if status == "COMPLETED":
                    result_response = requests.get(
                        status_data.get("response_url"),
                        headers=headers
                    )
                    result_response.raise_for_status()
                    return result_response.json()["images"][0]["url"]

                elif status == "FAILED":
                    print("Image generation task failed")
                    return None

                # Still processing, wait and retry
                time.sleep(0.4)
                retry_count += 1

            except requests.exceptions.RequestException as e:
                print(f"Error checking status: {str(e)}")
                return None
            except json.JSONDecodeError as e:
                print(f"Error decoding status response: {str(e)}")
                return None

        print("Timeout waiting for image generation")
        return None

    def check_for_image_request(self,messages: List[dict], agent_config: AgentConfig) -> tuple[bool, str, str,bool]:
        """check for Image request"""
    
        local_messages = messages.copy()
        prompt = textwrap.dedent(f"""\
    ## Instruction
    Analyze if the last user and character message warrants visual generation based on these criteria:
    
    1. Visual Changes:
       - Outfit changes or significant wardrobe adjustments
       - Physical transformations or appearance changes
       - Clear showing/revealing actions ("shows", "reveals", "displays")
       - Dramatic poses or intentional presentation

    2. Character Actions:
       - Intentional display or presentation
       - Taking photos or selfies
       - Meaningful changes in expression or pose
       - Dramatic or theatrical actions
    AND
    
    3. User Requests/Interest:
       - Direct requests ("show me", "send a picture")
       - Suggestive requests ("I wonder what you look like right now")
       - Implied interest ("You must look beautiful in that")
       - Questions about appearance or attire
    

    
    Do NOT trigger for:
    - Casual gestures or minor movements
    - Basic conversational actions
    - Generic scene descriptions
    - Subtle mood changes
    
    Review the last message from user and character and respond with TRUE only if there are notable visual elements AND user interest in appearance.
    
    User: {local_messages[-2]['content']}
    Character: {local_messages[-1]['content']}
    
    Additionally, review the character's message to determine R-rating, if character is clearly naked (NC-17) or if unsure then rating is R.
    Format response as a structured JSON:
    {{
        "result": true/false,
        "r-rating": "NC-17,R"
    }}""").rstrip()
        #print(prompt)
        local_messages.append({"role": "user", "content": prompt})    
    
        reasoning_response = ""
        for token in self.llm_handler.generate(local_messages,agent_config,
                                               temperature=0.2,
                                               model=agent_config.llm_config.reasoning_model,
                                               provider=agent_config.llm_config.reasoning_provider,
                                               max_tokens=50
                                              ):
            reasoning_response += token
        #print(f"Reasoning response: {reasoning_response}")
        explicit = False
        try:
            json_data = json.loads(reasoning_response.strip())
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response: {str(e)}")
            
        json_data = json.loads(reasoning_response)
        nudity_status = json_data.get("r-rating", "R").lower()
        
        if "nc-17" in nudity_status.lower():
            explicit = True

        result_status = json_data.get("result", False)
        if result_status:
            preallocated_name, public_url = self.file_service.generate_preallocated_url(agent_config, "images")
            return True, preallocated_name, public_url, explicit
        return False, "", "", False
    
    def request_image_generation(self,messages: List[dict],agent_config: AgentConfig,preallocated_image_name:str = "",explicit= False):
        """request image generation"""
    
        def get_last_ten_messages_with_system(self, messages: List[dict]) -> List[dict]:
            # Preserve the first system message
            first_system_message = next((msg for msg in messages if msg.get('role') == 'system'), None)
    
            # Filter out only non-system messages from the list
            non_system_messages = [msg for msg in messages if msg.get('role') != 'system']
            
            # Take the last 10 messages
            last_ten_messages = non_system_messages[1:11]
    
            return last_ten_messages
    
        local_messages = messages.copy()
        sfw_prompt =  " Ensure the image description does not contain explicit content."
        if explicit:
            sfw_prompt= ""
            
        prompt = textwrap.dedent(f"""
        <Instruction>
        Analyze the current scene and character appearance to generate a updated detailed visual description optimized for image generation, making sure description is up to date to the latest message.

        Format your response as a structured JSON with the following categories:

        {{
            "ImageDescriptionKeywords": {{
                "Character": [
                    "identity descriptors",
                    "age and gender",
                    "ethnicity if relevant",
                    "body type and build"
                ],
                "Pose & Expression": [
                    "current body posture",
                    "facial expression",
                    "emotional state",
                    "current gestural details"
                ],
                "Physical Features": [
                    "facial features, generate if missing",
                    "body features and details, generate if missing",
                    "hair style and color, generate if missing",
                    "eye details, generate if missing",
                    "skin details, generate if missing"
                ],
                "Clothes": [
                    "detailed descriptions of clothing items if any, including types and visible named garments",
                    "specify colors and patterns, if any", 
                    "materials and textures, if any",
                    "accessories, if any"
                ],
                "Environment": [
                    "location details",
                    "lighting conditions",
                    "time of day",
                    "weather,only if relevant"
                ],
                "Artistic Style": [
                    "art direction",
                    "rendering style",
                    "mood and atmosphere"
                ]
                "MessageSummary": "concise summary of recent visual actions and gestures by the character",
            }}
        }}

        Keep descriptions precise and visually focused, using specific keywords that translate well to image generation.
        Track all appearance changes from previous interactions, omit removed details e.g. clothes and add new details.
        Maintain consistency with established character traits.
        Use high-density visual descriptors for maximum detail.
        Match the descriptors to the align with last message context: {local_messages[-1]['content']}
        Keep it short and consise and avoid repeating descriptions.{sfw_prompt}""").rstrip()

        image_description_response = ""
        local_messages.append({"role": "user", "content": prompt})
        for token in self.llm_handler.generate(local_messages,
                                               agent_config,
                                               temperature=0.2,
                                               model=agent_config.llm_config.reasoning_model,
                                               provider=agent_config.llm_config.reasoning_provider,
                                               max_tokens=800):
            image_description_response += token

        #print(image_description_response)
        cleaned_response = image_description_response.replace("```json", "").replace("```", "").strip()
        json_data = json.loads(cleaned_response)
        image_prompt_list = self.parse_appearance(json_data)
        image_prompt = ', '.join(filter(None, self.remove_duplicate_strings(image_prompt_list)))
        
        if not explicit:
            agent_config.image_config.negative_prompt = "((nsfw, explicit, uncensored, nudity, partial clothes)), "+ agent_config.image_config.negative_prompt
            agent_config.image_config.anime_negative_prompt = "((nsfw, explicit, uncensored, nudity,partial clothes)), "+ agent_config.image_config.anime_negative_prompt
            
        image_url = self.generate_image(image_prompt,agent_config,preallocated_image_name=preallocated_image_name)
        #print(image_url)
        if image_url:
            return f"![{image_prompt}]({image_url})"
        return None

    def remove_duplicate_strings(self, image_prompt_list: List[str]) -> List[str]:
        """Remove duplicate strings from the list."""
        return list(dict.fromkeys(image_prompt_list))
    
    def parse_appearance(self, appearance_obj):
        """Recursively traverse appearance object and join all non-empty string values, with error handling."""
        parts = []
        try:
            if isinstance(appearance_obj, dict):
                for value in appearance_obj.values():
                    parts.extend(self.parse_appearance(value))
            elif isinstance(appearance_obj, list):
                for item in appearance_obj:
                    parts.extend(self.parse_appearance(item))
            elif isinstance(appearance_obj, str) and appearance_obj.strip():
                parts.append(appearance_obj.strip())
        except Exception as e:
            print(f"Error processing appearance object: {str(e)}")
        return [part for part in parts if part]
        

    
    def format_messages_to_display(self,local_messages: list) -> str:
        formatted_messages = []
        for msg in local_messages:
            if msg['role'] != 'system':  # Skip system messages
                role = 'User' if msg['role'] == 'user' else 'Assistant'
                formatted_messages.append(f"{role}: {msg['content']}")
    
        return "\n".join(formatted_messages)

    def update_appearance_from_image_message(self, local_messages: List[dict],agent_config:AgentConfig):
        # Step 1: Find the last image message
        last_image_message = next((msg for msg in reversed(local_messages) if msg.get('tag') == 'image'), None)
        if last_image_message:
            # Step 2: Extract keywords from the markdown
            image_content = last_image_message['content']
            match = re.search(r'!\[(.*?)\]\((.*?)\)', image_content)
            if match:
                keywords = match.group(1)  # Extract keywords
                # Step 3: Update the appearance variable in system prompt
                for msg in local_messages:
                    if msg.get('role') == 'system':
                        # Append extracted keywords to appearance
                        if 'Appearance' in msg['content']:
                            msg['content'] = msg['content'].replace('Appearance: ', f"Appearance: {keywords},")
                            #print(msg['content'])
                        break  # Update only the first system message
        local_messages = [msg for msg in local_messages if msg.get('tag') != 'image'] 
        return local_messages
