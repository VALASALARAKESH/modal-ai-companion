from typing import Optional,Union
import modal
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import os
import json


from src.agent.modal_agent import ModalAgent
from src.models.schemas import app,  AgentConfig,PromptConfig, LLMConfig, volume

web_app = FastAPI()
modal_agent = ModalAgent()

image = modal.Image.debian_slim(python_version="3.10").pip_install(
   "pydantic==2.6.4",
   "fastapi==0.114.0",
    "requests",
   "shortuuid",
)


# Set up the HTTP bearer scheme
http_bearer = HTTPBearer(
    scheme_name="Bearer Token",
    description="Enter your API token",
)


# Authentication dependency
async def authenticate(credentials: HTTPAuthorizationCredentials = Depends(http_bearer)):
    if credentials.credentials != os.environ["auth_token"]:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
        )
    return credentials.credentials

@web_app.post("/moderate_character")
async def moderate_character(
    agent_config:PromptConfig,
    credentials: str = Depends(authenticate)):
    print("POST /moderate_input")
    if not check_agent_config(agent_config):
        raise HTTPException(status_code=400, detail="Invalid agent configuration POST /prompt")
    return modal_agent.moderate_character.remote(agent_config)
    
@web_app.post("/generate_avatar")
async def generate_avatar(
    agent_config:PromptConfig,credentials: str = Depends(authenticate)):
    print("POST /generate_avatar")
    try:   
        avatar_url = modal_agent.generate_avatar.remote(agent_config)
        return avatar_url
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@web_app.post("/init_agent")
async def init_agent(agent_config: AgentConfig, credentials: str = Depends(authenticate)):
    if not check_agent_config(agent_config):
        raise HTTPException(status_code=400, detail="Invalid agent configuration POST /init_agent")
    print("POST /init_agent ", agent_config.context_id, agent_config.agent_id, agent_config.workspace_id)
    return modal_agent.get_or_create_agent_config.remote(agent_config, update_config=True)
    
@web_app.post("/get_chat_history")
async def get_chat_history(agent_config: AgentConfig, credentials: str = Depends(authenticate)):
    if not check_agent_config(agent_config):
        raise HTTPException(status_code=400, detail="Invalid agent configuration POST /get_chat_history")
    print("POST /get_chat_history ", agent_config.context_id, agent_config.agent_id, agent_config.workspace_id)
    return modal_agent.get_chat_history.remote(agent_config)

@web_app.post("/append_chat_history")
async def append_chat_history(agent_config: AgentConfig, credentials: str = Depends(authenticate)):
    #TODO get messages from kwargs
    
    if not check_agent_config(agent_config):
        raise HTTPException(status_code=400, detail="Invalid agent configuration POST /append_chat_history")
    print("POST /append_chat_history ", agent_config.context_id, agent_config.agent_id, agent_config.workspace_id)
    return modal_agent.append_chat_history.remote(agent_config,**agent_config.kwargs)
                           
@web_app.post("/delete_chat_history")
async def delete_chat_history(agent_config: AgentConfig,credentials: str = Depends(authenticate)):
    is_valid = check_agent_config(agent_config)
    if not is_valid:
        raise HTTPException(status_code=400, detail="Invalid agent configuration POST /delete_chat_history")
    print("POST delete_chat_history ", agent_config.context_id, agent_config.agent_id, agent_config.workspace_id)
    return modal_agent.delete_chat_history.remote(agent_config)

@web_app.post("/delete_workspace")
async def delete_workspace(agent_config: AgentConfig,credentials: str = Depends(authenticate)):
    is_valid = check_agent_config(agent_config)
    if not is_valid:
        raise HTTPException(status_code=400, detail="Invalid agent configuration POST /delete_workspace")
    print("POST delete_workspace ", agent_config.context_id, agent_config.agent_id,agent_config.workspace_id)
    return modal_agent.delete_workspace.remote(agent_config)

@web_app.post("/delete_message_pairs")
async def delete_message_pairs(agent_config: AgentConfig,credentials: str = Depends(authenticate)):
    is_valid = check_agent_config(agent_config)
    if not is_valid:
        raise HTTPException(status_code=400, detail="Invalid agent configuration POST /delete_message_pairs")
    print("POST /delete_message_pairs ", agent_config.context_id, agent_config.agent_id, agent_config.workspace_id)
    try:
        result = modal_agent.delete_message_pairs.remote(agent_config, **agent_config.kwargs)
        return {"success": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@web_app.post("/prompt")
async def prompt(agent_config:PromptConfig, token: str = Depends(authenticate)):
    
    if not check_agent_config(agent_config):
        raise HTTPException(status_code=400, detail="Invalid agent configuration POST /prompt")
    
    print("POST /prompt ", agent_config.context_id, agent_config.agent_id, agent_config.workspace_id)
    def stream_generator():
        try:
            for token in modal_agent.run.remote_gen(agent_config):
                yield token
        except Exception as e:
            yield f"\ndata: Error: {str(e)}\n\n"
    return StreamingResponse(stream_generator(), media_type="text/event-stream")
    
def check_agent_config(agent_config: Union[AgentConfig, PromptConfig]) -> bool:
    """
    Check if essential agent configuration parameters are defined.
    Returns True if all required fields are present, False otherwise.
    """
    missing_fields = []
    if not agent_config:
        return False
    if not agent_config.context_id:
        missing_fields.append("context_id")
    if not agent_config.workspace_id:
        missing_fields.append("workspace_id")
    if not agent_config.agent_id:
        missing_fields.append("agent_id")
        
    if isinstance(agent_config, PromptConfig) and not agent_config.prompt:
        missing_fields.append("prompt")
        
    if missing_fields:
        print(f"Warning: Missing required agent configuration fields: {', '.join(missing_fields)}")
        return False

    return True
    

@app.function(
    timeout=60 * 5,
    container_idle_timeout=60 * 15,
    allow_concurrent_inputs=100,
    image=image,
    secrets=[modal.Secret.from_name("fast-api-secret")],
    volumes={"/data": volume}
)
@modal.asgi_app()
def fastapi_app():
    return web_app


if __name__ == "__main__":
    app.deploy("webapp")