import httpx
import asyncio
import os
import shortuuid
from character_loader import load_character_from_yaml, Character

API_URL = "https://mikpoik--modal-agent-fastapi-app-dev.modal.run/prompt"
AUTH_TOKEN = os.environ["API_KEY"]
WORKSPACE = "default704"
AGENT_ID = shortuuid.uuid()
CONTEXT_ID = AGENT_ID
TIMEOUT_SETTINGS = httpx.Timeout(
    timeout=300.0,  # 5 minutes total timeout
    connect=60.0,   # connection timeout
    read=300.0,     # read timeout
    write=60.0      # write timeout
)

async def send_message(client: httpx.AsyncClient, message: str, workspace_id: str = WORKSPACE):
    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": "application/json"
    }

    data = {
        "prompt": message,
        "workspace_id": workspace_id,
        "context_id": CONTEXT_ID,
        "agent_id": AGENT_ID,
    }

    try:
        print("***Response:***\n")
        async with client.stream('POST', API_URL, headers=headers, json=data, timeout=TIMEOUT_SETTINGS) as response:
            async for line in response.aiter_lines():
                if line:
                    print(line)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

async def init_character(client: httpx.AsyncClient, character_yaml: str):
    """Initialize the agent with a character from YAML"""
    character = load_character_from_yaml(character_yaml)
    if not character:
        print("Failed to load character, using defaults")
        return

    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": "application/json"
    }

    agent_config = {
        "context_id": CONTEXT_ID,
        "agent_id": AGENT_ID,
        "workspace_id": WORKSPACE,
        "character": character.to_dict()
    }

    init_url = "https://mikpoik--modal-agent-fastapi-app-dev.modal.run/init_agent"
    response = await client.post(init_url, headers=headers, json=agent_config, timeout=TIMEOUT_SETTINGS)
    if response.status_code == 200:
        print("Agent initialized\n***Chat***")
    else:
        print("Failed to initialize agent")

async def main():
    async with httpx.AsyncClient() as client:
        # Initialize with character from YAML
        await init_character(client, "test/characters/velvet.yaml")

        # Initial scene setup with structured prompt
        initial_prompt = """Now begin role-play with me."""
        await send_message(client, initial_prompt)


        while True:
            prompt = input("\nEnter your message ('exit' to quit): ")
            if prompt.lower() == 'exit':
                break
            await send_message(client, prompt)

if __name__ == "__main__":
    asyncio.run(main())