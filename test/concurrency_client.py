# concurrency_client.py
import httpx
import asyncio
import os
import shortuuid
from character_loader import load_character_from_yaml, Character
from typing import List, Dict

API_URL = "https://valasalarakesh--modal-agent-fastapi-app-dev.modal.run/prompt"
AUTH_TOKEN = os.environ["API_KEY"]
TIMEOUT_SETTINGS = httpx.Timeout(
    timeout=300.0,  # 5 minutes total timeout
    connect=60.0,   # connection timeout
    read=300.0,     # read timeout
    write=60.0      # write timeout
)

class ConcurrentClient:
    def __init__(self, base_url: str = API_URL, auth_token: str = AUTH_TOKEN):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json"
        }

    def generate_workspace(self) -> Dict[str, str]:
        """Generate unique workspace identifiers"""
        return {
            "workspace_id": f"test_{shortuuid.uuid()}",
            "context_id": f"ctx_default",
            "agent_id": f"agent_{shortuuid.uuid()}"
        }

    async def init_character_concurrent(self, client: httpx.AsyncClient, character_yaml: str, workspace_ids: Dict[str, str]):
        """Initialize the agent with a character from YAML with specific workspace IDs"""
        character = load_character_from_yaml(character_yaml)
        if not character:
            print("Failed to load character, using defaults")
            return

        init_url = "https://valasalarakesh--modal-agent-fastapi-app-dev.modal.run/init_agent"
        agent_config = {
            **workspace_ids,
            "character": character.to_dict()
        }

        response = await client.post(init_url, headers=self.headers, json=agent_config, timeout=TIMEOUT_SETTINGS)
        if response.status_code == 200:
            print(f"Initialized agent in workspace: {workspace_ids['workspace_id']}")
        else:
            print(f"Failed to initialize agent in workspace: {workspace_ids['workspace_id']}")

    async def send_message_concurrent(self, client: httpx.AsyncClient, message: str, workspace_ids: Dict[str, str]) -> str:
        """Send a message with specific workspace IDs and return the response"""
        data = {
            "prompt": message,
            **workspace_ids
        }

        response_text = ""
        try:
            async with client.stream('POST', self.base_url, headers=self.headers, json=data, timeout=TIMEOUT_SETTINGS) as response:
                async for line in response.aiter_lines():
                    if line:
                        response_text += line + "\n"
        except Exception as e:
            response_text = f"Error in workspace {workspace_ids['workspace_id']}: {str(e)}"

        return response_text

    async def run_concurrent_test(self, num_users: int, character_yaml: str, test_message: str):
        """Run a concurrent test with multiple simulated users"""
        async with httpx.AsyncClient() as client:
            # Generate unique workspaces for each user
            workspaces = [self.generate_workspace() for _ in range(num_users)]

            # Initialize characters
            init_tasks = [
                self.init_character_concurrent(client, character_yaml, workspace)
                for workspace in workspaces
            ]
            await asyncio.gather(*init_tasks)

            # Send test messages
            message_tasks = [
                self.send_message_concurrent(
                    client,
                    f"{test_message} (User {i+1})",
                    workspace
                )
                for i, workspace in enumerate(workspaces)
            ]

            # Collect responses
            responses = await asyncio.gather(*message_tasks)

            # Print results
            for i, response in enumerate(responses):
                print(f"\n--- User {i+1} (Workspace: {workspaces[i]['workspace_id']}) ---")
                print(response[:200] + "..." if len(response) > 200 else response)

async def main():
    # Create concurrent test client
    concurrent_client = ConcurrentClient()

    # Run concurrent test
    print("Starting concurrent user test...")
    await concurrent_client.run_concurrent_test(
        num_users=3,
        character_yaml="test/characters/velvet.yaml",
        test_message="""Narrate a brief scene showing the character's personality.
Keep it concise but engaging."""
    )

if __name__ == "__main__":
    asyncio.run(main())