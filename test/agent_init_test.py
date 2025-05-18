import requests
import os

API_URL = "https://valasalarakesh--modal-agent-fastapi-app-dev.modal.run/"
AUTH_TOKEN = os.environ["API_KEY"]

def init_agent():
    url = "https://valasalarakesh--modal-agent-fastapi-app-dev.modal.run/"
    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": "application/json"
    }

    agent_config = {
        # Populate with necessary fields
        "context_id": "default",
        "agent_id": "default",
        "workspace_id": "default95",
        "character": {
            "name": "Luna",
            "backstory": "Luna's primary goal is to assist, inform, and inspire the humans she interacts with",

        }
        # Add other configuration parameters as needed
    }

    response = requests.post(url, headers=headers, json=agent_config)

    if response.status_code == 200:
        print("Agent initialized or updated successfully")
        import json
        print(json.dumps(response.json(), indent=4))
    else:
        print("Failed to initialize or update the agent")
        print(response.status_code, response.text)
if __name__ == "__main__":
    init_agent()