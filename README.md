# AI Character Interaction System

A modal-based system for creating and interacting with AI characters with support for text and image generation.

## Features

- Character-based AI interactions with configurable personalities
- YAML-based character configuration
- Image generation support for character avatars
- Streaming responses
- Modal-based deployment architecture
- FastAPI backend integration

## Prerequisites

- Python 3.10 or higher
- Poetry for dependency management
- Modal.com account
- Required API keys:
  - DeepInfra API key
  - FAL.ai API key
  - OpenAI API key (optional)
  - Google Cloud Storage credentials

## Installation

1. Clone the repository
2. Install dependencies using Poetry:
```bash
poetry install
```

## Configuration

Environment Variables
Required environment variables :

- API_KEY: Authentication token (made up token to authenticate with app endpoint)
- MODAL_TOKEN_ID (modal auth)
- MODAL_TOKEN_SECRET
- Deep Infra API key (to Modal secrets)
- FAL.ai API key (to Modal secrets)
- GCP credentials (for image storage in Modal secrets)


## Usage
Start the server:
```
modal serve src.app
```
To deploy:
```
modal deploy src.app
```

## Project Structure
- src/
  - app.py: Main FastAPI application
  - agent/: Modal agent implementation
  - handlers/: Image and LLM handling logic
  - models/: Data models and schemas
- test/
  - characters/: Character YAML files
  - client.py: Test client implementation
  - character_loader.py: YAML character loading utilities
- API Endpoints
  - /init_agent: Initialize or update an agent
  - /prompt: Send prompts to the agent
  - /generate_avatar: Generate character avatars