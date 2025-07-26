# EgoStarSea Agents - OpenAI Compatible Chat API

This document describes the OpenAI-compatible chat API implemented in `main.py`.

## Overview

The chat API provides OpenAI-compatible endpoints for interacting with A2A (Agent-to-Agent) agents. It supports both streaming and non-streaming chat completions.

## Features

- ✅ OpenAI-compatible `/v1/chat/completions` endpoint
- ✅ Streaming and non-streaming responses
- ✅ Model/agent management endpoints
- ✅ Health check endpoint
- ✅ CORS support
- 🚧 A2A agent integration (currently using echo responses)
- 🚧 Agent discovery service integration
- 🚧 Supabase database integration

## API Endpoints

### Chat Completions

#### POST `/v1/chat/completions`

OpenAI-compatible chat completions endpoint.

**Request Body:**
```json
{
  "model": "default-agent",
  "messages": [
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "stream": false,
  "temperature": 0.7,
  "max_tokens": null,
  "user": "optional-user-id"
}
```

**Response (Non-streaming):**
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "default-agent",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! I'm doing well, thank you for asking."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

**Response (Streaming):**
```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1234567890,"model":"default-agent","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1234567890,"model":"default-agent","choices":[{"index":0,"delta":{"content":" there!"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1234567890,"model":"default-agent","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### Models

#### GET `/v1/models`

List available models/agents.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "default-agent",
      "object": "model",
      "created": 1234567890,
      "owned_by": "egostarsea",
      "permission": [],
      "root": "default-agent",
      "parent": null
    }
  ]
}
```

### Agent Management

#### GET `/agents`

List all registered agents.

#### POST `/agents/register`

Register a new agent.

**Parameters:**
- `model_id`: Agent identifier
- `agent_url`: Agent URL

#### DELETE `/agents/{model_id}`

Unregister an agent.

#### GET `/agents/{model_id}/status`

Get agent status.

### System

#### GET `/health`

Health check endpoint.

## Usage Examples

### Python with requests

```python
import requests

# Non-streaming chat
response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "default-agent",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": False
})
print(response.json())

# Streaming chat
response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "default-agent",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": True
}, stream=True)

for line in response.iter_lines():
    if line.startswith(b'data: '):
        data = line[6:].decode('utf-8')
        if data == '[DONE]':
            break
        # Process chunk...
```

### curl

```bash
# Non-streaming
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default-agent",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'

# Streaming
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default-agent",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## Running the Server

```bash
# Install dependencies
pip install fastapi uvicorn loguru python-dotenv

# Run the server
python main.py

# Or with uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Testing

Run the test script to verify all endpoints:

```bash
python test_chat_api.py
```

## Current Limitations

1. **Mock Responses**: Currently using echo responses instead of actual A2A agent calls
2. **Agent Discovery**: Not yet integrated with the agent discovery service
3. **Database**: Not yet connected to Supabase for agent management
4. **Authentication**: No authentication/authorization implemented yet

## Next Steps

1. Integrate with A2A agent system
2. Connect to agent discovery service
3. Implement Supabase integration
4. Add authentication/authorization
5. Add proper error handling and logging
6. Add rate limiting and request validation
