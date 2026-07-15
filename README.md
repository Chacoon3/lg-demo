# lg-demo

A Python starter project for building and serving multi-agent workflows with LangGraph.

This repo includes:
- A reusable graph runtime in `src/lg_demo/core`
- Three example agents (general arithmetic, finance, travel)
- A FastAPI service for invoking agents over HTTP
- A local interactive CLI agent for workspace tasks
- Unit tests for runtime and API layers

## Project Layout

- `src/lg_demo/agents`: Example agent definitions and the `AgentRegistry`
- `src/lg_demo/core`: Runtime graph builder, routers, nodes, model providers, tools, and state models
- `src/lg_demo/utils`: Disk caching and DAG helpers used by agents
- `src/agent_api`: FastAPI app, routes, middleware, dependencies, and response helpers
- `src/agent_cli/cli_entry.py`: Interactive CLI software-engineering agent
- `tests/unit`: Unit tests for API and core runtime behavior
- `.vscode/launch.json`: Preconfigured debug profiles

## How It Works

At startup, the API builds an `AgentRegistry` and initializes:
- `general_agent`: arithmetic tool-augmented agent
- `finance_agent`: web-search-enabled market summary agent
- `travel_agent`: plan-and-execute travel workflow with DAG state

The runtime uses a graph model where:
- nodes are inference or tool execution steps
- routers define entry, direct edges, and tool-call transitions
- `RuntimeBuilder` compiles the graph into a `CompiledStateGraph`

## Requirements

- Python 3.11+
- A virtual environment (recommended)
- One model backend configured via environment variables

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
# or: .venv\\Scripts\\activate  # Windows cmd

pip install -e .
pip install -e .[dev]
```

## Environment Variables

Create `.env.local` in the project root.

### Minimum for API default (Ollama)

```env
OLLAMA_MODEL=llama3.1
```

### Optional model providers

```env
# OpenAI provider
OPENAI_MODEL=gpt-5-mini
OPENAI_API_KEY=...

# Hugging Face router provider
HF_CLOUD_MODEL=...
HF_TOKEN=...
```

### Optional web search tool (finance/travel quality)

```env
TAVILY_API_KEY=...
```

## Run the API

```bash
uvicorn agent_api.main:app --host 127.0.0.1 --port 8000 --reload
```

Open docs at:
- `http://127.0.0.1:8000/docs`

Health check:

```bash
curl http://127.0.0.1:8000/agent_api/prompt/health_check
```

## API Endpoints

Base path: `/agent_api/prompt`

- `GET /health_check`: service health
- `GET /graph?agent_class=general|finance`: returns graph PNG
- `POST /general`: run arithmetic/general agent
- `POST /finance`: run finance agent
- `POST /travel`: run travel agent

Example request:

```bash
curl -X POST http://127.0.0.1:8000/agent_api/prompt/general \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is (7 + 9) * 3?","debug":false}'
```

## Run the CLI Agent

```bash
python src/agent_cli/cli_entry.py
```

CLI commands:
- `/new`: start a fresh conversation thread
- `/clear`: clear terminal output
- `/exit` or `/quit`: exit

The CLI agent exposes sandboxed tools:
- `list_files`
- `read_file`
- `write_file`
- `run_command` (allowlist: `python`, `pytest`, `ruff`, `git`)

## Testing

Run all tests:

```bash
pytest -q tests
```

Targeted suites:

```bash
pytest -q tests/unit/agent_api
pytest -q tests/unit/lg_demo/core
```

## Lint and Formatting

```bash
ruff check src tests
black src tests
isort src tests
```

## Debugging in VS Code

Use predefined launch configs in `.vscode/launch.json`:
- `LG Demo: FastAPI (Uvicorn)`
- `LG Demo: CLI Agent`
- `LG Demo: Pytest (all tests)`

## Notes for New Contributors

- Keep graph changes small and testable; update related tests in `tests/unit/lg_demo/core`.
- If you add new tools that require secrets, document required env vars in this README.
- API responses are wrapped by response helpers and middleware; preserve that pattern for new routes.
