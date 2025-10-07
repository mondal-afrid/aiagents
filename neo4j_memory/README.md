# LlamaIndex + Neo4j + Streamlit Memory Agent
# README — streamlit_memory_agent_with_neo_4_j.py

This README documents how to run the Streamlit app `memory_agent_with_neo_4_j.py` in this repository.

## Purpose
This Streamlit app demonstrates an AI Agent Memory System that stores and queries user memories in Neo4j and (optionally) uses Gemini / Google generative models via the llama-index client for embeddings and generation.

## Prerequisites
- Neo4j must be running and accessible to the app (Bolt protocol). The app expects to connect using the `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD` environment variables (see below).
- Python dependencies listed in `requirements.txt` should be installed.
- (Optional) A Gemini/Google API key for embeddings/generation — if not provided the app runs in demo mode with heuristic classification and random embeddings.

## .env (copy to project root and fill)
Create a `.env` file in the repository root and add the following (fill values where required):

# Copy to .env and fill
GEMINI_API_KEY=
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=
STREAMLIT_PORT=8501

Notes:
- The code also accepts `GOOGLE_API_KEY` (it checks `GOOGLE_API_KEY` first, then `GEMINI_API_KEY`).
- `STREAMLIT_PORT` is the port used when launching Streamlit. The default shown above is `8501`.

## Quick start (PowerShell)
1. Install dependencies:

```powershell
pip install -r requirements.txt
```

2. Create a `.env` file in the repository root with the values shown above.

3. Start the Streamlit app (uses the `STREAMLIT_PORT` environment variable):

```powershell
$env:STREAMLIT_PORT = 8501; streamlit run streamlit_memory_agent_with_neo_4_j.py --server.port $env:STREAMLIT_PORT
```

4. Open your browser at `http://localhost:8501` (or the port you set).

## Troubleshooting
- If the app cannot connect to Neo4j, confirm Neo4j is running and that the `NEO4J_URI`, `NEO4J_USER` and `NEO4J_PASSWORD` values are correct and reachable from this machine.
- If you don't provide an API key (`GEMINI_API_KEY` or `GOOGLE_API_KEY`) the app will run in demo mode (random embeddings and canned responses).