"""
Streamlit app: AI Agent Memory System with Neo4j + Generative Language (Gemini/Google)
embeddings & completions, now using the llama-index client for better API management.

Set your API key as an environment variable named `GOOGLE_API_KEY` (or `GEMINI_API_KEY` for compatibility).

Run: streamlit run streamlit_memory_agent_with_llamaindex_gemini.py
"""

import os
import time
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

import streamlit as st
import numpy as np
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# --- New Imports for llama-index client ---
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
# -----------------------------------------

load_dotenv()

# ---------------------- Configuration ----------------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
# Accept either name for the key to be flexible
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
API_KEY = (GOOGLE_API_KEY or "").strip()

# Memory types (canonical set)
MEMORY_TYPES = [
    "Short-term",
    "Semantic",
    "Episodic",
    "Procedural",
    "Working",
    "Contextual",
    "Temporal",
]

EMBEDDING_DIM = 1536

# ---------------------- Generative Language / Gemini Client ----------------------
class LLMClient:
    """
    Client wrapper for llama-index GoogleGenAI components.
    """
    def __init__(self, api_key: str):
        self.model = GoogleGenAI(model="gemini-2.5-flash", api_key=api_key)
        self.embed_model = GoogleGenAIEmbedding(model="embedding-001", api_key=api_key)

# Global Client Instance
LLM_CLIENT = None
if API_KEY:
    try:
        LLM_CLIENT = LLMClient(api_key=API_KEY)
    except Exception as e:
        print(f"Error initializing LLM client: {e}. Falling back to demo mode.")
        LLM_CLIENT = None
# ---------------------------------------------------------------------------------


# ---------------------- Neo4j Helper ----------------------
class Neo4jClient:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_memory_node(self, memory_id: str, user_id: str, text: str, memory_type: str, timestamp: float, embedding: List[float]):
        with self.driver.session() as session:
            session.run(
                """
                MERGE (u:User {id: $user_id})
                CREATE (m:Memory {id: $memory_id, text: $text, memory_type: $memory_type, timestamp: $timestamp, embedding: $embedding})
                MERGE (u)-[:HAS_MEMORY]->(m)
                RETURN m
                """,
                user_id=user_id,
                memory_id=memory_id,
                text=text,
                memory_type=memory_type,
                timestamp=timestamp,
                embedding=embedding,
            )

    def get_user_memories(self, user_id: str) -> List[Dict[str, Any]]:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (u:User {id: $user_id})-[:HAS_MEMORY]->(m:Memory)
                RETURN m.id AS id, m.text AS text, m.memory_type AS memory_type, m.timestamp AS timestamp, m.embedding AS embedding
                ORDER BY m.timestamp DESC
                """,
                user_id=user_id,
            )
            rows = result.data()
            for r in rows:
                if r.get("embedding") is None:
                    r["embedding"] = []
            return rows

    def get_memory_counts_by_type(self, user_id: str) -> Dict[str, int]:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (u:User {id: $user_id})-[:HAS_MEMORY]->(m:Memory)
                RETURN m.memory_type AS type, count(*) as c
                """,
                user_id=user_id,
            )
            return {r["type"]: r["c"] for r in result.data()}


def gemini_embedding(texts: List[str]) -> List[List[float]]:
    if not LLM_CLIENT:
        st.info("LLM Client not initialized — using random embeddings for demo.")
        rng = np.random.default_rng(42)
        return (rng.standard_normal((len(texts), EMBEDDING_DIM))).tolist()

    embeddings: List[List[float]] = []
    try:
        embeddings = [LLM_CLIENT.embed_model.get_text_embedding(text) for text in texts]
    except Exception as e:
        st.error(f"Embedding API request failed — using demo embeddings. Error: {e}")
        rng = np.random.default_rng(123)
        return (rng.standard_normal((len(texts), EMBEDDING_DIM))).tolist()

    if len(embeddings) != len(texts):
        st.warning("Embedding API returned unexpected count; using demo embeddings as fallback.")
        rng = np.random.default_rng(999)
        return (rng.standard_normal((len(texts), EMBEDDING_DIM))).tolist()

    return embeddings


def gemini_classify_memory_type(text: str) -> str:
    """
    Ask Generative Language API to classify the input text into one of MEMORY_TYPES via llama-index client.
    If client is not available, use a heuristic fallback.
    """
    if not LLM_CLIENT:
        t = text.lower()
        if any(k in t for k in ["how to", "step", "install", "setup", "configure", "to do", "instructions"]):
            return "Procedural"
        if any(k in t for k in ["remember", "once", "yesterday", "last", "met", "ago", "today", "tomorrow"]):
            return "Episodic"
        if any(k in t for k in ["definition", "is", "means", "refers", "term", "what is"]):
            return "Semantic"
        if len(text.split()) < 8:
            return "Short-term"
        return "Contextual"

    prompt = (
        "Classify the following user interaction into exactly ONE of these memory types:\n"
        + ", ".join(MEMORY_TYPES)
        + "\n\nInteraction:\n"
        + text
        + "\n\nAnswer with ONLY the exact label, e.g. 'Episodic'"
    )

    label = ""
    try:
        # Use llama-index complete method
        response = LLM_CLIENT.model.complete(prompt, max_tokens=64, temperature=0.0)
        label = response.text
    except Exception as e:
        st.error(f"Classification API request failed — using heuristic fallback. Error: {e}")
        return "Contextual"

    label = (label or "").strip()

    # Normalize the returned label to one of MEMORY_TYPES (existing logic)
    for mt in MEMORY_TYPES:
        if mt.lower() in label.lower():
            return mt

    # Try token matching
    if label:
        for mt in MEMORY_TYPES:
            if mt.split("-")[0].lower() in label.lower():
                return mt

    return "Contextual"


def gemini_generate_response(prompt: str, context: Optional[str] = None, max_tokens: int = 256) -> str:
    """
    Generate a response using Generative Language via llama-index client. Inserts selected memories into the prompt as context.
    Fallback: canned response when client fails.
    """
    combined_prompt = ""
    if context:
        combined_prompt += "Context:\n" + context + "\n\n"
    combined_prompt += "User prompt:\n" + prompt + "\n\nAssistant:"

    if not LLM_CLIENT:
        return "[DEMO] Generated response using provided memories: \n" + (context or "(no context)") + "\n\nAnswer: This is a demo answer."

    out_text = ""
    try:
        # Use llama-index complete method
        response = LLM_CLIENT.model.complete(combined_prompt, max_tokens=max_tokens, temperature=0.2)
        out_text = response.text
    except Exception as e:
        st.error(f"Generation API request failed — using demo response. Error: {e}")
        return "[DEMO] Could not call generation API; returning a demo response."

    return (out_text or "").strip() or "[DEMO] Generation returned empty text."

# ---------------------- Utility Functions ----------------------

def ensure_embedding_vector(vec: List[float]) -> List[float]:
    # ensure correct dimension
    if not isinstance(vec, list):
        return [0.0] * EMBEDDING_DIM
    if len(vec) == EMBEDDING_DIM:
        return vec
    # if returned dim differs, pad/truncate
    if len(vec) > EMBEDDING_DIM:
        return vec[:EMBEDDING_DIM]
    return vec + [0.0] * (EMBEDDING_DIM - len(vec))


def compute_cosine_similarities(query_emb: List[float], candidate_embs: List[List[float]]) -> List[float]:
    if len(candidate_embs) == 0:
        return []
    q = np.array(query_emb).reshape(1, -1)
    c = np.array(candidate_embs)
    sims = cosine_similarity(q, c)[0]
    return sims.tolist()

# ---------------------- Memory Decision Logic ----------------------

def select_memories_for_response(user_memories: List[Dict[str, Any]], user_query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    texts = [m["text"] for m in user_memories]
    embeddings = [ensure_embedding_vector(m.get("embedding", [])) for m in user_memories]

    if len(texts) == 0:
        return []

    q_emb = gemini_embedding([user_query])[0]
    q_emb = ensure_embedding_vector(q_emb)

    sims = compute_cosine_similarities(q_emb, embeddings)

    type_boost = {
        "Episodic": 1.2,
        "Contextual": 1.15,
        "Procedural": 1.1,
        "Semantic": 1.05,
        "Short-term": 1.1,
        "Working": 1.05,
        "Temporal": 1.05,
    }

    scores = []
    for m, s in zip(user_memories, sims):
        boost = type_boost.get(m.get("memory_type"), 1.0)
        age_seconds = time.time() - float(m.get("timestamp", time.time()))
        age_hours = age_seconds / 3600.0
        recency_boost = 1.0
        if age_hours < 1:
            recency_boost = 1.15
        elif age_hours < 24:
            recency_boost = 1.08
        score = s * boost * recency_boost
        scores.append(score)

    for m, score in zip(user_memories, scores):
        m["score"] = float(score)

    user_memories_sorted = sorted(user_memories, key=lambda x: x["score"], reverse=True)
    return user_memories_sorted[:top_k]

# ---------------------- Streamlit UI ----------------------

st.set_page_config(page_title="AI Agent Memory System", layout="wide")
st.title("AI Agent Memory System — Neo4j + llama-index/Gemini")

# Initialize Neo4j client
try:
    neo = Neo4jClient(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
except Exception as e:
    st.error(f"Could not connect to Neo4j: {e}")
    neo = None

with st.sidebar:
    st.header("Settings & Demo Data")
    user_id = st.text_input("User ID", value="user_123")
    ingest_sample = st.button("Ingest sample dataset for user")
    clear_user = st.button("Clear displayed user memories (UI only)")
    top_k = st.slider("Number of memories to use", 1, 10, 5)

# Sample dataset ingestion
if ingest_sample and neo:
    st.info("Ingesting sample interactions...")
    sample_texts = [
        "I met Anna at the cafe last Saturday and we talked about traveling to Italy.",
        "How to reset my router: unplug, wait 20s, plug back in.",
        "Photosynthesis is the process where plants convert light into chemical energy.",
        "Reminder: buy milk tomorrow morning.",
        "I updated my password on 2025-01-10 and saved it in my password manager.",
        "During the meeting, we decided to prioritize the Q3 roadmap including the search feature.",
        "To change the tire, loosen lug nuts, jack up car, replace tire, tighten nuts.",
        "User asked about the weather next week and we checked forecast for Budapest.",
        "I feel frustrated when my code doesn't run on first try — debugging helps me understand mistakes.",
    ]
    for t in sample_texts:
        mem_id = str(uuid.uuid4())
        ts = time.time()
        mtype = gemini_classify_memory_type(t)
        emb = gemini_embedding([t])[0]
        emb = ensure_embedding_vector(emb)
        neo.create_memory_node(mem_id, user_id, t, mtype, ts, emb)
    st.success("Sample dataset ingested.")

# Main interaction area
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("New interaction")
    user_query = st.text_area("User message / query", height=160)
    ingest_button = st.button("Ingest this interaction as memory")
    classify_button = st.button("Classify memory type (preview)")

    if classify_button and user_query:
        try:
            mtype = gemini_classify_memory_type(user_query)
            st.info(f"Predicted memory type: {mtype}")
        except Exception as e:
            st.error(f"Classification failed: {e}")

    if ingest_button and user_query and neo:
        mem_id = str(uuid.uuid4())
        ts = time.time()
        try:
            mtype = gemini_classify_memory_type(user_query)
        except Exception:
            mtype = "Contextual"
        emb = gemini_embedding([user_query])[0]
        emb = ensure_embedding_vector(emb)
        neo.create_memory_node(mem_id, user_id, user_query, mtype, ts, emb)
        st.success(f"Ingested memory as {mtype}")

    st.markdown("---")
    st.subheader("Ask a question (use memories to answer)")
    question = st.text_input("Question to the agent")
    answer_button = st.button("Get answer using selected memories")

    if answer_button:
        user_memories = neo.get_user_memories(user_id) if neo else []
        selected = select_memories_for_response(user_memories, question, top_k=top_k)
        context = "\n\n".join([f"[{m['memory_type']}] {m['text']}" for m in selected])
        response = gemini_generate_response(question, context=context)
        st.subheader("Agent response")
        st.write(response)
        st.markdown("---")
        st.subheader("Memories used (ranked)")
        for idx, m in enumerate(selected):
            st.markdown(f"**{idx+1}.** ({m['memory_type']}) — score: {m.get('score', 0):.3f}")
            st.write(m['text'])

with col2:
    st.subheader("User memory store (recent first)")
    if neo:
        rows = neo.get_user_memories(user_id)
        if rows:
            counts = neo.get_memory_counts_by_type(user_id)
            st.write("Memory counts by type:")
            st.write(counts)

            for r in rows:
                ts = datetime.fromtimestamp(float(r['timestamp'])).isoformat()
                st.markdown(f"- **{r['memory_type']}** — {ts} — id: {r['id']}")
                st.write(r['text'])
        else:
            st.info("No memories found for this user. Ingest some interactions.")
    else:
        st.warning("Neo4j not configured — cannot show stored memories.")

st.markdown("---")
if not API_KEY:
    st.warning("API Key not set. Running in demo mode with heuristic classification, random embeddings, and canned responses.")
else:
    st.info("API Key found. Using llama-index/Gemini for embedding and generation.")


# Export helper
import io
import zipfile

if st.button("Export user memories as JSON ZIP"):
    if neo:
        mems = neo.get_user_memories(user_id)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr(f"{user_id}_memories.json", json.dumps(mems, indent=2))
        buf.seek(0)
        st.download_button(label="Download memories ZIP", data=buf, file_name=f"{user_id}_memories.zip")
    else:
        st.warning("Neo4j not configured — nothing to export.")