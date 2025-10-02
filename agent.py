import os
from memory_manager import MemoryManager
from google import genai

#NOTE: Use your own API_KEY
os.environ["GOOGLE_API_KEY"] = ""

class AIAgent:
    def __init__(self):
        self.memory = MemoryManager()
        self.client = genai.Client()

    def process_request(self, request: str):
        episodic = self.memory.search_episodic(request, top_k=2)
        semantic = self.memory.get_semantic("refund_policy")
        procedure = self.memory.get_procedure("refund")

        context = f"""
        User request: {request}

        Episodic memory: {episodic}
        Semantic knowledge: {semantic}
        Procedural knowledge: {procedure}

        Based on the above, respond in a helpful, human-like way.
        """

        completion = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{{\"text\":\"{context}\"}}",
        )

        return {
            "episodic_memory": episodic,
            "semantic_memory": semantic,
            "procedural_memory": procedure,
            "response": completion.text
        }
