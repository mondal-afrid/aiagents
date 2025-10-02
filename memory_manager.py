import os
import faiss
import numpy as np
from google import genai

# Note: Use your own API_KEY
os.environ["GOOGLE_API_KEY"] = ""

class MemoryManager:
    # FIX: Change the default dimension (dim) to 768 to match the 
    # output dimension of the 'text-embedding-004' model.
    def __init__(self, dim=768): 
        self.index = faiss.IndexFlatL2(dim)
        self.vectors = []
        self.texts = []
        self.semantic_knowledge = {}
        self.procedures = {}
        self.client = genai.Client()

    def embed(self, text: str):
        response = self.client.models.embed_content(
            model="text-embedding-004",
            contents=text
        )
        # Ensure we return the list of float values, not the embedding object
        embedding_values = response.embeddings[0].values
        # print(embedding_values) # For debugging
        return embedding_values

    def add_episodic(self, text):
        vector = self.embed(text)
        
        # vector is now a list of 768 floats, which FAISS can process.
        self.index.add(np.array([vector], dtype='float32'))
        self.vectors.append(vector)
        self.texts.append(text)

    def search_episodic(self, query, top_k=1):
        # Embed the query and ensure it's formatted as a 1x768 numpy array
        query_vector = np.array([self.embed(query)], dtype='float32')
        D, I = self.index.search(query_vector, top_k)
        return [self.texts[i] for i in I[0] if i < len(self.texts)]

    def add_semantic(self, knowledge):
        self.semantic_knowledge.update(knowledge)

    def get_semantic(self, key):
        return self.semantic_knowledge.get(key, None)

    def add_procedure(self, name, steps):
        self.procedures[name] = steps

    def get_procedure(self, name):
        return self.procedures.get(name, [])
