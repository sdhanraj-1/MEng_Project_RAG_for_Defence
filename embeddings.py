# embedding.py  
from os import environ as env
from langchain_ollama import OllamaEmbeddings
import time

class embedder:
    def __init__(self, max_retries=3): 
        self.max_retries = max_retries
        self.ollama_embed = None
        self._initialize_embeddings()

    def _initialize_embeddings(self):
        for attempt in range(self.max_retries):
            try:
                self.ollama_embed = OllamaEmbeddings(model="all-minilm", num_gpu=-1)
                return
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise Exception(f"Failed to initialize embeddings after {self.max_retries} attempts: {str(e)}")
                time.sleep(2)  # Wait 2 seconds before retrying

    def embedding(self, input: str):
        try:
            return self.ollama_embed.embed_documents([input])
        except Exception as e:
            raise Exception(f"Error during embedding: {str(e)}")

if __name__ == "__main__":
    OllamaEmbedder = embedder()
    embeddedRes = OllamaEmbedder.embedding("checking embeddings")
    print(embeddedRes)
    