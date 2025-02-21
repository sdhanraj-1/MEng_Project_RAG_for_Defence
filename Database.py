# chroma.py
from chromadb import PersistentClient, ClientAPI

class ChromaDataBase():
    """
      Chroma class to instantiate a vector db in memory.
    """
    def __init__(self, default_database: str = "default", first_collection_name: str = "test", top_k: int = 1):
        self.api: ClientAPI = PersistentClient("./Chromadb")
        self.collection_pointer = self.api.get_or_create_collection(first_collection_name)
        self.top_k = top_k
    
    def new_collection(self ,name: str, **kwargs):
        try:
            self.api.create_collection(name, **kwargs)
        except Exception as e:
            print(e)
    
    def add_data_to(self, data):
        try:
            self.collection_pointer.add(
                embeddings=data.get("embeddings"),
                documents=data.get("contents"),
                metadatas=data.get("metadatas"),
                ids=data.get("ids")
            )
        except Exception as e:
            print(e)
    
    def switch_collection(self, new_pointer: str):
        try:
            self.collection_pointer = self.api.get_collection(new_pointer)
        except Exception as e:
            print(e)
    
    def query(self, embedding: list[float], **kwargs):
        try:
            result = self.collection_pointer.query(query_embeddings=embedding, n_results=self.top_k, **kwargs)
            print(result)
        except Exception as e:
            print(e)