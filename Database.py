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
    
    def add_data_to(self, collection_name: str, data: dict):
        """
        Add data elements to a collection individually.
        
        Args:
            collection_name (str): Name of the collection
            data (dict): Dictionary containing:
                - embeddings: List of embedding vectors
                - contents: List of document contents
                - metadatas: List of metadata dictionaries
                - ids: List of unique identifiers (UUIDs)
        """
        try:
            collection = self.get_collection(collection_name)
            if collection is None:
                print(f"Creating new collection: {collection_name}")
                collection = self.api.create_collection(collection_name)
            
            # Verify all lists are the same length
            lengths = {
                'embeddings': len(data.get('embeddings', [])),
                'contents': len(data.get('contents', [])),
                'metadatas': len(data.get('metadatas', [])),
                'ids': len(data.get('ids', []))
            }
            
            if len(set(lengths.values())) != 1:
                raise ValueError(f"Inconsistent data lengths: {lengths}")
                
            if lengths['embeddings'] == 0:
                raise ValueError("No data to add")
                
            total_items = lengths['embeddings']
            
            for i in range(total_items):
                single_item = {
                    'embeddings': [data['embeddings'][i]],
                    'documents': [data['contents'][i]],
                    'metadatas': [data['metadatas'][i]],
                    'ids': [data['ids'][i]]
                }
                collection.add(**single_item)
            
            print(f"Successfully added {total_items} items to collection '{collection_name}'")
            
        except Exception as e:
            print(f"Error adding data to collection: {e}")
            raise
    
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

    def get_collection(self, collection_name: str):
        try:
            return self.api.get_collection(collection_name)
        except Exception as e:
            print(f"Error getting collection '{collection_name}': {e}")
            return None
    