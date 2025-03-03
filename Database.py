# chroma.py
from langchain_chroma import Chroma 
from langchain_community.embeddings import OllamaEmbeddings

class ChromaDataBase():
    """
    Chroma class using Langchain's implementation.
    """
    def __init__(self, persist_directory: str = "./Chromadb", collection_name: str = "test"):
        self.embeddings = OllamaEmbeddings(
            model="all-minilm", 
            num_gpu=-1,
        )
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
    
    def add_texts(self, texts: list[str], metadatas: list[dict] = None, batch_size: int = 100):
        """
        Add texts to the vector store in batches.
        
        Args:
            texts (list[str]): List of text content to add
            metadatas (list[dict], optional): List of metadata dictionaries
            batch_size (int): Size of batches for processing
        """
        try:
            total = len(texts)
            for i in range(0, total, batch_size):
                end_idx = min(i + batch_size, total)
                batch_texts = texts[i:end_idx]
                batch_metadatas = metadatas[i:end_idx] if metadatas else None
                
                self.vectorstore.add_texts(
                    texts=batch_texts,
                    metadatas=batch_metadatas
                )
                print(f"Processed batch {i//batch_size + 1} ({end_idx}/{total} documents)")
            
            print(f"Successfully added all {total} documents")
        except Exception as e:
            print(f"Error adding texts: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 4):
        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            print(f"Error during search: {e}")
            return []
    
    def get_collection(self):
        return self.vectorstore
    