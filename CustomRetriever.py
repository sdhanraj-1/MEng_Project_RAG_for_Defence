from typing import List
from pydantic import Field, BaseModel

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever

class CompoundRetriever(BaseRetriever, BaseModel):
    k: int = Field(default=5, description="Number of documents to retrieve")
    embedder: OllamaEmbeddings = Field(default_factory=lambda: OllamaEmbeddings(
        model="all-minilm", 
        num_gpu=-1
    ))
    vectorstore: Chroma = Field(default_factory=lambda: Chroma(
        persist_directory="./Chromadb",
        embedding_function=OllamaEmbeddings(model="all-minilm", num_gpu=-1),
        collection_name="test"
    ))

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        Docs = self.vectorstore.similarity_search(query, k=200)
        retriever = BM25Retriever.from_documents(Docs)
        result = retriever.invoke(query)
        return result

# Example usage
if __name__ == "__main__": 
    # Using default k=5
    retriever = CompoundRetriever(k=5)
    
    result = retriever.invoke("software safety")
    print(result[0].page_content)