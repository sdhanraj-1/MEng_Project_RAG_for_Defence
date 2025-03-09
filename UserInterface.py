from data_loader import Loader
from Database import ChromaDataBase
from embeddings import embedder
from LLM import LLM_Chain
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import os

ChromaInstance = ChromaDataBase()
LoaderInstance = Loader(chroma_instance=ChromaInstance)

def create_database():
    folder_path = './LocalDocs'
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            LoaderInstance.load_document(root + "/" + file)
    
    pass


#create_database()

if __name__ == "__main__":
    llm = LLM_Chain()
    embeddings = OllamaEmbeddings(model="all-minilm") 
    db = Chroma(persist_directory="./Chromadb", embedding_function=embeddings)
    chain = llm.get_qa_chain()

    while True:
        user_query = input("- Prompt: ")
        response = chain.invoke({"input": user_query})
        print("\nAI Response:", response['answer'])
        
        print("\nSources:")
        for doc in response['context']:
            print("\n---")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Title: {doc.metadata.get('title', 'Unknown')}")
            print(f"Relevance: {doc.metadata.get('relevance_score', 'Unknown')}")
            print(f"Content: {doc.page_content[:200]}...")  # First 200 chars of content
  