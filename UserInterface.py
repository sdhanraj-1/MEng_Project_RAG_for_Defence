from data_loader import Loader
from Database import ChromaDataBase
from embeddings import embedder
from LLM import LLM_Chain
from langchain_chroma import Chroma
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
    chain= llm.get_qa_chain(db)


    while True:
        user_query = input("- Prompt: ")
        #query = {"input":user_query}
        response = chain.invoke(user_query)
        print("AI Response:", response['result'], "/n Source Docs:", response['source_documents'])
  