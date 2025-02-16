from data_loader import Loader
from Database import ChromaDataBase
from embeddings import embedder
from LLM import LLM_Chain
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

ChromaInstance = ChromaDataBase()
LoaderInstance = Loader(chroma_instance=ChromaInstance)

# change it with your text file or pdf file path.
example_path = "./LocalDocs/MIL-STD-882E.pdf"

LoaderInstance.load_document(example_path)


llm = LLM_Chain()
embeddings = OllamaEmbeddings(model="all-minilm") 
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
chain= llm.get_qa_chain(db)


while True:
    user_query = input("- Prompt: ")
    query = {"input":user_query}
    response = chain.invoke(query)
    print("AI Response:", response)
  