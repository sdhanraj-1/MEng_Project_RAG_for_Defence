from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate,FewShotChatMessagePromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from Retriever import retrieverPipeline

class LLM_Chain():
    """
        LLM Base class - Use this class to instance multiple-llms.
        [See HF repository to view all llm models]
    """
    def __init__(self): 
        self.retriever = retrieverPipeline()

    # text generation
    def get_qa_chain(self):
        #retriever = db.as_retriever()
        llm = OllamaLLM(model="llama3.2:3b", num_gpu = -1)

        # Create a more detailed prompt that explicitly uses the context
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant. Use the following pieces of context to answer the user's question. 
            If you don't know the answer or can't find it in the context, just say "I don't have enough information to answer that."
            Always base your answer on the context provided, not on prior knowledge.
            
            Context: {context}"""),
            ("human", "{input}")
        ])

        # Create the document chain
        document_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=prompt,
        )

        # Create the retrieval chain
        retrieval_chain = create_retrieval_chain(
            retriever=self.retriever.getPipeline(),
            combine_docs_chain=document_chain
        )

        return retrieval_chain



if __name__ == "__main__":
    Chain = LLM_Chain()

    embeddings = OllamaEmbeddings(model="all-minilm")  # Using model for embeddings
    persist_directory = "./Chromadb"
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    qa_chain = Chain.get_qa_chain()
