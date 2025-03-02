from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import RetrievalQA

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
    def get_qa_chain(self,db):
        #retriever = db.as_retriever()
        llm = OllamaLLM(model="llama3.2:1b", num_gpu = -1)


        '''system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Use three sentence maximum and keep the answer concise. "
        "Context: {context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        chain = create_retrieval_chain(self.retriever.getPipeline(), question_answer_chain)'''

        chain = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=self.retriever.getPipeline(),
                                 return_source_documents=True)

        return chain



if __name__ == "__main__":
    Chain = LLM_Chain()

    embeddings = OllamaEmbeddings(model="all-minilm")  # Using model for embeddings
    persist_directory = "./Chromadb"
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    qa_chain = Chain.get_qa_chain(db)
