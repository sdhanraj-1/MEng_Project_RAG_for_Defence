from langchain_community.document_transformers.embeddings_redundant_filter import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_transformers.long_context_reorder import LongContextReorder
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever
from Reranker import BgeRerank
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

class retrieverPipeline(): 
    def __init__(self): 
        self.embedder = OllamaEmbeddings(model="all-minilm") 
        self.vectorstore = Chroma(persist_directory="./Chromadb", embedding_function= self.embedder)
        self.redundant_filter = EmbeddingsRedundantFilter(embeddings= self.embedder)
        self.reordering = LongContextReorder()
        self.reranker = BgeRerank()
        

    def singularRetriever(self): 
        singular_retriever = self.vectorstore.as_retriever(search_kwargs={"k":10})
        return singular_retriever

    def multipleSourceRetriever(self):
        #vs_retriever = vectorstore.as_retriever(search_kwargs={"k":10})
        #bm25_retriever = 
        #ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever,vs_retriever], weight=[0.5,0.5])
        #return ensemble_retriever
        pass

    def getPipeline(self): 
        retriever = self.singularRetriever()
        #pipeline_compressor = DocumentCompressorPipeline(transformers=[self.redundant_filter,self.reordering,self.reranker])
        #compression_pipeline = ContextualCompressionRetriever(base_compressor=pipeline_compressor,
        #                                               base_retriever=retriever)
        return retriever

    def pretty_print_docs(self,docs):
        print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n + {d.page_content}" for i,d in enumerate(docs)]))