# data_loader.py
from chroma import Chroma
from chunking import get_file_chunks
from langchain.schema import Document
from embeddings import embedding

class Loader():

    def __init__(self, chroma_instance: Chroma):
        self.chroma = chroma_instance
        self.files_read = []
    
    def load_document(self, path: str):
        chunks = self.chunk_document(path)
        self.insert_data_to_chroma(chunks)
        print("Data added successfully.")

    def chunk_document(self, path: str):
        file = open(path, "rb")
        bytes = file.read()
        if bytes:
            self.files_read.append(path)
            print("File read successfully. Loading into vector store...")

        chunks: list[Document] = get_file_chunks(bytes, path)
        return chunks
    
    def insert_data_to_chroma(self, chunks: list[Document]):
        documents = []
        for index, chunk in enumerate(chunks):
            data={
                    "embedding": embedding(chunk.page_content),
                    "content": chunk.page_content,
                    "metadata": chunk.metadata,
                    "id": index
                }
            documents.append(data)

        embeddings = [data["embedding"] for data in documents]
        contents = [data["content"] for data in documents]
        metadatas = [data["metadata"] for data in documents]
        ids = [str(data["id"]) for data in documents]

        data={
            "embeddings":embeddings,
            "contents":contents,
            "metadatas":metadatas,
            "ids":ids
        }
        try:
            self.chroma.add_data_to(
                "test",
                data=data
            )
        except Exception as e:
            print(e)