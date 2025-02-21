# data_loader.py
from Database import ChromaDataBase
from PreProcessing import preprocessor
from langchain.schema import Document
from embeddings import embedder

class Loader():

    def __init__(self, chroma_instance: ChromaDataBase):
        self.ChromaDataBase = chroma_instance
        self.files_read = []
        self.embedder = embedder()


    def load_document(self, path: str):
        chunks = self.chunk_document(path)
        self.insert_data_to_chroma(chunks)
        print("Data added successfully.")


    def chunk_document(self, path: str):
        #file = open(path, "r")
        """bytes = file.read()
        if bytes:
            self.files_read.append(path)
            print("File read successfully. Loading into vector store...") """

        processor = preprocessor()
        chunks: list[Document] = processor.get_file_chunks(path)
        return chunks
    
    def insert_data_to_chroma(self, chunks: list[Document]):
        documents = []
        for index, chunk in enumerate(chunks):
            data={
                    "embedding": self.embedder.embedding(chunk.page_content),
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
            self.ChromaDataBase.add_data_to(
                "test",
                data=data
            )
        except Exception as e:
            print(e)


if __name__ == "__main__": 
    db = ChromaDataBase()
    DBLoader = Loader(db)
    DBLoader.chunk_document(path = "./LocalDocs/MIL-STD-882E.pdf")
    DBLoader.load_document(path = "./LocalDocs/MIL-STD-882E.pdf")
