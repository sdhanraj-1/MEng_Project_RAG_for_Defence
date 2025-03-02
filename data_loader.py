# data_loader.py
from Database import ChromaDataBase
from PreProcessing import preprocessor
from langchain.schema import Document
from embeddings import embedder
from uuid import uuid4

class Loader():

    def __init__(self, chroma_instance: ChromaDataBase):
        self.ChromaDataBase = chroma_instance
        self.files_read = []
        self.embedder = embedder()


    def load_document(self, path: str):
        print(f"Starting to load document from: {path}")
        chunks = self.chunk_document(path)
        if not chunks:
            print("No chunks were created from the document")
            return
        print(f"Created {len(chunks)} chunks")
        self.insert_data_to_chroma(chunks)


    def chunk_document(self, path: str):
        try:
            processor = preprocessor()
            chunks: list[Document] = processor.get_file_chunks(path)
            if not chunks:
                print("Warning: No chunks were generated from the file")
            return chunks
        except Exception as e:
            print(f"Error during chunking: {e}")
            return []
    
    def insert_data_to_chroma(self, chunks: list[Document]):
        if not chunks:
            print("No chunks to insert")
            return
            
        documents = []
        print(f"Processing {len(chunks)} chunks for insertion...")
        
        for chunk in chunks:
            try:
                embedding = self.embedder.embedding(chunk.page_content)[0]
                data = {
                    "embedding": embedding,
                    "content": chunk.page_content,
                    "metadata": chunk.metadata,
                    "id": str(uuid4())
                }
                documents.append(data)
            except Exception as e:
                print(f"Error processing chunk: {e}")

        if not documents:
            print("No documents were processed successfully")
            return

        print(f"Preparing {len(documents)} documents for insertion")
        
        data = {
            "embeddings": [d["embedding"] for d in documents],
            "contents": [d["content"] for d in documents],
            "metadatas": [d["metadata"] for d in documents],
            "ids": [d["id"] for d in documents]
        }
        
        try:
            self.ChromaDataBase.add_data_to(
                "test",
                data=data
            )
            print(f"Successfully added {len(documents)} documents to Chroma")
        except Exception as e:
            print(f"Error adding to ChromaDB: {e}")


if __name__ == "__main__": 
    try:
        db = ChromaDataBase()
        DBLoader = Loader(db)
        
        test_path = "./LocalDocs/MIL-STD-882E.pdf"
        print(f"\nTesting with file: {test_path}")
        
        # First test chunking
        print("\n=== Testing Chunking ===")
        chunks = DBLoader.chunk_document(path=test_path)
        if chunks:
            print(f"Successfully created {len(chunks)} chunks")
            print("First chunk preview:", chunks[0].page_content[:100])
        
        # Then test full loading
        print("\n=== Testing Full Loading ===")
        DBLoader.load_document(path=test_path)
        
        # Check if data exists in the database
        print("\n=== Checking Database ===")
        try:
            collection = db.get_collection("test")
            count = collection.count()
            print(f"Documents in collection: {count}")
        except Exception as e:
            print(f"Error checking collection: {e}")
            
    except Exception as e:
        print(f"Main execution error: {e}")

