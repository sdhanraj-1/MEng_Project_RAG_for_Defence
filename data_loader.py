# data_loader.py
import os
from Database import ChromaDataBase
from PreProcessing import preprocessor
from langchain.schema import Document

class Loader():

    def __init__(self, chroma_instance: ChromaDataBase):
        self.ChromaDataBase = chroma_instance
        self.files_read = []


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
    
    def insert_data_to_chroma(self, chunks: list[Document], batch_size: int = 100):
        if not chunks:
            print("No chunks to insert")
            return
            
        print(f"Processing {len(chunks)} chunks for insertion...")
        
        try:
            texts = [chunk.page_content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            
            self.ChromaDataBase.add_texts(
                texts=texts,
                metadatas=metadatas,
                batch_size=batch_size
            )
            
        except Exception as e:
            print(f"Error adding to ChromaDB: {e}")

    def load_all_documents(self, directory: str = "./LocalDocs"):
        """
        Loads all documents from the specified directory into the Chroma Database.
        
        Args:
            directory (str): Path to the directory containing documents. Defaults to "./LocalDocs"
        """
        print(f"Loading all documents from: {directory}")
        
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist")
            return
            
        # Walk through the directory and process all files
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                print(f"\nProcessing file: {file_path}")
                self.load_document(file_path)
                
        print("\nFinished loading all documents")


if __name__ == "__main__": 
    try:
        db = ChromaDataBase()
        DBLoader = Loader(db)
        
        # Test loading all documents
        print("\n=== Testing Loading All Documents ===")
        DBLoader.load_all_documents()
        
        """ test_path = "./LocalDocs/MIL-STD-882E.pdf"
        print(f"\nTesting with single file: {test_path}")
        
        # First test chunking
        print("\n=== Testing Chunking ===")
        chunks = DBLoader.chunk_document(path=test_path)
        if chunks:
            print(f"Successfully created {len(chunks)} chunks")
            print("First chunk preview:", chunks[0].page_content[:100])
        
        # Then test full loading
        print("\n=== Testing Full Loading ===")
        DBLoader.load_document(path=test_path) """
            
    except Exception as e:
        print(f"Main execution error: {e}")

