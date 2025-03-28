# chunking.py
import os
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
import PyPDF2
from langchain.schema import Document
import re


class preprocessor: 
    def __init__(self):
        self.use_semantic_chunking = 1
        if self.use_semantic_chunking:
            self.embeddings = OpenAIEmbeddings()

    def get_text_splitter(self, chunk_size: int, chunk_overlap: int):
        """Create a consistent text splitter with the given parameters."""
        if self.use_semantic_chunking:
            return SemanticChunker(
                embeddings=self.embeddings,
                breakpoint_threshold_type="percentile"
            )
        else:
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ".", " ", ""]
            )

    def extract(self, file, ext: str, chunking:int=5000, overlap:int=50, **kwargs):
        if ext == "pdf":
            return self.extract_pdf(file, chunking, overlap, **kwargs)
        if ext == "txt":
            return self.extract_text(file, chunking, overlap, **kwargs)
        return self.extract_csv(file, chunking, overlap, **kwargs)

    def extension(self, file_name:str):
        return file_name.split(".")[-1]

    def clean_text(self, text: str) -> str:
        """Clean extracted text by removing extra whitespace and unwanted characters."""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove multiple newlines
        text = re.sub(r'\n\s*\n', '\n', text)
        # Remove form feed characters
        text = text.replace('\f', '')
        # Strip whitespace from start and end
        text = text.strip()
        return text

    def extract_pdf(self, file: str, chunking: int, overlap: int, **kwargs):
        """Extract text from PDF using PyPDF2 and create chunks."""
        try:
            # Open the PDF file
            with open(file, 'rb') as pdf_file:
                # Create PDF reader object
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                
                # Extract text from each page
                full_text = []
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text:
                        cleaned_text = self.clean_text(text)
                        if cleaned_text:  # Only add non-empty text
                            full_text.append(cleaned_text)
                
                # Join all text with newlines
                complete_text = '\n'.join(full_text)
                
                # Get text splitter and create chunks
                text_splitter = self.get_text_splitter(chunking, overlap)
                chunks = text_splitter.create_documents(
                    texts=[complete_text],
                    metadatas=[{"source": file}]
                )
                
                return chunks
                
        except Exception as e:
            print(f"Error extracting PDF text: {e}")
            return []

    def extract_text(self,file,chunking, overlap, **kwargs):
        loader = TextLoader(file, 'utf-8')
        return self.load(loader, chunking, overlap, **kwargs)

    def extract_csv(self,file,chunking, overlap, **kwargs):
        loader = CSVLoader(file)
        return self.load(loader, chunking, overlap, **kwargs)

    def load(self, loader: TextLoader | CSVLoader | PyMuPDFLoader, chunking, overlap, **kwargs):
        text_splitter = self.get_text_splitter(chunking, overlap)
        return loader.load_and_split(text_splitter=text_splitter, **kwargs)

    def get_file_chunks(self, file_path: str, chunk_size: int = 200, chunk_overlap: int = 10):
        try:
            chunks = self.extract(file_path, self.extension(file_path), chunk_size, chunk_overlap)
            return chunks
        except Exception as e:
            print(f"ERROR TRYING TO GET CHUNKS FROM FILE: {e}")
            return []

if __name__ == "__main__": 
    processor = preprocessor()
    test_file = "./LocalDocs/MIL-STD-882E.pdf"
    print(f"\nProcessing file: {test_file}")
    chunks = processor.extract(test_file, processor.extension(test_file), 5000, 50)
    
    print(f"\nTotal chunks created: {len(chunks)}")
    
    # Print content of each chunk with formatting
    for i, chunk in enumerate(chunks, 1):
        print(f"\n{'='*80}")
        print(f"Chunk {i}/{len(chunks)}")
        print(f"Metadata: {chunk.metadata}")
        print(f"Content length: {len(chunk.page_content)} characters")
        print(f"{'='*80}")
        print(chunk.page_content[:500] + "..." if len(chunk.page_content) > 500 else chunk.page_content)
        
        # Ask if user wants to continue after each chunk
        if i < len(chunks):
            response = input("\nPress Enter to see next chunk (or 'q' to quit): ")
            if response.lower() == 'q':
                break