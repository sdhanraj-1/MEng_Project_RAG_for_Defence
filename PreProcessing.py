# chunking.py
import os
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader


class preprocessor: 
    def extract(self, file, ext: str, chunking:int=5000,overlap:int=50, **kwargs):
        if ext == "pdf":
            return self.extract_pdf(file, chunking, overlap, **kwargs)
        if ext == "txt":
            return self.extract_text(file,chunking, overlap, **kwargs)
        return self.extract_csv(file,chunking, overlap, **kwargs)

    def extension(self,file_name:str):
        return file_name.split(".")[-1]

    def extract_pdf(self,file,chunking, overlap, **kwargs):
        loader = PyMuPDFLoader(file)
        return self.load(loader, chunking, overlap, **kwargs)

    def extract_text(self,file,chunking, overlap, **kwargs):
        loader = TextLoader(file, 'utf-8')
        return self.load(loader, chunking, overlap, **kwargs)

    def extract_csv(self,file,chunking, overlap, **kwargs):
        loader = CSVLoader(file)
        return self.load(loader, chunking, overlap, **kwargs)

    def load(self, loader: TextLoader | CSVLoader | PyMuPDFLoader, chunking, overlap, **kwargs):
        return loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunking, chunk_overlap=overlap),**kwargs)


    def get_file_chunks(self, file_path: str, chunk_size: int = 200, chunk_overlap: int = 10):
        try:
            """             # Using temp-directory to store our read file for a little while
            # This is because langchain loaders only acept file paths and not its raw bytes.
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = os.path.join(temp_dir, file_name)
                
                with open(temp_file_path, "wb") as temp_file:
                    temp_file.write(file_bytes) """
                    
            chunks = self.extract(file_path, self.extension(file_path), chunk_size, chunk_overlap)

            return chunks
        except Exception as e:
            print("ERROR TRYING TO GET CHUNKS FROM FILE")

if __name__ == "__main__": 
    processor = preprocessor()
    chunks = processor.extract("./LocalDocs/MIL-STD-882E.pdf", processor.extension("./LocalDocs/MIL-STD-882E.pdf"),5000,50)
    print(chunks)