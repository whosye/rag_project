from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from pathlib import Path

def load_api_key():
    from dotenv import load_dotenv
    import os
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
    return os.getenv("OPENAI_API_KEY")

def load_documents(source: Path):
    if isinstance(source, Path) and source.is_dir():
        
        directory_loader = DirectoryLoader(str(source), glob="*.txt", show_progress=True, loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
        documents = directory_loader.load()
        return documents
    else:
        raise ValueError(f"Source {source} is not a valid directory.")
    
def split_documents(documents: list):
    chunks = []
    for doc in documents:
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separator="\n")
        chunks.extend(text_splitter.split_documents([doc]))
    return chunks

def create_embeddings(chunks: list, api_key: str):
    model_name = "text-embedding-3-small"
    embedding_model = OpenAIEmbeddings(model=model_name, api_key=api_key)
    db_path = str(Path(__file__).parent.parent / "db")
    db = Chroma(persist_directory=db_path, collection_name="books", embedding_function=embedding_model)
    db.add_documents(chunks)
    print("Vector store created and persisted to 'db' directory.")



def main():
   
    source = Path(__file__).parent.parent / "docs"

    # 1. Load files 
    documents = load_documents(source)

    # 2. Chunk text (if needed)
    chunks = split_documents(documents)

    # 3. Create embeddings and store
    create_embeddings(chunks, load_api_key())


if __name__ == "__main__":
    main()