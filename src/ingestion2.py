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
    return os.getenv("OPENAPI_KEY")

def load_docuemnts(source: Path):
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
    vector_store = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory="db", collection_name="bible")
    print("Vector store created and persisted to 'db' directory.")
    return vector_store


def main():
   
    source = Path(__file__).parent.parent / "source"
    db_path = Path(__file__).parent.parent / "db"

    # 1. Load files 
    documents = load_docuemnts(source)

    # 2. Chunk text (if needed)
    chunks = split_documents(documents)

    # 3. Create embeddings and store
    embeddings = create_embeddings(chunks, load_api_key())


if __name__ == "__main__":
    main()