
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from pathlib import Path

def load_api_key():
    from dotenv import load_dotenv
    import os
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
    return os.getenv("OPENAI_API_KEY")

def get_similar(query: str, api_key: str):
    model_name = "text-embedding-3-small"
    embedding_model = OpenAIEmbeddings(model=model_name, api_key=api_key)
    db_path = str(Path(__file__).parent.parent / "db")
    db = Chroma(persist_directory=db_path, collection_name="books", embedding_function=embedding_model)
    retriver = db.as_retriever(search_kwargs={"k": 5},
                               search_type="similarity")

    
    return retriver.invoke(query)
# if __name__ == "__main__":
#     res = get_similar("what is the meaning of life?", load_api_key())
#     if not res:
#         print("No relevant chunks found.")
#     for r in res:
#         print(r.page_content[:200].replace(chr(10), " "))