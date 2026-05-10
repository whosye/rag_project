
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from pathlib import Path

def load_api_key():
    from dotenv import load_dotenv
    import os
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
    return os.getenv("OPENAPI_KEY")

def get_similar(querry: str, api_key: str):
    model_name = "text-embedding-3-small"
    embedding_model = OpenAIEmbeddings(model=model_name, api_key=api_key)
    db_path = str(Path(__file__).parent.parent / "db")
    db = Chroma(persist_directory=db_path, collection_name="bible", embedding_function=embedding_model)
    retriver = db.as_retriever(search_kwargs={"k": 5},
                               search_type="similarity")
    
    return retriver.invoke(querry)



def main():
    results = get_similar("Jaké jsou nejdůležitější myšlenky z Janova evangelia?", load_api_key())
    for result in results:
         print(result.metadata["source"])
         print(result.page_content)
         print("-" * 80) 

# if __name__ == "__main__":
#     main()