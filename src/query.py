import os
from pathlib import Path

import chromadb
from fastembed import TextEmbedding
from openai import OpenAI

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 5


def retrieve(query: str, collection: chromadb.Collection, model: TextEmbedding) -> list[dict]:
    query_embedding = next(model.embed([query]))
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=TOP_K,
        include=["documents", "metadatas"],
    )
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": doc, "source": meta.get("source", "")})
    return chunks


def generate(query: str, chunks: list[dict], client: OpenAI) -> str:
    context = "\n\n".join(
        f"[{c['source']}]\n{c['text']}" for c in chunks
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Jsi pomocník pro studium Bible Kralické. "
                    "Odpovídej pouze na základě poskytnutého kontextu. "
                    "Pokud odpověď v kontextu není, řekni to."
                ),
            },
            {
                "role": "user",
                "content": f"Kontext:\n{context}\n\nOtázka: {query}",
            },
        ],
    )
    return response.choices[0].message.content


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Nastav proměnnou prostředí OPENAI_API_KEY")

    db_path = Path(__file__).parent.parent / "db"
    client_chroma = chromadb.PersistentClient(path=str(db_path))
    collection = client_chroma.get_collection(name="bible")

    model = TextEmbedding(MODEL_NAME)
    client_openai = OpenAI(api_key=api_key)

    print("RAG – Bible Kralická (napište 'konec' pro ukončení)\n")
    while True:
        query = input("Otázka: ").strip()
        if query.lower() in ("konec", "exit", "quit"):
            break
        if not query:
            continue
        chunks = retrieve(query, collection, model)
        answer = generate(query, chunks, client_openai)
        print(f"\nOdpověď: {answer}\n")
        print("Zdroje: " + ", ".join(c["source"] for c in chunks if c["source"]) + "\n")


if __name__ == "__main__":
    main()
