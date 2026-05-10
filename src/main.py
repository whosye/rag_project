
from dotenv import load_dotenv
import os
from pathlib import Path
from openai import OpenAI
from retrival import get_similar

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


API = os.getenv("OPENAPI_KEY")

client = OpenAI(api_key=API)

response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {
            "role": "user",
            "content": "Hello!"
        }
    ]
)

print(response.choices[0].message.content)


if __name__ == "__main__":
    
    while True:
        query = input("Otázka: ").strip()
        if query.lower() in ("konec", "exit", "quit"):
            break
        if not query:
            continue
        print(f"Hledám relevantní informace pro: {query}")
        context = get_similar(query, API)
        print(f"Nalezeno {len(context)} relevantních chunků.")
        for i, c in enumerate(context):
            print(f"  [{i+1}] {c.page_content[:120].replace(chr(10), ' ')}")
        query = query + "\n\nKontext:\n" + "\n\n".join(
            f"[{c.metadata['source']}]\n{c.page_content}" for c in context
        )
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Odpovídej pouze na základě poskytnutého kontextu. Pokud odpověď v kontextu není, řekni, že nevíš."
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
        )
        print(f"Odpověď: {response.choices[0].message.content}\n")