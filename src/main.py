
from dotenv import load_dotenv
import os
from pathlib import Path
from openai import OpenAI
from retrival import get_similar
from user_chat import Chat

env_path = Path(__file__).parent.parent / ".env"

def load_api_key():
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
    return os.getenv("OPENAI_API_KEY")

API = load_api_key()
client = OpenAI(api_key=API)

if __name__ == "__main__":
    chat = Chat(get_similar, client)
    while True:
        query = input("Otázka: ").strip()
        if query.lower() in ("konec", "exit", "quit"):
            break
        if not query:
            continue

        response = chat.ask(query)
        print("\n" + "-"*50 + "\n")
        print("Odpověď:", response)