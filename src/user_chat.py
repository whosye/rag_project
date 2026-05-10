from retrival import get_similar

class SystemMessage(dict):
    def __init__(self, content: str):
        super().__init__({"role": "system", "content": content})

class UserMessage(dict):
    def __init__(self, content: str):
        super().__init__({"role": "user", "content": content})

class Chat:
    
    def __init__(self, retriever, client):
        self.retriever = retriever
        self.client = client
        self.chat_history = []

    def ask(self, query: str):

        if not query.strip():
            return "Please enter a valid question."
        
        if self.chat_history:
            messages = [
                SystemMessage("Given the following conversation history and the new question, rewrite the question to be standalone and searchable. If the question is already standalone, return it as is."),
                UserMessage(f"Conversation history:\n{self.chat_history}\n\nNew question: {query}")
            ]

            standalone_question = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )

            standalone_question = standalone_question.choices[0].message.content.strip()

        else:
            standalone_question = query
        

        context = get_similar(standalone_question, self.client.api_key)
        answer = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                SystemMessage("You are a helpful assistant. Answer the question based only on the provided context. If the answer is not in the context, say you don't know."),
                UserMessage(f"Question: {standalone_question}\n\nContext:\n" + "\n\n".join(
                    f"[{c.metadata['source']}]\n{c.page_content}" for c in context
                ))
            ]
        )

        self.chat_history.append({"question": query, "answer": answer.choices[0].message.content})
        return answer.choices[0].message.content