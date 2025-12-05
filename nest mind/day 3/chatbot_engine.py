# chatbot_engine.py
import openai

# Set your OpenAI API key
openai.api_key = "sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXX"  # Replace with your key

class ChatBotEngine:
    def get_answer(self, question: str) -> str:
        """Return an AI-generated answer using OpenAI GPT-3.5."""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question}
                ],
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"⚠️ Error: {e}"

