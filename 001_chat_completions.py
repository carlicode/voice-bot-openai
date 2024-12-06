from dotenv import load_dotenv
from openai import OpenAI
import os

class OpenAIChat:
    def __init__(self, api_key: str, model: str, system_prompt: str):
        """
        Clase para manejar interacciones con el modelo de OpenAI Chat.
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt

    def create_chat_completion(self, user_message: str, temperature: float = 0.5, max_tokens: int = 150):
        """
        Crea un chat completion basado en el mensaje del usuario.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": 'system', "content": self.system_prompt},
                {"role": 'user', "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

if __name__ == "__main__":
    # Cargar la API Key
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')

    # Configurar el cliente de chat
    system_prompt = "Eres un asistente experto en física, eres conciso en tu respuesta"
    model = "gpt-4o"
    chat_client = OpenAIChat(api_key=api_key, model=model, system_prompt=system_prompt)

    # Solicitar la fórmula más importante
    user_message = "Dime la fórmula más importante."
    response = chat_client.create_chat_completion(user_message)

    # Imprimir la respuesta
    print(response)
