from dotenv import load_dotenv
from openai import OpenAI
import os
import json

class OpenAIChat:
    def __init__(self, api_key: str, model: str, system_prompt: str):
        """
        Clase para manejar interacciones con el modelo de OpenAI Chat.
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt

    def get_chat_completion(self, user_message: str, temperature: float = 0, max_tokens: int = 150):
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
        return response

    @staticmethod
    def save_response_to_json(response: dict, file_name: str):
        """
        Guarda la respuesta de OpenAI en un archivo JSON.
        """
        with open(file_name, 'w', encoding='utf-8') as json_file:
            json.dump(response, json_file, ensure_ascii=False, indent=4)
        print(f"Respuesta guardada en {file_name}")

if __name__ == "__main__":
    # Cargar la API Key
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')

    # Configurar el cliente de chat
    system_prompt = "Eres un asistente experto en física, eres conciso en tu respuesta"
    model = "gpt-4o"
    chat_client = OpenAIChat(api_key=api_key, model=model, system_prompt=system_prompt)

    # Solicitar la fórmula más importante
    user_message = "genera un json con 3 keys de las 3 formulas mas importantes, no pongas ningun mensaje adicional para que pueda guardar en un json directamente, no pongas saltos de linea ni ningun caracter adicional porque quiero formatear el resultado con la libreria json de python y guardarlo en un fichero en mi servidor"
    response = chat_client.get_chat_completion(user_message)

    # Imprimir la respuesta
    print(response.choices[0].message.content)

    '''# Preguntar al usuario si desea guardar la respuesta
    save_to_json = input("¿Deseas guardar la respuesta en un archivo JSON? (s/n): ").strip().lower()
    if save_to_json == 's':
        file_name = input("Introduce el nombre del archivo (ejemplo: respuesta.json): ").strip()
        chat_client.save_response_to_json(response, file_name)'''
