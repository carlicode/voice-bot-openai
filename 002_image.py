from dotenv import load_dotenv
from openai import OpenAI
import base64
import os

class OpenAIChat:
    def __init__(self, api_key: str, model: str):
        """
        Clase para manejar interacciones con el modelo de OpenAI Chat.
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def encode_image(self, image_path: str) -> str:
        """
        Codifica una imagen en formato Base64.
        """
        with open(image_path, 'rb') as image_file:
            image_binary_data = image_file.read()
            return base64.b64encode(image_binary_data).decode('utf-8')

    def create_chat_completion(self, system_prompt: str, user_prompt: str, image_data: dict = None, temperature: float = 1):
        """
        Crea un chat completion basado en el mensaje del usuario, con opción de incluir imágenes.
        """
        messages = [{"role": "system", "content": system_prompt}]
        
        # Agregar el mensaje del usuario
        user_message = {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
        
        # Agregar datos de la imagen si están presentes
        if image_data:
            user_message["content"].append({"type": "image_url", "image_url": image_data})
        
        messages.append(user_message)

        # Realizar la solicitud al modelo
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content

if __name__ == "__main__":
    # Cargar la API Key
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')

    # Configurar el cliente de chat
    model = "gpt-4o"
    chat_client = OpenAIChat(api_key=api_key, model=model)

    # Caso 1: Cálculo del área de un triángulo con una imagen codificada
    system_prompt_1 = "Eres un profesor de álgebra."
    prompt_1 = "Calcula el área del triángulo."
    image_path = 'triangle.png'
    base64_image = chat_client.encode_image(image_path)
    image_data_1 = {"url": f"data:image/png;base64,{base64_image}"}

    response_1 = chat_client.create_chat_completion(system_prompt_1, prompt_1, image_data=image_data_1)
    print("Respuesta del cálculo del área del triángulo:")
    print(response_1)

    # Caso 2: Descripción de una imagen desde una URL
    system_prompt_2 = "Eres un asistente útil."
    prompt_2 = "Describe la imagen."
    image_url = "https://static.fundacion-affinity.org/cdn/farfuture/PVbbIC-0M9y4fPbbCsdvAD8bcjjtbFc0NSP3lRwlWcE/mtime:1643275542/sites/default/files/los-10-sonidos-principales-del-perro.jpg"
    image_data_2 = {"url": image_url}

    response_2 = chat_client.create_chat_completion(system_prompt_2, prompt_2, image_data=image_data_2)
    print("\nRespuesta de la descripción de la imagen:")
    print(response_2)
