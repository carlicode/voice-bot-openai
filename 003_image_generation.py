from dotenv import load_dotenv
from openai import OpenAI
import os


class OpenAIImageGeneration:
    def __init__(self, api_key: str, model: str):
        """
        Clase para manejar interacciones con el modelo de OpenAI Image.
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_image(self, prompt: str, size: str = '1024x1024', n: int = 1):
        """
        Crea una imagen a partir de un prompt del usuario.

        :param prompt: Descripción de la imagen a generar.
        :param size: Tamaño de la imagen (por defecto '1024x1024').
        :param n: Número de imágenes a generar (por defecto 1).
        :return: URL de la imagen generada.
        """
        try:
            response = self.client.images.generate(
                model=self.model,
                prompt=prompt,
                size=size,
                n=n
            )
            # Verificar si hay datos en la respuesta
            if response and response.data:
                return response.data[0].url  # Retorna la URL de la primera imagen generada
            else:
                raise ValueError("No se recibieron datos válidos de la API.")
        except Exception as e:
            print(f"Error al generar la imagen: {e}")
            return None


if __name__ == "__main__":
    # Cargar la API Key
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')

    # Configurar el cliente de generación de imágenes
    prompt = "una imagen fotorrealista de un gato naranja acurrucado en una frazada al lado de una chimenea"
    model = "dall-e-3"
    generator = OpenAIImageGeneration(api_key, model)

    # Generar la imagen
    image_url = generator.generate_image(prompt)

    # Mostrar la URL de la imagen generada
    if image_url:
        print(f"Imagen generada con éxito: {image_url}")
    else:
        print("No se pudo generar la imagen.")
