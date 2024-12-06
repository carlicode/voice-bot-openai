from dotenv import load_dotenv
from openai import OpenAI
import os
from PIL import Image
import io


class OpenAIImageVariation:
    def __init__(self, api_key: str):
        """
        Clase para manejar interacciones con el modelo de OpenAI Image.
        """
        self.client = OpenAI(api_key=api_key)

    def convert_to_png_bytes(self, image_path: str):
        """
        Convierte una imagen a formato PNG y devuelve los bytes.

        :param image_path: Ruta de la imagen original.
        :return: Bytes de la imagen convertida a PNG.
        """
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGBA")  # Convertir a formato RGBA
                byte_stream = io.BytesIO()
                img.save(byte_stream, format="PNG", optimize=True)
                byte_stream.seek(0)
                return byte_stream
        except Exception as e:
            print(f"Error al convertir la imagen a PNG: {e}")
            return None

    def variation_image(self, image_bytes, size: str = '1024x1024', n: int = 1):
        """
        Genera una variación de una imagen dada.

        :param image_bytes: Bytes de la imagen en formato PNG.
        :param size: Tamaño de la imagen generada.
        :param n: Número de variaciones a generar.
        :return: URL de la imagen variada.
        """
        try:
            response = self.client.images.create_variation(
                image=image_bytes,
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

    # Configurar el cliente de generación de variaciones de imágenes
    input_image_path = "jc.jpeg"  # Imagen original
    generator = OpenAIImageVariation(api_key)

    # Convertir la imagen a PNG en memoria
    png_image_bytes = generator.convert_to_png_bytes(input_image_path)

    if png_image_bytes:
        # Generar una variación de la imagen
        image_variated_url = generator.variation_image(image_bytes=png_image_bytes)

        # Mostrar la URL de la imagen variada generada
        if image_variated_url:
            print(f"Imagen variada generada con éxito: {image_variated_url}")
        else:
            print("No se pudo generar una variación de la imagen.")
    else:
        print("No se pudo convertir la imagen a PNG.")
