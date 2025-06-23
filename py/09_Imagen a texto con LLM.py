# %% [markdown]
# ### Introducción
# 
# En esta práctica, automatizamos el proceso de convertir imágenes de menús en una hoja de cálculo de Excel estructurada utilizando el modelo GPT de OpenAI. El script lee imágenes de un directorio específico, procesa cada imagen para extraer los datos de la carta de un restaurante y compila la información extraída en un archivo de Excel siguiendo una plantilla predefinida.

# %% [markdown]
# ### Setup

# %%
from dotenv import load_dotenv
import os

load_dotenv(override=True)
api_key = os.getenv('GOOGLE_API_KEY')

# %% [markdown]
# A continuación, importamos todas las librerías necesarias para el procesamiento de imágenes, la gestión de datos y la interacción con la API de OpenAI.

# %%
# Load the libraries
from openai import OpenAI
import os
import base64
from IPython.display import Image, display, Markdown
import pandas as pd

# %% [markdown]
# Usamos un modelo de Google Gemini capaz de procesar imágenes y extraer texto de ellas.

# %%
# Set up the OpenAI client and specify the model to use
MODEL = "gemini-2.5-flash-preview-05-20"
client = OpenAI(
    api_key=api_key, # Use the provided API key for authentication
    base_url="https://generativelanguage.googleapis.com/v1beta" 
)

# %% [markdown]
# ### Definiendo el mensaje del sistema

# %%
system_prompt = """
Convierte la imagen del menú en un formato estructurado de hoja de Excel siguiendo la plantilla e instrucciones proporcionadas.
Este asistente convierte datos de menús de restaurantes o cafeterías en una hoja de Excel estructurada que se ajusta a una plantilla específica.
La plantilla incluye categorías, subcategorías, nombres de los artículos, precios, descripciones y más, garantizando la consistencia de los datos.
Este asistente ayuda a los usuarios a completar correctamente cada fila, siguiendo las instrucciones detalladas proporcionadas.

Resumen:
- Cada fila en la hoja de cálculo de Excel representa un artículo único, categorizado bajo una categoría o subcategoría.
- Los nombres de categorías y subcategorías se repiten para los artículos dentro de la misma subcategoría.
- Ciertas columnas deben dejarse en blanco cuando no correspondan, como los detalles de subcategoría para artículos directamente bajo una categoría.
- Los detalles de cada artículo, incluidos nombres, precios y descripciones, deben ser únicos para cada entrada.
- El contenido del menú cargado se añadirá al menú existente sin eliminar ninguna entrada actual.
- Las columnas que terminan en "Es" son para traducciones al español, "Pt" para portugués y "En" para inglés.
- Deberás traducir al español los valores que estén en portugués.

IMPORTANTE: La salida debe ser ÚNICAMENTE una tabla de Markdown 
con EXACTAMENTE las siguientes columnas (ni más ni menos):

Nombre de columna                  | Descripción                                   | Valores aceptados                | Ejemplo
-----------------------------------|-----------------------------------------------|----------------------------------|-----------------------
CategoryTitleEs (Columna A)        | Traducciones al español de los nombres de categoría              | Texto, máximo 256 caracteres     | Bebidas
CategoryTitlePt (Columna B)        | Nombres de categorías en portugués            | Texto, máximo 256 caracteres     | Bebidas
CategoryTitleEn (Columna C) (Opcional) | Traducciones al inglés de los nombres de categoría | Texto, máximo 256 caracteres | Beverages
SubcategoryTitleEs (Columna D) (Opcional) | Traducciones al español de los nombres de subcategoría   | Texto, máximo 256 caracteres o en blanco | Zumos
SubcategoryTitlePt (Columna E) (Opcional) | Nombres de subcategorías en portugués   | Texto, máximo 256 caracteres o en blanco | Sucos
SubcategoryTitleEn (Columna F) (Opcional) | Traducciones al inglés de los nombres de subcategoría | Texto, máximo 256 caracteres o en blanco | Juices
ItemNameEs (Columna G)             | Traducciones al español de los nombres de los artículos         | Texto, máximo 256 caracteres     | Agua Mineral
ItemNamePt (Columna H)             | Nombres de los artículos en portugués         | Texto, máximo 256 caracteres     | Água Mineral
ItemNameEn (Columna I) (Opcional)  | Traducciones al inglés de los nombres de los artículos | Texto, máximo 256 caracteres o en blanco | Mineral Water
ItemPrice (Columna J)              | Precio de cada artículo sin símbolo de moneda | Texto                           | 2.50 o 2,50
Calories (Columna K) (Opcional)    | Contenido calórico de cada artículo           | Numérico                        | 150
PortionSize (Columna L)            | Tamaño de la porción de cada artículo en unidades | Texto                      | 500ml, 1, 2-3
Availability (Columna M) (Opcional)| Disponibilidad actual del artículo            | Numérico: 1 para Sí, 0 para No  | 1
ItemDescriptionEs (Columna N) (Opcional) | Descripción detallada en español      | Texto, máximo 500 caracteres     | Contiene minerales esenciales
ItemDescriptionPt (Columna O) (Opcional) | Descripción detallada en portugués      | Texto, máximo 500 caracteres     | Contém minerais essenciais
ItemDescriptionEn (Columna P) (Opcional) | Descripción detallada en inglés        | Texto, máximo 500 caracteres     | Contains essential minerals

Notas:
- Asegúrate de que todos los datos ingresados sigan los formatos especificados para mantener la integridad de la base de datos.
- Revisa los datos para garantizar su precisión y consistencia antes de enviar la hoja de Excel.
"""

# %% [markdown]
# Este `prompt` proporciona al modelo instrucciones completas sobre cómo procesar las imágenes del menú y el formato exacto esperado para la salida en Excel. Incluye una descripción general, descripciones de las columnas y ejemplos para garantizar la coherencia y precisión en el proceso de extracción de datos.

# %% [markdown]
# Creamos una función que convierte una imagen a Base64

# %%
# IMAGE_DIR = "imgs/menus/Regatta"
IMAGE_DIR = "../imgs/menus/Dim Sum"

def encode_image(image_path):
    # Open the image file in binary mode and encode it in Base64
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Process imaged in the directory
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
image_files

# %% [markdown]
# Codificar imágenes en Base64 nos permite incluir los datos de la imagen directamente en nuestras solicitudes a la API sin depender de URLs externas.
# 
# Este código escanea el directorio en busca de archivos que terminen en `.png`, `.jpg` o `.jpeg`, asegurando que solo procesamos los archivos de imagen relevantes para nuestra tarea.

# %% [markdown]
# Mostramos una imagen de ejemplo.

# %%
display(Image(filename=os.path.join(IMAGE_DIR, image_files[0]), width=300))

# %% [markdown]
# Probamos a enviar una imagen y comprobamos que se genera una tabla de Markdown con los datos del menú.

# %%
image_path = os.path.join(IMAGE_DIR, image_files[0])  # Use the first image in the directory
image_data = encode_image(image_path)

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {'type': 'text',
             'text': "Convierte esta imagen de menú en un formato estructurado de hoja de Excel."},
            {'type': 'image_url',
             'image_url': {'url': f'data:image/png;base64,{image_data}'}}
        ]}
    ],
    temperature=0
)

response.choices[0].message.content

# %% [markdown]
# Convertimos la respuesta del modelo a un DataFrame de Pandas para facilitar su manipulación y exportación a Excel.

# %%
df = pd.DataFrame(columns=['CategoryTitleEs', 'CategoryTitlePt', 'CategoryTitleEn', 'SubcategoryTitleEs', 'SubcategoryTitlePt', 'SubcategoryTitleEn',
                           'ItemNameEs', 'ItemNamePt', 'ItemNameEn', 'ItemPrice', 'Calories', 'PortionSize', 'Availability',
                           'ItemDescriptionEs', 'ItemDescriptionPt', 'ItemDescriptionEn'])

print(f"Procesando la respuesta del modelo para {image_path}")
for row in response.choices[0].message.content.split('\n'):
    if row.startswith('|') and not row.startswith('|-'): # Ensure that the data is a row and not a header format
      columns = [col.strip() for col in row.split('|')[1:-1]]
      if len(columns) == len(df.columns):
        if 'CategoryTitleEs' in columns:
          headers_added = True
          continue
        if headers_added and 'CategoryTitleEs' in columns:
          continue # skip the row
        new_row = pd.Series(columns, index=df.columns)
        print(f"\tAñadiendo nuevo plato: {new_row['CategoryTitlePt']}/{new_row['ItemNamePt']}")
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
      else:
        print(f"Saltándose fila { row}")
df

# %% [markdown]
# Solicitamos al usuario que ingrese un nombre para el nuevo archivo Excel donde se guardarán los datos extraídos.
# 
# Recorremos cada archivo de imagen, lo codificamos, lo enviamos a la API de OpenAI para su procesamiento y analizamos la respuesta para llenar nuestro DataFrame.

# %%
# Prompt the user for the excel file name
new_excel_file_name = input("Introduce el nombre del nuevo archivo Excel (sin extensión): ")
EXCEL_PATH = os.path.join(IMAGE_DIR, f"{new_excel_file_name}.xlsx")

# Crear el dataframe de PANDAS
df = pd.DataFrame(columns=['CategoryTitleEs', 'CategoryTitlePt', 'CategoryTitleEn', 'SubcategoryTitleEs', 'SubcategoryTitlePt', 'SubcategoryTitleEn',
                           'ItemNameEs', 'ItemNamePt', 'ItemNameEn', 'ItemPrice', 'Calories', 'PortionSize', 'Availability',
                           'ItemDescriptionEs', 'ItemDescriptionPt', 'ItemDescriptionEn'])


for image in image_files:
  # Retrieve and encode the image
  image_path = os.path.join(IMAGE_DIR, image)
  image_data = encode_image(image_path)

  # Adding a flag for the headers
  headers_added = False

  display(Image(filename=image_path, width=100))
  print(f"Enviando imagen al LLM {image_path}")
  response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {'type': 'text',
             'text': "Convierte esta imagen de menú en un formato estructurado de hoja de Excel."},
            {'type': 'image_url',
             'image_url': {'url': f'data:image/png;base64,{image_data}'}}
        ]}
    ],
    temperature=0
  )

  print(f"Procesando la respuesta del LLM para {image_path}")
  for row in response.choices[0].message.content.split('\n'):
      if row.startswith('|') and not row.startswith('|-'): # Ensure that the data is a row and not a header format
        columns = [col.strip() for col in row.split('|')[1:-1]]
        if len(columns) == len(df.columns):
          if 'CategoryTitleEs' in columns:
            headers_added = True
            continue
          if headers_added and 'CategoryTitleEs' in columns:
            continue # skip the row
          new_row = pd.Series(columns, index=df.columns)
          print(f"\033[92m\tAñadiendo nuevo plato: {new_row['CategoryTitlePt']}/{new_row['ItemNamePt']}\033[0m")
          df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
          print(f"\033[91mSaltándose fila. Se esperaban {len(df.columns)} columnas, pero se encontraron {len(columns)}.\n { row}\033[0m")

df.to_excel(EXCEL_PATH, index=False)
print(f"Fichero de Excel grabado en: {EXCEL_PATH}")


