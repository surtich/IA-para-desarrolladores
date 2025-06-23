# %% [markdown]
# ### ¿Qué es RAG?
# 
# **RAG (Retrieval-Augmented Generation)** es una técnica avanzada en el campo de la inteligencia artificial generativa que potencia las capacidades de los LLM. Su objetivo principal es mejorar la precisión, actualidad y relevancia de las respuestas generadas por estos modelos, permitiéndoles acceder a información externa más allá de los datos con los que fueron entrenados.
# 
# **¿Cómo funciona RAG?**
# 
# El proceso de RAG se compone de tres etapas principales:
# 
# - Recuperación (Retrieval): Ante una consulta del usuario, el sistema busca información relevante en fuentes externas, como bases de datos, repositorios de documentos, o bases de datos vectoriales. Esta búsqueda puede realizarse mediante técnicas de búsqueda semántica, embeddings o consultas tradicionales.
# - Augmentación (Augmentation): La información recuperada se combina con la consulta original del usuario, creando un contexto enriquecido que se utiliza como entrada para el modelo de lenguaje.
# - Generación (Generation): El LLM genera una respuesta utilizando tanto su conocimiento interno como la información externa proporcionada, lo que permite respuestas más precisas, actualizadas y con posibilidad de citar fuentes específicas.
# 
# **Ventajas de RAG**
# 
# - Actualización y precisión: Permite que los LLM accedan a información reciente y específica de dominio sin necesidad de reentrenar el modelo, lo que ahorra recursos y tiempo.
# - Reducción de alucinaciones: Al basar las respuestas en datos recuperados, disminuye la probabilidad de que el modelo genere información incorrecta o inventada.
# - Transparencia y verificabilidad: RAG puede incluir citas o referencias a las fuentes de la información, facilitando la verificación de la respuesta por parte del usuario.
# - Adaptabilidad: Es posible personalizar el sistema para que consulte bases de datos internas, repositorios corporativos o cualquier fuente relevante para el contexto de uso.
# 
# **Ejemplo de aplicación**
# 
# Imagina una empresa que necesita que su chatbot responda preguntas sobre normativas internas. Un sistema RAG permite que el LLM consulte documentos internos actualizados y genere respuestas precisas, citando la fuente correspondiente, sin exponer la información a internet ni requerir reentrenamiento del modelo.
# 
# En resumen, RAG es una arquitectura fundamental para aprovechar al máximo los LLM en contextos donde la precisión, la actualidad y la trazabilidad de la información son críticas, especialmente en entornos corporativos, legales, médicos y de atención al cliente.
# 

# %% [markdown]
# ### Introducción
# 
# Este proyecto tiene como objetivo crear un sistema que permita incorporar recetas de cocina a una base de datos vectorial para su posterior consulta. Utilizando técnicas de RAG, se busca mejorar la precisión y relevancia de las respuestas generadas al consultar recetas específicas.

# %% [markdown]
# ### Setup

# %%
from dotenv import load_dotenv
import os

load_dotenv(override=True)
gemini_api_key = os.getenv('GOOGLE_API_KEY')

# %%
# Import the libraries
from openai import OpenAI
import base64

# %%
# Set up connection to OpenAI API
chat_client = OpenAI(
    api_key=gemini_api_key, # Use the provided API key for authentication
    base_url="https://generativelanguage.googleapis.com/v1beta" 
)
# Specify the model to be used
model = "gemini-2.5-flash-preview-05-20"

# %% [markdown]
# ### Realizar OCR y transformar a imágenes
# 
# Algunos PDFs se crean a partir de texto escaneados o mediante imágenes. Para estos casos, es necesario realizar un OCR (Reconocimiento Óptico de Caracteres) para extraer el texto de las imágenes. Incluso si el PDF mezcla texto e imágenes a veces es recomendable usar OCR ya que así el LLM puede entender mejor el contenido.
# 
# A la siguiente función se le pasa un PDF y se trocea en páginas. Cada página se convierte a imagen.

# %%
from pdf2image import convert_from_path

# Function to converts pdfs into images and stores the image paths
def pdf_to_images(pdf_path, output_folder):
  # Create the output folder if it doesn't exist
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)

  # Convert PDF into images
  images = convert_from_path(pdf_path) # Convert each page of the PDF to an image
  image_paths = []

  # Save images and store their paths
  for i, image in enumerate(images):
    image_path = os.path.join(output_folder, f"page{i+1}.jpg") # Generate the image file path
    image.save(image_path, "JPEG") # Save the image as a JPEG file
    image_paths.append(image_path) # Append the image path to the list

  return image_paths # Return the list of image paths

# %%
# Define the path to the PDF and the output folder for images
book_name = "Southern Cookbook of Fine Recipes"
pdf_path = f"../pdfs/recipes/{book_name}.pdf"
output_folder = f"../imgs/recipes/{book_name}"

# Convert the PDF into images and store the image paths
image_paths = pdf_to_images(pdf_path, output_folder)

# %% [markdown]
# Leemos una imagen y obtenemos el Base64 de la imagen.

# %%
# Read and encode one image
book_page = 22 # Specify the page number to read
image_file = os.path.join(output_folder, f"page{book_page}.jpg") # Path to the image to be encoded

# Encode the image in base64 and decode to string
with open(image_file, "rb") as image_file:
  image_data = base64.b64encode(image_file.read()).decode('utf-8')
image_data

# %% [markdown]
# Definimos el `system_prompt`:

# %%
# Define the system prompt
system_prompt = """
Por favor, analiza el contenido de esta imagen y extrae cualquier información relacionada con recetas.
La salida se espera completa en español: Nombre de la receta, ingredientes y preparación.
Si una receta no tiene ingredientes, obténgalos de la preparación.
"""

# %% [markdown]
# Llamamos al LLM. Observe cómo se pasa la imagen en Base64.

# %%
# Call the OpenAI API use the chat completion method
response = chat_client.chat.completions.create(
    model = model,
    messages = [
        # Proporciona el mensaje del sistema
        {"role": "system", "content": system_prompt},

        # El mensaje del usuario contiene tanto el texto como la URL / ruta de la imagen
        {"role": "user", "content": [
            {"type": "text", "text": "Esta es la imagen de la página de la receta."},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{image_data}",
                           "detail": "low"}}
        ]}
    ]
)

gpt_response = response.choices[0].message.content

# %%
from IPython.display import Markdown, Image, HTML, display

import markdown2

def display_image_and_text(image_path, text_html):
    """Display an image and text side by side"""
    image_html = f'<img src="{image_path}" width="300" style="vertical-align:top; margin-right:20px;">'

    display(HTML(f"""
    <div style="display:flex; align-items:flex-start;">
        {image_html}
        <div style="margin-left:20px;">{text_html}</div>
    </div>
    """))

text_html = markdown2.markdown(gpt_response)
display_image_and_text(image_file.name, text_html)

# %% [markdown]
# Creamos una función para mostrar el resultado de la llamada al LLM.

# %%
# Define a function to get the GPT response and display it in Markdown
def get_gpt_response():
  gpt_response = response.choices[0].message.content # Extract the response content from the API response
  return display(Markdown(gpt_response)) # Display the response as Markdown

# Call the function to display the GPT response
get_gpt_response()

# %% [markdown]
# Definimos un `prompt` mejorado.

# %%
system_prompt = """
Eres un asistente que extrae recetas en formato estructurado según el esquema proporcionado.
Si una receta no tiene ingredientes, los obtienes de la preparación.
Los valores de todos los campos, incluido recipe_name, estarán en español,
excepto book_name, que estará en el idioma original.
Si algún campo no está presente en la información, pon valor None.
Por ejemplo, campos como los siguientes deben ponerse con None:
- 'quantity': 'al gusto'
- 'prep_time': 'No especificado',
- 'cook_time': 'NA',
- 'total_time': 'Desconocido',
- 'servings': '',

Si en algún ingrediente, no se indica la cantidad, el campo quantity tendrá valor None.
Ejemplo:

```json
{
  ...
  "ingredients": [
    {"item": "manzanas", "quantity": "3"},
    {"item": "harina", "quantity": None}
  ]
}
``` 

IMPORTANTE: Si no puedes deducir un campo con certeza a partir de la información dada, 
no inventes ni asumas información bajo ninguna circunstancia.
Por ejemplo, no hagas estimaciones del tiempo o de la dificultad si esas informaciones no están presentes en la receta.
El tiempo de preparación no debe ser estimado a partir de las instrucciones de preparación,
sino que se debe utilizar el que se diga explícitamente en la receta. Si no se menciona, pon None.

Ejemplo de entrada:

...
Tiempo de preparación: 30 minutos
...

Salida esperada:

```json
{
  ...
  "cook_time": "30 minutos",
}
```

Ejemplo de entrada:

...
Tiempo de cocción: 30 minutos
...

Salida esperada (el tiempo de cocción es un paso de la preparación, no el tiempo total de preparación):

```json
{
  ...
  "cook_time": None,
}
```

En ejemplo anterior, la cocción es sólo un paso de la preparación,
por lo que ese no es el tiempo total de preparación de la receta.
Esto mismo aplica para horneado y otros pasos de la preparación.
"""

def user_prompt(book_name, book_page):
  """
  Generate a user prompt for extracting recipe information from an image.
  """
  return f"""
  Extrae la información de la receta de la siguiente imagen. 
  La receta pertenece al libro '{book_name}' y está en la página {book_page}.
  Devuelve la información siguiendo el esquema estructurado.
  """


# %% [markdown]
# Definimos el formato que el LLM debe devolver. A esto se le conoce como salida estructurada.

# %%
from pydantic import BaseModel, Field
from typing import List, Optional

class Ingredient(BaseModel):
    quantity: Optional[str] = Field(description="Cantidad del ingrediente (por ejemplo, '2 tazas')")
    item: Optional[str] = Field(description="Nombre del ingrediente (por ejemplo, 'harina')")

class Recipe(BaseModel):
    is_recipe: bool = Field(description="""
                            Indica si el contenido es una receta.
                            Como mínimo, debe contener:
                             - el nombre del libro de recetas
                            -  el número de página
                            -  el nombre de la receta
                            -  los ingredientes
                            -  las instrucciones de preparación
                            """)
    book_name: str = Field(description="Nombre del libro de recetas")
    page_number: int = Field(description="Número de página de la receta")
    recipe_name: str = Field(description="Nombre de la receta")
    ingredients: List[Optional[Ingredient]] = Field(description="Lista de ingredientes", default=[])
    instructions: List[Optional[str]] = Field(description="Pasos de la receta", default=[])
    cook_time: Optional[str] = Field(description="Tiempo de cocción", default=None)
    servings: Optional[str] = Field(description="Número de comensales", default=None)
    difficulty: Optional[str] = Field(description="Nivel de dificultad de la receta (por ejemplo, 'Fácil')", default=None)
    prep_time: Optional[str] = Field(description="Tiempo de preparación (por ejemplo, '15 minutos')", default=None)
    cuisine_type: Optional[str] = Field(description="Tipo de cocina", default=None)
    dish_type: Optional[str] = Field(description="Tipo de plato", default=None)
    tags: Optional[List[Optional[str]]] = Field(description="Etiquetas relevantes", default=None)
    notes: Optional[str] = Field(description="Notas adicionales sobre la receta", default=None)

class Recipes(BaseModel):
    recipes: List[Recipe] = Field(description="Lista de recetas extraídas de la imagen", default=[])

# %% [markdown]
# Llamamos a la función adecuada para poder pasar el formato de respuesta. Observe que el formato de la respuesta estructurada se para en el parámetro `response_format`. Además, el método usado para obtener respuestas estructuradas en la API de OpenAI es `chat_client.beta.chat.completions.parse`.

# %%
import json 

def image_to_base64(image_path: str) -> str:
    """
    Convert an image file to a base64 encoded string.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        str: Base64 encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_recipes_from_image(book_name: str = book_name, page_number: int = book_page) -> Recipes:
    """
    Extract recipes from an image using the OpenAI API.
    
    Args:
        image_data (str): Base64 encoded image data.
        system_prompt (str): System prompt for the model.
        user_prompt (str): User prompt containing the image and instructions.
    
    Returns:
        Recipes: Structured recipes extracted from the image.
    """

    # Convert the image to base64
    image_path = os.path.join(output_folder, f"page{page_number}.jpg")
    image_data = image_to_base64(image_path)

    # Call the API to extract the information
    response = chat_client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt(book_name, page_number)},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{image_data}",
                               "detail": "low"}}
            ]}
        ],
        response_format=Recipes,
        temperature=0,  # Set the temperature to 0 for deterministic output
    )
    return image_path, json.loads(response.choices[0].message.content)

# %% [markdown]
# Probamos

# %%
image_file, recipes = extract_recipes_from_image(page_number = 23)
display_image_and_text(image_file, recipes)

# %% [markdown]
# Si intentamos procesar una imagen que no contiene recetas, el campo `is_recipe` será `False`.

# %%
image_file, recipes = extract_recipes_from_image(page_number = 6)
display_image_and_text(image_file, recipes)

# %% [markdown]
# 
# Procesamos todas las páginas.

# %%
import re

def extract_number(image_file):
    # Busca el primer grupo de dígitos en el nombre del archivo
    match = re.search(r'\d+', image_file)
    return int(match.group()) if match else -1

image_files = sorted([f for f in os.listdir(output_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))], key=extract_number)  # Sort files by page number extracted from filename
extracted_recipes = []  # List to store extracted recipes

for image_file in image_files:
  page_number = int(extract_number(image_file))  # Ensure the function is called to extract the number
  print(f"Procesando página {page_number}")
  display(Image(filename=os.path.join(output_folder, image_file), width=150))
  _, recipes = extract_recipes_from_image(page_number = page_number)

  count_true = 0
  count_false = 0

  for recipe in recipes.get("recipes"):
    if recipe.get("is_recipe"):
      print(f"\tReceta encontrada en la página {page_number}: {recipe.get('recipe_name')}")
      extracted_recipes.append(recipe)
      count_true += 1
    else:
        count_false += 1

  print(f"\033[92mRecetas correctamente formadas en la página {page_number}: {count_true}\033[0m")
  print(f"\033[91mRecetas incorrectamente formadas en la página {page_number}: {count_false}\033[0m")


# %%
len(extracted_recipes)

# %% [markdown]
# Almacenamos el resultado en un archivo JSON.

# %%
# Define the output file path
output_file = os.path.join("../pdfs", "recipe_info.json")

# Write the filtered list to a json file
with open(output_file, "w") as json_file:
  json.dump(extracted_recipes, json_file, indent = 4)

# %% [markdown]
# ### Embeddings
# 
# A partir de las recetas, vamos a crear los `embeddings` para poder almacenarlos en una base de datos vectorial.
# 
# Recuperamos los datos del archivo JSON generado anteriormente, para no tener que volver a procesar las imágenes si reiniciamos el libro.

# %%
output_file = os.path.join("../pdfs", "recipe_info.json")

# Write the filtered list to a json file
with open(output_file, "r") as json_file:
  recipes = json.load(json_file)

# %%
len(recipes)

# %%
recipes[0]

# %% [markdown]
# Creamos una función que dada una receta, genere genere el texto para el `embedding` uniendo en un texto el nombre de la receta, los ingredientes y los pasos de la receta.  

# %%
def prepare_recipe_for_embedding(recipe):
    """
    Generates the embedding text and metadata dictionary from a recipe dictionary.
    Removes 'is_recipe' from metadata.
    Handles None values for 'quantity' and 'item' in ingredients.
    """
    # Recipe name
    name = recipe.get('recipe_name', '') or ''

    # Ingredients as text
    ingredients_list = recipe.get('ingredients', [])
    ingredients_text = ''
    if ingredients_list and isinstance(ingredients_list, list):
        ingredient_strings = []
        for ing in ingredients_list:
            quantity = ing.get('quantity')
            item = ing.get('item')
            # Only include if item is present (item is essential)
            if item:
                quantity_str = str(quantity) if quantity is not None else ''
                item_str = str(item) if item is not None else ''
                ingredient_strings.append(f"{quantity_str.strip()} {item_str.strip()}".strip())
        ingredients_text = '; '.join(ingredient_strings)

    # Instructions as text
    instructions_list = recipe.get('instructions', [])
    instructions_text = ''
    if instructions_list and isinstance(instructions_list, list):
        instructions_text = ' '.join([str(instr) for instr in instructions_list if instr])

    # Text for embedding
    embedding_text = (
        f"Nombre de la receta: {name}. "
        f"Ingredientes: {ingredients_text}. "
        f"Instrucciones de preparación: {instructions_text}"
    )

    metadata = {}
    for k, v in recipe.items():
        if k == 'is_recipe':
            continue
        if v is None:
            metadata[k] = ''
        elif isinstance(v, (str, int, float, bool)):
            metadata[k] = v
        else:
            # Ignorar valores complejos
            pass

    # Añadir ingredientes como lista de strings
    metadata['ingredients'] = ingredient_strings

    metadata['ingredients'] = ingredients_text
    metadata['instructions'] = instructions_text
    
    return embedding_text, metadata


# %%
recipe_text, recipe_metadata = prepare_recipe_for_embedding(recipes[1])
print("Embedding text:\n", recipe_text)
print("\nMetadata:\n", recipe_metadata)


# %% [markdown]
# Creamos un cliente de Gemini para generar los embeddings. Observe que la URL es diferente de la habitual.

# %%
embedding_client = OpenAI(
    api_key=gemini_api_key,  # Reemplaza por tu clave real de Gemini
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# %% [markdown]
# Probamos a crear el `embedding` para una receta.

# %%
embedding_response = embedding_client.embeddings.create(
    input = recipe_text, # Provide the recipe text as input
    model = "text-embedding-004" # Specify the embedding model to use
)

# %%
embedding_response.data

# %% [markdown]
# Vamos a crear los `embeddings` de todas las recetas:

# %%
recipes_text, recipes_metadata = map(list, zip(*(prepare_recipe_for_embedding(recipe) for recipe in recipes)))

len(recipes_text)

# %% [markdown]
# El modelo `text-embedding-004` sólo permite crear `embeddings` para 100 elementos. Dividimos el proceso en bloques y unimos.

# %%
def batch_embedding_calls(texts, embedding_function, batch_size=100):
    """
    Procesa textos en lotes y concatena los resultados de embeddings.
    - texts: lista de textos a procesar.
    - embedding_function: función que recibe una lista de textos y devuelve una lista de embeddings.
    - batch_size: máximo de textos por llamada (100 para text-embedding-004).
    """
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    all_embeddings = []
    for batch in batches:
        embeddings = embedding_function(batch)  # Llama a la API aquí
        all_embeddings.extend(embeddings)
    return all_embeddings

def get_embeddings(text_batch):
    response = embedding_client.embeddings.create(
        input = text_batch, # Provide the recipe text as input
        model = "text-embedding-004" # Specify the embedding model to use
    )
    return [item.embedding for item in response.data]

# %%
recipes_embedding = batch_embedding_calls(recipes_text, get_embeddings, batch_size=100)


# %% [markdown]
# ### Almacenamiento en una base de datos vectorial
# 
# El siguiente paso es almacenar los `embeddings` y los metadatos en una base de datos vectorial. En este caso, usaremos ChromaDB.

# %% [markdown]
# Creamos el cliente de ChromaDB.

# %%
import chromadb

chroma_client = chromadb.PersistentClient(path="db/recipes")

# %% [markdown]
# Creamos la colección de recetas. Si ya existe, la eliminamos y la volvemos a crear.

# %%
collection_name = "recipes"

if collection_name in [c.name for c in chroma_client.list_collections()]:
    chat_client.delete_collection(name=collection_name)

recipes_collection = chroma_client.create_collection(name=collection_name)

# %%
collection_name = "recipes"

recipes_collection = chroma_client.get_collection(name=collection_name)

# %% [markdown]
# Insertamos los datos en la colección. Cada elemento en Chroma debe tener un identificador único.

# %%
import uuid

recipes_ids = [str(uuid.uuid4()) for _ in recipes]


# %%
recipes_collection.add(
    ids=recipes_ids,                # Lista de IDs únicos (str)
    embeddings=recipes_embedding,  # Lista de vectores (listas de floats)
    documents=recipes_text,         # Lista de textos originales (opcional, pero recomendable)
    metadatas=recipes_metadata      # Lista de diccionarios de metadatos (opcional)
)

# %% [markdown]
# ### Consultar la base de datos vectorial

# %% [markdown]
# Primero, generamos el `embedding` de la consulta.

# %%
# Generate the embeddings for the query
query = "¿Cómo se hace pan?"

query_embedding = embedding_client.embeddings.create(
    input = [query],
    model = "text-embedding-004"
).data[0].embedding

# %% [markdown]
# Después consultados la base de datos vectorial con el `embedding` de la consulta.

# %%
k = 5 # Number of top results to retrieve
results = recipes_collection.query(
    query_embeddings=[query_embedding],
    n_results=k,
    include=["documents", "metadatas", "distances"]  # Puedes incluir lo que necesites
)

# %% [markdown]
# Se muestran los 5 resultados más relevantes ordenados de menor a mayor distancia.

# %%
results

# %% [markdown]
# Creamos una función que reciba un `document` y lo muestre formateado en Markdown.

# %%
def recipe_document_to_markdown(doc_str):
    """
    Recibe un string con la receta y devuelve Markdown con nombre, ingredientes e instrucciones.
    """
    # Extraer nombre de la receta
    name_match = re.search(r'Nombre de la receta:\s*([^.]+)\.', doc_str)
    name = name_match.group(1).strip() if name_match else "Receta sin nombre"

    # Extraer ingredientes
    ing_match = re.search(r'Ingredientes:\s*([^.]+)\.', doc_str)
    ingredients_raw = ing_match.group(1).strip() if ing_match else ""
    # Separar ingredientes por ';'
    ingredients = [f"- {ing.strip()}" for ing in ingredients_raw.split(';') if ing.strip()]

    # Extraer instrucciones
    instr_match = re.search(r'Instrucciones de preparación:\s*(.+)', doc_str)
    instructions_raw = instr_match.group(1).strip() if instr_match else ""
    # Separar instrucciones por puntos, pero mantener frases completas
    # (puedes ajustar esto según tus necesidades)
    instructions = [f"{i+1}. {step.strip()}" for i, step in enumerate(re.split(r'\.([ ]|$)\s*', instructions_raw)) if step.strip()]

    # Construir el Markdown
    md = f"# {name}\n\n"
    md += "## Ingredientes\n"
    md += "\n".join(ingredients) + "\n\n"
    md += "## Instrucciones de preparación\n"
    md += "\n".join(instructions)

    return md


# %%
display(Markdown(recipe_document_to_markdown(results["documents"][0][-1])))

# %% [markdown]
# Creamos una función que reciba todos los `documents` y los muestre formateados los documentos en Markdown en orden de distancia.

# %%
def recipe_documents_to_markdown(documents):
    """
    Convierte una lista de documentos de recetas a Markdown.
    """
    return "\n\n".join(recipe_document_to_markdown(doc) for doc in reversed(documents))

# %%
display(Markdown(recipe_documents_to_markdown(results["documents"][0])))

# %% [markdown]
# Creamos una función que realiza todo el proceso

# %%
def search_recipes(query, k=5):
    """
    Busca recetas similares a la consulta dada.
    
    Args:
        query (str): Consulta de búsqueda.
        k (int): Número de resultados a devolver.
    
    Returns:
        list: Lista de documentos de recetas encontrados.
    """
    query_embedding = embedding_client.embeddings.create(
        input=[query],
        model="text-embedding-004"
    ).data[0].embedding

    results = recipes_collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    
    return recipe_documents_to_markdown(results["documents"][0])  # Devuelve los documentos encontrados

# %%
display(Markdown(search_recipes("¿Cómo se hace pan?")))

# %% [markdown]
# ### Sistema generativo
# 
# Ya tenemos las recetas que más se parecen a nuestra consulta. Pero eso no quiere decir que hayamos respondido a la pregunta. Vamos consultar a un LLM para que nos responda a la pregunta, pero que sólo use la información obtenida de las recetas.

# %%
# Define el prompt del sistema
def query_system_prompt(combined_content, query):
    return f"""
    Eres un chef altamente experimentado y experto, especializado en brindar consejos culinarios.
    Tu tarea principal es proporcionar información precisa y exacta sobre el contenido combinado.
    Tu objetivo es ayudar al usuario y responder la pregunta: "{query}"
    Respondes directamente a la consulta utilizando solo la información proporcionada en las
    siguientes recetas: [recetas].
    Si no sabes la respuesta, simplemente di que no lo sabes.
    El usuario no debe saber que consultas la información en una lista de recetas.
    No hagas aclaraciones al usuario si no sabes la respuesta.

    recetas: {combined_content}
    """


# %%
def generate_response(query, combined_content):
  response = chat_client.chat.completions.create(
      model = model,
      messages = [
          {"role": "system", "content": query_system_prompt(combined_content, query)}, # Provide system prompt for guidance
          {"role": "user", "content": query}
    ],
    temperature=0
  )
  return response

# %%
# Get the results from the API
query = "¿Cómo se hace masa de pan?"
combined_content = search_recipes(query)
response = generate_response(query, combined_content)

display(Markdown(response.choices[0].message.content))

# %%
# Get the results from the API
query = "Dime la mejor receta de pastel de chocolate"
combined_content = search_recipes(query)
response = generate_response(query, combined_content)

display(Markdown(response.choices[0].message.content))

# %%
# Get the results from the API
query = "Dime recetas con tortuga y cebolla. Muestra las palabras que hagan referencia a tortuga y cebolla en verde"
combined_content = search_recipes(query)
response = generate_response(query, combined_content)

display(Markdown(response.choices[0].message.content))


