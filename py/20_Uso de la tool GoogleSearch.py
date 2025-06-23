# %% [markdown]
# ### Tools
# 
# Una `Tool`  en el contexto de un LLM es una **función externa o capacidad específica que un LLM puede "llamar" o "utilizar" para extender sus propias habilidades e interactuar con el mundo real o con sistemas externos.**
# 
# Los LLM son geniales para generar texto, comprender el lenguaje y razonar sobre información con la que fueron entrenados. Sin embargo, por sí solos, tienen limitaciones importantes:
# 
# * **No acceden a información en tiempo real**: Su conocimiento está limitado a la fecha de su último entrenamiento. No pueden buscar en Google ahora mismo.
# * **No realizan acciones**: No pueden ejecutar código, enviar correos electrónicos o interactuar con bases de datos.
# * **No son siempre precisos en tareas estructuradas**: Aunque pueden generar código, no siempre garantizan su corrección matemática o lógica.
# 
# Aquí es donde las herramientas se vuelven cruciales.
# 
# **¿Cómo Funcionan las Llamadas a Herramientas (Tool Calling)?**
# 
# Una "tool" es, en esencia, una **función predefinida** (escrita por el como desarrollador) que se le "presenta" al LLM. Cuando el LLM recibe una pregunta o una tarea que requiere una capacidad que no tiene inherentemente, puede "decidir" que necesita usar una de estas herramientas.
# 
# En esta práctica aprenderemos a usar `Tools` con un sencillo ejemplo.

# %% [markdown]
# Realizamos los `imports` y creamos una instancia de la API.

# %%
# imports

import os
from dotenv import load_dotenv
from openai import OpenAI
from IPython.display import Markdown, display, update_display

# %%
load_dotenv(override=True)
google_api_key = os.getenv('GOOGLE_API_KEY')

# %%
MODEL = "gemini-2.0-flash"
openai = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta", api_key=google_api_key)

# %% [markdown]
# Vamos a preguntar al modelo por la fecha actual:

# %%
response = openai.chat.completions.create(
 model=MODEL,
 messages=[{"role": "user", "content": "Dime la fecha y hora actual en Madrid (España)"}]
)

print(response.choices[0].message.content)

# %% [markdown]
# Vemos que el modelo da una respuesta incorrecta. Esto es porque el modelo no es capaz de hacer una búsqueda en Google.
# 
# La API de OpenAI para tools no es compatible con Google Gemini y tenemos que usar directamente la librería de Google Gemini:

# %%
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch

# Only run this block for Gemini Developer API
client = genai.Client(api_key=google_api_key)
model_id = "gemini-2.0-flash"

google_search_tool = Tool(
    google_search = GoogleSearch()
)

response = client.models.generate_content(
    model=model_id,
    contents="¿Cuál es la fecha y hora actual en Madrid (España)?",
    config=GenerateContentConfig(
        tools=[google_search_tool],
        response_modalities=["TEXT"],
    )
)

for each in response.candidates[0].content.parts:
    print(each.text)



