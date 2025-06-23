# %% [markdown]
# ### RAG (Retrieval-Augmented Generation o "Generación Aumentada por Recuperación")
# 
# RAG es un enfoque arquitectónico en inteligencia artificial que potencia los Modelos de Lenguaje al integrar un componente de recuperación de información. Esto permite que los LLM accedan y utilicen fuentes de datos externas (como bases de datos, repositorios de documentos o bases de conocimiento) en el momento de generar respuestas, en lugar de depender únicamente de su conocimiento preentrenado y estático.
# 
# ### ¿Cómo Funciona RAG?
# 
# 1. **Recuperación de Información:** Cuando el usuario envía una consulta, el sistema RAG primero busca y recupera documentos o datos relevantes de fuentes externas, utilizando técnicas de recuperación de información (a menudo mediante bases de datos vectoriales para búsqueda semántica).
# 2. **Generación Aumentada:** La información recuperada se combina con la consulta original, creando un *prompt aumentado* que ofrece contexto adicional y actualizado al LLM.
# 3. **Respuesta Mejorada:** El LLM genera una respuesta basada tanto en su conocimiento preentrenado como en la información recuperada, logrando respuestas más precisas, relevantes y actuales.
# 
# ### Principales Ventajas de usar RAG
# 
# - **Mayor Precisión y Relevancia:** Al fundamentar las respuestas en datos actualizados y autorizados, RAG reduce las alucinaciones (respuestas inventadas o inexactas) y mejora la veracidad.
# - **Adaptabilidad a Dominios Específicos:** Permite a los LLM responder usando información privada, específica de un dominio o que cambia rápidamente, sin necesidad de reentrenar el modelo.
# - **Transparencia:** Bien implementado, un sistema RAG puede proporcionar citas o referencias de la información utilizada, aumentando la confianza y verificabilidad.
# 
# ### Tabla comparativa entre LLM Estándar y LLM con RAG
# 
# | Característica           | LLM Estándar            | LLM con RAG                        |
# |--------------------------|-------------------------|-------------------------------------|
# | Fuente de Conocimiento   | Solo datos preentrenados| Preentrenado + datos externos       |
# | Actualización de Datos   | Reentrenamiento         | Solo actualizar la base de datos    |
# | Precisión Factual        | Limitada al entrenamiento| Mejorada por la recuperación        |
# | Adaptación a Dominios    | Difícil                 | Fácil, eligiendo fuentes de datos   |
# | Citas/Referencias        | Rara vez                | Posible con metadatos               |
# 
# 
# 
# 

# %% [markdown]
# ### Etapas de un sistema RAG
# 
# 1. **Knowledge Storage** (Almacenamiento del conocimiento)
# Se recopila y almacena la información relevante en una base de datos, normalmente como vectores para búsqueda semántica. Los datos pueden venir de documentos, bases de datos, páginas web, etc.
# 
# 2. **Question Reception** (Recepción de la pregunta)
# El sistema recibe la pregunta o consulta del usuario, lo que inicia el proceso de búsqueda y generación.
# 
# 3. **Information Retrieval** (Recuperación de información)
# La pregunta se transforma en un vector y se busca en la base de conocimiento los fragmentos más relevantes usando similitud semántica.
# 
# 4. **Prompt Construction** (Construcción del prompt)
# Se construye un mensaje (prompt) que combina la pregunta del usuario con los fragmentos de información recuperados, para proporcionar contexto al modelo.
# 
# 5. **Calling the Model** (Llamada al modelo)
# El prompt se envía al modelo generativo (por ejemplo, un LLM), que utiliza tanto su conocimiento preentrenado como la información contextual proporcionada.
# 
# 6. Response from the Model (Respuesta del modelo)
# El modelo genera y devuelve una respuesta fundamentada, utilizando tanto la información recuperada como su propio conocimiento. Esta respuesta es entregada al usuario.
# 

# %% [markdown]
# ### Selección de modelos en Hugging Face
# 
# Hugging Face tiene en el momento de escribir estas líneas (mayo 2025) casi 2.000.000 de modelos disponibles. Para encontrar un modelo existen los siguientes filtros.
# 
# **Tasks (Tareas):**
# Permiten filtrar modelos según la tarea de inteligencia artificial que resuelven, como clasificación de texto, generación de texto, traducción automática, análisis de sentimientos, procesamiento de imágenes, reconocimiento de voz, etc.
# 
# **Libraries (Librerías):**
# Permiten filtrar modelos según la librería o framework con el que fueron desarrollados y pueden ejecutarse, por ejemplo: Transformers, PyTorch, TensorFlow, JAX.
# 
# **Datasets (Conjuntos de datos):**
# Permiten buscar modelos que han sido entrenados o ajustados usando conjuntos de datos específicos, como Imagenet, GLUE, SQuAD, COCO, entre otros.
# 
# **Languages (Idiomas):**
# Permiten filtrar modelos según el idioma o los idiomas que soportan, como inglés, español, multilingüe, etc.
# 
# **Licenses (Licencias):**
# Permiten filtrar modelos según la licencia bajo la que se distribuyen, como MIT, Apache 2.0, GPL, entre otras.
# 

# %% [markdown]
# ### Bases de datos vectoriales
# 
# Una **base de datos vectorial** (o *embedding database*) es un tipo de base de datos diseñada específicamente para almacenar y buscar **vectores numéricos de alta dimensión** de manera eficiente. En el contexto de los LLMs, estos vectores se llaman **embeddings**.
# 
# ### ¿Cómo se realiza la búsqueda en una base de datos vectorial?
# 
# 
# 1. **Creación de embeddings:**  
#    Cada fragmento de texto (por ejemplo, un párrafo o documento) se transforma en un vector numérico usando un modelo de embeddings. Este vector representa el significado del texto en un espacio de muchas dimensiones.
# 
# 2. **Almacenamiento:**  
#    Los vectores generados se almacenan en la base de datos vectorial, en vez de guardar solo el texto original. Este almacenamiento permite búsquedas eficientes basadas en similitud semántica, no solo coincidencia exacta de palabras.
# 
# 3. **Búsqueda (Vector Search):**  
#    Cuando un usuario realiza una consulta, la pregunta también se convierte en un vector usando el mismo modelo de embeddings.  
#    La base de datos compara este vector de consulta con los vectores almacenados, utilizando métricas de similitud como la distancia coseno o euclidiana, para encontrar los fragmentos más cercanos en significado.
# 
# 4. **Recuperación de información:**  
#    Los fragmentos de texto más similares (los vectores más cercanos) se recuperan y pueden usarse como contexto para que el LLM genere una respuesta precisa y relevante.
# 
# ### Ventajas de las bases de datos vectoriales
# 
# - Permite búsquedas por significado, no solo por coincidencia exacta de palabras.
# - Es eficiente incluso con grandes volúmenes de datos no estructurados.
# - Es fundamental en sistemas de búsqueda semántica y en pipelines RAG.

# %% [markdown]
# ### Práctica de RAG
# 
# En esta práctica vamos a construir una aplicación con soporte RAG utilizando un modelo de lenguaje de código abierto de Hugging Face y la base de datos vectorial de código abierto creada por Hugging Face llamada Chroma Database.
# 
# ```mermaid
# flowchart LR
#     subgraph Documents
#         A[Doc]
#         B[Doc]
#         C[Doc]
#     end
#     A --> D[ChromaDB<br>Embedding Database]
#     B --> D
#     C --> D
#     E[QUERY] --> D
#     D --> F[AUGMENTED PROMPT]
#     F --> G[Large<br>Language Model]
# ```
# 
# Para ejecutar esta práctica se requiere GPU. Si no se dispone de ella se debe ejecutar en Google Colab o en Kaggle.

# %% [markdown]
# Suponiendo que se ha elegido ejecutarla en Google Colab. La siguiente celda monta `drive` para poder almacenar los datos en Google Drive.

# %%
from google.colab import drive
drive.mount('/content/drive')

# %% [markdown]
# El siguiente paso es descargar el dataset que vamos a utilizar. En este caso es el [Topic Labeled News Dataset
# ](https://www.kaggle.com/datasets/kotartemiy/topic-labeled-news-dataset) de Kaggle. Este conjunto de datos contiene 108.774 artículos etiquetados en 8 categorías en función de la temática del artículo. Se puede descargar el dataset y situarlo en el punto de montaje de `drive`, o se puede hacer eso mismo desde Python. Para hacerlo desde Python hay que instalar el cliente de Kaggle y generar una API Key. En este caso la API Key se baja en un fichero llamado `kaggle.json`. Ese fichero contiene un objeto con dos claves (`username` y `key`). La clave `username` del fichero `kaggle.json` hay que asignarla a la variable de entorno `KAGGLE_USERNAME` y la clave `key` a la variable de entorno `KAGGLE_KEY`. Con estas variables de entorno, se puede descargar el `dataset`:

# %%
from google.colab import userdata
import os

os.environ["KAGGLE_USERNAME"] = userdata.get('KAGGLE_USERNAME')
os.environ["KAGGLE_KEY"] = userdata.get('KAGGLE_KEY')

# %%
!kaggle datasets download -d kotartemiy/topic-labeled-news-dataset

# %% [markdown]
# El fichero descargado está en formato `zip`. Hay que descomprimirlo:

# %%
import zipfile

# Define the path to your zip file
file_path = '/content/topic-labeled-news-dataset.zip'

with zipfile.ZipFile(file_path, 'r') as zip_ref:
    zip_ref.extractall('/content/datasets')

# %% [markdown]
# El fichero descargado es `labelled_newscatcher_dataset.csv`.  La extensión corresponde a `Comma Separated Values`. Es decir, es un fichero de texto en el que en cada fila hay un registro y los campos están separados por comas (a veces, como en este caso, por punto y coma o espacio). Suelen tener una primera fila de cabacera que contiene los nombres de los campos. Para cargar el `dataset` y visualizarlo, la librería más popular es Pandas. Cuando leemos el fichero en memoria, hablamos de un `dataframe`:

# %%
import pandas as pd

df = pd.read_csv('/content/datasets/labelled_newscatcher_dataset.csv', sep=';')
df.head()

# %% [markdown]
# Observamos que los campos son autoexplicativos.

# %% [markdown]
# Definimos algunas constantes:

# %%
MAX_NEWS = 1000
DOCUMENT="title"
TOPIC="topic"
ID="id"

# %% [markdown]
# ChromaDB requiere que los datos tengan un identificador único. Se puede lograr esto con la siguiente sentencia, que creará una nueva columna llamada `id`.

# %%
df["id"] = df.index
df.head()

# %%
#Because it is just a example we select a small portion of News.
subset_news = df.head(MAX_NEWS)

# %% [markdown]
# Instalamos ChromaDB y la importamos junto con sus `settings` para poder configurar la base de datos.

# %%
!pip install -q chromadb

# %%
import chromadb
from chromadb.config import Settings

# %% [markdown]
# ChromaDB puede trabajar en memoria, pero si se desa persistir, descomentar y ejecutar la líena correspondiente de la siguiente celda apuntando a un directorio de Google Drive.

# %%
chroma_client = chromadb.EphemeralClient() # Este es el cliente específicamente para uso en memoria
#chroma_client = chromadb.PersistentClient(path="/content/drive/MyDrive/chromadb")

# %% [markdown]
# ChromaDB trabaja con colecciones, normalmente cada `dataset` se almacena en una colección. Si se intenta crear una colección con un nombre existente, se produce un error. En la siguiente celda se crea una colección con un nombre diferente para cada ejecución. También se borran las colecciones anteriores si se hubieran creado:

# %%
from datetime import datetime

collection_name = "news_collection"+datetime.now().strftime("%s")
if len(chroma_client.list_collections()) > 0 and collection_name in [chroma_client.list_collections()[0].name]:
        chroma_client.delete_collection(name=collection_name)

collection = chroma_client.create_collection(name=collection_name)

# %% [markdown]
# Una colección es un contenedor de `embeddings` con un identificador único y, opcionalmente, metadatos. Los metadatos son datos adicionales clave-valor asociados a cada `embedding`. Esto es muy útil para filtrar resultados antes o después de la búsqueda de similitud (ej., `{"autor": "Juan", "fecha": "2023-01-15"}`). Junto con el `embedding` se pueden almacenar los documentos originales que los produjeron. Las colecciones proporcionan un aislamiento lógico, lo que significa que las búsquedas de similitud se realizan solo dentro de la colección específica que se está consultando. Los resultados de diferentes conjuntos de datos no se mezclan a menos que así se explicite.
# 
# Dependiendo de la longitud de los documentos almacenados se puede considerar dividirlos en fragmentos más pequeños (como páginas o capítulos). Esto se debe a que los LLM tienen una "ventana de contexto" limitada para la información que pueden procesar en un `prompt`. En proyectos más avanzados, en lugar de pasar directamente los fragmentos recuperados al `prompt`, a menudo se usa otro modelo para generar un resumen más corto y relevante de esa información, que luego se incluye en el `prompt` para el LLM principal. En esta práctica, se va a usar el documento entero para crear el `propmt`.
# 

# %% [markdown]
# Para añadir los datos a la colección se una la función ***add***, se debe informar, al menos, de ***documents***, ***metadatas*** e ***ids***. En nuestro caso:
# 
# * En **document** será el `title` de la noticia.
# * En **metadatas** será el `topic` de la noticia.
# * En **id** debe informarse un identificador único para cada fila. Campo `id`

# %%
collection.add(
    documents=subset_news[DOCUMENT].tolist(),
    metadatas=[{TOPIC: topic} for topic in subset_news[TOPIC].tolist()],
    ids=subset_news[ID].map(lambda x: f"id-{x}").tolist()
)

# %% [markdown]
# Vamos a consultar los 10 documentos que más relación tengan con la palabra `laptop`:

# %%
results = collection.query(query_texts=["laptop"], n_results=10 )

print(results)

# %% [markdown]
# Ahora vamos importar de la librería `transformers` de Hugging Face los siguientes elementos:
# 
# * **AutoTokenizer**: Esta herramienta se usa para tokenizar texto y es compatible con muchos de los modelos preentrenados disponibles en la librería de Hugging Face.
# * **AutoModelForCasualLM**: Proporciona una interfaz para usar modelos diseñados específicamente para tareas de generación de texto, como los basados en GPT. En esta práctica, se usará el modelo `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.
# * **Pipeline**: Al usar pipelines, la librería transformers se encarga de la mayoría de las tareas. Cuando creas un pipeline de generación de texto, solo se necesita pasar el prompt al modelo y recibir el resultado.

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# %% [markdown]
# Para poder usar un modelo de Hugging Face, la API Key `HF_TOKEN` debe estar cargada en el entorno.

# %%
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
lm_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

# %% [markdown]
# Pasamos estas variables al `pipeline`. El primer parámetro es la tarea que la `pipeline` debe realizar, y debe coincidir con la tarea para la que el modelo ha sido entrenado.

# %%
pipe = pipeline(
    "text-generation",
    model=lm_model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    device_map="auto",
)

# %% [markdown]
# Después aumentamos el `prompt` con los resutados de la consulta:

# %%
question = "¿Cuánto cuesta un Acer 3 en India?"
context = " ".join([f"#{str(i)}" for i in results["documents"][0]])
#context = context[0:5120]
prompt_template = f"""
Relevant context: {context}
Considering the relevant context, answer the question.
Question: {question}
Answer: """
prompt_template

# %% [markdown]
# Para obtener una respuesta, le pasamos al `pipeline` el `prompt`:

# %%
lm_response = pipe(prompt_template)
print(lm_response[0]["generated_text"])


