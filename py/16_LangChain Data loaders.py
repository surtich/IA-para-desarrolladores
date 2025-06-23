# %% [markdown]
# ### Introducción
# 
# En esta práctica vamos a proporcionar a un modelo conversacional información proveniente de distintos tipos de fuentes de información no estructurada, como puede ser una hoja de cálculo, un documento de Word, un EPUB, ... El proceso siempre va a ser el mismo:
# 
# 1. **Extracción de la información**: Extraeremos el texto de la fuente de información no estructurada.
# 2. **Trocear el texto**: Trocearemos el texto en fragmentos de tamaño adecuado para que el modelo pueda procesarlo.
# 3. **Creación de los embeddings**: Crearemos los embeddings de los fragmentos de texto.
# 4. **Almacenamiento de los embeddings**: Almacenaremos los embeddings en una base de datos vectorial.
# 5. **Consulta a la base de datos vectorial**: Consultaremos la base de datos vectorial para obtener los fragmentos de texto más similares.
# 6. **Generación de la respuesta**: Pasamos como contexto los fragmentos de texto más similares al modelo conversacional para que genere una respuesta.

# %% [markdown]
# ### LangChain con Excel

# %% [markdown]
# Empezamos haciendo importaciones

# %%
# Import functions / modules
from langchain_community.document_loaders.excel import UnstructuredExcelLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

import os

# %% [markdown]
# Leemos el fichero.

# %%
# Load the excel data
# Use UnstructuredExcelLoader to parse the data into individual components
# mode = "elements" will load each cell as a separate document
# mode = "table" will load the entire table as a single document
loader = UnstructuredExcelLoader(os.path.join("../excel", "Reviews.xlsx"), mode = "elements")
docs = loader.load() # Load the data from the Excel file into `docs`

# Display the first 5 elements
docs[:5]

# %% [markdown]
# Partimos el contenido en `chuncks`. Los `embeddings` se crean por cada `chunk`. Cuanto más grande sea el `chunk`, más lento será el proceso de creación de `embeddings` y algunos proveedores no permiten `chunks` grandes, pero más contexto se proporciona. El `overlap` es el número de tokens que se repiten entre `chunks` consecutivos. Cuanto mayor sea, más contexto se mantiene entre `chucks`.

# %%
# Initiate text splitters
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2000,    # Define the maximum size of each chunk
    chunk_overlap = 200   # Define the overlap between chunks
)
chunks = text_splitter.split_documents(docs) # Split the documents into chunks

# Display the first 5 chunks
chunks[:5]

# %% [markdown]
# Chroma sólo acepta como metadatos, tipos simples, así que convertimos los tipos complejos a cadenas. El inconveniente es que no se pueden hacer consultas sobre los metadatos complejos, pero sí se pueden hacer consultas sobre los metadatos simples.

# %%
import json

for doc in chunks:
    for k, v in list(doc.metadata.items()):
        if isinstance(v, (list, dict)):
            doc.metadata[k] = json.dumps(v)


# %% [markdown]
# Creamos los `embeddings`.

# %%
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from a .env file
google_api_key = os.getenv("GOOGLE_API_KEY")  # Get the Gemini API key from environment variables

# %%
# Create embeddings using Google Gemini embedding model
embeddings_client = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# %%
db_chroma = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings_client
)

# %% [markdown]
# Hacemos una consulta.

# %%
# Try the Retrieval System
query = "give me my worst reviews with comments"

# Retrieve the context using Chroma; Langchain uses the Cosine Distance Metric by default
docs_chroma = db_chroma.similarity_search_with_score(query, k = 5) # Retrieve the top 5 relevant documents based on similarity

docs_chroma[0][1]

# %%
# Merge the retrieved documents into a single context string for the Generation system
context_text = "\n\n".join([doc.page_content for doc, _score in docs_chroma])
context_text

# %%
# Crea un prompt simple para un sistema RAG
prompt = f"""
basado en este contexto {context_text}
por favor, responde esta pregunta {query}
si no sabes la respuesta simplemente di que no lo sabes.
"""


# %%
# Call the Google Gemini API with the Langchain
model = ChatGoogleGenerativeAI(
    google_api_key = google_api_key,
    model="gemini-2.0-flash",

    temperature = 0 # Set temperature to 0 for deterministic responses
)
# Generate a response using the provided prompt
response_text = model.invoke(prompt)

# %%
from IPython.display import Markdown, display

display(Markdown(response_text.content))

# %% [markdown]
# Creamos funciones para hacer el mismo proceso.

# %%
# Preparing the unstructured data
def prepare_excel(file_path):
    # Loading the data from the specified Excel file
    loader = UnstructuredExcelLoader(file_path, mode = "elements")
    docs = loader.load()

    # Split the text into chunks for more manageable processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 2000, # Size of each chunk
        chunk_overlap = 200 # Overlap between chunks
    )
    chunks = text_splitter.split_documents(docs)

    for doc in chunks:
        for k, v in list(doc.metadata.items()):
            if isinstance(v, (list, dict)):
                doc.metadata[k] = json.dumps(v)


    embeddings_client = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    db_chroma = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings_client
    )

    # Return the Chroma index for further use
    return db_chroma

# %%
# Prepare a function to retrieve and generate (RAG)
def ask(df, query, k):
  # Retrieve relevant documents from the Chroma index
  docs_chroma = df.similarity_search_with_score(query, k = k)

  # Merge the content of retrieved documents into a single context string
  context_text = "\n\n".join([doc.page_content for doc, _score in docs_chroma])

  # Define the prompt for the language model
  prompt = f"""
  based on this context {context_text}
  please answer this question {query}
  if the information is not in the context, say that you don't have information.
  """

  # Call the Google Gemini API with the Langchain
  model = ChatGoogleGenerativeAI(
      google_api_key = google_api_key,
      model="gemini-2.0-flash",

      temperature = 0 # Set temperature to 0 for deterministic responses
  )
  # Generate a response using the provided prompt
  response_text = model.invoke(prompt)

  # Display the response as Markdown
  return display(Markdown(response_text.content))

# %%
# Preparing the excel Data
df_excel = prepare_excel(os.path.join("../excel", "Reviews.xlsx"))

# %%
# Define the query
query = """
Analyse the comments for the Econometrics course and tell me the top 3 improvements
"""

# %%
# Ask the question
ask(df_excel, query, 20)

# %% [markdown]
# ### LangChain con Word

# %% [markdown]
# Empezamos importando `nltk`para `tokenizar`el texto.

# %%
# Import nltk library
import nltk
nltk.download('punkt')  # Download the 'punkt' tokenizer model for sentence and word tokenization

# Import the document loader for unstructured Word documents
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

# %% [markdown]
# Creamos una función que hace lo mismo que hicimos con Excel.

# %%
# Create a function to prepare the word document
def prepare_word(file_path):
    # Loading the data from the Word document
    loader = UnstructuredWordDocumentLoader(
        file_path, # Path to the Word document
        mode = "elements"
    )
    # Load the document content into `docs`
    docs = loader.load()

    # Split the document into chunks for processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    # Split the documents into chunks
    chunks = text_splitter.split_documents(docs)

    for doc in chunks:
        for k, v in list(doc.metadata.items()):
            if isinstance(v, (list, dict)):
                doc.metadata[k] = json.dumps(v)

    embeddings_client = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    db_chroma = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings_client
    )

    # Return the Chroma index for further use
    return db_chroma

# %%
db_doc = prepare_word(os.path.join("../word", "Declaration of independence.docx"))

# %%
# Define a couple of queries
query1 = "Señala en español los tres puntos más importantes del documento"
query2 = "Dime la mejor receta de pastel de chocolate"

# %%
ask(db_doc, query1, k = 5)

# %%
ask(db_doc, query2, k = 5)

# %% [markdown]
# ### LangChain con PowerPoint
# 
# El proceso es muy similar

# %%
from langchain_community.document_loaders import UnstructuredPowerPointLoader

# %%
# Build a function to prepare powerpoint
def prepare_ppt(file_path):
    # Loading the data from the PowerPoint presentation
    loader = UnstructuredPowerPointLoader(file_path, mode = "elements")
    docs = loader.load()

    # Split the document into chunks for processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    # Split the documents into chunks
    chunks = text_splitter.split_documents(docs)

    for doc in chunks:
            for k, v in list(doc.metadata.items()):
                if isinstance(v, (list, dict)):
                    doc.metadata[k] = json.dumps(v)

    for doc in chunks:
        for k, v in list(doc.metadata.items()):
            if isinstance(v, (list, dict)):
                doc.metadata[k] = json.dumps(v)

    embeddings_client = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    db_chroma = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings_client
    )

    # Return the Chroma index for further use
    return db_chroma

# %%
# Prepare the presentation data
db_pp = prepare_ppt(os.path.join("../ppt", "Bitte pitch deck EN.pptx"))

# %%
# Define a couple of queries to test the presentation
query1 = "What is Bitte's competitive advantage?"
query2 = "What is indepence"

# %%
# Ask the questions
ask(db_pp, query1, k = 5)

# %%
# Ask the second question
ask(db_pp, query2, 5)

# %% [markdown]
# # LangChain con EPUB

# %%
# Import the libraries
from langchain_community.document_loaders import UnstructuredEPubLoader
import pypandoc

# Download Pandoc if it is not already installed
pypandoc.download_pandoc()

# %%
# Build function to prepare epub
def prepare_epub(file_path):
    # Loading the data from the ePub
    loader = UnstructuredEPubLoader(file_path, mode = "elements")
    docs = loader.load()

    # Split the document into chunks for processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 20
    )
    # Split the documents into chunks
    chunks = text_splitter.split_documents(docs)

    for doc in chunks:
        for k, v in list(doc.metadata.items()):
            if isinstance(v, (list, dict)):
                doc.metadata[k] = json.dumps(v)

    embeddings_client = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    db_chroma = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings_client
    )

    # Return the Chroma index for further use
    return db_chroma

# %%
db_epub = prepare_epub(os.path.join("../epub", "Alice’s Adventures in Wonderland.epub"))

# %%
# Prepare a couple of queries
query1 = "What is the main point of the story?"
query2 = "Does Alice like Bitte's digital menus?"

# %%
# Answer query 1
ask(db_epub, query1, 5)

# %%
# Answer query 1
ask(db_epub, query1, 30)

# %%
# Answer query 1
ask(db_epub, query2, 10)

# %% [markdown]
# ### LangChain con PDFs

# %% [markdown]
# Se necesitan estos paquetes de Python y del sistema. Los paquetes de Python ya se han instalado con `uv`. Comprobar que están instaladas los paquetes del sistema.

# %%
# Install libraries
%pip install pymupdf pdfminer.six pillow_heif unstructured_inference unstructured_pytesseract pi_heif
%apt-get install poppler-utils
%apt install tesseract-ocr

# %%
# Import the libraries
from langchain_community.document_loaders import UnstructuredPDFLoader
from pdfminer import psparser

# %%
# Build function to prepare pdf
def prepare_pdf(file_path):
    # Loading the data from the PDF document
    loader = UnstructuredPDFLoader(file_path, mode = "elements")
    docs = loader.load()

    # Split the document into chunks for processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    
    # Split the documents into chunks
    chunks = text_splitter.split_documents(docs)

    for doc in chunks:
        for k, v in list(doc.metadata.items()):
            if isinstance(v, (list, dict)):
                doc.metadata[k] = json.dumps(v)

    embeddings_client = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    db_chroma = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings_client
    )

    # Return the Chroma index for further use
    return db_chroma



# %%
# Apply the function
db_pdf = prepare_pdf(os.path.join("../pdfs/recipes", "Famous old receipts.pdf"))

# %%
# Query inspirations
query1 = "Which recipes are no longer common today? Tell me which and tell me how to make them"
query2 = "Which recipes would impress my friends"

# %%
# Try query 1
ask(db_pdf, query1, 5)

# %%
# Try query 1
ask(db_pdf, query2, 5)


