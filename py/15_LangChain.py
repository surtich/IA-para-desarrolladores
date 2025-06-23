# %% [markdown]
# ### ¿Qué es LangChain?
# 
# **LangChain** es un `framework` de código abierto diseñado para facilitar el desarrollo de aplicaciones que utilizan modelos de lenguaje. Permite unificar la interfaz de acceso a distintos proveedores.
# 
# ### Características principales LangChain
# 
# - **Orquestación modular:** LangChain proporciona componentes modulares que se pueden combinar para crear flujos de trabajo complejos, llamados "cadenas" (pipelines). Cada cadena es una secuencia de pasos que pueden incluir llamadas a modelos de lenguaje, consultas a bases de datos, procesamiento de texto.
# - **Integración sencilla:** Permite integrar casi cualquier modelo de lenguaje, tanto de código abierto como comercial, usando una interfaz estándar y sencilla.
# - **Gestión de contexto y memoria:** Facilita la gestión del estado de la conversación y el contexto, permitiendo que las aplicaciones recuerden interacciones anteriores y ofrezcan respuestas más coherentes y personalizadas.
# - **Automatización y agentes:** Permite crear agentes inteligentes que pueden tomar decisiones, consultar diferentes fuentes de datos y ejecutar acciones de forma autónoma.
# - **Soporte para Python y JavaScript:** Está disponible principalmente para estos lenguajes, facilitando su adopción en proyectos modernos.
# 
# 

# %% [markdown]
# ### Input y Output

# %% [markdown]
# En el siguiente ejemplo, comparamos el uso de un LLM con la API de OpenAI con la de LangChain.

# %%
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# %%
from openai import OpenAI

openai = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta", api_key=os.getenv('GOOGLE_API_KEY'))

response = openai.chat.completions.create(
    model="gemini-2.0-flash",
    messages=[
        {"role": "system", "content": "Eres un asistente útil especializado en proporcionar información sobre el Restaurante Italiano BellaVista."},
        {"role": "user", "content": "¿Qué hay en el menú?"},
        {"role": "assistant", "content": "BellaVista ofrece una variedad de platos italianos que incluyen pasta, pizza y mariscos."},
        {"role": "user", "content": "¿Tienen opciones veganas?"}
    ]
)

response.model_dump()

# %%
print(response.choices[0].message.content)

# %%
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=os.getenv('GOOGLE_API_KEY'))

llm.invoke([
        ("system",  "Eres un asistente útil especializado en proporcionar información sobre el Restaurante Italiano BellaVista."),
        ("human",  "¿Qué hay en el menú?"),
        ("ai",  "BellaVista ofrece una variedad de platos italianos que incluyen pasta, pizza y mariscos."),
        ("human",  "¿Tienen opciones veganas?")
    ])

# %% [markdown]
# Un forma alternativa de enviar mensajes.

# %%
from langchain.schema import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage(content="Eres un asistente útil especializado en proporcionar información sobre el Restaurante Italiano BellaVista."),
    HumanMessage(content="¿Cuál es el menú?"),
    AIMessage(content="BellaVista ofrece una variedad de platos italianos que incluyen pasta, pizza y mariscos."),
    HumanMessage(content="¿Tienen opciones veganas?")
]

llm_result = llm.invoke(input=messages)
llm_result

# %% [markdown]
# Los mensajes se pueden eviar en `batch`

# %%
batch_messages = [
    [
        SystemMessage(content="Eres un asistente útil que traduce del inglés al alemán"),
        HumanMessage(content="Do you have vegan options?")
    ],
    [
        SystemMessage(content="Eres un asistente útil que traduce del español al inglés."),
        HumanMessage(content="¿Tienen opciones veganas?")
    ],
]
batch_result = llm.generate(batch_messages)
batch_result


# %%
translations = [generation[0].text for generation in batch_result.generations]
translations

# %% [markdown]
# ### Prompt templates
# 
# LangChain permite crear plantillas de prompts que pueden ser reutilizadas y parametrizadas. Esto facilita la creación de mensajes complejos y dinámicos para los modelos de lenguaje.

# %%
from langchain.prompts.prompt import PromptTemplate

TEMPLATE = """
Eres un asistente útil que traduce del {input_language} al {output_language}
"""

prompt_template = PromptTemplate(
    input_variables=["input_language", "output_language", "text"],
    template="Eres un asistente útil que traduce del {input_language} al {output_language}. Traduce: {text}"
)

prompt = prompt_template.format(
    input_language="español",
    output_language="inglés",
    text="¿A qué te dedicas?"
)

llm.invoke(prompt)

# %% [markdown]
# LangChain facilita la creación de prompts con ejemplo (Few Shot Prompt)

# %%
from langchain_core.prompts import FewShotPromptTemplate

# Ejemplos de clasificación
examples = [
    {
        "text": "El restaurante BellaVista ofrece una experiencia culinaria exquisita. Los sabores son ricos y la presentación es impecable.",
        "sentiment": "positive",
        "subject": "BellaVista"
    },
    {
        "text": "El restaurante BellaVista estuvo bien. La comida era decente, pero nada destacaba.",
        "sentiment": "neutral",
        "subject": "BellaVista"
    },
    {
        "text": "Me decepcionó BellaVista. El servicio fue lento y los platos carecían de sabor.",
        "sentiment": "negative",
        "subject": "BellaVista"
    },
    {
        "text": "SeoulSavor ofreció los sabores coreanos más auténticos que he probado fuera de Seúl. El kimchi estaba perfectamente fermentado y picante.",
        "sentiment": "positive",
        "subject": "SeoulSavor"
    },
    {
        "text": "SeoulSavor estuvo bien. El bibimbap era bueno, pero el bulgogi era un poco demasiado dulce para mi gusto.",
        "sentiment": "neutral",
        "subject": "SeoulSavor"
    },
    {
        "text": "No disfruté mi comida en SeoulSavor. El tteokbokki estaba demasiado blando y el servicio no fue atento.",
        "sentiment": "negative",
        "subject": "SeoulSavor"
    },
    {
        "text": "MunichMeals tiene la mejor bratwurst y sauerkraut que he probado fuera de Baviera. Su ambiente de jardín de cerveza es verdaderamente auténtico.",
        "sentiment": "positive",
        "subject": "MunichMeals"
    },
    {
        "text": "MunichMeals estuvo bien. La weisswurst estaba bien, pero he probado mejores en otros lugares.",
        "sentiment": "neutral",
        "subject": "MunichMeals"
    },
    {
        "text": "Me decepcionó MunichMeals. La ensalada de patatas carecía de sabor y el personal parecía desinteresado.",
        "sentiment": "negative",
        "subject": "MunichMeals"
    }
]

# Plantilla para cada ejemplo
example_prompt = PromptTemplate(
    input_variables=["text", "sentiment", "subject"],
    template=(
        "text: {text}\n"
        "sentiment: {sentiment}\n"
        "subject: {subject}\n"
    )
)

# Plantilla FewShot con sufijo para el nuevo caso a clasificar
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="text: {text}\nsentiment:",
    input_variables=["text"]
)

# Opinión a clasificar
nueva_opinion = "El ambiente de BellaVista era agradable, pero la comida llegó fría y tardó mucho."

# Formatear el prompt final
prompt_final = few_shot_prompt.format(text=nueva_opinion)

llm.invoke(prompt_final)


# %% [markdown]
# Los prompts se puede componer para facilitar la reutilización.

# %%
from langchain.prompts.pipeline import PipelinePromptTemplate

# Introducción
introduction_template = """
Interpreta el texto y evalúalo. Determina si el texto tiene un sentimiento positivo, neutral o negativo. Además, identifica el tema del texto en una palabra.
"""
introduction_prompt = PromptTemplate.from_template(introduction_template)

# Ejemplo
example_template = """
Instrucciones de Cadena de Pensamiento:
Comencemos evaluando una declaración. Considera: "{example_text}". ¿Cómo te hace sentir esto sobre {example_subject}?
Respuesta: {example_evaluation}

Basado en la naturaleza {example_sentiment} de esa declaración, ¿cómo formatearías tu respuesta?
Respuesta: {example_format}
"""
example_prompt = PromptTemplate.from_template(example_template)

# Ejecución
execution_template = """
Ahora, ejecuta este proceso para el texto: "{input}".
"""
execution_prompt = PromptTemplate.from_template(execution_template)

# Componiendo el prompt completo
full_template = """{introduction}

{example}

{execution}"""
full_prompt = PromptTemplate.from_template(full_template)

# Prompts de Pipeline
input_prompts = [
    ("introduction", introduction_prompt),
    ("example", example_prompt),
    ("execution", execution_prompt)
]
pipeline_prompt = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)

prompt = pipeline_prompt.format(
    example_text="El restaurante BellaVista ofrece una experiencia culinaria exquisita. Los sabores son ricos y la presentación es impecable.",
    example_subject="BellaVista",
    example_evaluation="Suena como una crítica positiva para BellaVista.",
    example_sentiment="positive",
    example_format='{ "sentiment": "positive", "subject": "BellaVista" }',
    input="El nuevo restaurante del centro tiene platos insípidos y el tiempo de espera es demasiado largo."
)

print(prompt)

llm.invoke(prompt)  


# %% [markdown]
# Los prompts se pueden almacenar en disco duro y recuperar.

# %%
prompt_template = PromptTemplate(input_variables=["input"], template="Cuéntame un chiste sobre {input}")
prompt_template.save("prompt.yaml")
prompt_template.save("prompt.json")

# %%
from langchain.prompts import load_prompt

prompt_template = load_prompt("prompt.yaml")
prompt = prompt_template.format(input="gatos")

llm.invoke(prompt)

# %% [markdown]
# ### Chains
# 
# LangChain permite crear "cadenas" (chains) que son secuencias de pasos que pueden incluir llamadas a modelos de lenguaje. Las cadenas pueden ser simples o complejas, y permiten orquestar el flujo de trabajo de la aplicación. La forma más sencilla de crear una cade es usar LCEL (LangChain Expression Language), que permite definir cadenas de forma declarativa.

# %%
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("cuéntame un chiste corto sobre {topic}")
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

chain.invoke({"topic": "helado"})

# %%
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel

chain = RunnableParallel({"x": RunnablePassthrough(), "y": RunnablePassthrough()})
chain.invoke({"input": "hello", "input2": "goodbye"})


# %% [markdown]
# ### Callbacks
# 
# LangChain permite registrar callbacks para monitorear y depurar el flujo de trabajo de las cadenas. Los callbacks pueden ser utilizados para registrar información, manejar errores o realizar acciones específicas en diferentes etapas del proceso.

# %%
from langchain.callbacks import StdOutCallbackHandler

prompt_template = PromptTemplate(input_variables=["input"], template="Cuéntame un chiste sobre {input}")
chain = prompt | llm

handler = StdOutCallbackHandler()

config = {
    'callbacks' : [handler]
}

chain.invoke(input="león", config=config)

# %% [markdown]
# Se puede personalizar la función del `callback`.

# %%
from langchain.callbacks.base import BaseCallbackHandler

class MyCustomHandler(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs) -> None:
        print(f"REPONSE: ", response)

handler = MyCustomHandler()

config = {
    'callbacks' : [handler]
}

chain.invoke(input="pingüinos", config=config)

# %% [markdown]
# ### Memoria
# 
# LangChain proporciona mecanismos para gestionar el estado de la conversación y el contexto, permitiendo que las aplicaciones recuerden interacciones anteriores.
# 
# Algunos de los tipos de memoria disponibles en LangChain son los siguientes:
# 
# | Tipo de memoria                    | Descripción breve                                                                                         | Uso típico                                                  |
# |------------------------------------|----------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
# | ConversationBufferMemory           | Guarda toda la conversación en un buffer (lista o cadena).                                               | Chatbots, asistentes, historias interactivas                |
# | BufferWindowMemory                 | Similar al buffer, pero solo mantiene las últimas k interacciones.                                       | Limitar contexto a lo más reciente                          |
# | ConversationSummaryMemory          | Resume la conversación usando un LLM para obtener un resumen compacto y relevante.                       | Conversaciones largas o multitópico                         |
# | EntityMemory / EntityStoreMemory   | Extrae y almacena entidades (nombres, lugares, fechas) y sus atributos a lo largo de la conversación.    | Asistentes personalizados, CRM, sistemas de recomendación   |
# | VectorStore-Backed Memory          | Almacena recuerdos en una base de datos vectorial y recupera los más relevantes según el contexto.        | Recuperación de información, QA, chatbots con memoria larga |
# | DynamoDB/Momento/Redis/Upstash     | Variantes que almacenan la memoria en bases de datos externas para persistencia a largo plazo y escalabilidad. | Soporte de sesiones largas, multiusuario, persistencia real |
# | Motörhead / Zep                    | Servidores de memoria avanzados que permiten sumarización incremental, embedding, indexación y enriquecimiento de historiales. | Aplicaciones avanzadas, análisis de conversaciones           |
# 

# %%
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.chat_memory.add_user_message("¡Hola! Me llamo Juan.")
memory.chat_memory.add_ai_message("¡Hola, Juan!")
memory.load_memory_variables({})

# %%
from langchain.chains.conversation.base import ConversationChain

conversation = ConversationChain(
    llm=llm, verbose=True, memory=memory
)
conversation.invoke(input="¿Cómo me llamo?")

# %%
conversation.invoke(input="Quiero que me llames Juanito.")

# %%
conversation.invoke(input="¿Cómo me llamo?")

# %% [markdown]
# Cuando las entradas se hacen largas, quizás no queramos enviar la conversación completa, sino un resumen.

# %%
from langchain.memory import ConversationSummaryBufferMemory

review = "Pedí Pizza Salami por 9.99$ y ¡estaba increíble! \
La pizza fue entregada a tiempo y todavía estaba caliente cuando la recibí. \
La masa era fina y crujiente, y los ingredientes eran frescos y sabrosos. \
El Salami estaba bien cocido y complementaba el queso perfectamente. \
El precio era razonable y creo que obtuve el valor de mi dinero. \
En general, estoy muy satisfecho con mi pedido y recomendaría esta pizzería a otros."

summary_memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100) # Si el token es 1000, no se creará ningún resumen
summary_memory.save_context(
    {"input": "Hola, ¿cómo puedo ayudarte hoy?"},
    {"output": "¿Podrías analizar una reseña por favor?"},
)
summary_memory.save_context(
    {"input": "Claro, con gusto. ¿Podrías proporcionar la reseña?"},
    {"output": f"{review}"},
)

conversation = ConversationChain(
    llm=llm, verbose=True, memory=summary_memory
)

conversation.invoke(input="Muchas gracias")


# %% [markdown]
# ### Tools
# 
# LangChain permite definir herramientas que pueden ser utilizadas por los modelos de lenguaje para realizar acciones específicas, como consultar bases de datos, llamar a APIs externas o ejecutar código. Estas herramientas pueden ser integradas en las cadenas y utilizadas por los agentes para tomar decisiones informadas.

# %% [markdown]
# Sin tools, el modelo no puede responder a esta pregunta.

# %%
llm.invoke("¿Qué tiempo hace en Majadahonda ahora mismo?")

# %%
from langchain_core.tools import tool


@tool
def fake_weather_api(city: str) -> str:
    """
    Verifica el clima en una ciudad especificada.

    Args:
        city (str): El nombre de la ciudad donde quieres verificar el clima.

    Returns:
        str: Una descripción del clima actual en la ciudad especificada.
    """
    return "Soleado, 22°C"


@tool
def outdoor_seating_availability(city: str) -> str:
    """
    Verifica si hay asientos al aire libre disponibles en un restaurante específico en una ciudad dada.

    Args:
        city (str): El nombre de la ciudad donde quieres verificar la disponibilidad de asientos al aire libre.

    Returns:
        str: Un mensaje indicando si hay asientos al aire libre disponibles o no.
    """
    return "Asientos al aire libre disponibles."


tools = [fake_weather_api, outdoor_seating_availability]

# %%
llm_with_tools = llm.bind_tools(tools)

# %%
results = llm_with_tools.invoke("¿Qué tiempo hace en Majadahonda ahora mismo?")
results

# %% [markdown]
# El modelo puede pedir que se invoquen varias herramientas.

# %%
results = llm_with_tools.invoke("¿Qué tiempo hace en Majadahonda ahora mismo? ¿Hay asientos al aire libre disponibles?")
results

# %% [markdown]
# Otra forma de hacerlo.

# %%
from langchain_core.messages import HumanMessage, ToolMessage

messages = [
    HumanMessage(
        "¿Qué tiempo hace en Majadahonda ahora mismo? ¿Hay asientos al aire libre disponibles?"
    )
]
llm_output = llm_with_tools.invoke(messages)
llm_output


# %% [markdown]
# Añadimos la respuesta del modelo.

# %%
messages.append(llm_output)

# %% [markdown]
# Somos nosotros los que llamamos a las tools y proporcionamos el resultado al modelo.

# %%
tool_mapping = {
    "fake_weather_api": fake_weather_api,
    "outdoor_seating_availability": outdoor_seating_availability,
}

# %%
from langchain_core.messages import ToolMessage

for tool_call in llm_output.tool_calls:
    tool = tool_mapping[tool_call["name"].lower()]
    tool_output = tool.invoke(tool_call["args"])
    messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

# %%
messages

# %%
llm_with_tools.invoke(messages)

# %% [markdown]
# ### RAG
# 
# LangChain facilita la implementación de Retrieval-Augmented Generation (RAG), que combina la generación de texto con la recuperación de información relevante de bases de datos o documentos.

# %%
from google.colab import userdata
import os

os.environ["KAGGLE_USERNAME"] = userdata.get('KAGGLE_USERNAME')
os.environ["KAGGLE_KEY"] = userdata.get('KAGGLE_KEY')

# %%
!kaggle datasets download -d kotartemiy/topic-labeled-news-dataset

# %%
import zipfile

# Define the path to your zip file
file_path = '/content/topic-labeled-news-dataset.zip'

with zipfile.ZipFile(file_path, 'r') as zip_ref:
    zip_ref.extractall('/content/datasets')

# %%
import pandas as pd

df = pd.read_csv('/content/datasets/labelled_newscatcher_dataset.csv', sep=';')

# %%
MAX_NEWS = 1000
DOCUMENT="title"
TOPIC="topic"

subset_news = df.head(MAX_NEWS)

# %% [markdown]
# Aunque hemos leído en `dataset` en Pandas, LangChain puede cargar directamente el fichero `csv` con la librería `document_loader` y cargarlo en ChromaDB:

# %%
!pip install -q langchain
!pip install -q langchain_community

# %%
from langchain.document_loaders import DataFrameLoader
from langchain.vectorstores import Chroma

# %% [markdown]
# Creamos el`loader`, indicando la fuente de datos y el nombre de la columna en el `dataframe` que contiene la información.

# %%
df_loader = DataFrameLoader(subset_news, page_content_column=DOCUMENT)

# %% [markdown]
# Cargamos y mostramos el documento. Se observa que usa como `metadata` el resto de campos.

# %%
df_document = df_loader.load()
display(df_document[:2])

# %% [markdown]
# Ahora generamos los embeddings. Para ello, será necesario importar **CharacterTextSplitter:** para agrupar la información en `chunks`.
# 

# %%
from langchain.text_splitter import CharacterTextSplitter

# %% [markdown]
# No existe una forma 100% correcta de dividir los documentos en chunks). La clave está en equilibrar el contexto y el uso de memoria:
# 
# - **Fragmentos más grandes:** Proporcionan al modelo más contexto, lo que puede llevar a una mejor comprensión y respuestas más precisas. Sin embargo, consumen más memoria.
# - **Fragmentos más pequeños:** Reducen el uso de memoria, pero pueden limitar la comprensión contextual del modelo si la información queda demasiado fragmentada.
# 
# Se ha decidido usar un tamaño medio de 250 caracteres para cada `chunk` con un `overloap` de 10 caracteres. Es decir, los 10 caracteres finales de un `chunk`, serán los 10 primeros del siguiente.
# 

# %%
text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=10)
texts = text_splitter.split_documents(df_document)
display(texts[:2])

# %% [markdown]
# Ahora creamos los `embeddings`. Se puede usar directamente LangChain para hacer esto.

# %%
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# %% [markdown]
# Creamos la base de datos. Esta instrucción también crea los índices.

# %%
!pip install -q chromadb

# %%
chroma_db = Chroma.from_documents(
    texts, embedding_function
)

# %% [markdown]
# El siguiente paso es especificar el `retriever`, que recupera información de los documentos que le proporcionemos. En este caso hace una búsqueda por proximidad de los `embbeddings` almacenados en ChromaDB. El último paso es seleccionar el modelo de lenguaje que recibirá la `pipeline` de Hugging Face.

# %%
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

# %%
retriever = chroma_db.as_retriever()

# %%
model_id = "google/flan-t5-large"
task="text2text-generation"

hf_llm = HuggingFacePipeline.from_model_id(
    model_id=model_id,
    task=task,
    device_map="auto",
    pipeline_kwargs={
        "max_new_tokens": 256,
        "repetition_penalty":1.1, # penaliza que el modelo repita respuestas en el prompt. Parece que algunos modeos lo hacen
    },
)

# %% [markdown]
# Ahora configuramos la  `pipeline`:

# %%
document_qa = RetrievalQA.from_chain_type(
    llm=hf_llm, retriever=retriever, chain_type='stuff'
)

# %% [markdown]
# `chain_type` puede tener los siguientes valores:
# 
# - **stuff:** La opción más sencilla; simplemente toma los documentos que considera apropiados y los utiliza en el prompt que se pasa al modelo.
# - **refine:** Realiza múltiples llamadas al modelo con diferentes documentos, intentando obtener una respuesta más refinada cada vez. Puede ejecutar un número elevado de llamadas al modelo, por lo que debe usarse con precaución.
# - **map_reduce:** Intenta reducir todos los documentos en uno solo, posiblemente a través de varias iteraciones. Puede comprimir y condensar los documentos para que quepan en el prompt enviado al modelo.
# - **map_rerank:** Llama al modelo para cada documento y los clasifica, devolviendo finalmente el mejor. Similar a refine, puede ser arriesgado dependiendo del número de llamadas que se prevea realizar.
# 

# %% [markdown]
# Ahora, podemos hacer la pregunta:

# %%
response = document_qa.invoke("Can I buy a Toshiba laptop?")

display(response)

# %% [markdown]
# La respuesta es correcta. No se obtiene mucha información porque el modelo usado, T5, no está específicamente preparado para la generación de texto.

# %%
response = document_qa.invoke("Can I buy a Acer 3 laptop?")

display(response)

# %% [markdown]
# ### Agents
# 
# LangChain permite crear agentes que pueden tomar decisiones, consultar diferentes fuentes de datos y ejecutar acciones de forma autónoma. Los agentes pueden utilizar herramientas y modelos de lenguaje para interactuar con el entorno y resolver tareas complejas.

# %% [markdown]
# Los agentes pueden usar tools predefinidas.

# %%
from langchain.agents import load_tools
from langchain.agents import AgentType

tool_names = ["llm-math"]
tools = load_tools(tool_names, llm=llm)
tools

# %%
from langchain.agents import initialize_agent

agent = initialize_agent(tools,
                         llm,
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                         verbose=True,
                         max_iterations=3)

# %%
agent.invoke("¿Qué día es hoy?")

# %%
agent.invoke("¿Cuánto es 2 elevado a la potencia de 10?")

# %% [markdown]
# También pueden usar tools personalizadas.

# %%
from langchain_core.tools import tool
from langchain.agents import initialize_agent, AgentType

# 1. Define tu herramienta personalizada usando @tool
@tool
def invertir_texto(texto: str) -> str:
    """Invierte el texto proporcionado."""
    return texto[::-1]


# 3. Inicializa el agente con la herramienta custom
agent = initialize_agent(
    tools=[invertir_texto],  # Lista de herramientas; aquí solo la custom
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 4. Usa el agente
resultado = agent.run("Invierte el texto: LangChain es genial")
print(resultado)

# %% [markdown]
# El agente puede combinarse con un chat.

# %%
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent_chain = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

agent_chain.run("Invierte esta cadena: LangChain es genial")


