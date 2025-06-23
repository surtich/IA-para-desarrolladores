# %% [markdown]
# ### ¿Qué es un Agente LLM?
# 
# Un **Agente LLM** es un sistema avanzado de inteligencia artificial que utiliza LLM (como GPT-4, Llama, Claude, etc.) para comprender, generar y actuar sobre lenguaje humano de manera autónoma y contextual.
# 
# A diferencia de un simple `chatbot`, un agente LLM no solo responde preguntas, sino que puede planificar, razonar, recordar información previa, interactuar con herramientas externas (APIs, bases de datos, buscadores, etc.) y ejecutar tareas complejas de varios pasos, todo ello guiado por el lenguaje natural.
# 
# Una definición simple de un Agente:
# 
# > Un agente es un programa en el que el LLM controla el flujo de trabajo.
# 
# ### Características principales de un Agente LLM
# 
# - **Comprensión profunda del lenguaje:** Interpreta intenciones, matices y contexto de las conversaciones humanas.
# - **Generación de texto:** Produce respuestas coherentes y naturales, desde simples aclaraciones hasta textos complejos o resúmenes.
# - **Memoria y contexto:** Puede recordar interacciones previas y mantener coherencia a lo largo de una conversación o flujo de trabajo.
# - **Planificación y razonamiento:** Descompone problemas complejos en pasos más pequeños y decide qué acciones tomar y en qué orden.
# - **Uso de herramientas externas:** Integra APIs, bases de datos, sistemas de búsqueda y otras utilidades para ampliar sus capacidades más allá del texto.
# - **Autonomía:** Es capaz de ejecutar flujos de trabajo completos, tomar decisiones y adaptarse a los cambios en el entorno o en las instrucciones del usuario.
# 
# 

# %% [markdown]
# ### Agentes de Dataframes Pandas
# 
# En esta práctica vamos a usar un agente LLM para trabajar con dataframes de Pandas. El agente LLM va a poder ejecutar código Python para manipular dataframes, y va a poder usar herramientas como `pandas`, `numpy`, `matplotlib`, etc. para realizar tareas complejas de análisis de datos.

# %% [markdown]
# Leemos las variables de entorno:

# %%
from dotenv import load_dotenv
load_dotenv(override=True)

# %%
import getpass
import os

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Introduce la APY Key de Gemini: ")
if "KAGGLE_USERNAME" not in os.environ:
    os.environ["KAGGLE_USERNAME"] = getpass.getpass("Introduce el usuario de Kaggle: ")
if "KAGGLE_KEY" not in os.environ:
    os.environ["KAGGLE_KEY"] = getpass.getpass("Introduce la API Key de Kaggle: ")


# %% [markdown]
# Leemos un `dataset` de Kaggle:

# %%
!kaggle datasets download -d goyaladi/climate-insights-dataset -p tmp  

# %%
import zipfile

# Define the path to your zip file
file_path = 'tmp/climate-insights-dataset.zip'
with zipfile.ZipFile(file_path, 'r') as zip_ref:
    zip_ref.extractall('../datasets/climate_change')


# %% [markdown]
# Vamos a usar [este](https://www.kaggle.com/datasets/goyaladi/climate-insights-dataset) `dataset`:

# %%
import pandas as pd
csv_file='../datasets/climate_change/climate_change_data.csv'
document = pd.read_csv(csv_file)
document.head(5)

# %% [markdown]
# Creamos el agente:

# %%
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

sm_ds_Chat = create_pandas_dataframe_agent(
    ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=1),
    document,
    verbose=True,
    allow_dangerous_code=True, # LangChain obliga a poner esto. Ver: please see: https://python.langchain.com/docs/security/
)

# %% [markdown]
# Probamos el agente:

# %%
sm_ds_Chat.invoke("Analiza estos datos y haz un resumen de 100 palabras en español")

# %% [markdown]
# Podemos hacer gráficos. Observe que el agente falla y descubre que la columna que recoge la temperatura tiene un espacio al final ("Temperature "). Lo tiene en cuenta y vuelve a ejecutar el gráfico:

# %%
sm_ds_Chat.invoke("Dibuja la relación entre la temperatura y la humedad para todo el dataset")

# %% [markdown]
# Los campos que puede usar un agente son:
# Estudiando el rastro (trace), se pueden observar tres campos principales:
# 
# - **Thought (Pensamiento):** Muestra el razonamiento interno del agente, indicando lo que planea hacer y cuál es su objetivo inmediato. Es como su diálogo interno, donde analiza la situación y decide el siguiente paso a seguir.
# 
# - **Action (Acción):** Refleja las acciones que realiza el agente, normalmente llamando a funciones de Python o herramientas externas a las que tiene acceso, como búsquedas en internet, consultas a bases de datos, cálculos, etc
# 
# - **Observation (Observación):** Es la información o los datos que obtiene como resultado de la acción realizada. El agente utiliza esta observación para ajustar su razonamiento y decidir su próximo objetivo o acción, repitiendo el ciclo hasta alcanzar la meta final.
#   
# Este ciclo de Pensamiento → Acción → Observación permite a los agentes LLM resolver tareas complejas de manera iterativa, integrando razonamiento, ejecución y aprendizaje en tiempo real.
# 

# %% [markdown]
# Otra pregunta:

# %%
sm_ds_Chat.invoke("¿Crees que se puede predecir la temperatura?, ¿Cómo?")

# %% [markdown]
# Otra más:

# %%
sm_ds_Chat.invoke("""Selecciona un modelo estadístico de predicción para pronosticar la temperatura.
Utiliza este tipo de modelo para predecir la temperatura media
anual en South David, Vietnam, para los próximos 5 años.
Escribe las temperaturas pronosticadas en una tabla.""")



