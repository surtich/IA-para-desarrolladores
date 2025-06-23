# %% [markdown]
# ### LangGraph
# 
# LangGraph es una biblioteca del ecosistema LangChain diseñada para construir flujos de trabajo complejos y colaborativos entre múltiples agentes de IA. Utiliza una arquitectura basada en grafos, donde cada nodo puede ser un agente o una función, y las conexiones (aristas) definen el flujo y la comunicación entre ellos.
# 
# Esto permite crear sistemas donde varios agentes pueden interactuar, compartir información y tomar decisiones de manera coordinada, con soporte para memoria y control de estado. LangGraph es útil para aplicaciones como asistentes inteligentes, agentes autónomos, sistemas de decisión y automatización avanzada, facilitando la orquestación y escalabilidad de soluciones basadas en IA.

# %% [markdown]
# ### Elementos de LangGraph
# 
# * **Graph**: En LangGraph, un "Graph" (grafo) es la estructura principal que modela el flujo de trabajo de agentes como un conjunto de nodos (Nodes) conectados por aristas (Edges). Permite definir rutas, ciclos y la lógica de transición entre agentes o funciones, facilitando la coordinación y ejecución de sistemas complejos multiagente.
# 
# * **State**: El "State" (estado) es una estructura de datos compartida que representa una instantánea actual de la aplicación. Contiene toda la información relevante que se va actualizando conforme los nodos procesan datos y toman decisiones. Puede ser un diccionario de Python, un TypedDict o un modelo Pydantic, y es fundamental para mantener el contexto a lo largo de la ejecución del grafo.   El estado en LangGraph es inmutable. Asociado a cada campo del estado, se puede definir un `reducer`. LangGraph usa el `reducer` de cada campo para combinarlo con el estado actual.
# 
# * **Node**: Un "Node" (nodo) en LangGraph es típicamente una función de Python que implementa la lógica de un agente o un paso del flujo de trabajo. Recibe el estado actual como entrada, realiza un procesamiento o acción, y devuelve un estado actualizado.
# 
# * **Edge**: Una "Edge" (arista) es una función o conexión que determina qué nodo se ejecuta a continuación, en función del estado actual. Las aristas pueden ser transiciones fijas o condicionales, y permiten definir flujos de trabajo complejos, incluyendo bifurcaciones y ciclos. Son responsables de guiar el paso de información y la secuencia de ejecución entre nodos.
# 

# %% [markdown]
# ### Pasos para definir un grafo
# 
#  1. **Definir el estado**: Crear una clase que represente el estado compartido de la aplicación, utilizando un diccionario, TypedDict o Pydantic.
#  2. Empezar el `graph buiilder` para crear un nuevo grafo.
#  3. Crear un nodo.
#  4. Crear Edges.
#  5. Compilar el grafo

# %% [markdown]
# ### Ejemplo de uso
# 
# 

# %%
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from IPython.display import Image, display
import gradio as gr
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
import random


# %%
# Algunas constantes útiles

nouns = ["Coles", "Unicornios", "Tostadoras", "Pingüinos", "Plátanos", "Zombis", "Arcoíris", "Anguilas", "Pepinillos", "Muffins"]
adjectives = ["escandaloso", "maloliente", "pedante", "existencial", "malhumorado", "brillante", "poco fiable", "sarcástico", "blandito", "embrujado"]


# %%
# Our favorite first step! Crew was doing this for us, by the way.
load_dotenv(override=True)


# %%
def shout(text: Annotated[str, "something to be shouted"]) -> str:
    print(text.upper())
    return text.upper()

shout("hello")

# %% [markdown]
# ### Hola Mundo LangGraph
# 
# Vamos a hacer un ejemplo sencillo para comprender todos los pasos necesarios para crear un grafo en LangGraph. Este ejemplo demuestra que LangGraph sirve para construir flujos de trabajo que no requieren de un LLM. 

# %% [markdown]
# **Paso 1: Definir el estado**
# 
# El estado, en este caso, va a ser una clase de tipo `Pydantic` que contendrá un campo `messages`. El tipo de `messages` será una lista. Una forma alternativa a definir `type hint`, es usar campos "anotados". Los tipos anotados normalmente incluyen un comentario y son ignorados por Python, pero pueden ser usados en otro contexto. En este caso, LangGraph permite que definir el `reducer` de tipo. En este caso es una función ya proporcionada por LangGraph.
# 

# %%
class State(BaseModel):
        
    messages: Annotated[list, add_messages]


# %% [markdown]
# **Paso 2: Iniciar el `graph builder`**

# %%
graph_builder = StateGraph(State)

# %% [markdown]
# **Paso 3: Crear un nodo**

# %%
def our_first_node(old_state: State) -> State:

    reply = f"{random.choice(nouns)} {random.choice(adjectives)}"
    messages = [{"role": "assistant", "content": reply}]

    new_state = State(messages=messages)

    return new_state

graph_builder.add_node("first_node", our_first_node)

# %% [markdown]
# **Crear Edges**

# %%
graph_builder.add_edge(START, "first_node")
graph_builder.add_edge("first_node", END)

# %% [markdown]
# **Compilar el grafo**

# %%
graph = graph_builder.compile()

# %%
display(Image(graph.get_graph().draw_mermaid_png()))

# %% [markdown]
# **Probamos**

# %%
def chat(user_input: str, history):
    message = {"role": "user", "content": user_input}
    messages = [message]
    state = State(messages=messages)
    result = graph.invoke(state)
    print(result)
    return result["messages"][-1].content



# %%
gr.ChatInterface(chat, type="messages").launch()

# %% [markdown]
# ### Uso de LLMs en 
# 
# En este ejercicio incorporamos un LLM para que el grafo pueda interactuar con un modelo de lenguaje.

# %%
# Step 1: Define the State object
class State(BaseModel):
    messages: Annotated[list, add_messages]


# %%
# Step 2: Start the Graph Builder with this State class
graph_builder = StateGraph(State)

# %%
# Step 3: Create a Node

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def chatbot_node(old_state: State) -> State:
    response = llm.invoke(old_state.messages)
    new_state = State(messages=[response])
    return new_state

graph_builder.add_node("chatbot", chatbot_node)

# %%
# Step 4: Create Edges
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# %%
# Step 5: Compile the Graph
graph = graph_builder.compile()
display(Image(graph.get_graph().draw_mermaid_png()))

# %%
def chat(user_input: str, history):
    initial_state = State(messages=history + [{"role": "user", "content": user_input}])
    result = graph.invoke(initial_state)
    print(result)
    return result['messages'][-1].content


gr.ChatInterface(chat, type="messages").launch()

# %% [markdown]
# ### Tools y LangSmith
# En este ejercicio vamos a aprender a usar `tools`. Además, configuraremos la herramienta de `tracing` de LangSmith. Para poder depurar y analizar el grafo. LangSmith es una herramienta del ecosistema LangChain que permite registrar, depurar y analizar flujos de trabajo de IA. Para usarla hay que registrarse y obtener una clave de API, habilitando "setup tracing". Después de ponerlas en el fichero `.env`, hay que cargarlas en memoria.

# %%
load_dotenv(override=True)

# %% [markdown]
# Vamos la tool que permite hacer búsquedas con Google.

# %%
from langchain_community.utilities import GoogleSerperAPIWrapper

serper = GoogleSerperAPIWrapper()
serper.run("¿Cuál es la capital de España?")

# %% [markdown]
# Utilizamos esta función de LangChain que permite convertir funciones en `tools`

# %%
from langchain.agents import Tool

tool_search = Tool(
        name="search",
        func=serper.run,
        description="Útil cuando necesitas más información de una búsqueda en línea"
    )


# %% [markdown]
# Se puede probar que la `tool` funciona correctamente.

# %%
tool_search.run("¿Cuál es la capital de España?")

# %% [markdown]
# Hacemos lo mismo con la función `send_email`:

# %%
import os
from typing import Dict
import requests 

def send_email(subject: str, html_body: str, to: str, name: str = None) -> Dict[str, str]:
    """Enviar un correo electrónico"""
    from_email = os.getenv('MAILGUN_FROM')
    to_email = f"{name} <{to}>" if name else to
    content = html_body

    requests.post(
  		f"https://api.mailgun.net/v3/{os.getenv('MAILGUN_SANDBOX')}/messages",
  		auth=("api", os.getenv('MAILGUN_API_KEY')),
  		data={"from": from_email,
			"to": to_email,
  			"subject": subject,
  			"html": content})

    return {"status": "éxito"}



# %% [markdown]
# Cuando la función reciba más de un parámetro, se puede usar `StructuredTool`.

# %%
from langchain_community.tools import StructuredTool

tool_send_email = StructuredTool.from_function(send_email, description="Útil para enviar correos electrónicos", name="send_email")

tool_send_email.invoke({
    "subject": "Hola desde LangGraph",
    "html_body": "<h1>Hola, mundo!</h1><p>Este es un correo electrónico enviado desde LangGraph.</p>",
    "to": "surtich@gmail.com"
})

# %% [markdown]
# Ponemos las `tools` en una lista.

# %%
tools = [tool_search, tool_send_email]

# %% [markdown]
# Definimos el grafo. Empezamos definiendo el estado. Esta vez hemos usado `TypedDict` para definir el estado. Se podría haber usado `Pydantic`. Es simplemente otra forma de definir el estado.

# %%
# Step 1: Define the State object
from typing import TypedDict


class State(TypedDict):
    messages: Annotated[list, add_messages]

# %%
# Step 2: Start the Graph Builder with this State class
graph_builder = StateGraph(State)

# %% [markdown]
# Ahora creamos los nodos. Como tenemos `tools` hay que asociarlas tanto al modelo como a un nodo especial. Recordemos que cuando creamos una `tool` de forma nativa, se define la  `tool` en un fichero JSON y que luego hay que saber si la razón de finalización del modelo es que se ha invocado una `tool`. Estos dos pasos tienen aquí su equivalente en LangGraph.

# %%
# This is different:

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
llm_with_tools = llm.bind_tools(tools)

# %%
# Step 3: Create a Node
from langgraph.prebuilt import ToolNode, tools_condition

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))

# %% [markdown]
# Creamos los `edges`. En este caso unimos el nodo `chatbot` con el nodo `tools` a través de una condición que verifica si el modelo ha respondido invocando un `tool` (`finish_reason` == `tool_calls`). Tenemos que volver al chatbot después de usar una herramienta, para que el chatbot pueda decidir qué hacer a continuación. Por ejemplo volver a llamar a esa o a otra tool.
# 

# %%
# Step 4: Create Edges


graph_builder.add_conditional_edges( "chatbot", tools_condition, "tools")

# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot") 
graph_builder.add_edge(START, "chatbot")

# %% [markdown]
# Las líneas discontinuas del grafo son condicionales.

# %%
# Step 5: Compile the Graph
graph = graph_builder.compile()
display(Image(graph.get_graph().draw_mermaid_png()))

# %% [markdown]
# Probamos

# %%
def chat(user_input: str, history):
    result = graph.invoke({"messages": [{"role": "user", "content": user_input}]})
    return result["messages"][-1].content


gr.ChatInterface(chat, type="messages").launch()

# %% [markdown]
# ### Memoria
# 
# En el ejercicio anterior no hemos implementado la memoria del grafo. Por ejemplo, si en la interacción le da al LLM su nombre y luego se lo pregunta, el LLM no lo recordará. Se podría haber hecho invocando simplemente así:
# 
# ```python
# graph.invoke({"messages": history + [{"role": "user", "content": user_input}]})
# ```
# 
# Pero, haciéndolo así, no estaríamos aprovechando los estados de LangChain.

# %% [markdown]
# La memoria en LangGraph se implementa mediante `checkpointing`. En este caso, vamos a crear una memoria en memoria (valga la redundancia).

# %%
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

# %% [markdown]
# Asociamos la memoria al construir el grafo

# %%
graph = graph_builder.compile(checkpointer=memory)

# %% [markdown]
# Creamos un objeto de configuración para asociar la memoria a una conversación (nos inventamos un id)

# %%
config = {"configurable": {"thread_id": "1"}}


# %% [markdown]
# Probamos pasando el objeto config

# %%
def chat(user_input: str, history):
    result = graph.invoke({"messages": [{"role": "user", "content": user_input}]}, config=config)
    return result["messages"][-1].content


gr.ChatInterface(chat, type="messages").launch()

# %% [markdown]
# Podemos ver el estado de la conversación

# %%
graph.get_state(config)

# %% [markdown]
# También lo podemos ver paso a paso. Cada vez que se completa el grafo se almacena un `snapshot`.

# %%
# Most recent first

list(graph.get_state_history(config))

# %% [markdown]
# LangGraph permite retrotraer la conversación al momento deseado:
# 
# ```python
# config = {"configurable": {"thread_id": "1", "checkpoint_id": ...}}
# graph.invoke(None, config=config)
# ```
# 


