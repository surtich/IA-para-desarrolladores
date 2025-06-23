### LangGraph

LangGraph es una biblioteca del ecosistema LangChain diseñada para construir flujos de trabajo complejos y colaborativos entre múltiples agentes de IA. Utiliza una arquitectura basada en grafos, donde cada nodo puede ser un agente o una función, y las conexiones (aristas) definen el flujo y la comunicación entre ellos.

Esto permite crear sistemas donde varios agentes pueden interactuar, compartir información y tomar decisiones de manera coordinada, con soporte para memoria y control de estado. LangGraph es útil para aplicaciones como asistentes inteligentes, agentes autónomos, sistemas de decisión y automatización avanzada, facilitando la orquestación y escalabilidad de soluciones basadas en IA.

### Elementos de LangGraph

* **Graph**: En LangGraph, un "Graph" (grafo) es la estructura principal que modela el flujo de trabajo de agentes como un conjunto de nodos (Nodes) conectados por aristas (Edges). Permite definir rutas, ciclos y la lógica de transición entre agentes o funciones, facilitando la coordinación y ejecución de sistemas complejos multiagente.

* **State**: El "State" (estado) es una estructura de datos compartida que representa una instantánea actual de la aplicación. Contiene toda la información relevante que se va actualizando conforme los nodos procesan datos y toman decisiones. Puede ser un diccionario de Python, un TypedDict o un modelo Pydantic, y es fundamental para mantener el contexto a lo largo de la ejecución del grafo.   El estado en LangGraph es inmutable. Asociado a cada campo del estado, se puede definir un `reducer`. LangGraph usa el `reducer` de cada campo para combinarlo con el estado actual.

* **Node**: Un "Node" (nodo) en LangGraph es típicamente una función de Python que implementa la lógica de un agente o un paso del flujo de trabajo. Recibe el estado actual como entrada, realiza un procesamiento o acción, y devuelve un estado actualizado.

* **Edge**: Una "Edge" (arista) es una función o conexión que determina qué nodo se ejecuta a continuación, en función del estado actual. Las aristas pueden ser transiciones fijas o condicionales, y permiten definir flujos de trabajo complejos, incluyendo bifurcaciones y ciclos. Son responsables de guiar el paso de información y la secuencia de ejecución entre nodos.


### Pasos para definir un grafo

 1. **Definir el estado**: Crear una clase que represente el estado compartido de la aplicación, utilizando un diccionario, TypedDict o Pydantic.
 2. Empezar el `graph buiilder` para crear un nuevo grafo.
 3. Crear un nodo.
 4. Crear Edges.
 5. Compilar el grafo

### Ejemplo de uso




```python
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

```

    /home/surtich/projects/IA para desarrolladores/.venv/lib/python3.13/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm



```python
# Algunas constantes útiles

nouns = ["Coles", "Unicornios", "Tostadoras", "Pingüinos", "Plátanos", "Zombis", "Arcoíris", "Anguilas", "Pepinillos", "Muffins"]
adjectives = ["escandaloso", "maloliente", "pedante", "existencial", "malhumorado", "brillante", "poco fiable", "sarcástico", "blandito", "embrujado"]

```


```python
# Our favorite first step! Crew was doing this for us, by the way.
load_dotenv(override=True)

```




    True




```python
def shout(text: Annotated[str, "something to be shouted"]) -> str:
    print(text.upper())
    return text.upper()

shout("hello")
```

    HELLO





    'HELLO'



### Hola Mundo LangGraph

Vamos a hacer un ejemplo sencillo para comprender todos los pasos necesarios para crear un grafo en LangGraph. Este ejemplo demuestra que LangGraph sirve para construir flujos de trabajo que no requieren de un LLM. 

**Paso 1: Definir el estado**

El estado, en este caso, va a ser una clase de tipo `Pydantic` que contendrá un campo `messages`. El tipo de `messages` será una lista. Una forma alternativa a definir `type hint`, es usar campos "anotados". Los tipos anotados normalmente incluyen un comentario y son ignorados por Python, pero pueden ser usados en otro contexto. En este caso, LangGraph permite que definir el `reducer` de tipo. En este caso es una función ya proporcionada por LangGraph.



```python
class State(BaseModel):
        
    messages: Annotated[list, add_messages]

```

**Paso 2: Iniciar el `graph builder`**


```python
graph_builder = StateGraph(State)
```

**Paso 3: Crear un nodo**


```python
def our_first_node(old_state: State) -> State:

    reply = f"{random.choice(nouns)} {random.choice(adjectives)}"
    messages = [{"role": "assistant", "content": reply}]

    new_state = State(messages=messages)

    return new_state

graph_builder.add_node("first_node", our_first_node)
```




    <langgraph.graph.state.StateGraph at 0x7fac0ded56a0>



**Crear Edges**


```python
graph_builder.add_edge(START, "first_node")
graph_builder.add_edge("first_node", END)
```




    <langgraph.graph.state.StateGraph at 0x7fac0ded56a0>



**Compilar el grafo**


```python
graph = graph_builder.compile()
```


```python
display(Image(graph.get_graph().draw_mermaid_png()))
```


    
![png](22_LangGraph_files/22_LangGraph_19_0.png)
    


**Probamos**


```python
def chat(user_input: str, history):
    message = {"role": "user", "content": user_input}
    messages = [message]
    state = State(messages=messages)
    result = graph.invoke(state)
    print(result)
    return result["messages"][-1].content


```


```python
gr.ChatInterface(chat, type="messages").launch()
```

    * Running on local URL:  http://127.0.0.1:7863
    * To create a public link, set `share=True` in `launch()`.



<div><iframe src="http://127.0.0.1:7863/" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>





    



    {'messages': [HumanMessage(content='Hola', additional_kwargs={}, response_metadata={}, id='dba0e520-b9f3-4658-9a0b-d0c0874cb1bf'), AIMessage(content='Muffins blandito', additional_kwargs={}, response_metadata={}, id='773d49e2-c49f-4a1c-935a-75e73bf6240b')]}


### Uso de LLMs en 

En este ejercicio incorporamos un LLM para que el grafo pueda interactuar con un modelo de lenguaje.


```python
# Step 1: Define the State object
class State(BaseModel):
    messages: Annotated[list, add_messages]

```


```python
# Step 2: Start the Graph Builder with this State class
graph_builder = StateGraph(State)
```


```python
# Step 3: Create a Node

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def chatbot_node(old_state: State) -> State:
    response = llm.invoke(old_state.messages)
    new_state = State(messages=[response])
    return new_state

graph_builder.add_node("chatbot", chatbot_node)
```




    <langgraph.graph.state.StateGraph at 0x7fd58d1c6e40>




```python
# Step 4: Create Edges
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
```




    <langgraph.graph.state.StateGraph at 0x7fd58d1c6e40>




```python
# Step 5: Compile the Graph
graph = graph_builder.compile()
display(Image(graph.get_graph().draw_mermaid_png()))
```


    
![png](22_LangGraph_files/22_LangGraph_28_0.png)
    



```python
def chat(user_input: str, history):
    initial_state = State(messages=history + [{"role": "user", "content": user_input}])
    result = graph.invoke(initial_state)
    print(result)
    return result['messages'][-1].content


gr.ChatInterface(chat, type="messages").launch()
```

    * Running on local URL:  http://127.0.0.1:7860
    * To create a public link, set `share=True` in `launch()`.



<div><iframe src="http://127.0.0.1:7860/" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>





    



    {'messages': [HumanMessage(content='Hola', additional_kwargs={}, response_metadata={}, id='2a5d9460-8166-49c1-9728-cf3f8ace4259'), AIMessage(content='¡Hola! ¿En qué puedo ayudarte hoy? 😊', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'safety_ratings': []}, id='run--4742ca91-0e6f-444c-8a1c-b194eecc8f40-0', usage_metadata={'input_tokens': 1, 'output_tokens': 12, 'total_tokens': 13, 'input_token_details': {'cache_read': 0}})]}


### Tools y LangSmith
En este ejercicio vamos a aprender a usar `tools`. Además, configuraremos la herramienta de `tracing` de LangSmith. Para poder depurar y analizar el grafo. LangSmith es una herramienta del ecosistema LangChain que permite registrar, depurar y analizar flujos de trabajo de IA. Para usarla hay que registrarse y obtener una clave de API, habilitando "setup tracing". Después de ponerlas en el fichero `.env`, hay que cargarlas en memoria.


```python
load_dotenv(override=True)
```




    True



Vamos la tool que permite hacer búsquedas con Google.


```python
from langchain_community.utilities import GoogleSerperAPIWrapper

serper = GoogleSerperAPIWrapper()
serper.run("¿Cuál es la capital de España?")
```




    'Madrid'



Utilizamos esta función de LangChain que permite convertir funciones en `tools`


```python
from langchain.agents import Tool

tool_search = Tool(
        name="search",
        func=serper.run,
        description="Útil cuando necesitas más información de una búsqueda en línea"
    )

```

Se puede probar que la `tool` funciona correctamente.


```python
tool_search.run("¿Cuál es la capital de España?")
```




    'Madrid'



Hacemos lo mismo con la función `send_email`:


```python
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


```

Cuando la función reciba más de un parámetro, se puede usar `StructuredTool`.


```python
from langchain_community.tools import StructuredTool

tool_send_email = StructuredTool.from_function(send_email, description="Útil para enviar correos electrónicos", name="send_email")

tool_send_email.invoke({
    "subject": "Hola desde LangGraph",
    "html_body": "<h1>Hola, mundo!</h1><p>Este es un correo electrónico enviado desde LangGraph.</p>",
    "to": "surtich@gmail.com"
})
```




    {'status': 'éxito'}



Ponemos las `tools` en una lista.


```python
tools = [tool_search, tool_send_email]
```

Definimos el grafo. Empezamos definiendo el estado. Esta vez hemos usado `TypedDict` para definir el estado. Se podría haber usado `Pydantic`. Es simplemente otra forma de definir el estado.


```python
# Step 1: Define the State object
from typing import TypedDict


class State(TypedDict):
    messages: Annotated[list, add_messages]
```


```python
# Step 2: Start the Graph Builder with this State class
graph_builder = StateGraph(State)
```

Ahora creamos los nodos. Como tenemos `tools` hay que asociarlas tanto al modelo como a un nodo especial. Recordemos que cuando creamos una `tool` de forma nativa, se define la  `tool` en un fichero JSON y que luego hay que saber si la razón de finalización del modelo es que se ha invocado una `tool`. Estos dos pasos tienen aquí su equivalente en LangGraph.


```python
# This is different:

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
llm_with_tools = llm.bind_tools(tools)
```


```python
# Step 3: Create a Node
from langgraph.prebuilt import ToolNode, tools_condition

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))
```




    <langgraph.graph.state.StateGraph at 0x7fd55062efd0>



Creamos los `edges`. En este caso unimos el nodo `chatbot` con el nodo `tools` a través de una condición que verifica si el modelo ha respondido invocando un `tool` (`finish_reason` == `tool_calls`). Tenemos que volver al chatbot después de usar una herramienta, para que el chatbot pueda decidir qué hacer a continuación. Por ejemplo volver a llamar a esa o a otra tool.



```python
# Step 4: Create Edges


graph_builder.add_conditional_edges( "chatbot", tools_condition, "tools")

# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot") 
graph_builder.add_edge(START, "chatbot")
```




    <langgraph.graph.state.StateGraph at 0x7fd55062efd0>



Las líneas discontinuas del grafo son condicionales.


```python
# Step 5: Compile the Graph
graph = graph_builder.compile()
display(Image(graph.get_graph().draw_mermaid_png()))
```


    
![png](22_LangGraph_files/22_LangGraph_53_0.png)
    


Probamos


```python
def chat(user_input: str, history):
    result = graph.invoke({"messages": [{"role": "user", "content": user_input}]})
    return result["messages"][-1].content


gr.ChatInterface(chat, type="messages").launch()
```

    * Running on local URL:  http://127.0.0.1:7861
    * To create a public link, set `share=True` in `launch()`.



<div><iframe src="http://127.0.0.1:7861/" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>





    



### Memoria

En el ejercicio anterior no hemos implementado la memoria del grafo. Por ejemplo, si en la interacción le da al LLM su nombre y luego se lo pregunta, el LLM no lo recordará. Se podría haber hecho invocando simplemente así:

```python
graph.invoke({"messages": history + [{"role": "user", "content": user_input}]})
```

Pero, haciéndolo así, no estaríamos aprovechando los estados de LangChain.

La memoria en LangGraph se implementa mediante `checkpointing`. En este caso, vamos a crear una memoria en memoria (valga la redundancia).


```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
```

Asociamos la memoria al construir el grafo


```python
graph = graph_builder.compile(checkpointer=memory)
```

Creamos un objeto de configuración para asociar la memoria a una conversación (nos inventamos un id)


```python
config = {"configurable": {"thread_id": "1"}}

```

Probamos pasando el objeto config


```python
def chat(user_input: str, history):
    result = graph.invoke({"messages": [{"role": "user", "content": user_input}]}, config=config)
    return result["messages"][-1].content


gr.ChatInterface(chat, type="messages").launch()
```

    * Running on local URL:  http://127.0.0.1:7862
    * To create a public link, set `share=True` in `launch()`.



<div><iframe src="http://127.0.0.1:7862/" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>





    



Podemos ver el estado de la conversación


```python
graph.get_state(config)
```




    StateSnapshot(values={'messages': [HumanMessage(content='Hola me llamo Pepe', additional_kwargs={}, response_metadata={}, id='07a9e750-3bf6-4120-97c6-391b10bbd9d2'), AIMessage(content='Hola Pepe, ¿cómo puedo ayudarte hoy?', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--21af6c8b-7ba9-4354-ad95-fa0b46432d10-0', usage_metadata={'input_tokens': 49, 'output_tokens': 10, 'total_tokens': 59, 'input_token_details': {'cache_read': 0}}), HumanMessage(content='Dime cómo me llamo', additional_kwargs={}, response_metadata={}, id='44b1ecfd-fea9-4023-8bbc-17f1b1d78dee'), AIMessage(content='Te llamas Pepe.', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--aee5f131-7d89-4f9a-ac02-cda88709f315-0', usage_metadata={'input_tokens': 62, 'output_tokens': 5, 'total_tokens': 67, 'input_token_details': {'cache_read': 0}})]}, next=(), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f03cb68-6486-6840-8004-a3dbcc7d4887'}}, metadata={'source': 'loop', 'writes': {'chatbot': {'messages': [AIMessage(content='Te llamas Pepe.', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--aee5f131-7d89-4f9a-ac02-cda88709f315-0', usage_metadata={'input_tokens': 62, 'output_tokens': 5, 'total_tokens': 67, 'input_token_details': {'cache_read': 0}})]}}, 'step': 4, 'parents': {}, 'thread_id': '1'}, created_at='2025-05-29T17:58:28.309176+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f03cb68-5f8f-680b-8003-42b83fafe2de'}}, tasks=(), interrupts=())



También lo podemos ver paso a paso. Cada vez que se completa el grafo se almacena un `snapshot`.


```python
# Most recent first

list(graph.get_state_history(config))
```




    [StateSnapshot(values={'messages': [HumanMessage(content='Hola me llamo Pepe', additional_kwargs={}, response_metadata={}, id='07a9e750-3bf6-4120-97c6-391b10bbd9d2'), AIMessage(content='Hola Pepe, ¿cómo puedo ayudarte hoy?', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--21af6c8b-7ba9-4354-ad95-fa0b46432d10-0', usage_metadata={'input_tokens': 49, 'output_tokens': 10, 'total_tokens': 59, 'input_token_details': {'cache_read': 0}}), HumanMessage(content='Dime cómo me llamo', additional_kwargs={}, response_metadata={}, id='44b1ecfd-fea9-4023-8bbc-17f1b1d78dee'), AIMessage(content='Te llamas Pepe.', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--aee5f131-7d89-4f9a-ac02-cda88709f315-0', usage_metadata={'input_tokens': 62, 'output_tokens': 5, 'total_tokens': 67, 'input_token_details': {'cache_read': 0}}), HumanMessage(content='Escribe una transcripción de nuestra conversación', additional_kwargs={}, response_metadata={}, id='0c27c514-069e-44aa-93a1-a1c858987ce0'), AIMessage(content='Aquí está la transcripción de nuestra conversación:\n\nUsuario: Hola me llamo Pepe\nModelo: Hola Pepe, ¿cómo puedo ayudarte hoy?\nUsuario: Dime cómo me llamo\nModelo: Te llamas Pepe.\nUsuario: Escribe una transcripción de nuestra conversación\nModelo: Aquí está la transcripción de nuestra conversación:\n\nUsuario: Hola me llamo Pepe\nModelo: Hola Pepe, ¿cómo puedo ayudarte hoy?\nUsuario: Dime cómo me llamo\nModelo: Te llamas Pepe.\nUsuario: Escribe una transcripción de nuestra conversación', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--5c4d427d-d0fb-40f6-8657-dafa843e0190-0', usage_metadata={'input_tokens': 74, 'output_tokens': 107, 'total_tokens': 181, 'input_token_details': {'cache_read': 0}})]}, next=(), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f03cb6a-201e-61c6-8007-fdc49f0c9766'}}, metadata={'source': 'loop', 'writes': {'chatbot': {'messages': [AIMessage(content='Aquí está la transcripción de nuestra conversación:\n\nUsuario: Hola me llamo Pepe\nModelo: Hola Pepe, ¿cómo puedo ayudarte hoy?\nUsuario: Dime cómo me llamo\nModelo: Te llamas Pepe.\nUsuario: Escribe una transcripción de nuestra conversación\nModelo: Aquí está la transcripción de nuestra conversación:\n\nUsuario: Hola me llamo Pepe\nModelo: Hola Pepe, ¿cómo puedo ayudarte hoy?\nUsuario: Dime cómo me llamo\nModelo: Te llamas Pepe.\nUsuario: Escribe una transcripción de nuestra conversación', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--5c4d427d-d0fb-40f6-8657-dafa843e0190-0', usage_metadata={'input_tokens': 74, 'output_tokens': 107, 'total_tokens': 181, 'input_token_details': {'cache_read': 0}})]}}, 'step': 7, 'parents': {}, 'thread_id': '1'}, created_at='2025-05-29T17:59:14.823202+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f03cb6a-1653-625a-8006-bcea6db81b10'}}, tasks=(), interrupts=()),
     StateSnapshot(values={'messages': [HumanMessage(content='Hola me llamo Pepe', additional_kwargs={}, response_metadata={}, id='07a9e750-3bf6-4120-97c6-391b10bbd9d2'), AIMessage(content='Hola Pepe, ¿cómo puedo ayudarte hoy?', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--21af6c8b-7ba9-4354-ad95-fa0b46432d10-0', usage_metadata={'input_tokens': 49, 'output_tokens': 10, 'total_tokens': 59, 'input_token_details': {'cache_read': 0}}), HumanMessage(content='Dime cómo me llamo', additional_kwargs={}, response_metadata={}, id='44b1ecfd-fea9-4023-8bbc-17f1b1d78dee'), AIMessage(content='Te llamas Pepe.', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--aee5f131-7d89-4f9a-ac02-cda88709f315-0', usage_metadata={'input_tokens': 62, 'output_tokens': 5, 'total_tokens': 67, 'input_token_details': {'cache_read': 0}}), HumanMessage(content='Escribe una transcripción de nuestra conversación', additional_kwargs={}, response_metadata={}, id='0c27c514-069e-44aa-93a1-a1c858987ce0')]}, next=('chatbot',), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f03cb6a-1653-625a-8006-bcea6db81b10'}}, metadata={'source': 'loop', 'writes': None, 'step': 6, 'parents': {}, 'thread_id': '1'}, created_at='2025-05-29T17:59:13.796339+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f03cb6a-164f-6485-8005-7272d71418b4'}}, tasks=(PregelTask(id='74b6677d-7a48-fbed-e068-e88e88feee24', name='chatbot', path=('__pregel_pull', 'chatbot'), error=None, interrupts=(), state=None, result={'messages': [AIMessage(content='Aquí está la transcripción de nuestra conversación:\n\nUsuario: Hola me llamo Pepe\nModelo: Hola Pepe, ¿cómo puedo ayudarte hoy?\nUsuario: Dime cómo me llamo\nModelo: Te llamas Pepe.\nUsuario: Escribe una transcripción de nuestra conversación\nModelo: Aquí está la transcripción de nuestra conversación:\n\nUsuario: Hola me llamo Pepe\nModelo: Hola Pepe, ¿cómo puedo ayudarte hoy?\nUsuario: Dime cómo me llamo\nModelo: Te llamas Pepe.\nUsuario: Escribe una transcripción de nuestra conversación', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--5c4d427d-d0fb-40f6-8657-dafa843e0190-0', usage_metadata={'input_tokens': 74, 'output_tokens': 107, 'total_tokens': 181, 'input_token_details': {'cache_read': 0}})]}),), interrupts=()),
     StateSnapshot(values={'messages': [HumanMessage(content='Hola me llamo Pepe', additional_kwargs={}, response_metadata={}, id='07a9e750-3bf6-4120-97c6-391b10bbd9d2'), AIMessage(content='Hola Pepe, ¿cómo puedo ayudarte hoy?', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--21af6c8b-7ba9-4354-ad95-fa0b46432d10-0', usage_metadata={'input_tokens': 49, 'output_tokens': 10, 'total_tokens': 59, 'input_token_details': {'cache_read': 0}}), HumanMessage(content='Dime cómo me llamo', additional_kwargs={}, response_metadata={}, id='44b1ecfd-fea9-4023-8bbc-17f1b1d78dee'), AIMessage(content='Te llamas Pepe.', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--aee5f131-7d89-4f9a-ac02-cda88709f315-0', usage_metadata={'input_tokens': 62, 'output_tokens': 5, 'total_tokens': 67, 'input_token_details': {'cache_read': 0}})]}, next=('__start__',), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f03cb6a-164f-6485-8005-7272d71418b4'}}, metadata={'source': 'input', 'writes': {'__start__': {'messages': [{'role': 'user', 'content': 'Escribe una transcripción de nuestra conversación'}]}}, 'step': 5, 'parents': {}, 'thread_id': '1'}, created_at='2025-05-29T17:59:13.794746+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f03cb68-6486-6840-8004-a3dbcc7d4887'}}, tasks=(PregelTask(id='3dedb12f-b952-c129-adb7-2b85e7ef1bd0', name='__start__', path=('__pregel_pull', '__start__'), error=None, interrupts=(), state=None, result={'messages': [{'role': 'user', 'content': 'Escribe una transcripción de nuestra conversación'}]}),), interrupts=()),
     StateSnapshot(values={'messages': [HumanMessage(content='Hola me llamo Pepe', additional_kwargs={}, response_metadata={}, id='07a9e750-3bf6-4120-97c6-391b10bbd9d2'), AIMessage(content='Hola Pepe, ¿cómo puedo ayudarte hoy?', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--21af6c8b-7ba9-4354-ad95-fa0b46432d10-0', usage_metadata={'input_tokens': 49, 'output_tokens': 10, 'total_tokens': 59, 'input_token_details': {'cache_read': 0}}), HumanMessage(content='Dime cómo me llamo', additional_kwargs={}, response_metadata={}, id='44b1ecfd-fea9-4023-8bbc-17f1b1d78dee'), AIMessage(content='Te llamas Pepe.', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--aee5f131-7d89-4f9a-ac02-cda88709f315-0', usage_metadata={'input_tokens': 62, 'output_tokens': 5, 'total_tokens': 67, 'input_token_details': {'cache_read': 0}})]}, next=(), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f03cb68-6486-6840-8004-a3dbcc7d4887'}}, metadata={'source': 'loop', 'writes': {'chatbot': {'messages': [AIMessage(content='Te llamas Pepe.', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--aee5f131-7d89-4f9a-ac02-cda88709f315-0', usage_metadata={'input_tokens': 62, 'output_tokens': 5, 'total_tokens': 67, 'input_token_details': {'cache_read': 0}})]}}, 'step': 4, 'parents': {}, 'thread_id': '1'}, created_at='2025-05-29T17:58:28.309176+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f03cb68-5f8f-680b-8003-42b83fafe2de'}}, tasks=(), interrupts=()),
     StateSnapshot(values={'messages': [HumanMessage(content='Hola me llamo Pepe', additional_kwargs={}, response_metadata={}, id='07a9e750-3bf6-4120-97c6-391b10bbd9d2'), AIMessage(content='Hola Pepe, ¿cómo puedo ayudarte hoy?', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--21af6c8b-7ba9-4354-ad95-fa0b46432d10-0', usage_metadata={'input_tokens': 49, 'output_tokens': 10, 'total_tokens': 59, 'input_token_details': {'cache_read': 0}}), HumanMessage(content='Dime cómo me llamo', additional_kwargs={}, response_metadata={}, id='44b1ecfd-fea9-4023-8bbc-17f1b1d78dee')]}, next=('chatbot',), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f03cb68-5f8f-680b-8003-42b83fafe2de'}}, metadata={'source': 'loop', 'writes': None, 'step': 3, 'parents': {}, 'thread_id': '1'}, created_at='2025-05-29T17:58:27.788551+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f03cb68-5f8a-6b6c-8002-e1d32e31fc31'}}, tasks=(PregelTask(id='53cf3d64-d662-2a17-4e41-421290cdbc89', name='chatbot', path=('__pregel_pull', 'chatbot'), error=None, interrupts=(), state=None, result={'messages': [AIMessage(content='Te llamas Pepe.', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--aee5f131-7d89-4f9a-ac02-cda88709f315-0', usage_metadata={'input_tokens': 62, 'output_tokens': 5, 'total_tokens': 67, 'input_token_details': {'cache_read': 0}})]}),), interrupts=()),
     StateSnapshot(values={'messages': [HumanMessage(content='Hola me llamo Pepe', additional_kwargs={}, response_metadata={}, id='07a9e750-3bf6-4120-97c6-391b10bbd9d2'), AIMessage(content='Hola Pepe, ¿cómo puedo ayudarte hoy?', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--21af6c8b-7ba9-4354-ad95-fa0b46432d10-0', usage_metadata={'input_tokens': 49, 'output_tokens': 10, 'total_tokens': 59, 'input_token_details': {'cache_read': 0}})]}, next=('__start__',), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f03cb68-5f8a-6b6c-8002-e1d32e31fc31'}}, metadata={'source': 'input', 'writes': {'__start__': {'messages': [{'role': 'user', 'content': 'Dime cómo me llamo'}]}}, 'step': 2, 'parents': {}, 'thread_id': '1'}, created_at='2025-05-29T17:58:27.786601+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f03cb67-7a55-6109-8001-5fa9f91d24a5'}}, tasks=(PregelTask(id='9986d684-629f-7fa0-93f2-f08d351d1116', name='__start__', path=('__pregel_pull', '__start__'), error=None, interrupts=(), state=None, result={'messages': [{'role': 'user', 'content': 'Dime cómo me llamo'}]}),), interrupts=()),
     StateSnapshot(values={'messages': [HumanMessage(content='Hola me llamo Pepe', additional_kwargs={}, response_metadata={}, id='07a9e750-3bf6-4120-97c6-391b10bbd9d2'), AIMessage(content='Hola Pepe, ¿cómo puedo ayudarte hoy?', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--21af6c8b-7ba9-4354-ad95-fa0b46432d10-0', usage_metadata={'input_tokens': 49, 'output_tokens': 10, 'total_tokens': 59, 'input_token_details': {'cache_read': 0}})]}, next=(), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f03cb67-7a55-6109-8001-5fa9f91d24a5'}}, metadata={'source': 'loop', 'writes': {'chatbot': {'messages': [AIMessage(content='Hola Pepe, ¿cómo puedo ayudarte hoy?', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--21af6c8b-7ba9-4354-ad95-fa0b46432d10-0', usage_metadata={'input_tokens': 49, 'output_tokens': 10, 'total_tokens': 59, 'input_token_details': {'cache_read': 0}})]}}, 'step': 1, 'parents': {}, 'thread_id': '1'}, created_at='2025-05-29T17:58:03.752260+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f03cb67-5b08-689a-8000-827d4f4dac59'}}, tasks=(), interrupts=()),
     StateSnapshot(values={'messages': [HumanMessage(content='Hola me llamo Pepe', additional_kwargs={}, response_metadata={}, id='07a9e750-3bf6-4120-97c6-391b10bbd9d2')]}, next=('chatbot',), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f03cb67-5b08-689a-8000-827d4f4dac59'}}, metadata={'source': 'loop', 'writes': None, 'step': 0, 'parents': {}, 'thread_id': '1'}, created_at='2025-05-29T17:58:00.470311+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f03cb67-5b04-629b-bfff-fe3a221e5983'}}, tasks=(PregelTask(id='23df3b48-36fe-2388-52f2-7f478f0bdc02', name='chatbot', path=('__pregel_pull', 'chatbot'), error=None, interrupts=(), state=None, result={'messages': [AIMessage(content='Hola Pepe, ¿cómo puedo ayudarte hoy?', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--21af6c8b-7ba9-4354-ad95-fa0b46432d10-0', usage_metadata={'input_tokens': 49, 'output_tokens': 10, 'total_tokens': 59, 'input_token_details': {'cache_read': 0}})]}),), interrupts=()),
     StateSnapshot(values={'messages': []}, next=('__start__',), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f03cb67-5b04-629b-bfff-fe3a221e5983'}}, metadata={'source': 'input', 'writes': {'__start__': {'messages': [{'role': 'user', 'content': 'Hola me llamo Pepe'}]}}, 'step': -1, 'parents': {}, 'thread_id': '1'}, created_at='2025-05-29T17:58:00.468526+00:00', parent_config=None, tasks=(PregelTask(id='cde9f820-9680-848f-3590-7c3edb4f1e49', name='__start__', path=('__pregel_pull', '__start__'), error=None, interrupts=(), state=None, result={'messages': [{'role': 'user', 'content': 'Hola me llamo Pepe'}]}),), interrupts=())]



LangGraph permite retrotraer la conversación al momento deseado:

```python
config = {"configurable": {"thread_id": "1", "checkpoint_id": ...}}
graph.invoke(None, config=config)
```

