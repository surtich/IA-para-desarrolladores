
En esta práctica vamos a aprender a crear un Tool para usarla desde nuestra aplicación Web con Gradio. Una Tool es una función externa que un LLM puede decidir llamar para interactuar con el mundo exterior o realizar tareas específicas. Es muy importante entender que el LLM no llama directamente a la Tool, sino que le pide al cliente que la invoque. Es el cliente quien decide se ejecutar la tool y pasar al LLM el resultado de su ejecución.

Estas herramientas extienden las capacidades del LLM más allá de su entrenamiento, permitiéndole, por ejemplo, buscar información en la web, enviar correos electrónicos, interactuar con API, o realizar cálculos complejos.

Cuando un LLM tiene la capacidad de seleccionar y utilizar de forma autónoma una o varias Tools para alcanzar un objetivo, decimos que se comporta como un Agente. Un **Agente** no solo "sabe" qué herramientas existen, sino que también puede razonar sobre cuándo, cómo y en qué orden usarlas para resolver un problema o completar una tarea, a menudo en un proceso iterativo de planificación, ejecución y auto-corrección.

Realizamos los `imports` y creamos una instancia de la API.


```python
# imports

import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import json
```

    /home/surtich/projects/IA para desarrolladores/.venv/lib/python3.13/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm



```python
load_dotenv(override=True)
google_api_key = os.getenv('GOOGLE_API_KEY')
```


```python
MODEL = "gemini-2.0-flash"
geminiai = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta", api_key=google_api_key)
```

Definimos una función sencilla que devuelve los precios de viajar a varias cuiudades:


```python
ticket_prices = {"london": "799€", "paris": "899€", "tokyo": "1400€", "berlin": "499€"}

def get_ticket_price(destination_city):
    print(f"Tool get_ticket_price called for {destination_city}")
    city = destination_city.lower()
    return ticket_prices.get(city, "Unknown")
```


```python
get_ticket_price("London")
```

    Tool get_ticket_price called for London





    '799€'



Para ayudar al agente a decidir cuando llamar a una Tool específica, la API de OpenAI requiere un formato concreto de diccionario. Los campos `description` tanto de la función como de los parámetros serán utilizados para que el LLM entienda qué hace la función y cómo debe ser llamada.
```python


```python
# There's a particular dictionary structure that's required to describe our function:

price_function = {
    "name": "get_ticket_price",
    "description": "Obtén el precio de un billete de ida y vuelta a la ciudad de destino. Llama a esta función siempre que necesites saber el precio del billete, por ejemplo, cuando un cliente pregunte '¿Cuánto cuesta un billete a esta ciudad?'",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "La ciudad a la que el cliente desea viajar",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False
    }
}
```

Se debe crear una lista con todas las Tools del agente:


```python
# And this is included in a list of tools:

tools = [{"type": "function", "function": price_function}]
```

La función `handle_tool_call` recibe el `message` devuelto por la Tool y genera un respuesta en el formato requerido por la API de OpenAI. Observe que en la función `chat`, si el mensaje procede de la Tool, lo extrae y se lo vuelve a pasar al LLM para que lo procese y genere una respuesta final. El LLM no tiene acceso directo a la Tool, sino que el cliente es quien maneja la llamada y el resultado.


```python
system_message = "Eres un asistente útil para una aerolínea llamada FlightAI. "
system_message += "Ofrece respuestas cortas y educadas de no más de 1 oración. "
system_message += "Sé siempre preciso. Si no sabes la respuesta, dilo."

def handle_tool_call(message):
    tool_call = message.tool_calls[0] # the first tool call
    arguments = json.loads(tool_call.function.arguments) # the arguments are a JSON string. json.loads() parses it to a dict
    city = arguments.get('destination_city')
    price = get_ticket_price(city)
    response = {
        "role": "tool",
        "content": json.dumps({"destination_city": city,"price": price}), # the content is a JSON string. json.dumps() converts a dict to a JSON string
        "tool_call_id": tool_call.id # the ID of the tool call. This is used to identify the tool call in the response
    }
    return response, city

def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = geminiai.chat.completions.create(model=MODEL, messages=messages, tools=tools)

    if response.choices[0].finish_reason=="tool_calls": # the model has called a tool
        message = response.choices[0].message # Gets the message that contains the tool call
        response, city = handle_tool_call(message) 
        messages.append(message) 
        messages.append(response)
        response = geminiai.chat.completions.create(model=MODEL, messages=messages)
    
    return response.choices[0].message.content
```

Usamos Gradio para probarlo:


```python
gr.ChatInterface(fn=chat, type="messages").launch()
```

    * Running on local URL:  http://127.0.0.1:7866
    * To create a public link, set `share=True` in `launch()`.



<div><iframe src="http://127.0.0.1:7866/" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>





    



    Tool get_ticket_price called for London

