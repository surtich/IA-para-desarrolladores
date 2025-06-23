# %% [markdown]
# ### Gradio
# 
# En esta práctica vamos a aprender a construir aplicaciones Web con Gradio, una librería de Python que permite crear interfaces gráficas de usuario (GUI) para modelos de machine learning y otras funciones de Python. Gradio es especialmente útil para crear prototipos rápidos y compartir aplicaciones con otros usuarios.

# %% [markdown]
# Realizamos los `imports` y creamos una instancia de la API.

# %%
# imports

import os
from dotenv import load_dotenv
from openai import OpenAI

# %%
load_dotenv(override=True)
google_api_key = os.getenv('GOOGLE_API_KEY')

# %%
MODEL = "gemini-2.0-flash"
openai = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta", api_key=google_api_key)

# %% [markdown]
# El uso básico de Gradio es bastante simple: Creamos una función en Python y la asociamos a Gradio. El valor de entrada de la función será el componente del parámetro `inputs` de Gradio y su salida mostrará en el componente `ouputs`:

# %%
import gradio as gr

# here's a simple function
def shout(text):
    print(f"Shout has been called with input {text}")
    return text.upper()


# Adding inbrowser=True opens up a new browser window automatically
gr.Interface(fn=shout, inputs="textbox", outputs="textbox", flagging_mode="never").launch(inbrowser=True)


# %% [markdown]
# Podemos personalizar los componentes `inputs`y `outputs`:

# %%
view = gr.Interface(
    fn=shout,
    inputs=[gr.Textbox(label="Your message:", lines=6)],
    outputs=[gr.Textbox(label="Response:", lines=8)],
    flagging_mode="never"
)
view.launch()

# %% [markdown]
# La función podría ser la llamada a una API de un LLM:

# %%
def message_llm(prompt):
    response = openai.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

view = gr.Interface(
    fn=message_llm,
    inputs=[gr.Textbox(label="Your message:", lines=6)],
    outputs=[gr.Textbox(label="Response:", lines=8)],
    flagging_mode="never"
)
view.launch()

# %% [markdown]
# Lar respuesta también pode ser un `stream`:

# %%
system_message = "Eres un asistente que respondes en markdown"

def stream_llm(prompt):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
      ]
    stream = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True
    )
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result

view = gr.Interface(
    fn=stream_llm,
    inputs=[gr.Textbox(label="Your message:")],
    outputs=[gr.Markdown(label="Response:")],
    flagging_mode="never"
)
view.launch()

# %% [markdown]
# Hasta ahora hemos hecho una integración de Gradio con modelos LLM manual. Pero Gradio también permite hacer esto de forma automatizada:

# %%
system_message = "Eres un asistente"

def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]

    print("Historia::")
    print(history)
    print("Mensaje:")
    print(messages)

    stream = openai.chat.completions.create(model=MODEL, messages=messages, stream=True)

    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response

gr.ChatInterface(fn=chat, type="messages").launch()

# %% [markdown]
# ### NL2SQL (Natural Language to SQL)
# 
# Como ejemplo uso de Gradio como desarrollador, vamos a crear una aplicación que traduzca preguntas en lenguaje natural a consultas SQL.
# 
# Empezamos definiendo el `system` al que informamos de la estructura de la base de datos.

# %%
context = [
	{
		'role': 'system',
		'content': """
Eres un bot para ayudar a crear comandos SQL, todas tus respuestas deben comenzar con
Esto es tu SQL, y después de eso un SQL que haga lo que el usuario solicita.

Tu base de datos SQL está compuesta por algunas tablas.
Intenta mantener el orden del SQL simple.
Contesta únicamente con el SQL.
Explica el SQL solo si el usuario lo pide.
Si el usuario pide algo que no se puede responder con la información de la base de datos,
solo responde algo amable y simple, máximo 10 palabras, pidiéndole una nueva pregunta que
pueda resolverse con SQL.
"""
	}
]

context.append({
	'role': 'system',
	'content': """
Tablas de la base de datos:
[{
	"tableName": "empleados",
	"fields": [
		{
			"nombre": "id",
			"tipo": "int"
		},
		{
			"nombre": "name",
			"tipo": "string"
		}
	]
},
{
	"tableName": "salarios",
	"fields": [
		{
			"nombre": "id",
			"type": "int"
		},
		{
			"name": "año",
			"type": "date"
		},
		{
			"name": "salario",
			"type": "float"
		}
	]
},
{
	"tablename": "titulaciones",
	"fields": [
		{
			"name": "id",
			"type": "int"
		},
		{
			"name": "empleadoId",
			"type": "int"
		},
		{
			"name": "nivelEducativo",
			"type": "int"
		},
		{
			"name": "Institución",
			"type": "string"
		},
		{
			"name": "Fecha",
			"type": "date"
		},
		{
			"name": "Especialización",
			"type": "string"
		}
	]
}]
"""
})


# %% [markdown]
# Definimos las funciones que se encargarán de realizar la petición al modelo y mantener el contexto.

# %%
def continue_conversation(messages, temperature=0):
    response = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content

def gradio_chat(message, history):
    history_chat = context + history
    history_chat.append({"role":"user", "content":message})    
    return continue_conversation(history_chat, 0.3)

# %% [markdown]
# Creamos la interfaz de Gradio. Observe que podemos pasar ejemplos de preguntas que el usuario puede seleccionar.
# 
# 

# %%
examples = [
	"¿Quién es el empleado mejor pagado?",
	"¿Cuántos empleados son titulados?"
]

view = gr.ChatInterface(
    fn=gradio_chat,
    type="messages",    
    textbox=gr.Textbox(placeholder="Escribe tu consulta aquí"),
    title="SQL Generator",
    examples=examples,
    flagging_mode="never"
)
view.launch()


