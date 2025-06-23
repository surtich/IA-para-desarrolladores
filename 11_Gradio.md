### Gradio

En esta práctica vamos a aprender a construir aplicaciones Web con Gradio, una librería de Python que permite crear interfaces gráficas de usuario (GUI) para modelos de machine learning y otras funciones de Python. Gradio es especialmente útil para crear prototipos rápidos y compartir aplicaciones con otros usuarios.

Realizamos los `imports` y creamos una instancia de la API.


```python
# imports

import os
from dotenv import load_dotenv
from openai import OpenAI
```


```python
load_dotenv(override=True)
google_api_key = os.getenv('GOOGLE_API_KEY')
```


```python
MODEL = "gemini-2.0-flash"
openai = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta", api_key=google_api_key)
```

El uso básico de Gradio es bastante simple: Creamos una función en Python y la asociamos a Gradio. El valor de entrada de la función será el componente del parámetro `inputs` de Gradio y su salida mostrará en el componente `ouputs`:


```python
import gradio as gr

# here's a simple function
def shout(text):
    print(f"Shout has been called with input {text}")
    return text.upper()


# Adding inbrowser=True opens up a new browser window automatically
gr.Interface(fn=shout, inputs="textbox", outputs="textbox", flagging_mode="never").launch(inbrowser=True)

```

    /home/surtich/projects/IA para desarrolladores/.venv/lib/python3.13/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


    * Running on local URL:  http://127.0.0.1:7860
    * To create a public link, set `share=True` in `launch()`.



<div><iframe src="http://127.0.0.1:7860/" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>





    



    Shout has been called with input Hola


Podemos personalizar los componentes `inputs`y `outputs`:


```python
view = gr.Interface(
    fn=shout,
    inputs=[gr.Textbox(label="Your message:", lines=6)],
    outputs=[gr.Textbox(label="Response:", lines=8)],
    flagging_mode="never"
)
view.launch()
```

    * Running on local URL:  http://127.0.0.1:7861
    * To create a public link, set `share=True` in `launch()`.



<div><iframe src="http://127.0.0.1:7861/" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>





    



La función podría ser la llamada a una API de un LLM:


```python
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
```

    * Running on local URL:  http://127.0.0.1:7862
    * To create a public link, set `share=True` in `launch()`.



<div><iframe src="http://127.0.0.1:7862/" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>





    



Lar respuesta también pode ser un `stream`:


```python
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
```

    * Running on local URL:  http://127.0.0.1:7863
    * To create a public link, set `share=True` in `launch()`.



<div><iframe src="http://127.0.0.1:7863/" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>





    



Hasta ahora hemos hecho una integración de Gradio con modelos LLM manual. Pero Gradio también permite hacer esto de forma automatizada:


```python
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
```

    * Running on local URL:  http://127.0.0.1:7864
    * To create a public link, set `share=True` in `launch()`.



<div><iframe src="http://127.0.0.1:7864/" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>





    



    Historia::
    []
    Mensaje:
    [{'role': 'system', 'content': 'Eres un asistente'}, {'role': 'user', 'content': 'Hola'}]


### NL2SQL (Natural Language to SQL)

Como ejemplo uso de Gradio como desarrollador, vamos a crear una aplicación que traduzca preguntas en lenguaje natural a consultas SQL.

Empezamos definiendo el `system` al que informamos de la estructura de la base de datos.


```python
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

```

Definimos las funciones que se encargarán de realizar la petición al modelo y mantener el contexto.


```python
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
```

Creamos la interfaz de Gradio. Observe que podemos pasar ejemplos de preguntas que el usuario puede seleccionar.




```python
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
```

    * Running on local URL:  http://127.0.0.1:7865
    * To create a public link, set `share=True` in `launch()`.



<div><iframe src="http://127.0.0.1:7865/" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>





    


