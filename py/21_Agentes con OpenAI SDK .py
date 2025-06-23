# %% [markdown]
# ### OpenAI SDK Agents
# 
# El SDK Agents es una herramienta que permite crear agentes de IA personalizados utilizando los modelos de OpenAI. Algunos proveedores de modelos de lenguaje como Google Gemini pueden usar esta API con soporte limitado.
# 
# SDK Agents tiene pocas abstracciones y es fácil de usar. El Agents SDK tiene un conjunto muy pequeño de primitivas:
# 
# * Agentes (`Agents`), que son LLMs equipados con instrucciones y herramientas.
# * Delegaciones (`Handoffs`), que permiten a los agentes delegar tareas específicas a otros agentes.
# * Barreras de seguridad (`Guardrails`), que permiten validar las entradas y salidas de los agentes.
# 
# Además, el SDK viene con trazabilidad incorporada que permite visualizar y depurar los flujos de agentes.
# 
# En esta práctica vamos a utilizar el SDK Agents para crear una aplicación que genere un informe profesional a partir de una pregunta formulada por el usuario. Varios agentes se coordinarán para realizar una búsqueda en Internet y componer un informe que será enviado al usuario.

# %% [markdown]
# Realizamos las importacines necesarias

# %%
from openai import AsyncOpenAI
from agents import Agent, trace, Runner, gen_trace_id, function_tool, OpenAIChatCompletionsModel
from agents.model_settings import ModelSettings
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import asyncio
import os
from typing import Dict
from IPython.display import display, Markdown
import requests
from google import genai
from google.genai.types import Tool, GoogleSearch, GenerateContentConfig
import json
import gradio as gr

# %% [markdown]
# ### `Searcher Agent`

# %% [markdown]
# Creamos el primer agente que será capaz de buscar información en Internet. Este agente realiza la búsqueda usando Google Gemini. Para hacer búsquedas en Internet con el SDK Agents hay que usar la clase `WebSearchAgent`. Pero esta clase no es compatible con `OpenAIChatCompletionsModel`. `OpenAIChatCompletionsModel` es una función que se debe usar cuando se crean agentes con un proveedor distinto de OpenAI. Por lo tanto, se ha tenido que utilizar otra estrategia para poder realizar búsquedas en Internet. Se ha realizado usando la API de Google Gemini llamada Google GenAI. Los pasos son los siguientes:
# 
# * 1 Importamos la clave de Google Gemini.
# * 2 Creamos el modelo que se usará para crear los agentes. Normalmente este modelo sería una simple cadena de texto con el nombre del modelo. Pero cuando el modelo no es de OpenAI el proceso es un poco más complicado.
# * 3 Creamos el cliente que se conectará a la API de Google GenAI.
# * 4 Creamos la `tool` de GenAI que se usará para realizar las búsquedas en Internet.
# * 5 Creamos una función de búsqueda que utiliza la `tool`.
# * 6 ...
# 
# ... Aunque no hayamos terminado de crear el agente, tenemos la función de búsqueda que se usará en el agente y podemos probarla.

# %%
# 1.- Importamos la clave de Google Gemini.
load_dotenv()
google_api_key = os.getenv('GOOGLE_API_KEY')

if google_api_key:
    print(f"La clave de API de Google existe y comienza con {google_api_key[:2]}")
else:
    print("La clave de API de Google no está configurada (y esto no es opcional)")

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


# 2.- Creamos el modelo que se usará para crear los agentes.
chat_client = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)
chat_model = OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=chat_client)

# 3.- Creamos el cliente que se conectará a la API de Google GenAI.
search_client = genai.Client(api_key=google_api_key)

# 4.- Creamos la `tool` de GenAI que se usará para realizar las búsquedas en Internet.
INSTRUCTIONS = "Eres un asistente de investigación. Dado un término de búsqueda, buscas en la web ese término y \
produces un resumen conciso de los resultados. El resumen debe tener 2-3 párrafos y menos de 300 \
palabras. Captura los puntos principales. Escribe de forma sucinta, no es necesario usar oraciones completas ni buena \
gramática. Esto será utilizado por alguien que está sintetizando un informe, así que es vital que captures la \
esencia y ignores cualquier información irrelevante. No incluyas ningún comentario adicional aparte del propio resumen."

search_tool = Tool(google_search=GoogleSearch())

config = GenerateContentConfig(
    system_instruction=INSTRUCTIONS,
    tools=[search_tool],
    response_modalities=["TEXT"]
)

# 5.- Creamos una función de búsqueda que utiliza la `tool`.
def search_term(term: str) -> Dict[str, str]:
    """Realiza una búsqueda en Google y devuelve los resultados."""
    response = search_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=term,
        config=config
    )
    return response

# %%
response = search_term("Día y hora actual en Madrid (España) y previsión del tiempo para los próximos 3 días.")
display(Markdown(response.text))

# %% [markdown]
# ... continuamos:
# 
# * 6 Convertimos la función de búsqueda en una `tool` que se usará en el agente. Para ello usamos el decorador `@function_tool`.
# * 7 ...

# %%
# 6.- Convertimos la función de búsqueda en una `tool
@function_tool
def search_term(term: str) -> Dict[str, str]:
    """Realiza una búsqueda en Google y devuelve los resultados."""
    response = search_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=term,
        config=config
    )
    return response

# %%
# search_term ya no es una función normal de Python sino una tool que se puede usar en un agente.
search_term

# %% [markdown]
# * 7 Creamos un agente que usa la `tool` de búsqueda. Con `ModelSettings` forzamos al agente a que use la `tool` siempre.

# %%
# 7.- Creamos un agente que usa la `tool` de búsqueda
search_agent = Agent(
    name="Agente de Búsqueda",
    instructions=INSTRUCTIONS,
    tools=[search_term],
    model=chat_model,
    model_settings=ModelSettings(tool_choice="required"),
)

# %% [markdown]
# * 8 Probamos el agente. Está metido en un contexto `trace` que permite trazabilidad. La trazabilidad es muy fácil de implementar con OpenAI Agents SDK pero no he sido capaz de configurarlo en Google Gemini.

# %%
message = "Frameworks de agentes de IA nuevos y más populares en el año actual"

with trace("Search"):
    result = await Runner.run(search_agent, message)

display(Markdown(result.final_output))

# %% [markdown]
# ### `Planner Agent`
# 
# Este agente, dado un término de búsqueda, propone un conjunto de preguntas que se pueden hacer para afinar la búsqueda. En la construcción del agente se utiliza el parámetro `output_type` para especificar el formato de salida esperado. A esto se le llama salidas estructuradas. En este caso, se espera una lista de preguntas, donde cada pregunta tiene los campos `reason` y `query`. Observe que el esquema se define usando `Pydantic` y es el agente el que tiene que entender el esquemea y devolver la salida en el formato correcto.

# %%
HOW_MANY_SEARCHES = 3

INSTRUCTIONS = f"Eres un asistente de investigación útil. Dada una consulta, genera un conjunto de búsquedas web en español\
para responder mejor a la pregunta. Devuelve {HOW_MANY_SEARCHES} términos de búsqueda a consultar."

# Usamos Pydantic para definir el esquema de respuesta - conocido como "Salidas Estructuradas"
class WebSearchItem(BaseModel):
    reason: str = Field(description="Tu razonamiento sobre por qué esta búsqueda es importante para la consulta.")
    
    query: str = Field(description="El término de búsqueda a utilizar para la búsqueda web.")


class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem] = Field(description="Una lista de búsquedas web para realizar y responder mejor a la consulta.")

# Este agente no hace búsquedas web directamente, sino que genera un plan de búsqueda con tres preguntas.
planner_agent = Agent(
    name="Agente Planificador",
    instructions=INSTRUCTIONS,
    model=chat_model,
    output_type=WebSearchPlan,
)

# %% [markdown]
# Probamos el agente. Observe que propone tres preguntas con el esquema definido.

# %%
message = "Frameworks de agentes de IA nuevos y más populares en el año actual"

with trace("Plan"):
    result = await Runner.run(planner_agent, message)

result.final_output

# %% [markdown]
# ### Agente `Emailer Agent`
# 
# Este agente es capaz de enviar correos electrónicos. Pare ello creamos la `tool` `send_email` que se encargará de enviar el correo electrónico utilizando la API de Mailgun.

# %%
@function_tool
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

# %%
INSTRUCTIONS = """Eres capaz de enviar un correo electrónico en HTML bien formateado a partir de un informe detallado.
Se te proporcionará un informe detallado. Debes usar tu herramienta para enviar un solo correo, convirtiendo el informe
en un HTML limpio y bien presentado, con una línea de asunto adecuada."""

email_agent = Agent(
    name="Agente de correo",
    instructions=INSTRUCTIONS,
    tools=[send_email],
    model=chat_model,
    model_settings=ModelSettings(tool_choice="required")
)

# %% [markdown]
# Probamos el agente. Revisar la carpeta de `spam` porque suele llegar allí.

# %%
with trace("Email"):
    params = json.dumps({
        "subject": "Informe de investigación sobre frameworks de agentes de IA",
        "html_body": "bla **bla** bla",
        "to": "surtich@gmail.com",
        "name": "Javier"
    })
    result = await Runner.run(email_agent, params)

# %% [markdown]
# ### `Writer Agent`
# 
# Este agente es capaz de redactar un informe profesional a partir de un conjunto de preguntas y respuestas.

# %%
INSTRUCTIONS = (
    "Eres un investigador senior encargado de redactar un informe cohesivo para una consulta de investigación. "
    "Se te proporcionará la consulta original y una investigación inicial realizada por un asistente de investigación.\n"
    "Primero debes elaborar un esquema para el informe que describa la estructura y el flujo del mismo. Luego, genera el informe y devuélvelo como salida final.\n"
    "La salida final debe estar en formato markdown y debe ser extensa y detallada. Apunta a 5-10 páginas de contenido, al menos 1000 palabras."
)


class ReportData(BaseModel):
    short_summary: str = Field(description="Un resumen breve de 2-3 frases sobre los hallazgos.")

    markdown_report: str = Field(description="El informe final")

    follow_up_questions: list[str] = Field(description="Temas sugeridos para investigar más a fondo")


writer_agent = Agent(
    name="AgenteRedactor",
    instructions=INSTRUCTIONS,
    model=chat_model,
    output_type=ReportData,
)

# %% [markdown]
# ### Funciones auxiliares
# 
# Las siguientes funciones permiten conectar los agentes entre sí. El SDK Agents es de bajo nivel y este proceso hay que hacerlo manualmente.

# %% [markdown]
# La `plan_searches` utilza el planificador para generar un conjunto de preguntas a partir de un término de búsqueda.

# %%
async def plan_searches(query: str) -> WebSearchPlan:
    """ Utiliza el planner_agent para planificar qué búsquedas realizar para la consulta """
    print("Planificando búsquedas...")
    result = await Runner.run(planner_agent, f"Consulta: {query}")
    print(f"Se realizarán {len(result.final_output.searches)} búsquedas")
    return result.final_output_as(WebSearchPlan) # no es estrictamente necesario. Se puede ejecutar simplemente result.final_output

# %% [markdown]
# Observe que la función `plan_searches` es una corutina.

# %%
query = "Informe sobre institutos públicos de la Comunidad de Madrid que imparten DAW. Nombre de los institutos, reseñas, ..."
plan_searches(query)

# %% [markdown]
# Para ejecutala hay que usar:
# 
# * `asyncio.run(plan_searches("Python programming language"))` si estamos en un programa de Python.
# * `await plan_searches("Python programming language")` si estamos en un entorno asíncrono como Jupyter Notebook.

# %%
search_plan = await plan_searches(query)
search_plan

# %% [markdown]
# La función `perform_searches` utiliza la función `search` para realizar las búsquedas en Internet.

# %%
async def perform_searches(search_plan: WebSearchPlan):
    """ Llama a search() para cada elemento en el plan de búsqueda """
    print("Buscando...")
    tasks = [asyncio.create_task(search(item)) for item in search_plan.searches] # Ejecuta en paralelo las búsquedas
    num_completed = 0
    results = []
    for task in asyncio.as_completed(tasks):
        result = await task
        if result is not None:
            results.append(result)
        num_completed += 1
        print(f"Buscando... {num_completed}/{len(tasks)} completado")
    print("Búsquedas finalizadas")
    return results

async def search(item: WebSearchItem):
    """ Utiliza el search_agent para realizar una búsqueda web por cada elemento del plan de búsqueda """
    input = f"Término de búsqueda: {item.query}\nRazón para buscar: {item.reason}"
    result = await Runner.run(search_agent, input)
    return result.final_output

# %% [markdown]
# Probamos

# %%
search_results = await perform_searches(search_plan)
search_results

# %% [markdown]
# La siguiente función escribe el informe a partir de las preguntas y respuestas obtenidas.

# %%
async def write_report(query: str, search_results: list[str]):
    """ Utiliza el agente redactor para escribir un informe basado en los resultados de búsqueda """
    print("Pensando en el informe...")
    input = f"Consulta original: {query}\nResultados de búsqueda resumidos: {search_results}"
    result = await Runner.run(writer_agent, input)
    print("Informe terminado")
    return result.final_output

# %% [markdown]
# Probamos

# %%
await write_report(query, search_results)

# %% [markdown]
# Por último, escribimos la función que envía el correo electrónico con el informe generado.

# %%
async def send_report(report: ReportData, to: str, name: str = None):
    """ Utiliza el agente de correo para enviar un correo con el informe """
    print("Redactando correo...")
    send_email_params = {
        "html_body": report.markdown_report,
        "to": to,
        "name": name
    }
    if (name):
        send_email_params["name"] = name
    result = await Runner.run(email_agent, json.dumps(send_email_params))
    print("Correo enviado")
    return report

# %% [markdown]
# Probamos

# %%
report = ReportData(
    short_summary="Resumen breve de los hallazgos del informe.",
    markdown_report="Informe sobre institutos públicos de la Comunidad de Madrid que imparten DAW\n\nEste es un informe detallado...",
    follow_up_questions=["¿Qué otros institutos ofrecen DAW?", "¿Cuáles son las tasas de empleo de los graduados?"]
)
await send_report(report, "surtich@gmail.com")

# %% [markdown]
# ### Prueba de la aplicación
# 
# Finalmente, probamos la aplicación completa. Creamos función que coordina todos los agentes anteriores y envía el informe por correo electrónico.

# %%
query = "Aplicaciones del lenguaje Python para Administradores de Sistemas en 2025"
to = "surtich@gmail.com"

with trace("Rastreo de investigación"):
    print("Iniciando investigación...")
    search_plan = await plan_searches(query)
    search_results = await perform_searches(search_plan)
    report = await write_report(query, search_results)
    await send_report(report, to)
    print("¡Hurra!")

# %% [markdown]
# ### Integración con Gradio
# 
# En este caso no podemos usar la función `ChatInterface` de Gradio porque necesitamos recoger la entrada del usuario. Creamos una función `run` y se la asociamos a la función `launch`. La función `run` emite (`yield`) valores que Gradio mostrará en la interfaz.
# 
# 

# %%
async def run(query: str, to: str, name: str = None):
    """ Ejecuta el proceso de investigación profunda, generando actualizaciones de estado y el informe final """
    print("Iniciando investigación...")
    search_plan = await plan_searches(query)
    yield "Búsquedas planificadas, iniciando búsqueda..."     
    search_results = await perform_searches(search_plan)
    yield "Búsquedas completadas, redactando informe..."
    report = await write_report(query, search_results)
    yield "Informe redactado, enviando correo..."
    await send_report(report, to, name)
    yield "Correo enviado, investigación completada"
    yield report.markdown_report

# %%
with gr.Blocks(theme=gr.themes.Default(primary_hue="sky")) as ui:
    gr.Markdown("# Investigación Profunda")
    query_textbox = gr.Textbox(label="¿Sobre qué tema te gustaría investigar?")
    query_email = gr.Textbox(label="¿Cuál es tu correo electrónico?")
    query_name = gr.Textbox(label="¿Cuál es tu nombre? (opcional)")
    run_button = gr.Button("Ejecutar", variant="primary")
    report = gr.Markdown(label="Informe")
    
    run_button.click(fn=run, inputs=[query_textbox, query_email, query_name], outputs=report)
    query_textbox.submit(fn=run, inputs=query_textbox, outputs=report)

ui.launch(inbrowser=True)


