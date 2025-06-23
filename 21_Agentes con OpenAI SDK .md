### OpenAI SDK Agents

El SDK Agents es una herramienta que permite crear agentes de IA personalizados utilizando los modelos de OpenAI. Algunos proveedores de modelos de lenguaje como Google Gemini pueden usar esta API con soporte limitado.

SDK Agents tiene pocas abstracciones y es fácil de usar. El Agents SDK tiene un conjunto muy pequeño de primitivas:

* Agentes (`Agents`), que son LLMs equipados con instrucciones y herramientas.
* Delegaciones (`Handoffs`), que permiten a los agentes delegar tareas específicas a otros agentes.
* Barreras de seguridad (`Guardrails`), que permiten validar las entradas y salidas de los agentes.

Además, el SDK viene con trazabilidad incorporada que permite visualizar y depurar los flujos de agentes.

En esta práctica vamos a utilizar el SDK Agents para crear una aplicación que genere un informe profesional a partir de una pregunta formulada por el usuario. Varios agentes se coordinarán para realizar una búsqueda en Internet y componer un informe que será enviado al usuario.

Realizamos las importacines necesarias


```python
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
```

    /home/surtich/projects/IA para desarrolladores/.venv/lib/python3.13/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


### `Searcher Agent`

Creamos el primer agente que será capaz de buscar información en Internet. Este agente realiza la búsqueda usando Google Gemini. Para hacer búsquedas en Internet con el SDK Agents hay que usar la clase `WebSearchAgent`. Pero esta clase no es compatible con `OpenAIChatCompletionsModel`. `OpenAIChatCompletionsModel` es una función que se debe usar cuando se crean agentes con un proveedor distinto de OpenAI. Por lo tanto, se ha tenido que utilizar otra estrategia para poder realizar búsquedas en Internet. Se ha realizado usando la API de Google Gemini llamada Google GenAI. Los pasos son los siguientes:

* 1 Importamos la clave de Google Gemini.
* 2 Creamos el modelo que se usará para crear los agentes. Normalmente este modelo sería una simple cadena de texto con el nombre del modelo. Pero cuando el modelo no es de OpenAI el proceso es un poco más complicado.
* 3 Creamos el cliente que se conectará a la API de Google GenAI.
* 4 Creamos la `tool` de GenAI que se usará para realizar las búsquedas en Internet.
* 5 Creamos una función de búsqueda que utiliza la `tool`.
* 6 ...

... Aunque no hayamos terminado de crear el agente, tenemos la función de búsqueda que se usará en el agente y podemos probarla.


```python
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
```

    La clave de API de Google existe y comienza con AI



```python
response = search_term("Día y hora actual en Madrid (España) y previsión del tiempo para los próximos 3 días.")
display(Markdown(response.text))
```


En Madrid, España, el día y la hora actuales son el miércoles 25 de junio de 2025, a las 12:54 PM (hora de verano de Europa Central, CEST).

Para los próximos 3 días (miércoles 25, jueves 26 y viernes 27 de junio), se espera un tiempo mayormente soleado. Las temperaturas diurnas irán en aumento, alcanzando máximas de 28°C a 35°C (83°F a 97°F). Las temperaturas mínimas nocturnas oscilarán entre los 15°C y los 25°C (59°F y 77°F). La probabilidad de lluvia será muy baja, cercana al 0%.


... continuamos:

* 6 Convertimos la función de búsqueda en una `tool` que se usará en el agente. Para ello usamos el decorador `@function_tool`.
* 7 ...


```python
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
```


```python
# search_term ya no es una función normal de Python sino una tool que se puede usar en un agente.
search_term
```




    FunctionTool(name='search_term', description='Realiza una búsqueda en Google y devuelve los resultados.', params_json_schema={'properties': {'term': {'title': 'Term', 'type': 'string'}}, 'required': ['term'], 'title': 'search_term_args', 'type': 'object', 'additionalProperties': False}, on_invoke_tool=<function function_tool.<locals>._create_function_tool.<locals>._on_invoke_tool at 0x7f816367ba60>, strict_json_schema=True, is_enabled=True)



* 7 Creamos un agente que usa la `tool` de búsqueda. Con `ModelSettings` forzamos al agente a que use la `tool` siempre.


```python
# 7.- Creamos un agente que usa la `tool` de búsqueda
search_agent = Agent(
    name="Agente de Búsqueda",
    instructions=INSTRUCTIONS,
    tools=[search_term],
    model=chat_model,
    model_settings=ModelSettings(tool_choice="required"),
)
```

* 8 Probamos el agente. Está metido en un contexto `trace` que permite trazabilidad. La trazabilidad es muy fácil de implementar con OpenAI Agents SDK pero no he sido capaz de configurarlo en Google Gemini.


```python
message = "Frameworks de agentes de IA nuevos y más populares en el año actual"

with trace("Search"):
    result = await Runner.run(search_agent, message)

display(Markdown(result.final_output))
```

    OPENAI_API_KEY is not set, skipping trace export



Los frameworks de agentes de IA están evolucionando para crear sistemas inteligentes capaces de razonar, tomar decisiones y ejecutar tareas complejas con mínima intervención humana. La tendencia actual, especialmente para 2025, se centra en soluciones que facilitan la orquestación de múltiples agentes, la gestión de estado y memoria, la integración con APIs y sistemas externos, y la capacidad de aprendizaje y adaptación. Estos frameworks son esenciales para automatizar flujos de trabajo, mejorar la toma de decisiones y permitir la colaboración multi-agente en tareas complejas.

Entre los frameworks más populares y destacados en el año actual se encuentran LangChain, AutoGen y Semantic Kernel. LangChain es valorado por su flexibilidad en la creación de aplicaciones basadas en modelos de lenguaje, permitiendo la definición de agentes con herramientas específicas, cadenas de razonamiento y memorias. Su extensión LangGraph es clave para arquitecturas multi-agente con flujos de estado complejos. AutoGen, de Microsoft, es popular para sistemas multi-agente autónomos o asistidos por humanos, con APIs flexibles y una interfaz gráfica (AutoGen Studio) para prototipado y evaluación. Semantic Kernel, también de Microsoft, es ideal para integrar agentes de IA en el ecosistema de herramientas de Microsoft.

Otros frameworks notables incluyen CrewAI, LlamaIndex (especialmente para RAG y consulta de datos privados), OpenAI Agents SDK, Haystack y OpenDevin.


    OPENAI_API_KEY is not set, skipping trace export
    OPENAI_API_KEY is not set, skipping trace export
    OPENAI_API_KEY is not set, skipping trace export
    OPENAI_API_KEY is not set, skipping trace export
    OPENAI_API_KEY is not set, skipping trace export
    OPENAI_API_KEY is not set, skipping trace export
    OPENAI_API_KEY is not set, skipping trace export
    OPENAI_API_KEY is not set, skipping trace export
    OPENAI_API_KEY is not set, skipping trace export
    OPENAI_API_KEY is not set, skipping trace export


### `Planner Agent`

Este agente, dado un término de búsqueda, propone un conjunto de preguntas que se pueden hacer para afinar la búsqueda. En la construcción del agente se utiliza el parámetro `output_type` para especificar el formato de salida esperado. A esto se le llama salidas estructuradas. En este caso, se espera una lista de preguntas, donde cada pregunta tiene los campos `reason` y `query`. Observe que el esquema se define usando `Pydantic` y es el agente el que tiene que entender el esquemea y devolver la salida en el formato correcto.


```python
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
```

Probamos el agente. Observe que propone tres preguntas con el esquema definido.


```python
message = "Frameworks de agentes de IA nuevos y más populares en el año actual"

with trace("Plan"):
    result = await Runner.run(planner_agent, message)

result.final_output
```




    WebSearchPlan(searches=[WebSearchItem(reason='Búsqueda directa de frameworks de agentes de IA relevantes para el año en curso.', query='frameworks agentes IA 2024'), WebSearchItem(reason='Para identificar herramientas y librerías de código abierto que puedan estar ganando popularidad.', query='novedades agentes IA open source'), WebSearchItem(reason='Python es un lenguaje popular para IA. Identificar frameworks basados en Python es crucial.', query='mejores frameworks agentes IA python')])



### Agente `Emailer Agent`

Este agente es capaz de enviar correos electrónicos. Pare ello creamos la `tool` `send_email` que se encargará de enviar el correo electrónico utilizando la API de Mailgun.


```python
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
```


```python
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
```

Probamos el agente. Revisar la carpeta de `spam` porque suele llegar allí.


```python
with trace("Email"):
    params = json.dumps({
        "subject": "Informe de investigación sobre frameworks de agentes de IA",
        "html_body": "bla **bla** bla",
        "to": "surtich@gmail.com",
        "name": "Javier"
    })
    result = await Runner.run(email_agent, params)
```

### `Writer Agent`

Este agente es capaz de redactar un informe profesional a partir de un conjunto de preguntas y respuestas.


```python
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
```

### Funciones auxiliares

Las siguientes funciones permiten conectar los agentes entre sí. El SDK Agents es de bajo nivel y este proceso hay que hacerlo manualmente.

La `plan_searches` utilza el planificador para generar un conjunto de preguntas a partir de un término de búsqueda.


```python
async def plan_searches(query: str) -> WebSearchPlan:
    """ Utiliza el planner_agent para planificar qué búsquedas realizar para la consulta """
    print("Planificando búsquedas...")
    result = await Runner.run(planner_agent, f"Consulta: {query}")
    print(f"Se realizarán {len(result.final_output.searches)} búsquedas")
    return result.final_output_as(WebSearchPlan) # no es estrictamente necesario. Se puede ejecutar simplemente result.final_output
```

Observe que la función `plan_searches` es una corutina.


```python
query = "Informe sobre institutos públicos de la Comunidad de Madrid que imparten DAW. Nombre de los institutos, reseñas, ..."
plan_searches(query)
```




    <coroutine object plan_searches at 0x7f813ad6b220>



Para ejecutala hay que usar:

* `asyncio.run(plan_searches("Python programming language"))` si estamos en un programa de Python.
* `await plan_searches("Python programming language")` si estamos en un entorno asíncrono como Jupyter Notebook.


```python
search_plan = await plan_searches(query)
search_plan
```

    Planificando búsquedas...
    Se realizarán 3 búsquedas





    WebSearchPlan(searches=[WebSearchItem(reason='Busca listados de institutos públicos en la Comunidad de Madrid que ofrecen el ciclo formativo de grado superior de Desarrollo de Aplicaciones Web (DAW).', query='institutos públicos DAW Comunidad Madrid'), WebSearchItem(reason='Busca reseñas y opiniones sobre la calidad de la enseñanza de DAW en institutos públicos de Madrid, proporcionando información sobre la experiencia de otros estudiantes.', query='reseñas institutos públicos DAW Madrid'), WebSearchItem(reason='Busca rankings o comparativas de los mejores institutos públicos en Madrid para estudiar DAW, lo que podría ayudar a identificar los centros con mejor reputación.', query='ranking mejores institutos DAW Madrid')])



La función `perform_searches` utiliza la función `search` para realizar las búsquedas en Internet.


```python
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
```

Probamos


```python
search_results = await perform_searches(search_plan)
search_results
```

    Buscando...
    Buscando... 1/3 completado
    Buscando... 2/3 completado
    Buscando... 3/3 completado
    Búsquedas finalizadas





    ['Varios institutos en Madrid destacan para estudiar el Grado Superior en Desarrollo de Aplicaciones Web (DAW). Entre los más mencionados están IFP, Universidad Europea, UCJC, CES, MEDAC, CESUR, Linkia FP y CEU FP.\n\nOtros centros que también se mencionan son Davante Medac, MasterD Davante, Upgrade Hub, DIGITECH y Sinergia FP. La elección del mejor centro depende de las preferencias individuales, como la modalidad de estudio (presencial u online), el precio y las opiniones de otros estudiantes.\n\nSe recomienda investigar a fondo cada opción para encontrar el centro que mejor se adapte a las necesidades y objetivos profesionales del estudiante.\n',
     'Las reseñas sobre institutos públicos que imparten DAW en Madrid varían. Factores importantes incluyen la calidad del profesorado, la actualización de los contenidos, la metodología de enseñanza, la infraestructura y las prácticas.\n\nLa FP pública es más económica, incluso gratuita en algunas comunidades. Algunos usuarios señalan que la calidad de la enseñanza online puede ser inferior a la privada, con profesores menos involucrados y falta de corrección de ejercicios. La FP pública presencial puede ser una buena opción. Algunos institutos públicos mencionados son el IES Alonso de Avellaneda, el IES Virgen de la Paz, el IES Clara del Rey y el IES Islas Filipinas.\n\nTambién se mencionan centros privados, algunos con becas por excelencia académica, como Davante Medac, Upgrade Hub, DIGITECH y Sinergia FP. La elección depende de recursos, preferencias y necesidades individuales.\n',
     'El ciclo formativo de grado superior en Desarrollo de Aplicaciones Web (DAW) se ofrece en varios institutos públicos de la Comunidad de Madrid. Estos estudios capacitan al alumno para desarrollar, implantar y mantener aplicaciones web, garantizando el acceso seguro a los datos y cumpliendo con los estándares de accesibilidad, usabilidad y calidad.\n\nPara acceder a estos estudios, se requiere el título de Bachiller, Técnico Especialista, Técnico Superior o equivalente. También se puede acceder mediante la superación de una prueba de acceso para mayores de 19 años o para mayores de 18 años con el título de Técnico de grado medio. La duración del ciclo es de 2.000 horas, distribuidas en dos cursos académicos, incluyendo la Formación en Centros de Trabajo (FCT).\n\nAlgunos de los institutos públicos que imparten DAW en la Comunidad de Madrid son el IES Alonso de Avellaneda (Alcalá de Henares), IES Virgen de la Paz (Alcobendas), IES Josefina Aldecoa (Alcorcón), IES Ángel Corella (Colmenar Viejo), IES Lázaro Cárdenas (Collado Villalba), IES Luis Braille (Coslada), IES Federica Montseny (Fuenlabrada), IES Alarnes (Getafe), IES Satafi (Getafe), IES Luis Vives (Leganés) e IES Pablo Neruda (Leganés).\n']



La siguiente función escribe el informe a partir de las preguntas y respuestas obtenidas.


```python
async def write_report(query: str, search_results: list[str]):
    """ Utiliza el agente redactor para escribir un informe basado en los resultados de búsqueda """
    print("Pensando en el informe...")
    input = f"Consulta original: {query}\nResultados de búsqueda resumidos: {search_results}"
    result = await Runner.run(writer_agent, input)
    print("Informe terminado")
    return result.final_output
```

Probamos


```python
await write_report(query, search_results)
```

    Pensando en el informe...
    Informe terminado





    ReportData(short_summary='Este informe resume los institutos públicos en la Comunidad de Madrid que imparten el ciclo formativo de grado superior en Desarrollo de Aplicaciones Web (DAW).  Identifica algunos institutos, resume las reseñas disponibles y describe el plan de estudios, destacando la importancia de la investigación individual para la elección del centro educativo.', markdown_report='# Informe sobre Institutos Públicos de la Comunidad de Madrid que imparten DAW\n\n## Introducción\n\nEste informe tiene como objetivo proporcionar una visión general de los institutos públicos en la Comunidad de Madrid que ofrecen el ciclo formativo de grado superior en Desarrollo de Aplicaciones Web (DAW). El informe abordará la identificación de estos institutos, las reseñas y opiniones disponibles sobre ellos, las características del plan de estudios y otros aspectos relevantes para los estudiantes interesados en esta formación.\n\n## Metodología\n\nLa información contenida en este informe se ha recopilado a partir de diversas fuentes, incluyendo:\n\n*   Resultados de búsqueda en internet.\n*   Bases de datos de la Consejería de Educación de la Comunidad de Madrid (si es posible acceder a ellas).\n*   Foros y comunidades online de estudiantes de FP.\n*   Reseñas y opiniones de usuarios en plataformas como Google Maps y portales educativos.\n\n## Institutos Públicos que imparten DAW en la Comunidad de Madrid\n\nSegún los resultados de la búsqueda inicial, varios institutos públicos en la Comunidad de Madrid ofrecen el ciclo formativo de DAW. Algunos de los más destacados son:\n\n*   **IES Alonso de Avellaneda (Alcalá de Henares)**\n*   **IES Virgen de la Paz (Alcobendas)**\n*   **IES Josefina Aldecoa (Alcorcón)**\n*   **IES Ángel Corella (Colmenar Viejo)**\n*   **IES Lázaro Cárdenas (Collado Villalba)**\n*   **IES Luis Braille (Coslada)**\n*   **IES Federica Montseny (Fuenlabrada)**\n*   **IES Alarnes (Getafe)**\n*   **IES Satafi (Getafe)**\n*   **IES Luis Vives (Leganés)**\n*   **IES Pablo Neruda (Leganés)**\n*   **IES Clara del Rey (Madrid)**\n*   **IES Islas Filipinas (Madrid)**\n\nEs importante señalar que esta lista no es exhaustiva y puede haber otros institutos públicos que también ofrezcan este ciclo formativo. Se recomienda consultar la página web de la Consejería de Educación de la Comunidad de Madrid para obtener una lista completa y actualizada.\n\n## Reseñas y Opiniones sobre los Institutos Públicos\n\nLas reseñas y opiniones sobre los institutos públicos que imparten DAW en Madrid varían considerablemente. Algunos factores importantes que influyen en la percepción de los estudiantes son:\n\n*   **Calidad del profesorado:** La experiencia, la dedicación y la capacidad de transmitir los conocimientos por parte de los profesores son aspectos cruciales.\n*   **Actualización de los contenidos:** El sector del desarrollo web está en constante evolución, por lo que es fundamental que los contenidos del ciclo formativo estén actualizados y reflejen las últimas tecnologías y tendencias.\n*   **Metodología de enseñanza:** La combinación de clases teóricas, prácticas en laboratorio y proyectos reales es esencial para un aprendizaje efectivo.\n*   **Infraestructura:** La disponibilidad de equipos informáticos modernos, software actualizado y acceso a internet de alta velocidad son importantes para el desarrollo de las actividades prácticas.\n*   **Prácticas en empresas (FCT):** La calidad y la relevancia de las prácticas en empresas son determinantes para la adquisición de experiencia laboral y la inserción en el mercado laboral.\n\n**Ventajas de la FP Pública:**\n\n*   **Coste:** La FP pública es significativamente más económica que la privada, e incluso gratuita en algunas comunidades autónomas.\n*   **Acceso:** Los requisitos de acceso son generalmente menos exigentes que en las universidades.\n\n**Desventajas de la FP Pública:**\n\n*   **Recursos:** Algunos usuarios señalan que la FP pública puede tener menos recursos que la privada, lo que podría afectar a la calidad de la enseñanza en algunos casos.\n*   **Profesorado:** Existe la percepción de que algunos profesores en la FP pública pueden estar menos involucrados que en la privada, aunque esto no es una generalidad.\n*   **Tiempos de espera:** Puede haber listas de espera para acceder a determinados ciclos formativos en institutos públicos populares.\n\n**Ejemplos de reseñas (basadas en la información disponible):**\n\n*   **IES Alonso de Avellaneda:** Se destaca por su larga trayectoria y experiencia en la impartición de ciclos formativos de informática. Algunos usuarios valoran positivamente la dedicación de los profesores y la calidad de las prácticas en empresas.\n*   **IES Virgen de la Paz:** Se menciona por su buena ubicación y sus instalaciones modernas. Algunos estudiantes destacan la variedad de actividades complementarias que se ofrecen.\n*   **IES Clara del Rey:** Algunos usuarios señalan que los contenidos del ciclo formativo están actualizados y que los profesores están al día de las últimas tendencias del sector.\n\nEs importante tener en cuenta que estas son solo algunas reseñas y opiniones de usuarios. Se recomienda investigar a fondo cada instituto y contactar con antiguos alumnos para obtener una visión más completa y precisa.\n\n## Plan de Estudios del Ciclo Formativo de DAW\n\nEl ciclo formativo de grado superior en Desarrollo de Aplicaciones Web (DAW) tiene una duración de 2.000 horas, distribuidas en dos cursos académicos. El plan de estudios incluye los siguientes módulos profesionales:\n\n*   **Desarrollo web en entorno cliente:** Este módulo se centra en el aprendizaje de lenguajes de programación como HTML, CSS y JavaScript, así como en el uso de frameworks y librerías como React, Angular o Vue.js.\n*   **Desarrollo web en entorno servidor:** Este módulo aborda el desarrollo de aplicaciones web del lado del servidor, utilizando lenguajes como PHP, Java o Python, y frameworks como Laravel, Spring o Django.\n*   **Diseño de interfaces web:** Este módulo se centra en el diseño de interfaces de usuario atractivas y usables, teniendo en cuenta los principios de la experiencia de usuario (UX) y la accesibilidad.\n*   **Acceso a datos:** Este módulo enseña cómo acceder a bases de datos y manipular datos utilizando lenguajes como SQL y tecnologías como MySQL, PostgreSQL o MongoDB.\n*   **Despliegue de aplicaciones web:** Este módulo cubre el proceso de despliegue de aplicaciones web en servidores web como Apache o Nginx, así como el uso de plataformas de cloud computing como Amazon Web Services o Microsoft Azure.\n*   **Seguridad en aplicaciones web:** Este módulo aborda las principales vulnerabilidades de seguridad en aplicaciones web y las técnicas para prevenirlas.\n*   **Formación en Centros de Trabajo (FCT):** Este módulo consiste en la realización de prácticas en empresas del sector, lo que permite a los estudiantes adquirir experiencia laboral y aplicar los conocimientos aprendidos en el aula.\n*   **Empresa e iniciativa emprendedora:** Módulo destinado a proporcionar los conocimientos necesarios para crear y gestionar una empresa, así como para desarrollar habilidades emprendedoras.\n*   **Inglés técnico para el desarrollo de aplicaciones web:** Este módulo se centra en el aprendizaje del inglés técnico específico del sector del desarrollo web, lo que permite a los estudiantes comprender la documentación técnica y comunicarse con profesionales de otros países.\n\n**Requisitos de Acceso:**\n\nPara acceder al ciclo formativo de DAW, se requiere uno de los siguientes requisitos:\n\n*   Título de Bachiller.\n*   Título de Técnico Especialista o Técnico Superior.\n*   Título de Técnico de grado medio (requiere superar una prueba de acceso).\n*   Superar una prueba de acceso para mayores de 19 años (o mayores de 18 años si se posee el título de Técnico de grado medio).\n\n## Consideraciones Finales y Recomendaciones\n\nLa elección del instituto público para estudiar DAW es una decisión importante que debe basarse en una investigación exhaustiva y en la consideración de las necesidades y preferencias individuales de cada estudiante. Se recomienda:\n\n*   Visitar los institutos y hablar con profesores y alumnos.\n*   Investigar la calidad del profesorado y la actualización de los contenidos.\n*   Informarse sobre las oportunidades de prácticas en empresas.\n*   Considerar la ubicación y las instalaciones del instituto.\n*   Leer reseñas y opiniones de otros estudiantes.\n\nAdemás de los institutos públicos, también existen centros privados que ofrecen el ciclo formativo de DAW. Estos centros suelen tener un coste más elevado, pero pueden ofrecer ventajas como una mayor atención personalizada, instalaciones más modernas y una oferta de actividades complementarias más amplia. La elección entre un instituto público y un centro privado dependerá de los recursos económicos y las prioridades de cada estudiante.\n\n## Próximos Pasos\n\nPara complementar este informe, se sugiere realizar una investigación más profunda sobre los siguientes aspectos:\n\n*   Análisis comparativo de los planes de estudio de DAW en diferentes institutos públicos.\n*   Evaluación de la calidad del profesorado en cada instituto.\n*   Identificación de las empresas que colaboran con los institutos para las prácticas de los alumnos.\n*   Seguimiento de la inserción laboral de los graduados de DAW.\n*   Recopilación de información actualizada sobre las becas y ayudas disponibles para estudiantes de FP.\n\n\nEste informe pretende ser un punto de partida para aquellos interesados en estudiar DAW en un instituto público de la Comunidad de Madrid. Se espera que la información proporcionada les sea útil para tomar una decisión informada y elegir la mejor opción para su futuro profesional.', follow_up_questions=['¿Cuáles son los criterios específicos utilizados para evaluar la calidad del profesorado en los institutos públicos que imparten DAW?', '¿Cómo se comparan los planes de estudio de DAW en diferentes institutos públicos de la Comunidad de Madrid?', '¿Qué tipo de convenios tienen los institutos públicos con empresas para las prácticas de los alumnos de DAW?', '¿Cuáles son las tasas de empleo de los graduados de DAW de los institutos públicos en la Comunidad de Madrid?', '¿Cómo han evolucionado los contenidos del ciclo formativo de DAW en los últimos años para adaptarse a las nuevas tecnologías?'])



Por último, escribimos la función que envía el correo electrónico con el informe generado.


```python
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
```

Probamos


```python
report = ReportData(
    short_summary="Resumen breve de los hallazgos del informe.",
    markdown_report="Informe sobre institutos públicos de la Comunidad de Madrid que imparten DAW\n\nEste es un informe detallado...",
    follow_up_questions=["¿Qué otros institutos ofrecen DAW?", "¿Cuáles son las tasas de empleo de los graduados?"]
)
await send_report(report, "surtich@gmail.com")
```

    Redactando correo...
    Correo enviado





    ReportData(short_summary='Resumen breve de los hallazgos del informe.', markdown_report='Informe sobre institutos públicos de la Comunidad de Madrid que imparten DAW\n\nEste es un informe detallado...', follow_up_questions=['¿Qué otros institutos ofrecen DAW?', '¿Cuáles son las tasas de empleo de los graduados?'])



### Prueba de la aplicación

Finalmente, probamos la aplicación completa. Creamos función que coordina todos los agentes anteriores y envía el informe por correo electrónico.


```python
query = "Aplicaciones del lenguaje Python para Administradores de Sistemas en 2025"
to = "surtich@gmail.com"

with trace("Rastreo de investigación"):
    print("Iniciando investigación...")
    search_plan = await plan_searches(query)
    search_results = await perform_searches(search_plan)
    report = await write_report(query, search_results)
    await send_report(report, to)
    print("¡Hurra!")
```

    Iniciando investigación...
    Planificando búsquedas...
    Se realizarán 3 búsquedas
    Buscando...
    Buscando... 1/3 completado
    Buscando... 2/3 completado
    Buscando... 3/3 completado
    Búsquedas finalizadas
    Pensando en el informe...
    Informe terminado
    Redactando correo...
    Correo enviado
    ¡Hurra!


### Integración con Gradio

En este caso no podemos usar la función `ChatInterface` de Gradio porque necesitamos recoger la entrada del usuario. Creamos una función `run` y se la asociamos a la función `launch`. La función `run` emite (`yield`) valores que Gradio mostrará en la interfaz.




```python
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
```


```python
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
```

    /home/surtich/projects/IA para desarrolladores/.venv/lib/python3.13/site-packages/gradio/utils.py:1042: UserWarning: Expected at least 2 arguments for function <function run at 0x7ff4b5940680>, received 1.
      warnings.warn(


    * Running on local URL:  http://127.0.0.1:7860
    * To create a public link, set `share=True` in `launch()`.



<div><iframe src="http://127.0.0.1:7860/" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>





    



    Iniciando investigación...
    Planificando búsquedas...
    Se realizarán 3 búsquedas
    Buscando...
    Buscando... 1/3 completado
    Buscando... 2/3 completado
    Buscando... 3/3 completado
    Búsquedas finalizadas
    Pensando en el informe...
    Informe terminado
    Redactando correo...
    Correo enviado

