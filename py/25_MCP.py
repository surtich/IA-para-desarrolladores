# %% [markdown]
# ### Model Context Protocol
# 
# El ***Protocolo de Contexto de Modelo** (MCP, por sus siglas en inglés) es un estándar abierto desarrollado por Anthropic que busca estandarizar la forma en que los LLM interactúan con fuentes de datos y herramientas externas. MCP permite la integración de tools de forma sencilla, escalable y estandarizada y segura.
# 
# **Componentes de MCP**
# 
# - **MCP Hosts**: Aplicaciones o entornos donde los usuarios interactúan con modelos de IA y desean acceder a datos o herramientas externas a través de MCP (por ejemplo, Claude Desktop, IDEs, asistentes personalizados).
# - **MCP Clients**: Componentes ligeros dentro del host que mantienen conexiones 1:1 con servidores MCP, gestionando la comunicación y transmisión de solicitudes y datos.
# - **MCP Servers**: Programas independientes que exponen capacidades concretas (herramientas, recursos o prompts) a través del protocolo MCP estandarizado, pudiendo ejecutarse localmente o en la nube.
# - **Local Data Sources**: Archivos, bases de datos y servicios que residen en el ordenador del usuario y a los que los servidores MCP pueden acceder de forma segura.
# - **Remote Services**: Sistemas externos accesibles a través de Internet (por ejemplo, mediante APIs) a los que los servidores MCP pueden conectarse para obtener datos o ejecutar acciones.
# 

# %% [markdown]
# ```mermaid
# %%{init: {"themeVariables": {
#   "fontFamily": "Inter, Arial, sans-serif",
#   "clusterBkg": "#fff9c4",
#   "clusterBorder": "#bdb76b",
#   "clusterTextColor": "#000",
#   "nodeTextColor": "#000",
#   "nodeBorder": "#888",
#   "nodeBkg": "#fff"
# }}}%%
# flowchart LR
#     subgraph local["Your Computer"]
#         Host["Host with MCP Client\n(Claude, IDEs, Tools)"]
#         S1["MCP Server A"]
#         S2["MCP Server B"]
#         S3["MCP Server C"]
#         Host <-->|"MCP Protocol"| S1
#         Host <-->|"MCP Protocol"| S2
#         Host <-->|"MCP Protocol"| S3
#         S1 <--> D1[("Local\nData Source A")]
#         S2 <--> D2[("Local\nData Source B")]
#     end
#     subgraph remote["Internet"]
#         S3 <-->|"Web APIs"| D3[("Remote\nService C")]
#     end
#     style local color:#000;
#     style remote color:#000;
# 
# ```

# %% [markdown]
# ### Primer Servidor MCP
# 
# Vamos a crear un primer servidor MCP que exponga una herramienta simple para sumar dos números. Este servidor estará implementado en Python utilizando el paquete `FastMCP`.

# %% [markdown]
# Primero creamos el servidor MCP:

# %%
import os
os.makedirs("25_MCP/01_FirstMCPServer", exist_ok=True)

# %%
%%writefile "25_MCP/01_FirstMCPServer/server.py"
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Demo Server")


@mcp.tool(description="Add two integers")
def add(a: int, b: int) -> int:
    return a + b


if __name__ == "__main__":
    mcp.run(transport="streamable-http")

# %% [markdown]
# Creamos el cliente.

# %%
%%writefile "25_MCP/01_FirstMCPServer/client.py"
import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def main():
    url = "http://127.0.0.1:8000/mcp/"
    async with streamablehttp_client(url) as (read, write, get_session_id):
        async with ClientSession(read, write) as session:
            print("Before initialize:", get_session_id())

            await session.initialize()

            sid = get_session_id()
            print("Session ID after initialize:", sid)

            result = await session.call_tool("add", {"a": 21, "b": 21})
            print("Server result:", result)


if __name__ == "__main__":
    asyncio.run(main())

# %% [markdown]
# Para probar, situarse en el directorio y arrancar el servidor y el cliente en dos terminales diferentes.
# 
# ```python
# uv run server.py
# ```
# 
# ```python
# uv run client.py
# ```

# %% [markdown]
# También podemos ejecutar la tool desde la línea de comandos.

# %%
%%writefile "25_MCP/01_FirstMCPServer/script.sh"
#!/usr/bin/env bash
set -euo pipefail
S=http://127.0.0.1:8000/mcp/
ACCEPT='application/json, text/event-stream'
CT='application/json'

# 1) initialize
SID=$(curl -sS -D - -o /dev/null \
  -H "Accept: $ACCEPT" -H "Content-Type: $CT" \
  -X POST $S \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{
        "protocolVersion":"2025-03-26",
        "capabilities":{},
        "clientInfo":{"name":"bash","version":"1.0"}
      }}' | sed -nE 's/^Mcp-Session-Id:[[:space:]]*//Ip' | tr -d '\r')
echo "SID=$SID"

# 2) notifications/initialized
curl -sS \
  -H "Accept: $ACCEPT" \
  -H "Content-Type: $CT" \
  -H "Mcp-Session-Id: $SID" \
  -X POST $S \
  -d '{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}'

# 3) tools/call
curl -sS \
  -H "Accept: $ACCEPT" -H "Content-Type: $CT" -H "Mcp-Session-Id: $SID" \
  -X POST $S \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{
        "name":"add","arguments":{"a":2,"b":3}}}'
echo

# %% [markdown]
# Además, podemos probar el servidor MCP desde el inspector de MCP. Para ello, ejecutar en el terminal el siguiente comando.
# 
# ```bash
# DANGEROUSLY_OMIT_AUTH=true npx @modelcontextprotocol/inspector
# ```

# %% [markdown]
# O podemos ejecutar el inspector sin la variable de entorno `DANGEROUSLY_OMIT_AUTH`. En ese caso se ejecutaría así.
# 
# ```bash
# npx @modelcontextprotocol/inspector
# ```
# 
# Para poder conectar con el servidor MCP, hay que copiar el valor de la variable `MCP_PROXY_AUTH_TOKEN` que se genera en la línea de comandos al arrancar el inspector y copiarlo en el inspector pulsado sobre el botón `Configuration` y pegarlo en la caja `Proxy Session Token`.

# %% [markdown]
# También, podemos probar la Tool desde el Chat de Copilot si elegimos el Agente y luego añadimos la tool pulsado sobre el icono "Configurar herramientas". Actualmente VS Code no soporta el transporte `Streamable HTTP`. Si se desea probar, hay que cambiar el transporte a `SSE` o a `stdio`.Aunque el primero de ellos, está obsoleto y no se debe usar.

# %% [markdown]
# Por último, se puede configurar el servidor MCP en algunas herramientas de chat como Claude Desktop o Cherry Studio.

# %% [markdown]
# ### Transports
# 
# MCP utiliza JSON-RPC 2.0 como formato de mensajes. El transporte convierte los mensajes del protocolo MCP a JSON-RPC para transmitirlos y viceversa al recibirlos.
# 
# Hay tres tipos de mensajes:
# - **Requests (Solicitudes):** Incluyen método, parámetros e identificador.
# - **Responses (Respuestas):** Devuelven resultados o errores asociados a una solicitud.
# - **Notifications (Notificaciones):** Mensajes sin respuesta esperada.
# 
# 
# **Tipos de transporte integrados**
# 
# Actualmente, MCP define dos mecanismos estándar:
# 
# 1. **Standard Input/Output (stdio):**
#    - Usa los flujos estándar de entrada y salida del sistema operativo.
#    - Es útil para herramientas de línea de comandos, integraciones locales y scripts.
#    - Suele usarse para pruebas en la fase de desarrollo.
#    - Ejemplo: conectar un cliente o servidor MCP usando procesos locales y stdio[1][3].
# 
# 2. **Streamable HTTP:**
#    - Utiliza peticiones HTTP POST para la comunicación cliente-servidor.
#    - Opcionalmente, emplea Server-Sent Events (SSE) para transmitir mensajes del servidor al cliente.
#    - Soporta sesiones con estado, múltiples clientes concurrentes y conexiones reanudables.
#    - Permite la persistencia de sesión mediante un header `Mcp-Session-Id` y la reanudación de mensajes perdidos usando `Last-Event-ID`.

# %% [markdown]
# En el siguiente ejemplo, se muestra cómo crear un servidor MCP que no usa sesiones. Vemos que el cliente es mucho más sencillo.

# %%
import os
os.makedirs("25_MCP/02_TransportMethods/streamable_http", exist_ok=True)

# %%
%%writefile "25_MCP/02_TransportMethods/streamable_http/server.py"
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Demo Server", stateless_http=True)


@mcp.tool(description="Add two integers")
def add(a: int, b: int) -> int:
    return a + b


if __name__ == "__main__":
    mcp.run(transport="streamable-http")

# %%
%%writefile "25_MCP/02_TransportMethods/streamable_http/client.py"
import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def main():
    url = "http://127.0.0.1:8000/mcp/"
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            # await session.initialize()            # JSON-RPC „initialize“
            result = await session.call_tool("add", {"a": 21, "b": 21})
            print("Server result:", result)


if __name__ == "__main__":
    asyncio.run(main())

# %% [markdown]
# En el siguiente ejemplo se usa el transporte `stdio`. En este caso no se debe arrancar el servidor, ya que se ejecuta el fichero `server.py` directamente desde el cliente.

# %%
import os
os.makedirs("25_MCP/02_TransportMethods/stdinout", exist_ok=True)

# %%
%%writefile "25_MCP/02_TransportMethods/stdinout/server.py"
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Add STDIO Server")


@mcp.tool(description="Add two integers")
def add(a: int, b: int) -> int:
    return a + b


if __name__ == "__main__":
    mcp.run(transport="stdio")

# %%
%%writefile "25_MCP/02_TransportMethods/stdinout/client.py"
import asyncio
import sys

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def main() -> None:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["server.py"],
        env=None,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.call_tool("add", {"a": 7, "b": 5})
            print("7 + 5 =", res.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())

# %% [markdown]
# ### Componentes MCP: Tools, Resources y Prompts
# ====================================================
# 
# El Model Context Protocol (MCP) organiza la interacción entre modelos de lenguaje, aplicaciones y usuarios en tres componentes esenciales, cada uno actuando en un nivel diferente del ecosistema de IA. La imagen proporcionada ilustra cómo estos elementos se conectan y fluyen dentro de una arquitectura típica MCP.
# 
# ```mermaid
# graph TD
#     Tools --> LLM
#     Resources --> Aplicación
#     Prompts --> Usuario
# 
#     style LLM fill:transparent,stroke-width:0
#     style Aplicación fill:transparent,stroke-width:0
#     style Usuario fill:transparent,stroke-width:0
# 
# ```
# 
# **Tools (Herramientas)**
# 
# Las tools son funciones o acciones que el modelo de lenguaje (LLM) puede invocar directamente. Están diseñadas para que el modelo ejecute tareas externas como consultar una API, modificar una base de datos o realizar cálculos.
# 
# - Ejemplo: El LLM llama a una tool para buscar información en una base de datos de clientes o para enviar un correo electrónico.
# 
# **Resources (Recursos)**
# 
# Los resources representan datos estructurados que la aplicación expone al LLM. Son gestionados y controlados por la propia aplicación, y sirven como entradas de solo lectura, similares a endpoints GET en una API REST. Incluyen archivos, logs, respuestas de APIs o cualquier fuente de datos relevante.
# 
# - Ejemplo: La aplicación expone logs recientes, archivos de configuración o información de usuario como recursos para que el modelo los utilice al responder una consulta.
# 
# **Prompts (Plantillas)**
# 
# Los prompts son plantillas reutilizables y predefinidas que estructuran las interacciones entre el usuario y el modelo. Facilitan la estandarización y reutilización de tareas comunes.
# 
# - Ejemplo: Un usuario selecciona el prompt "Analizar logs y código", que solicita los archivos y logs relevantes como argumentos, y guía al modelo a través de un flujo de análisis estructurado.

# %% [markdown]
# En el siguiente ejemplo, se muestra cómo crear un servidor MCP que expone tools, resources y prompts. Los resources pueden recibir parámetros.

# %%
import os
os.makedirs("25_MCP/03_RessourcesPromptsTools", exist_ok=True)

# %%
%%writefile "25_MCP/03_RessourcesPromptsTools/server.py"
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base

mcp = FastMCP("Recipe-Stateless", stateless_http=True)

_FAKE_DB = {
    "chili_con_carne": "Chili con Carne\n• Beans\n• Ground meat\n• Chili\n…",
    "pancakes": "Pancakes\n• Flour\n• Milk\n• Eggs\n…",
}


@mcp.resource("recipes://list")
def list_recipes() -> str:
    """Returns a comma-separated list of all available recipes."""
    return ", ".join(sorted(_FAKE_DB))


@mcp.resource("recipe://{dish}")
def get_recipe(dish: str) -> str:
    """Returns the recipe for the specified dish."""
    return _FAKE_DB.get(dish, f"No recipe found for {dish!r}.")


@mcp.tool(description="Doubles an integer.")
def double(n: int) -> int:
    return n * 2


@mcp.prompt()
def review_recipe(recipe: str) -> list[base.Message]:
    return [
        base.UserMessage("Please review this recipe:"),
        base.UserMessage(recipe),
    ]


if __name__ == "__main__":
    mcp.run(transport="streamable-http")

# %% [markdown]
# El cliente permite listar las tools, resources y prompts disponibles y utilizarlos. Observe que cuando el listado de resources no incluye los parametrizados.

# %%
%%writefile "25_MCP/03_RessourcesPromptsTools/client.py"
import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

SERVER = "http://127.0.0.1:8000/mcp/"


async def main() -> None:
    async with streamablehttp_client(SERVER) as (read, write, _):
        async with ClientSession(read, write) as session:
            resources = await session.list_resources()
            print("Resources:", [r.uri for r in resources.resources])

            tools = await session.list_tools()
            print("Tools:", [t.name for t in tools.tools])

            prompts = await session.list_prompts()
            print("Prompts:", prompts.prompts)

            recipe_response = await session.read_resource("recipe://chili_con_carne")
            recipe_text = recipe_response.contents[0].text
            print("\nRecipe:\n", recipe_text)

            doubled_response = await session.call_tool("double", {"n": 21})
            doubled_value = doubled_response.content[0].text
            print(f"\n21 doubled -> {doubled_value}")

            prompt_response = await session.get_prompt(
                "review_recipe",
                {"recipe": recipe_text},
            )
            print("\nPrompt messages:")
            for message in prompt_response.messages:
                print(f"[{message.role}] {message.content.text}")


if __name__ == "__main__":
    asyncio.run(main())

# %% [markdown]
# ### Context
# 
# El contexto en MCP permite que el servidor comunique información al cliente durante la ejecución de una solicitud. Esto es útil para proporcionar datos adicionales o resultados intermedios que el cliente puede necesitar para completar la tarea.

# %% [markdown]
# En el siguiente ejemplo, el servidor simula el procesamiento de una lista de elementos y la comunicación al cliente durante la ejecución de una solicitud. Observe que el cliente recibe dos tipos de mensajes: uno de tipo log y otro de tipo progress.
# 

# %%
import os
os.makedirs("25_MCP/04_Context", exist_ok=True)

# %%
%%writefile "25_MCP/04_Context/server.py"
import asyncio

from mcp.server.fastmcp import Context, FastMCP

mcp = FastMCP(
    name="ProgressDemoServer",
    stateless_http=False,
)


@mcp.tool(
    name="process_items", description="Processes a list of items with progress updates"
)
async def process_items(items: list[str], ctx: Context) -> list[str]:
    total = len(items)
    results: list[str] = []
    for i, item in enumerate(items, start=1):
        await ctx.info(f"Processing item {i}/{total}: {item}")
        await ctx.report_progress(progress=i, total=total)
        await asyncio.sleep(0.5)
        results.append(item.upper())
    return results


if __name__ == "__main__":
    mcp.run(transport="streamable-http")

# %% [markdown]
# El cliente usa un `handler` para cada uno de los tipos de mensajes enviados por el servidor. Observe también que se ha usado el paquete `fastmcp` en lugar de `mcp`, ya que no se ha conseguido que el paquete `mcp` funcione correctamente con context.

# %%
%%writefile "25_MCP/04_Context/client.py"
import asyncio

import mcp.types as types
from fastmcp import Client
from fastmcp.client.logging import LogMessage
from fastmcp.client.transports import StreamableHttpTransport


async def message_handler(msg):
    if not isinstance(msg, types.ServerNotification):
        return

    root = msg.root
    if isinstance(root, types.ProgressNotification):
        p = root.params
        print(f"[Progress] {p.progress}/{p.total or '?'}")


async def log_handler(params: LogMessage):
    level = params.level.upper()
    print(f"[Log – {level}] {params.data}")


async def main():
    transport = StreamableHttpTransport(url="http://127.0.0.1:8000/mcp/")
    client = Client(transport, message_handler=message_handler, log_handler=log_handler)

    async with client:
        tools = await client.list_tools()
        print("→ Available tools:", [t.name for t in tools])

        print("→ Calling process_items…")
        items = ["one", "two", "three", "four", "five"]
        result = await client.call_tool("process_items", {"items": items})
        processed = [c.text for c in result]
        print("→ Result:", processed)


if __name__ == "__main__":
    asyncio.run(main())

# %% [markdown]
# ### Tools dinámicas
# 
# En ocasiones, es necesario que el servidor MCP pueda exponer tools dinámicamente, es decir, que el cliente pueda solicitar al servidor que le envíe una tool específica en lugar de tener que conocerla de antemano. Esto es útil cuando las tools dependen de datos específicos o del contexto de la solicitud.

# %%
import os
os.makedirs("25_MCP/05_Discovery", exist_ok=True)

# %%
%%writefile "25_MCP/05_Discovery/server.py"
import asyncio
import re
from fastmcp.tools import Tool
from typing import Callable
from fastmcp import Context, FastMCP

mcp = FastMCP(name="Dynamic-Tool-Router Demo")


async def to_upper(text: str) -> str:
    return text.upper()


async def count_words(text: str) -> int:
    await asyncio.sleep(0)
    return len(re.findall(r"\w+", text))


TOOLS: dict[str, tuple[Callable, str, str]] = {
    "uppercase": (to_upper, "upper_tool", "Convert text to uppercase."),
    "wordcount": (count_words, "wordcount_tool", "Count words in the text."),
}


def classify(text: str) -> str | None:
    if re.fullmatch(r"[A-ZÄÖÜÊẞ ]+", text):
        return "wordcount"
    if "words" in text.lower() or "count" in text.lower():
        return "wordcount"
    if text.islower() or "upper" in text.lower():
        return "uppercase"
    return None


@mcp.tool(
    name="router",
    description="Classifies text, registers the appropriate tool, executes it, and returns the result.",
)
async def router(text: str, ctx: Context):
    category = classify(text) or "uppercase"
    fn, tool_name, desc = TOOLS[category]

    # >= 2.7.0
    new_tool = Tool.from_function(fn, name=tool_name, description=desc)
    ctx.fastmcp.add_tool(new_tool)

    # ctx.fastmcp.add_tool(fn, name=tool_name, description=desc) # before 2.7.0
    result = await fn(text)
    await ctx.info(f"Result from {tool_name}: {result!r}")
    # await ctx.fastmcp.remove_tool(tool_name)  # remove the tool again if desired
    return result


if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8000)

# %%
%%writefile "25_MCP/05_Discovery/client.py"
import asyncio
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

async def main():
    async with Client(StreamableHttpTransport("http://127.0.0.1:8000/mcp/")) as c:
        print("Tools BEFORE :", [t.name for t in await c.list_tools()])
        response = await c.call_tool("router", {"text": "please make this upper CASE"})
        print("Response   :", response)
        print("Tools AFTER  :", [t.name for t in await c.list_tools()])


if __name__ == "__main__":
    asyncio.run(main())

# %% [markdown]
# ### Integración con LangChain
# 
# Un agente de LangChain puede usar las tools de un servidor MCP como si fueran tools de LangChain.

# %%
import os
os.makedirs("25_MCP/08_LangChain_MCP", exist_ok=True)

# %%
%%writefile "25_MCP/08_LangChain_MCP/server.py"
from fastmcp import FastMCP

mcp = FastMCP(name="WeatherServer", stateless_http=True)


@mcp.tool(
    name="get_weather",
    description="Returns a weather description for a given city",
)
def get_weather(city: str) -> str:
    """
    Args:
        city (str): Name of the city
    Returns:
        str: Description of the current weather (mock data)
    """
    return "Sunny, 22°C"


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=3000)

# %% [markdown]
# Podríamos haber guardado el cliente como hemos hecho en los anteriores ejemplos, aunque en este caso se ha preferido ejecutarlo directamente desde Jupyter.

# %%
import dotenv
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

# Busca el archivo .env (o el que especifiques) en el directorio actual y padres
dotenv_path = dotenv.find_dotenv()
print(dotenv_path)  # Imprime la ruta completa al archivo encontrado

# Carga las variables de entorno desde ese archivo
dotenv.load_dotenv(dotenv_path)

async def main():
    client = MultiServerMCPClient(
        {
            "weather": {
                "transport": "streamable_http",
                "url": "http://127.0.0.1:3000/mcp/",
            }
        }
    )

    model = ChatGoogleGenerativeAI(
       model="gemini-2.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),
    )
    tools = await client.get_tools()
    agent_executor = create_react_agent(model, tools)
    messages = await agent_executor.ainvoke({"messages": ["¿Cuál es el clima en Madrid?"]})
    for message in messages["messages"]:
        if isinstance(message, AIMessage):
            if message.tool_calls:
                print(f"AI calls: {message.tool_calls}")
            else:
                print(f"AI: {message.content}")
        elif isinstance(message, HumanMessage):
            print(f"Human: {message.content}")
        elif isinstance(message, ToolMessage):
            print(f"Tool: {message.content}")
        else:
            print(f"Message: {message.content}")

await main()

# %% [markdown]
# ### Integración con OpenAI SDK Agents

# %%
from openai import AsyncOpenAI
from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, Runner
from agents.mcp import MCPServerStreamableHttp
from IPython.display import display, Markdown

google_api_key = os.getenv('GOOGLE_API_KEY')
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

chat_client = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)
chat_model = OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=chat_client)

async with MCPServerStreamableHttp(    name="weather",
    params={"url": "http://127.0.0.1:3000/mcp"}
) as weather_server:

    chat_agent = Agent(
        name="Agente MCP",
        mcp_servers=[weather_server],
        model=chat_model,
        model_settings=ModelSettings(tool_choice="required"),
    )

    result = await Runner.run(chat_agent, "¿Qué tiempo hace en Madrid?")
    display(Markdown(result.final_output))

# %% [markdown]
# ### Autenticación con OAuth2
# 
# Los servidores MCP permiten autenticación de clientes MCP utilizando el protocolo OAuth2. La autenticación se hace a nivel de cliente, no de usuario. Para ello se requiere un servidor OAuth de autentificación (GitHub, Google, auth0, ...). 

# %% [markdown]
# ```mermaid
# sequenceDiagram
#     participant Cliente as 🤖 Cliente
#     participant ServidorAuth as Servidor de Autorización
#     participant ServidorMCP as Servidor MCP
# 
#     Cliente->>ServidorAuth: 1. Solicita token de acceso
#     ServidorAuth-->>Cliente: 2. Devuelve JWT (token)
#     Cliente->>ServidorMCP: 3. Usa token en cabecera HTTP para llamar a la tool
#     ServidorMCP->>ServidorAuth: 4. Valida el token (incluyendo "scope")
#     ServidorMCP-->>Cliente: 5. El cliente recibe la respuesta del servidor
# ```

# %% [markdown]
# Para registrar una nueva API y aplicación cliente en Auth0:
# 
# 1.- Registrarse e ir al [`Dashboard`](https://manage.auth0.com/dashboard) de Auth0.
# 2.- Pulsar sobre `Applications` en el menú de la izquierda.
# 3.- Pulsar sobre `APIs`.
# 4.- Pulsar sobre `+ Create API`.
# 5.- Introducir un nombre y una descripción para la API.
# 6.- En `Identifier`, introducir: `http://localhost:8000/mcp`.
# 7.- Pulsar  sobre `Create`.
# 8.- Pulsar sobre `Permissions`.
# 9.- Añadir un nuevo permiso con el nombre `read:add` y la descripción `Permite usar tool read del servidor MCP`.
# 10.- Pulsar sobre `Machine to Machine Applications`.
# 11.- Seleccionar el permiso y pulsar `Update` y `Continue`.
# 12.- Pulsar sobre `Applications` en el menú de la izquierda.
# 13.- Seleccionar la aplicación que se ha creado.
# 14.- Copiar `DOMAIN`, `Client ID` y `Client Secret` que se han generado en el fichero `.env` con los nombres de variables `AUTH0_DOMAIN`, `AUTH0_CLIENT_ID` y `AUTH0_CLIENT_SECRET`, respectivamente.

# %%
import os
os.makedirs("25_MCP/09_Authorization", exist_ok=True)

# %%
%%writefile "25_MCP/09_Authorization/server.py"
import os

from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

AUTH0_DOMAIN = os.environ["AUTH0_DOMAIN"]
API_AUDIENCE = os.environ.get("API_AUDIENCE", "http://localhost:8000/mcp")
REQUIRED_SCOPES = ["read:add"]

auth = BearerAuthProvider(
    jwks_uri=f"{AUTH0_DOMAIN.rstrip('/')}/.well-known/jwks.json",
    issuer=AUTH0_DOMAIN.rstrip("/") + "/",
    audience=API_AUDIENCE,
    required_scopes=REQUIRED_SCOPES,
)

mcp = FastMCP(
    name="SecureAddServer",
    stateless_http=True,
    auth=auth,
)


@mcp.tool(description="Add two integers")
def add(a: int, b: int) -> int:
    return a + b


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8000)

# %%
%%writefile "25_MCP/09_Authorization/client.py"
import asyncio
import os

import httpx
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from dotenv import load_dotenv, find_dotenv 

load_dotenv(find_dotenv())

AUTH0_DOMAIN = os.environ["AUTH0_DOMAIN"]
AUTH0_CLIENT_ID = os.environ["AUTH0_CLIENT_ID"]
AUTH0_CLIENT_SECRET = os.environ["AUTH0_CLIENT_SECRET"]
API_AUDIENCE = "http://localhost:8000/mcp"

async def get_auth0_token() -> str:
    """
    Request an access token from Auth0 using the Client Credentials Grant.
    """
    token_url = f"{AUTH0_DOMAIN}/oauth/token"
    payload = {
        "grant_type": "client_credentials",
        "client_id": AUTH0_CLIENT_ID,
        "client_secret": AUTH0_CLIENT_SECRET,
        "audience": API_AUDIENCE,
    }
    async with httpx.AsyncClient() as http:
        response = await http.post(token_url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["access_token"]


async def main():
    token = await get_auth0_token()
    print("Got Auth0 token:", token)

    transport = StreamableHttpTransport(
        url=API_AUDIENCE, headers={"Authorization": f"Bearer {token}"}
    )

    client = Client(transport)
    async with client:
        result = await client.call_tool("add", {"a": 5, "b": 7})
        print("5 + 7 =", result[0].text)


if __name__ == "__main__":
    asyncio.run(main())

# %% [markdown]
# Se puede probar desde el inspector de MCP añadiendo el Bearer Token en la configuración del inspector. Para obtener el token, en el `Dashboard` de Auth0, pulsar sobre `Applications` y luego sobre selección la aplicación; pulsar sobre `Quickstart` y con la pestaña `CURL` seleccionada, pulsar `Get Token`. Copiar el token que se genera y pegarlo en el inspector en la cada `Bearer` (pulsar `Authentication` para ver la caja).

# %% [markdown]
# ### Integración con FastAPI

# %%
import os
os.makedirs("25_MCP/10_Fastapi_Integration", exist_ok=True)

# %% [markdown]
# Observe que se puede definir endpoints de FastAPI y tools de MCP. Los endpoints de FastAPI se usan como `tools` de MCP.

# %%
%%writefile "25_MCP/10_Fastapi_Integration/server.py"
from fastapi import FastAPI, HTTPException
from fastmcp import FastMCP
from pydantic import BaseModel

app = FastAPI(title="Product API")
_products: dict[int, dict] = {}

class Product(BaseModel):
    name: str
    price: float

@app.get("/products")
def list_products():
    """List all products"""
    return list(_products.values())

@app.get("/products/{product_id}")
def get_product(product_id: int):
    """Get a product by its ID"""
    if product_id not in _products:
        raise HTTPException(status_code=404, detail="Product not found")
    return _products[product_id]

@app.post("/products")
def create_product(p: Product):
    """Create a new product"""
    new_id = len(_products) + 1
    _products[new_id] = {"id": new_id, **p.model_dump()}
    return _products[new_id]

mcp = FastMCP.from_fastapi(app=app, name="ProductMCP")

@mcp.tool(description="Add two integers")
def add(a: int, b: int) -> int:
    return a + b

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8000)

# %%
%%writefile "25_MCP/10_Fastapi_Integration/client.py"
import asyncio
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

SERVER = "http://127.0.0.1:8000/mcp/"


def section(title: str):
    print(f"\n{'=' * 10} {title} {'=' * 10}")


async def main() -> None:
    async with Client(StreamableHttpTransport(SERVER)) as session:
        
        tools = await session.list_tools()
        section("Available Tools")
        for tool in tools:
            print(f"Tool Name: {tool.name}")
        
        all_products = await session.call_tool(tools[0].name)
        section("All Products (Before)")
        print(all_products)


        create_tool_name = tools[1].name

        section(f"Calling Tool: {create_tool_name}")
        created = await session.call_tool(
            create_tool_name,
            {"name": "Widget", "price": 19.99},
        )
        print("Created product:", created[0].text)

        all_products = await session.call_tool(tools[0].name)
        section("All Products (After)")
        print(all_products)

if __name__ == "__main__":
    asyncio.run(main())

# %% [markdown]
# Otra forma de integrar FastAPI con MCP se muestra en el siguiente código. En este caso, los endpoint de FastAPI se sirven desde fuera de MCP. MCP está montado en `http://localhost:8000/mcp-server/mcp`.

# %%
%%writefile "25_MCP/10_Fastapi_Integration/server.py"
import uvicorn
from fastapi import FastAPI, HTTPException
from fastmcp import FastMCP
from pydantic import BaseModel

_products: dict[int, dict] = {}

mcp = FastMCP("AddServer", stateless_http=True)
mcp_app = mcp.http_app(path="/mcp")
app = FastAPI(lifespan=mcp_app.router.lifespan_context)
app.mount("/mcp-server", mcp_app)

class Product(BaseModel):
    name: str
    price: float


@app.get("/products")
def list_products():
    """List all products"""
    return list(_products.values())


@app.get("/products/{product_id}")
def get_product(product_id: int):
    """Get a product by its ID"""
    if product_id not in _products:
        raise HTTPException(status_code=404, detail="Product not found")
    return _products[product_id]


@app.post("/products")
def create_product(p: Product):
    """Create a new product"""
    new_id = len(_products) + 1
    _products[new_id] = {"id": new_id, **p.model_dump()}
    return _products[new_id]


@mcp.tool(description="Add two integers")
def add(a: int, b: int) -> int:
    return a + b


if __name__ == "__main__":
    uvicorn.run(app=app, host="127.0.0.1", port=8000)

# %% [markdown]
# 
# 


