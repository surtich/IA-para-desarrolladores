# %% [markdown]
# # Sidekick
# 
# Vamos a crear un proyecto completo con LangGraph que use Multi-Agents.

# %%
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from IPython.display import Image, display
import gradio as gr
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests
import os
from langchain.agents import Tool

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver

# %%
load_dotenv(override=True)

# %% [markdown]
# ### Asynchronous LangGraph
# 
# LangGraph puede funcionar en modo síncrono o asíncrono. Lo usaremos más adelante.
# 
# To run a tool:  
# Sync: `tool.run(inputs)`  
# Async: `await tool.arun(inputs)`
# 
# To invoke the graph:  
# Sync: `graph.invoke(state)`  
# Async: `await graph.ainvoke(state)`

# %%
class State(TypedDict):
    
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

# %%
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

import os
from langchain_community.tools import StructuredTool

tool_send_email = StructuredTool.from_function(send_email, description="Útil para enviar correos electrónicos", name="send_email")




# %% [markdown]
# El proyecto requiere Playwright instalado. Playwright es una herramienta que permite interactuar con navegadores web de manera programática, lo cual es útil para pruebas automatizadas y scraping.
# 
# On Windows and MacOS:  
# `playwright install`
# 
# On Linux:  
# `playwright install --with-Deps chromium`

# %% [markdown]
# El código asíncrono en Python funciona como JavaScript, con el uso del `event loop`. Sólo puede haber un `event loop` en una aplicación. En este proyecto vamos a ejecutar código asíncrono dentro del `event loop`, por lo que se requieren `event loops` anidados. Para salvar la limitación de Python hacemos lo siguiente.

# %%
import nest_asyncio
nest_asyncio.apply()

# %% [markdown]
# Hay librería de LangChain que permite usar PlayWright en forma de `tools`

# %%


# %%
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser

async_browser =  create_async_playwright_browser(headless=False)  # headful mode
toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
tools = toolkit.get_tools()

# %% [markdown]
# Empaquetamos las `tools` en un diccionario y probamos una de ellas:

# %%
tool_dict = {tool.name:tool for tool in tools}

navigate_tool = tool_dict.get("navigate_browser")
extract_text_tool = tool_dict.get("extract_text")

await navigate_tool.arun({"url": "https://www.cnn.com"})
text = await extract_text_tool.arun({})

# %%
import textwrap
print(textwrap.fill(text))

# %%
all_tools = [tool_send_email] + tools

# %%
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
llm_with_tools = llm.bind_tools(all_tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# %%
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=all_tools))
graph_builder.add_conditional_edges( "chatbot", tools_condition, "tools")
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
display(Image(graph.get_graph().draw_mermaid_png()))

# %%
# Esto evita el error: There is no current event loop in thread 'MainThread'
import uvicorn

uvicorn.config.LOOP_SETUPS = {
    "none": None,
    "auto": "uvicorn.loops.asyncio:asyncio_setup",
    "asyncio": "uvicorn.loops.asyncio:asyncio_setup",
    "uvloop": "uvicorn.loops.uvloop:uvloop_setup",
}

# %%
config = {"configurable": {"thread_id": "1"}}

async def chat(user_input: str, history):
    result = await graph.ainvoke({"messages": [{"role": "user", "content": user_input}]}, config=config)
    return result["messages"][-1].content


gr.ChatInterface(chat, type="messages").launch()


