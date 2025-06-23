from google.adk.agents import Agent
from google.adk.tools import google_search

news_analyst = Agent(
    name="news_analyst",
    model="gemini-2.0-flash",
    description="Agente analista de noticias",
    instruction="""
    Eres un asistente útil que puede analizar artículos de noticias y proporcionar un resumen de las noticias.

    Cuando se te pregunte sobre noticias, debes usar la herramienta google_search para buscar las noticias.

    Si el usuario pide noticias usando un tiempo relativo, debes usar la herramienta get_current_time para obtener la hora actual y usarla en la consulta de búsqueda.
    """,
    tools=[google_search],
)
