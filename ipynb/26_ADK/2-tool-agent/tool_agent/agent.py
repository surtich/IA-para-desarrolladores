from google.adk.agents import Agent
from google.adk.tools import google_search
from datetime import datetime

def get_current_time() -> dict:
    """
    Obtiene la hora actual en formato AAAA-MM-DD HH:MM:SS
    """
    return {
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

root_agent = Agent(
    name="tool_agent",
    model="gemini-2.0-flash",
    description="Agente de herramientas",
    instruction="""
    Eres un asistente útil que puede usar las siguientes herramientas:
    - get_current_time
    """,
    tools=[get_current_time],
)
