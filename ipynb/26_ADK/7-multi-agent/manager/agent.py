from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool

from .sub_agents.funny_nerd.agent import funny_nerd
from .sub_agents.news_analyst.agent import news_analyst
from .sub_agents.stock_analyst.agent import stock_analyst
from .tools.tools import get_current_time

root_agent = Agent(
    name="manager",
    model="gemini-2.0-flash",
    description="Agente gestor",
    instruction="""
    Eres un agente gestor responsable de supervisar el trabajo de los otros agentes.

    Siempre delega la tarea al agente apropiado. Usa tu mejor juicio
    para determinar a qué agente delegar.

    Eres responsable de delegar tareas a los siguientes agentes:
    - stock_analyst
    - funny_nerd

    También tienes acceso a las siguientes herramientas:
    - get_current_time
    - news_analyst
    """,
    sub_agents=[stock_analyst, funny_nerd],
    tools=[
        AgentTool(news_analyst),
        get_current_time,
    ],
)
