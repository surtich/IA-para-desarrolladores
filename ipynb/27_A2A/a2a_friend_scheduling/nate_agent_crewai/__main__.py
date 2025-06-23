"""Este archivo sirve como punto de entrada principal para la aplicación.

Inicializa el servidor A2A, define las capacidades del agente,
y arranca el servidor para manejar las solicitudes entrantes.
"""

import logging
import os

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from agent import SchedulingAgent
from agent_executor import SchedulingAgentExecutor
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Excepción para clave de API faltante."""


def main():
    """Punto de entrada para el Agente de Programación de Nate."""
    host = "localhost"
    port = 10003
    try:
        if not os.getenv("GOOGLE_API_KEY"):
            raise MissingAPIKeyError("La variable de entorno GOOGLE_API_KEY no está configurada.")

        capabilities = AgentCapabilities(streaming=False)
        skill = AgentSkill(
            id="availability_checker",
            name="Verificador de Disponibilidad",
            description="Revisa mi calendario para ver cuándo estoy disponible para un partido de pickleball.",
            tags=["agenda", "disponibilidad", "calendario"],
            examples=[
                "¿Estás libre mañana?",
                "¿Puedes jugar pickleball el próximo martes a las 5pm?",
            ],
        )

        agent_host_url = os.getenv("HOST_OVERRIDE") or f"http://{host}:{port}/"
        agent_card = AgentCard(
            name="Agente Nate",
            description="Un agente amigable para ayudarte a programar un partido de pickleball con Nate.",
            url=agent_host_url,
            version="1.0.0",
            defaultInputModes=SchedulingAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=SchedulingAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        request_handler = DefaultRequestHandler(
            agent_executor=SchedulingAgentExecutor(),
            task_store=InMemoryTaskStore(),
        )
        server = A2AStarletteApplication(
            agent_card=agent_card, http_handler=request_handler
        )

        uvicorn.run(server.build(), host=host, port=port)

    except MissingAPIKeyError as e:
        logger.error(f"Error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Ocurrió un error durante el inicio del servidor: {e}")
        exit(1)


if __name__ == "__main__":
    main()
