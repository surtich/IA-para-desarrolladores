import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, AsyncIterable, List

import httpx
import nest_asyncio
from a2a.client import A2ACardResolver
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
)
from dotenv import load_dotenv
from google.adk import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from .pickleball_tools import (
    book_pickleball_court,
    list_court_availabilities,
)
from .remote_agent_connection import RemoteAgentConnections

load_dotenv()
nest_asyncio.apply()


class HostAgent:
    """El agente Anfitrión."""

    def __init__(
        self,
    ):
        self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}
        self.agents: str = ""
        self._agent = self.create_agent()
        self._user_id = "host_agent"
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    async def _async_init_components(self, remote_agent_addresses: List[str]):
        async with httpx.AsyncClient(timeout=30) as client:
            for address in remote_agent_addresses:
                card_resolver = A2ACardResolver(client, address)
                try:
                    card = await card_resolver.get_agent_card()
                    remote_connection = RemoteAgentConnections(
                        agent_card=card, agent_url=address
                    )
                    self.remote_agent_connections[card.name] = remote_connection
                    self.cards[card.name] = card
                except httpx.ConnectError as e:
                    print(f"ERROR: Falló al obtener la tarjeta del agente de {address}: {e}")
                except Exception as e:
                    print(f"ERROR: Falló al inicializar la conexión para {address}: {e}")

        agent_info = [
            json.dumps({"name": card.name, "description": card.description})
            for card in self.cards.values()
        ]
        print("agent_info:", agent_info)
        self.agents = "\n".join(agent_info) if agent_info else "No se encontraron amigos"

    @classmethod
    async def create(
        cls,
        remote_agent_addresses: List[str],
    ):
        instance = cls()
        await instance._async_init_components(remote_agent_addresses)
        return instance

    def create_agent(self) -> Agent:
        return Agent(
            model="gemini-2.5-flash-preview-04-17",
            name="Host_Agent",
            instruction=self.root_instruction,
            description="Este agente Anfitrión orquesta la programación de pickleball con amigos.",
            tools=[
                self.send_message,
                book_pickleball_court,
                list_court_availabilities,
            ],
        )

    def root_instruction(self, context: ReadonlyContext) -> str:
        return f"""
        **Rol:** Eres el Agente Anfitrión, un experto programador de partidos de pickleball. Tu función principal es coordinar con los agentes amigos para encontrar un momento adecuado para jugar y luego reservar una cancha.

        **Directivas Principales:**

        *   **Iniciar Planificación:** Cuando se te pida programar un partido, primero determina a quién invitar y el rango de fechas deseado por el usuario.
        *   **Delegación de Tareas:** Usa la herramienta `send_message` para preguntar a cada amigo su disponibilidad.
            *   Formula tu solicitud claramente (ej., "¿Estás disponible para pickleball entre 2024-08-01 y 2024-08-03?").
            *   Asegúrate de pasar el nombre oficial del agente amigo para cada solicitud de mensaje.
        *   **Analizar Respuestas:** Una vez que tengas la disponibilidad de todos los amigos, analiza las respuestas para encontrar franjas horarias comunes.
        *   **Verificar Disponibilidad de Cancha:** Antes de proponer horarios al usuario, usa la herramienta `list_court_availabilities` para asegurarte de que la cancha también esté libre en las franjas horarias comunes.
        *   **Proponer y Confirmar:** Presenta las franjas horarias comunes y disponibles en la cancha al usuario para su confirmación.
        *   **Reservar la Cancha:** Después de que el usuario confirme un horario, usa la herramienta `book_pickleball_court` para hacer la reserva. Esta herramienta requiere una `start_time` y una `end_time`.
        *   **Comunicación Transparente:** Transmite la confirmación final de la reserva, incluyendo el ID de la reserva, al usuario. No pidas permiso antes de contactar a los agentes amigos.
        *   **Dependencia de Herramientas:** Confía estrictamente en las herramientas disponibles para atender las solicitudes del usuario. No generes respuestas basadas en suposiciones.
        *   **Legibilidad:** Asegúrate de responder en un formato conciso y fácil de leer (los puntos de viñeta son buenos).
        *   Cada agente disponible representa un amigo. Así que Bob_Agent representa a Bob.
        *   Cuando se te pregunte qué amigos están disponibles, debes devolver los nombres de los amigos disponibles (es decir, los agentes que están activos).
        *   Cuando obtengas

        **Fecha de Hoy (AAAA-MM-DD):** {datetime.now().strftime("%Y-%m-%d")}

        <Agentes Disponibles>
        {self.agents}
        </Agentes Disponibles>
        """

    async def stream(
        self, query: str, session_id: str
    ) -> AsyncIterable[dict[str, Any]]:
        """
        Transmite la respuesta del agente a una consulta dada.
        """
        session = await self._runner.session_service.get_session(
            app_name=self._agent.name,
            user_id=self._user_id,
            session_id=session_id,
        )
        content = types.Content(role="user", parts=[types.Part.from_text(text=query)])
        if session is None:
            session = await self._runner.session_service.create_session(
                app_name=self._agent.name,
                user_id=self._user_id,
                state={},
                session_id=session_id,
            )
        async for event in self._runner.run_async(
            user_id=self._user_id, session_id=session.id, new_message=content
        ):
            if event.is_final_response():
                response = ""
                if (
                    event.content
                    and event.content.parts
                    and event.content.parts[0].text
                ):
                    response = "\n".join(
                        [p.text for p in event.content.parts if p.text]
                    )
                yield {
                    "is_task_complete": True,
                    "content": response,
                }
            else:
                yield {
                    "is_task_complete": False,
                    "updates": "El agente anfitrión está pensando...",
                }

    async def send_message(self, agent_name: str, task: str, tool_context: ToolContext):
        """Envía una tarea a un agente amigo remoto."""
        if agent_name not in self.remote_agent_connections:
            raise ValueError(f"Agente {agent_name} no encontrado")
        client = self.remote_agent_connections[agent_name]

        if not client:
            raise ValueError(f"Cliente no disponible para {agent_name}")

        # Gestión simplificada de tareas e ID de contexto
        state = tool_context.state
        task_id = state.get("task_id", str(uuid.uuid4()))
        context_id = state.get("context_id", str(uuid.uuid4()))
        message_id = str(uuid.uuid4())

        payload = {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": task}],
                "messageId": message_id,
                "taskId": task_id,
                "contextId": context_id,
            },
        }

        message_request = SendMessageRequest(
            id=message_id, params=MessageSendParams.model_validate(payload)
        )
        send_response: SendMessageResponse = await client.send_message(message_request)
        print("send_response", send_response)

        if not isinstance(
            send_response.root, SendMessageSuccessResponse
        ) or not isinstance(send_response.root.result, Task):
            print("Se recibió una respuesta no exitosa o no de tarea. No se puede continuar.")
            return

        response_content = send_response.root.model_dump_json(exclude_none=True)
        json_content = json.loads(response_content)

        resp = []
        if json_content.get("result", {}).get("artifacts"):
            for artifact in json_content["result"]["artifacts"]:
                if artifact.get("parts"):
                    resp.extend(artifact["parts"])
        return resp


def _get_initialized_host_agent_sync():
    """Crea e inicializa sincrónicamente el HostAgent."""

    async def _async_main():
        # URLs codificadas para los agentes amigos
        friend_agent_urls = [
            "http://localhost:10002",  # Agente de Karley
            "http://localhost:10003",  # Agente de Nate
            "http://localhost:10004",  # Agente de Kaitlynn
        ]

        print("inicializando agente anfitrión")
        hosting_agent_instance = await HostAgent.create(
            remote_agent_addresses=friend_agent_urls
        )
        print("HostAgent inicializado")
        return hosting_agent_instance.create_agent()

    try:
        return asyncio.run(_async_main())
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            print(
                f"Advertencia: No se pudo inicializar HostAgent con asyncio.run(): {e}. "
                "Esto puede ocurrir si ya hay un bucle de eventos en ejecución (por ejemplo, en Jupyter). "
                "Considere inicializar HostAgent dentro de una función asíncrona en su aplicación."
            )
        else:
            raise


root_agent = _get_initialized_host_agent_sync()