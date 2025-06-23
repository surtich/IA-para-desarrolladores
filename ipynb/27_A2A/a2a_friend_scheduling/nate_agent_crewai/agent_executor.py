from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError,
    InvalidParamsError,
    Part,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils.errors import ServerError
from agent import SchedulingAgent


class SchedulingAgentExecutor(AgentExecutor):
    """Executor de Agente para el agente de programación."""

    def __init__(self):
        """Inicializa el SchedulingAgentExecutor."""
        self.agent = SchedulingAgent()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Ejecuta el agente de programación."""
        if not context.task_id or not context.context_id:
            raise ValueError("RequestContext debe tener task_id y context_id")
        if not context.message:
            raise ValueError("RequestContext debe tener un mensaje")

        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if not context.current_task:
            await updater.submit()
        await updater.start_work()

        if self._validate_request(context):
            raise ServerError(error=InvalidParamsError())

        query = context.get_user_input()
        try:
            result = self.agent.invoke(query)
            print(f"Resultado Final ===> {result}")
        except Exception as e:
            print(f"Error al invocar al agente: {e}")
            raise ServerError(error=InternalError()) from e

        parts = [Part(root=TextPart(text=result))]

        await updater.add_artifact(parts)
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Maneja la cancelación de tareas."""
        raise ServerError(error=UnsupportedOperationError())

    def _validate_request(self, context: RequestContext) -> bool:
        """Valida el contexto de la solicitud."""
        return False
