import random
from collections.abc import AsyncIterable
from datetime import date, datetime, timedelta
from typing import Any, List, Literal

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

memory = MemorySaver()


def generate_kaitlyns_calendar() -> dict[str, list[str]]:
    """Genera el calendario de Kaitlyn para los próximos 7 días."""
    calendar = {}
    today = date.today()
    # Disponibilidad de Kaitlyn: tardes entre semana, más libre los fines de semana.
    for i in range(7):
        current_date = today + timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")
        day_of_week = current_date.weekday()  # Lunes es 0 y Domingo es 6

        if day_of_week < 5:  # Día de semana
            possible_times = [f"{h:02}:00" for h in range(18, 22)]  # 6 PM a 10 PM
            available_slots = sorted(
                random.sample(possible_times, random.randint(2, 3))
            )
        else:  # Fin de semana
            possible_times = [f"{h:02}:00" for h in range(10, 20)]  # 10 AM a 8 PM
            available_slots = sorted(
                random.sample(possible_times, random.randint(4, 6))
            )

        calendar[date_str] = available_slots
    return calendar


KAITLYNS_CALENDAR = generate_kaitlyns_calendar()


class AvailabilityToolInput(BaseModel):
    """Esquema de entrada para la herramienta de disponibilidad."""

    date_range: str = Field(
        ...,
        description=(
            "La fecha o rango de fechas para verificar la disponibilidad, ej. "
            "'2024-07-28' o '2024-07-28 to 2024-07-30'."
        ),
    )


@tool(args_schema=AvailabilityToolInput)
def get_availability(date_range: str) -> str:
    """Usa esto para obtener la disponibilidad de Kaitlyn para una fecha o rango de fechas dado."""
    dates_to_check = [d.strip() for d in date_range.split("to")]
    start_date_str = dates_to_check[0]
    end_date_str = dates_to_check[-1]

    try:
        start = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end = datetime.strptime(end_date_str, "%Y-%m-%d").date()

        if start > end:
            return "Rango de fechas inválido. La fecha de inicio no puede ser posterior a la fecha de fin."

        results = []
        delta = end - start
        for i in range(delta.days + 1):
            day = start + timedelta(days=i)
            date_str = day.strftime("%Y-%m-%d")
            available_slots = KAITLYNS_CALENDAR.get(date_str, [])
            if available_slots:
                availability = (
                    f"El {date_str}, Kaitlyn está disponible a las: "
                    f"{', '.join(available_slots)}."
                )
                results.append(availability)
            else:
                results.append(f"Kaitlyn no está disponible el {date_str}.")

        return "\n".join(results)

    except ValueError:
        return (
            "No pude entender la fecha. "
            "Por favor, pregunta para verificar la disponibilidad para una fecha como 'AAAA-MM-DD'."
        )


class ResponseFormat(BaseModel):
    """Responde al usuario en este formato."""

    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str


class KaitlynAgent:
    """KaitlynAgent - un asistente especializado para la programación."""

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    SYSTEM_INSTRUCTION = (
        "Eres el asistente de programación de Kaitlyn. "
        "Tu único propósito es usar la herramienta 'get_availability' para responder preguntas sobre el horario de Kaitlyn para jugar pickleball. "
        "Se te proporcionará la fecha actual para ayudarte a entender consultas de fechas relativas como 'mañana' o 'la próxima semana'. "
        "Usa esta información para llamar correctamente a la herramienta con una fecha específica (ej., 'AAAA-MM-DD'). "
        "Si el usuario pregunta sobre algo que no sea programar pickleball, "
        "indica amablemente que no puedes ayudar con ese tema y que solo puedes ayudar con consultas de programación. "
        "No intentes responder preguntas no relacionadas ni usar herramientas para otros propósitos."
        "Establece el estado de la respuesta a 'input_required' si el usuario necesita proporcionar más información."
        "Establece el estado de la respuesta a 'error' si hay un error al procesar la solicitud."
        "Establece el estado de la respuesta a 'completed' si la solicitud está completa."
    )

    def __init__(self):
        self.model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        self.tools = [get_availability]

        self.graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=ResponseFormat,
        )

    def invoke(self, query, context_id):
        config: RunnableConfig = {"configurable": {"thread_id": context_id}}
        today_str = f"La fecha de hoy es {date.today().strftime('%Y-%m-%d')}."
        augmented_query = f"{today_str}\n\nConsulta del usuario: {query}"
        self.graph.invoke({"messages": [("user", augmented_query)]}, config)
        return self.get_agent_response(config)

    async def stream(self, query, context_id) -> AsyncIterable[dict[str, Any]]:
        today_str = f"La fecha de hoy es {date.today().strftime('%Y-%m-%d')}."
        augmented_query = f"{today_str}\n\nConsulta del usuario: {query}"
        inputs = {"messages": [("user", augmented_query)]}
        config: RunnableConfig = {"configurable": {"thread_id": context_id}}

        for item in self.graph.stream(inputs, config, stream_mode="values"):
            message = item["messages"][-1]
            if (
                isinstance(message, AIMessage)
                and message.tool_calls
                and len(message.tool_calls) > 0
            ):
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Verificando la disponibilidad de Kaitlyn...",
                }
            elif isinstance(message, ToolMessage):
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Procesando disponibilidad...",
                }

        yield self.get_agent_response(config)

    def get_agent_response(self, config):
        current_state = self.graph.get_state(config)
        structured_response = current_state.values.get("structured_response")
        if structured_response and isinstance(structured_response, ResponseFormat):
            if structured_response.status == "input_required":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": structured_response.message,
                }
            if structured_response.status == "error":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": structured_response.message,
                }
            if structured_response.status == "completed":
                return {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": structured_response.message,
                }

        return {
            "is_task_complete": False,
            "require_user_input": True,
            "content": (
                "No podemos procesar su solicitud en este momento. "
                "Por favor, inténtelo de nuevo."
            ),
        }
