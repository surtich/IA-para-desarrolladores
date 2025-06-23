import os
import random
from datetime import date, datetime, timedelta
from typing import Type

from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import BaseTool
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


def generate_calendar() -> dict[str, list[str]]:
    """Genera un calendario aleatorio para los próximos 7 días."""
    calendar = {}
    today = date.today()
    possible_times = [f"{h:02}:00" for h in range(8, 21)]  # 8 AM a 8 PM

    for i in range(7):
        current_date = today + timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")
        available_slots = sorted(random.sample(possible_times, 8))
        calendar[date_str] = available_slots
    print("---- Calendario Generado por Nate ----")
    print(calendar)
    print("---------------------------------")
    return calendar


MY_CALENDAR = generate_calendar()


class AvailabilityToolInput(BaseModel):
    """Esquema de entrada para AvailabilityTool."""

    date_range: str = Field(
        ...,
        description="La fecha o rango de fechas para verificar la disponibilidad, ej., '2024-07-28' o '2024-07-28 a 2024-07-30'.",
    )


class AvailabilityTool(BaseTool):
    name: str = "Verificador de Disponibilidad del Calendario"
    description: str = (
        "Verifica mi disponibilidad para una fecha o rango de fechas dado. "
        "Usa esto para saber cuándo estoy libre."
    )
    args_schema: Type[BaseModel] = AvailabilityToolInput

    def _run(self, date_range: str) -> str:
        """Verifica mi disponibilidad para un rango de fechas dado."""
        dates_to_check = [d.strip() for d in date_range.split("to")]
        start_date_str = dates_to_check[0]
        end_date_str = dates_to_check[-1]

        try:
            start = datetime.strptime(start_date_str, "%Y-%m-%d").date()
            end = datetime.strptime(end_date_str, "%Y-%m-%d").date()

            if start > end:
                return (
                    "Rango de fechas inválido. La fecha de inicio no puede ser posterior a la fecha de fin."
                )

            results = []
            delta = end - start
            for i in range(delta.days + 1):
                day = start + timedelta(days=i)
                date_str = day.strftime("%Y-%m-%d")
                available_slots = MY_CALENDAR.get(date_str, [])
                if available_slots:
                    availability = f"El {date_str}, estoy disponible a las: {', '.join(available_slots)}."
                    results.append(availability)
                else:
                    results.append(f"No estoy disponible el {date_str}.")

            return "\n".join(results)

        except ValueError:
            return (
                "No pude entender la fecha. "
                "Por favor, pregunta para verificar la disponibilidad para una fecha como 'AAAA-MM-DD'."
            )


class SchedulingAgent:
    """Agente que maneja las tareas de programación."""

    SUPPORTED_CONTENT_TYPES = ["text/plain"]

    def __init__(self):
        """Inicializa el SchedulingAgent."""
        if os.getenv("GOOGLE_API_KEY"):
            self.llm = LLM(
                model="gemini/gemini-2.0-flash",
                api_key=os.getenv("GOOGLE_API_KEY"),
            )
        else:
            raise ValueError("La variable de entorno GOOGLE_API_KEY no está configurada.")

        self.scheduling_assistant = Agent(
            role="Asistente Personal de Programación",
            goal="Revisar mi calendario y responder preguntas sobre mi disponibilidad.",
            backstory=(
                "Eres un asistente altamente eficiente y educado. Tu único trabajo es "
                "gestionar mi calendario. Eres un experto en el uso de la "
                "herramienta de Verificador de Disponibilidad del Calendario para saber cuándo estoy libre. Nunca "
                "participas en conversaciones fuera de la programación."
            ),
            verbose=True,
            allow_delegation=False,
            tools=[AvailabilityTool()],
            llm=self.llm,
        )

    def invoke(self, question: str) -> str:
        """Inicia el equipo para responder una pregunta de programación."""
        task_description = (
            f"Responde la pregunta del usuario sobre mi disponibilidad. El usuario preguntó: '{question}'. "
            f"La fecha de hoy es {date.today().strftime('%Y-%m-%d')}."
        )

        check_availability_task = Task(
            description=task_description,
            expected_output="Una respuesta educada y concisa a la pregunta del usuario sobre mi disponibilidad, basada en la salida de la herramienta del calendario.",
            agent=self.scheduling_assistant,
        )

        crew = Crew(
            agents=[self.scheduling_assistant],
            tasks=[check_availability_task],
            process=Process.sequential,
            verbose=True,
        )
        result = crew.kickoff()
        return str(result)
