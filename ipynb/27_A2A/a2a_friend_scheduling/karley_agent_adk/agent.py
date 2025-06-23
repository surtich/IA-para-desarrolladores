import random
from datetime import date, datetime, timedelta

from google.adk.agents import LlmAgent


def generate_karley_calendar() -> dict[str, list[str]]:
    """Genera un calendario aleatorio para Karley para los próximos 7 días."""
    calendar = {}
    today = date.today()
    possible_times = [f"{h:02}:00" for h in range(8, 21)]  # 8 AM a 8 PM

    for i in range(7):
        current_date = today + timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")

        # Selecciona 8 franjas horarias únicas aleatorias para aumentar la disponibilidad
        available_slots = sorted(random.sample(possible_times, 8))
        calendar[date_str] = available_slots

    print("Calendario de Karley:", calendar)

    return calendar


KARLEY_CALENDAR = generate_karley_calendar()


def get_availability(start_date: str, end_date: str) -> str:
    """
    Verifica la disponibilidad de Karley para un rango de fechas dado.

    Args:
        start_date: El inicio del rango de fechas a verificar, en formato YYYY-MM-DD.
        end_date: El final del rango de fechas a verificar, en formato YYYY-MM-DD.

    Returns:
        Una cadena que enumera los horarios disponibles de Karley para ese rango de fechas.
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()

        if start > end:
            return "Rango de fechas inválido. La fecha de inicio no puede ser posterior a la fecha de fin."

        results = []
        delta = end - start
        for i in range(delta.days + 1):
            day = start + timedelta(days=i)
            date_str = day.strftime("%Y-%m-%d")
            available_slots = KARLEY_CALENDAR.get(date_str, [])
            if available_slots:
                availability = f"El {date_str}, Karley está disponible a las: {', '.join(available_slots)}."
                results.append(availability)
            else:
                results.append(f"Karley no está disponible el {date_str}.")

        return "\n".join(results)

    except ValueError:
        return (
            "Formato de fecha inválido. Por favor, usa YYYY-MM-DD para ambas fechas de inicio y fin."
        )


def create_agent() -> LlmAgent:
    """Construye el agente ADK para Karley."""
    return LlmAgent(
        model="gemini-2.5-flash-preview-04-17",
        name="Karley_Agent",
        instruction="""
            **Rol:** Eres el asistente personal de programación de Karley.
            Tu única responsabilidad es gestionar su calendario y responder a las consultas
            sobre su disponibilidad para jugar al pickleball.

            **Directivas principales:**

            *   **Verificar disponibilidad:** Utiliza la herramienta `get_karley_availability` para determinar
                    si Karley está libre en una fecha solicitada o en un rango de fechas.
                    La herramienta requiere una `start_date` y una `end_date`. Si el usuario solo proporciona
                    una única fecha, usa esa fecha tanto para el inicio como para el final.
            *   **Educado y conciso:** Sé siempre educado y directo en tus respuestas.
            *   **Cíñete a tu rol:** No participes en ninguna conversación fuera de la programación.
                    Si te hacen otras preguntas, di amablemente que solo puedes ayudar con la programación.
        """,
        tools=[get_availability],
    )
