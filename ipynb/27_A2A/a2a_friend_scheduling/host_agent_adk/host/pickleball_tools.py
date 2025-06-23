from datetime import date, datetime, timedelta
from typing import Dict

# Base de datos en memoria para los horarios de las canchas, mapeando la fecha a un diccionario de franjas horarias y nombres de las partes
COURT_SCHEDULE: Dict[str, Dict[str, str]] = {}


def generate_court_schedule():
    """Genera un horario para la cancha de pickleball para los próximos 7 días."""
    global COURT_SCHEDULE
    today = date.today()
    possible_times = [f"{h:02}:00" for h in range(8, 21)]  # 8 AM a 8 PM

    for i in range(7):
        current_date = today + timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")
        COURT_SCHEDULE[date_str] = {time: "unknown" for time in possible_times}


# Inicializa el horario cuando se carga el módulo
generate_court_schedule()


def list_court_availabilities(date: str) -> dict:
    """
    Lista las franjas horarias disponibles y reservadas para una cancha de pickleball en una fecha determinada.

    Args:
        date: La fecha a verificar, en formato YYYY-MM-DD.

    Returns:
        Un diccionario con el estado y el horario detallado del día.
    """
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return {
            "status": "error",
            "message": "Formato de fecha inválido. Por favor, use YYYY-MM-DD.",
        }

    daily_schedule = COURT_SCHEDULE.get(date)
    if not daily_schedule:
        return {
            "status": "success",
            "message": f"La cancha no está abierta el {date}.",
            "schedule": {},
        }

    available_slots = [
        time for time, party in daily_schedule.items() if party == "unknown"
    ]
    booked_slots = {
        time: party for time, party in daily_schedule.items() if party != "unknown"
    }

    return {
        "status": "success",
        "message": f"Horario para el {date}.",
        "available_slots": available_slots,
        "booked_slots": booked_slots,
    }


def book_pickleball_court(
    date: str, start_time: str, end_time: str, reservation_name: str
) -> dict:
    """
    Reserva una cancha de pickleball para una fecha y rango de horas determinados bajo un nombre de reserva.

    Args:
        date: La fecha de la reserva, en formato YYYY-MM-DD.
        start_time: La hora de inicio de la reserva, en formato HH:MM.
        end_time: La hora de finalización de la reserva, en formato HH:MM.
        reservation_name: El nombre de la reserva.

    Returns:
        Un diccionario que confirma la reserva o proporciona un error.
    """
    try:
        start_dt = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M")
        end_dt = datetime.strptime(f"{date} {end_time}", "%Y-%m-%d %H:%M")
    except ValueError:
        return {
            "status": "error",
            "message": "Formato de fecha u hora inválido. Por favor, use YYYY-MM-DD y HH:MM.",
        }

    if start_dt >= end_dt:
        return {"status": "error", "message": "La hora de inicio debe ser anterior a la hora de finalización."}

    if date not in COURT_SCHEDULE:
        return {"status": "error", "message": f"La cancha no está abierta el {date}."}

    if not reservation_name:
        return {
            "status": "error",
            "message": "No se puede reservar una cancha sin un nombre de reserva.",
        }

    required_slots = []
    current_time = start_dt
    while current_time < end_dt:
        required_slots.append(current_time.strftime("%H:%M"))
        current_time += timedelta(hours=1)

    daily_schedule = COURT_SCHEDULE.get(date, {})
    for slot in required_slots:
        if daily_schedule.get(slot, "booked") != "unknown":
            party = daily_schedule.get(slot)
            return {
                "status": "error",
                "message": f"La franja horaria {slot} del {date} ya está reservada por {party}.",
            }

    for slot in required_slots:
        COURT_SCHEDULE[date][slot] = reservation_name

    return {
        "status": "success",
        "message": f"¡Éxito! La cancha de pickleball ha sido reservada para {reservation_name} de {start_time} a {end_time} el {date}.",
    }
