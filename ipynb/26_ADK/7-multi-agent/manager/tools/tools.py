from datetime import datetime


def get_current_time() -> dict:
    """
    Obtiene la hora actual en formato AAAA-MM-DD HH:MM:SS
    """
    return {
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
