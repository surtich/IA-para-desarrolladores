"""
Herramienta de Información de CPU

Este módulo proporciona una herramienta para recopilar información de la CPU.
"""

import time
from typing import Any, Dict

import psutil


def get_cpu_info() -> Dict[str, Any]:
    """
    Recopila información de la CPU, incluyendo el número de núcleos y el uso.

    Devuelve:
        Dict[str, Any]: Diccionario con información de la CPU estructurada para ADK
    """
    try:
        # Obtener información de la CPU
        cpu_info = {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "cpu_usage_per_core": [
                f"Núcleo {i}: {percentage:.1f}%"
                for i, percentage in enumerate(
                    psutil.cpu_percent(interval=1, percpu=True)
                )
            ],
            "avg_cpu_usage": f"{psutil.cpu_percent(interval=1):.1f}%",
        }

        # Calcular algunas estadísticas para el resumen del resultado
        avg_usage = float(cpu_info["avg_cpu_usage"].strip("%"))
        high_usage = avg_usage > 80

        # Formatear para la estructura de retorno de la herramienta ADK
        return {
            "result": cpu_info,
            "stats": {
                "physical_cores": cpu_info["physical_cores"],
                "logical_cores": cpu_info["logical_cores"],
                "avg_usage_percentage": avg_usage,
                "high_usage_alert": high_usage,
            },
            "additional_info": {
                "data_format": "dictionary",
                "collection_timestamp": time.time(),
                "performance_concern": (
                    "Uso de CPU alto detectado" if high_usage else None
                ),
            },
        }
    except Exception as e:
        return {
            "result": {"error": f"No se pudo recopilar información de la CPU: {str(e)}"},
            "stats": {"success": False},
            "additional_info": {"error_type": str(type(e).__name__)},
        }
