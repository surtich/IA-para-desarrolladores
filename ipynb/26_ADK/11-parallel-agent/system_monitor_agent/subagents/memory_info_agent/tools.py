"""
Herramienta de Información de Memoria

Este módulo proporciona una herramienta para recopilar información de la memoria.
"""

import time
from typing import Any, Dict

import psutil


def get_memory_info() -> Dict[str, Any]:
    """
    Recopila información de la memoria, incluyendo RAM y uso de swap.

    Devuelve:
        Dict[str, Any]: Diccionario con información de la memoria estructurada para ADK
    """
    try:
        # Obtener información de la memoria
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        memory_info = {
            "total_memory": f"{memory.total / (1024 ** 3):.2f} GB",
            "available_memory": f"{memory.available / (1024 ** 3):.2f} GB",
            "used_memory": f"{memory.used / (1024 ** 3):.2f} GB",
            "memory_percentage": f"{memory.percent:.1f}%",
            "swap_total": f"{swap.total / (1024 ** 3):.2f} GB",
            "swap_used": f"{swap.used / (1024 ** 3):.2f} GB",
            "swap_percentage": f"{swap.percent:.1f}%",
        }

        # Calcular estadísticas
        memory_usage = memory.percent
        swap_usage = swap.percent
        high_memory_usage = memory_usage > 80
        high_swap_usage = swap_usage > 80

        # Formatear para la estructura de retorno de la herramienta ADK
        return {
            "result": memory_info,
            "stats": {
                "memory_usage_percentage": memory_usage,
                "swap_usage_percentage": swap_usage,
                "total_memory_gb": memory.total / (1024**3),
                "available_memory_gb": memory.available / (1024**3),
            },
            "additional_info": {
                "data_format": "dictionary",
                "collection_timestamp": time.time(),
                "performance_concern": (
                    "Uso de memoria alto detectado" if high_memory_usage else None
                ),
                "swap_concern": "Uso de swap alto detectado" if high_swap_usage else None,
            },
        }
    except Exception as e:
        return {
            "result": {"error": f"No se pudo recopilar información de la memoria: {str(e)}"},
            "stats": {"success": False},
            "additional_info": {"error_type": str(type(e).__name__)},
        }
