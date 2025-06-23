"""
Herramienta de Información de Disco

Este módulo proporciona una herramienta para recopilar información del disco.
"""

import time
from typing import Any, Dict

import psutil


def get_disk_info() -> Dict[str, Any]:
    """
    Recopila información del disco, incluyendo particiones y uso.

    Devuelve:
        Dict[str, Any]: Diccionario con información del disco estructurada para ADK
    """
    try:
        # Obtener información del disco
        disk_info = {"partitions": []}
        partitions_over_threshold = []
        total_space = 0
        used_space = 0

        for partition in psutil.disk_partitions():
            try:
                partition_usage = psutil.disk_usage(partition.mountpoint)

                # Rastrear particiones con uso alto
                if partition_usage.percent > 85:
                    partitions_over_threshold.append(
                        f"{partition.mountpoint} ({partition_usage.percent:.1f}%)"
                    )

                # Añadir a los totales
                total_space += partition_usage.total
                used_space += partition_usage.used

                disk_info["partitions"].append(
                    {
                        "device": partition.device,
                        "mountpoint": partition.mountpoint,
                        "filesystem_type": partition.fstype,
                        "total_size": f"{partition_usage.total / (1024 ** 3):.2f} GB",
                        "used": f"{partition_usage.used / (1024 ** 3):.2f} GB",
                        "free": f"{partition_usage.free / (1024 ** 3):.2f} GB",
                        "percentage": f"{partition_usage.percent:.1f}%",
                    }
                )
            except (PermissionError, FileNotFoundError):
                # Algunas particiones pueden no ser accesibles
                pass

        # Calcular estadísticas generales del disco
        overall_usage_percent = (
            (used_space / total_space * 100) if total_space > 0 else 0
        )

        # Formatear para la estructura de retorno de la herramienta ADK
        return {
            "result": disk_info,
            "stats": {
                "partition_count": len(disk_info["partitions"]),
                "total_space_gb": total_space / (1024**3),
                "used_space_gb": used_space / (1024**3),
                "overall_usage_percent": overall_usage_percent,
                "partitions_with_high_usage": len(partitions_over_threshold),
            },
            "additional_info": {
                "data_format": "dictionary",
                "collection_timestamp": time.time(),
                "high_usage_partitions": (
                    partitions_over_threshold if partitions_over_threshold else None
                ),
            },
        }
    except Exception as e:
        return {
            "result": {"error": f"No se pudo recopilar información del disco: {str(e)}"},
            "stats": {"success": False},
            "additional_info": {"error_type": str(type(e).__name__)},
        }
