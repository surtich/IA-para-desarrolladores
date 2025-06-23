"""
Agente Raíz del Monitor del Sistema

Este módulo define el agente raíz para la aplicación de monitoreo del sistema.
Utiliza un agente paralelo para la recopilación de información del sistema y una
tubería secuencial para el flujo general.
"""

from google.adk.agents import ParallelAgent, SequentialAgent

from .subagents.cpu_info_agent import cpu_info_agent
from .subagents.disk_info_agent import disk_info_agent
from .subagents.memory_info_agent import memory_info_agent
from .subagents.synthesizer_agent import system_report_synthesizer

# --- 1. Crear Agente Paralelo para recopilar información concurrentemente ---
system_info_gatherer = ParallelAgent(
    name="system_info_gatherer",
    sub_agents=[cpu_info_agent, memory_info_agent, disk_info_agent],
)

# --- 2. Crear Tubería Secuencial para recopilar información en paralelo, luego sintetizar ---
root_agent = SequentialAgent(
    name="system_monitor_agent",
    sub_agents=[system_info_gatherer, system_report_synthesizer],
)
