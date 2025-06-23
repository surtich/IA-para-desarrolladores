# Demostración de Programación de Amigos A2A
Este documento describe una aplicación multi-agente que demuestra cómo orquestar conversaciones entre diferentes agentes para programar una reunión.

Esta aplicación contiene cuatro agentes:
*   **Agente Anfitrión**: El agente principal que orquesta la tarea de programación.
*   **Agente Kaitlynn**: Un agente que representa el calendario y las preferencias de Kaitlynn.
*   **Agente Nate**: Un agente que representa el calendario y las preferencias de Nate.
*   **Agente Karley**: Un agente que representa el calendario y las preferencias de Karley.

## Configuración y Despliegue

### Prerrequisitos

Antes de ejecutar la aplicación localmente, asegúrate de tener lo siguiente instalado:

1. **uv:** La herramienta de gestión de paquetes de Python utilizada en este proyecto. Sigue la guía de instalación: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)
2. **python 3.13** Se requiere Python 3.13 para ejecutar a2a-sdk
3. **configurar .env**

Crea un archivo `.env` en la raíz del directorio `a2a_friend_scheduling` con tu clave de API de Google:
```
GOOGLE_API_KEY="tu_clave_api_aquí"
```

## Ejecutar los Agentes

Necesitarás ejecutar cada agente en una ventana de terminal separada. La primera vez que ejecutes estos comandos, `uv` creará un entorno virtual e instalará todas las dependencias necesarias antes de iniciar el agente.

### Terminal 1: Ejecutar Agente Kaitlynn
```bash
cd kaitlynn_agent_langgraph
uv run app
```

### Terminal 2: Ejecutar Agente Nate
```bash
cd nate_agent_crewai
uv venv
source .venv/bin/activate
uv run --active .
```

### Terminal 3: Ejecutar Agente Karley
```bash
cd karley_agent_adk
uv venv
source .venv/bin/activate
uv run --active .
```

### Terminal 4: Ejecutar Agente Anfitrión
```bash
cd host_agent_adk
uv venv
source .venv/bin/activate
uv run --active adk web
```

## Interactuar con el Agente Anfitrión

Una vez que todos los agentes estén en ejecución, el agente anfitrión comenzará el proceso de programación. Puedes ver la interacción en la salida de la terminal del `host_agent`.

## Referencias
- https://github.com/google/a2a-python
- https://codelabs.developers.google.com/intro-a2a-purchasing-concierge#1
