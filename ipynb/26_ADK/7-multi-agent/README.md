# Sistemas Multi-Agente en ADK

Este ejemplo demuestra cómo crear un sistema multi-agente en ADK, donde agentes especializados colaboran para manejar tareas complejas, cada uno centrándose en su área de especialización.

## ¿Qué es un Sistema Multi-Agente?

Un Sistema Multi-Agente es un patrón avanzado en el Kit de Desarrollo de Agentes (ADK) que permite que múltiples agentes especializados trabajen juntos para manejar tareas complejas. Cada agente puede centrarse en un dominio o funcionalidad específica, y pueden colaborar a través de la delegación y la comunicación para resolver problemas que serían difíciles para un solo agente.

## Requisitos de Estructura del Proyecto

Para que los sistemas multi-agente funcionen correctamente con ADK, su proyecto debe seguir una estructura específica:

```
parent_folder/
├── root_agent_folder/           # Paquete del agente principal (ej., "manager")
│   ├── __init__.py              # Debe importar agent.py
│   ├── agent.py                 # Debe definir root_agent
│   ├── .env                     # Variables de entorno
│   └── sub_agents/              # Directorio para todos los sub-agentes
│       ├── __init__.py          # Vacío o importa sub-agentes
│       ├── agent_1_folder/      # Paquete del sub-agente
│       │   ├── __init__.py      # Debe importar agent.py
│       │   └── agent.py         # Debe definir una variable de agente
│       ├── agent_2_folder/
│       │   ├── __init__.py
│       │   └── agent.py
│       └── ...
```

### Componentes Esenciales de la Estructura:

1. **Paquete del Agente Raíz**
   - Debe tener la estructura de agente estándar (como en el ejemplo de agente básico)
   - El archivo `agent.py` debe definir una variable `root_agent`

2. **Directorio de Sub-agentes**
   - Típicamente organizado como un directorio llamado `sub_agents` dentro de la carpeta del agente raíz
   - Cada sub-agente debe estar en su propio directorio siguiendo la misma estructura que los agentes regulares

3. **Importación de Sub-agentes**
   - El agente raíz debe importar los sub-agentes para usarlos:
   ```python
   from .sub_agents.funny_nerd.agent import funny_nerd
   from .sub_agents.stock_analyst.agent import stock_analyst
   ```

4. **Ubicación del Comando**
   - Siempre ejecute `adk web` desde el directorio padre (`6-multi-agent`), no desde dentro de ningún directorio de agente

Esta estructura asegura que ADK pueda descubrir y cargar correctamente todos los agentes en la jerarquía.

## Opciones de Arquitectura Multi-Agente

ADK ofrece dos enfoques principales para construir sistemas multi-agente:

### 1. Modelo de Delegación de Sub-Agentes

Usando el parámetro `sub_agents`, el agente raíz puede delegar completamente las tareas a agentes especializados:

```python
root_agent = Agent(
    name="manager",
    model="gemini-2.0-flash",
    description="Agente gestor",
    instruction="Eres un agente gestor que delega tareas a agentes especializados...",
    sub_agents=[stock_analyst, funny_nerd],
)
```

**Características:**
- Delegación completa - el sub-agente se encarga de toda la respuesta
- La decisión del sub-agente es final y toma el control de la conversación
- El agente raíz actúa como un "enrutador" que determina qué especialista debe manejar la consulta

### 2. Modelo de Agente como Herramienta

Usando el envoltorio `AgentTool`, los agentes pueden ser usados como herramientas por otros agentes:

```python
from google.adk.tools.agent_tool import AgentTool

root_agent = Agent(
    name="manager",
    model="gemini-2.0-flash",
    description="Agente gestor",
    instruction="Eres un agente gestor que utiliza agentes especializados como herramientas...",
    tools=[
        AgentTool(news_analyst),
        get_current_time,
    ],
)
```

**Características:**
- El sub-agente devuelve los resultados al agente raíz
- El agente raíz mantiene el control y puede incorporar la respuesta del sub-agente en la suya propia
- Se pueden realizar múltiples llamadas a herramientas a diferentes herramientas de agente en una sola respuesta
- Le da al agente raíz más flexibilidad en cómo usa los resultados

## Limitaciones al Usar Multi-Agentes

### Restricciones de Sub-agentes

**Las herramientas incorporadas no se pueden usar dentro de un sub-agente.**

Por ejemplo, este enfoque que utiliza herramientas incorporadas dentro de sub-agentes **no** es compatible actualmente:

```python
search_agent = Agent(
    model='gemini-2.0-flash',
    name='SearchAgent',
    instruction="Eres un especialista en Búsqueda de Google",
    tools=[google_search],  # Herramienta incorporada
)
coding_agent = Agent(
    model='gemini-2.0-flash',
    name='CodeAgent',
    instruction="Eres un especialista en Ejecución de Código",
    tools=[built_in_code_execution],  # Herramienta incorporada
)
root_agent = Agent(
    name="RootAgent",
    model="gemini-2.0-flash",
    description="Agente Raíz",
    sub_agents=[
        search_agent,  # NO COMPATIBLE
        coding_agent   # NO COMPATIBLE
    ],
)
```

### Solución Usando Herramientas de Agente

Para usar múltiples herramientas incorporadas o para combinar herramientas incorporadas con otras herramientas, puede usar el enfoque `AgentTool`:

```python
from google.adk.tools import agent_tool

search_agent = Agent(
    model='gemini-2.0-flash',
    name='SearchAgent',
    instruction="Eres un especialista en Búsqueda de Google",
    tools=[google_search],
)
coding_agent = Agent(
    model='gemini-2.0-flash',
    name='CodeAgent',
    instruction="Eres un especialista en Ejecución de Código",
    tools=[built_in_code_execution],
)
root_agent = Agent(
    name="RootAgent",
    model="gemini-2.0-flash",
    description="Agente Raíz",
    tools=[
        agent_tool.AgentTool(agent=search_agent), 
        agent_tool.AgentTool(agent=coding_agent)
    ],
)
```

Este enfoque envuelve a los agentes como herramientas, permitiendo que el agente raíz delegue en agentes especializados que usan cada uno una única herramienta incorporada.

## Nuestro Ejemplo Multi-Agente

Este ejemplo implementa un agente gestor que trabaja con tres agentes especializados:

1. **Analista de Bolsa** (Sub-agente): Proporciona información financiera y conocimientos del mercado de valores
2. **Friki Divertido** (Sub-agente): Crea chistes frikis sobre temas técnicos
3. **Analista de Noticias** (Herramienta de Agente): Ofrece resúmenes de noticias tecnológicas actuales

El agente gestor enruta las consultas al especialista apropiado basándose en el contenido de la solicitud del usuario.

## Primeros Pasos

Este ejemplo utiliza el mismo entorno virtual creado en el directorio raíz. Asegúrese de haber:

1. Activado el entorno virtual desde el directorio raíz:
```bash
# macOS/Linux:
source ../.venv/bin/activate
# Windows CMD:
..\.venv\Scripts\activate.bat
# Windows PowerShell:
..\.venv\Scripts\Activate.ps1
```

2. Configurado su clave API:
   - Renombre `.env.example` a `.env` en la carpeta del gestor
   - Agregue su clave API de Google a la variable `GOOGLE_API_KEY` en el archivo `.env`

## Ejecutando el Ejemplo

Para ejecutar el ejemplo multi-agente:

1. Navegue al directorio `6-multi-agent` que contiene sus carpetas de agente.

2. Inicie la interfaz web interactiva:
```bash
adk web
```

3. Acceda a la interfaz web abriendo la URL que se muestra en su terminal (normalmente http://localhost:8000)

4. Seleccione el agente "manager" del menú desplegable en la esquina superior izquierda de la interfaz de usuario

5. Comience a chatear con su agente en el cuadro de texto en la parte inferior de la pantalla

### Solución de Problemas

Si su configuración multi-agente no aparece correctamente en el menú desplegable:
- Asegúrese de que está ejecutando `adk web` desde el directorio padre (`6-multi-agent`)
- Verifique que el `__init__.py` de cada agente importe correctamente su respectivo `agent.py`
- Compruebe que el agente raíz importa correctamente todos los sub-agentes

### Ejemplos de Prompts para Probar

- "¿Puedes hablarme sobre el mercado de valores hoy?"
- "Cuéntame algo divertido sobre programación"
- "¿Cuáles son las últimas noticias tecnológicas?"
- "¿Qué hora es ahora mismo?"

Puede salir de la conversación o detener el servidor presionando `Ctrl+C` en su terminal.

## Recursos Adicionales

- [Documentación de Sistemas Multi-Agente de ADK](https://google.github.io/adk-docs/agents/multi-agent-systems/)
- [Documentación de Herramientas de Agente](https://google.github.io/adk-docs/tools/function-tools/#3-agent-as-a-tool)