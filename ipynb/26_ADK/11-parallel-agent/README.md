# Agentes Paralelos en ADK

Este ejemplo demuestra cómo implementar un Agente Paralelo en el Kit de Desarrollo de Agentes (ADK). El agente principal en este ejemplo, `system_monitor_agent`, utiliza un Agente Paralelo para recopilar información del sistema de forma concurrente y luego la sintetiza en un informe completo de salud del sistema.

## ¿Qué son los Agentes Paralelos?

Los Agentes Paralelos son agentes de flujo de trabajo en ADK que:

1. **Ejecutan Concurrently**: Los sub-agentes se ejecutan simultáneamente en lugar de secuencialmente
2. **Operan Independientemente**: Cada sub-agente trabaja de forma independiente sin compartir estado durante la ejecución
3. **Mejoran el Rendimiento**: Aceleran drásticamente los flujos de trabajo donde las tareas pueden realizarse en paralelo

Utilice Agentes Paralelos cuando necesite ejecutar múltiples tareas independientes de manera eficiente y el tiempo sea un factor crítico.

## Ejemplo de Monitoreo del Sistema

En este ejemplo, hemos creado una aplicación de monitoreo del sistema que utiliza un Agente Paralelo para recopilar información del sistema. El flujo de trabajo consiste en:

1. **Recopilación Paralela de Información del Sistema**: Utilizando un `ParallelAgent` para recopilar datos concurrentemente sobre:
   - Uso y estadísticas de la CPU
   - Utilización de la memoria
   - Espacio y uso del disco

2. **Síntesis Secuencial del Informe**: Después de la recopilación de datos en paralelo, un agente sintetizador combina toda la información en un informe completo

### Sub-Agentes

1. **Agente de Información de CPU**: Recopila y analiza la información de la CPU
   - Recupera el número de núcleos, estadísticas de uso y métricas de rendimiento
   - Identifica posibles problemas de rendimiento (alto uso de CPU)

2. **Agente de Información de Memoria**: Recopila información sobre el uso de la memoria
   - Recopila la memoria total, usada y disponible
   - Analiza la presión de la memoria y el uso del intercambio

3. **Agente de Información de Disco**: Analiza el espacio y el uso del disco
   - Informa sobre el espacio total, usado y libre del disco
   - Identifica los discos que se están quedando sin espacio

4. **Sintetizador de Informes del Sistema**: Combina toda la información recopilada en un informe completo de salud del sistema
   - Crea un resumen ejecutivo de la salud del sistema
   - Organiza la información específica de los componentes en secciones
   - Proporciona recomendaciones basadas en las métricas del sistema

### Cómo Funciona

La arquitectura combina patrones de flujo de trabajo tanto paralelos como secuenciales:

1. Primero, el Agente Paralelo `system_info_gatherer` ejecuta los tres agentes de información concurrentemente
2. Luego, el `system_report_synthesizer` utiliza los datos recopilados para generar un informe final

Este enfoque híbrido demuestra cómo combinar tipos de agentes de flujo de trabajo para un rendimiento óptimo y un flujo lógico.

## Estructura del Proyecto

```
10-parallel-agent/
│
├── system_monitor_agent/          # Paquete principal del Agente de Monitoreo del Sistema
│   ├── __init__.py                # Inicialización del paquete
│   ├── agent.py                   # Definiciones del agente (root_agent)
│   │
│   └── subagents/                 # Carpeta de sub-agentes
│       ├── __init__.py            # Inicialización de sub-agentes
│       │
│       ├── cpu_info_agent/        # Agente de información de CPU
│       │   ├── __init__.py
│       │   ├── agent.py
│       │   └── tools.py           # Herramientas de recopilación de información de CPU
│       │
│       ├── memory_info_agent/     # Agente de información de memoria
│       │   ├── __init__.py
│       │   ├── agent.py
│       │   └── tools.py           # Herramientas de recopilación de información de memoria
│       │
│       ├── disk_info_agent/       # Agente de información de disco
│       │   ├── __init__.py
│       │   ├── agent.py
│       │   └── tools.py           # Herramientas de recopilación de información de disco
│       │
│       └── synthesizer_agent/     # Agente sintetizador de informes
│           ├── __init__.py
│           └── agent.py
│
├── .env.example                   # Ejemplo de variables de entorno
└── README.md                      # Esta documentación
```

## Primeros Pasos

### Configuración

1. Active el entorno virtual desde el directorio raíz:
```bash
# macOS/Linux:
source ../.venv/bin/activate
# Windows CMD:
..\.venv\Scripts\activate.bat
# Windows PowerShell:
..\.venv\Scripts\Activate.ps1
```

2. Copie el archivo `.env.example` a `.env` y agregue su clave API de Google:
```
GOOGLE_API_KEY=su_clave_api_aqui
```

### Ejecutando el Ejemplo

```bash
cd 10-parallel-agent
adk web
```

Luego seleccione "system_monitor_agent" del menú desplegable en la interfaz de usuario web.

## Interacciones de Ejemplo

Pruebe estas indicaciones de ejemplo:

```
Verifica la salud de mi sistema
```

```
Proporciona un informe completo del sistema con recomendaciones
```

```
¿Mi sistema se está quedando sin memoria o espacio en disco?
```

## Conceptos Clave: Ejecución Independiente

Un aspecto clave de los Agentes Paralelos es que **los sub-agentes se ejecutan de forma independiente sin compartir estado durante la ejecución**. En este ejemplo:

1. Cada agente de recopilación de información opera de forma aislada
2. Los resultados de cada agente se recopilan una vez que finaliza la ejecución paralela
3. El agente sintetizador luego utiliza estos resultados recopilados para crear el informe final

Este enfoque es ideal para escenarios donde las tareas son completamente independientes y no requieren interacción durante la ejecución.

## Cómo se Comparan los Agentes Paralelos con Otros Agentes de Flujo de Trabajo

ADK ofrece diferentes tipos de agentes de flujo de trabajo para diferentes necesidades:

- **Agentes Secuenciales**: Para una ejecución estricta y ordenada donde cada paso depende de las salidas anteriores
- **Agentes de Bucle**: Para la ejecución repetida de sub-agentes basada en condiciones
- **Agentes Paralelos**: Para la ejecución concurrente de sub-agentes independientes (como este ejemplo)

## Recursos Adicionales

- [Documentación de Agentes Paralelos de ADK](https://google.github.io/adk-docs/agents/workflow-agents/parallel-agents/)
- [Ejemplo Completo: Investigación Web Paralela](https://google.github.io/adk-docs/agents/workflow-agents/parallel-agents/#full-example-parallel-web-research)
