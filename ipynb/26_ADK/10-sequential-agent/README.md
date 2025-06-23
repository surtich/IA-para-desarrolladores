# Agentes Secuenciales en ADK

Este ejemplo demuestra cómo implementar un Agente Secuencial en el Kit de Desarrollo de Agentes (ADK). El agente principal en este ejemplo, `lead_qualification_agent`, es un Agente Secuencial que ejecuta sub-agentes en un orden predefinido, con la salida de cada agente alimentando al siguiente agente en la secuencia.

## ¿Qué son los Agentes Secuenciales?

Los Agentes Secuenciales son agentes de flujo de trabajo en ADK que:

1. **Ejecutan en un Orden Fijo**: Los sub-agentes se ejecutan uno tras otro en la secuencia exacta en que se especifican.
2. **Pasan Datos entre Agentes**: Utilizan la gestión de estado para pasar información de un sub-agente al siguiente.
3. **Crean Pipelines de Procesamiento**: Perfectos para escenarios donde cada paso depende de la salida del paso anterior.

Utilice Agentes Secuenciales cuando necesite un flujo de trabajo determinista, paso a paso, donde el orden de ejecución es importante.

## Ejemplo de Pipeline de Calificación de Leads

En este ejemplo, hemos creado `lead_qualification_agent` como un Agente Secuencial que implementa un pipeline de calificación de leads para equipos de ventas. Este Agente Secuencial orquesta tres sub-agentes especializados:

1. **Agente Validador de Leads**: Verifica si la información del lead está lo suficientemente completa para la calificación.
   - Valida la información requerida como detalles de contacto e interés.
   - Emite un simple "válido" o "inválido" con una razón.

2. **Agente Calificador de Leads**: Califica leads válidos en una escala del 1 al 10.
   - Analiza factores como la urgencia, la autoridad para tomar decisiones, el presupuesto y el cronograma.
   - Proporciona una puntuación numérica con una breve justificación.

3. **Agente Recomendador de Acciones**: Sugiere los siguientes pasos basándose en la validación y la puntuación.
   - Para leads inválidos: Recomienda qué información recopilar.
   - Para leads con puntuación baja (1-3): Sugiere acciones de fomento.
   - Para leads con puntuación media (4-7): Sugiere acciones de calificación.
   - Para leads con puntuación alta (8-10): Sugiere acciones de venta.

### Cómo Funciona

El Agente Secuencial `lead_qualification_agent` orquesta este proceso al:

1. Ejecutar primero el Validador para determinar si el lead está completo.
2. Ejecutar luego el Calificador (que puede acceder a los resultados de validación a través del estado).
3. Ejecutar finalmente el Recomendador (que puede acceder tanto a los resultados de validación como a los de calificación).

La salida de cada sub-agente se almacena en el estado de la sesión utilizando el parámetro `output_key`:
- `validation_status`
- `lead_score`
- `action_recommendation`

## Estructura del Proyecto

```
9-sequential-agent/
│
├── lead_qualification_agent/       # Paquete principal del Agente Secuencial
│   ├── __init__.py                 # Inicialización del paquete
│   ├── agent.py                    # Definición del Agente Secuencial (root_agent)
│   │
│   └── subagents/                  # Carpeta de sub-agentes
│       ├── __init__.py             # Inicialización de sub-agentes
│       │
│       ├── validator/              # Agente de validación de leads
│       │   ├── __init__.py
│       │   └── agent.py
│       │
│       ├── scorer/                 # Agente de calificación de leads
│       │   ├── __init__.py
│       │   └── agent.py
│       │
│       └── recommender/            # Agente de recomendación de acciones
│           ├── __init__.py
│           └── agent.py
│
├── .env.example                    # Ejemplo de variables de entorno
└── README.md                       # Esta documentación
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

2. Copie el archivo `.env.example` a `.env` y añada su clave API de Google:
```
GOOGLE_API_KEY=su_clave_api_aqui
```

### Ejecutando el Ejemplo

```bash
cd 9-sequential-agent
adk web
```

Luego seleccione "lead_qualification_agent" del menú desplegable en la interfaz de usuario web.

## Interacciones de Ejemplo

Pruebe estas interacciones de ejemplo:

### Ejemplo de Lead Calificado:
```
Información del Lead:
Nombre: Sarah Johnson
Correo electrónico: sarah.j@techinnovate.com
Teléfono: 555-123-4567
Empresa: Tech Innovate Solutions
Puesto: CTO
Interés: Buscando una solución de IA para automatizar el soporte al cliente
Presupuesto: $50K-100K disponible para la solución adecuada
Cronograma: Esperando implementar dentro del próximo trimestre
Notas: Actualmente usando el producto de un competidor pero insatisfecha con el rendimiento
```

### Ejemplo de Lead No Calificado:
```
Información del Lead:
Nombre: John Doe
Correo electrónico: john@gmail.com
Interés: Algo con IA quizás
Notas: Conocido en una conferencia, parecía interesado pero fue vago sobre sus necesidades
```

## Cómo se Comparan los Agentes Secuenciales con Otros Agentes de Flujo de Trabajo

ADK ofrece diferentes tipos de agentes de flujo de trabajo para diferentes necesidades:

- **Agentes Secuenciales**: Para una ejecución estricta y ordenada (como este ejemplo)
- **Agentes de Bucle**: Para la ejecución repetida de sub-agentes basada en condiciones
- **Agentes Paralelos**: Para la ejecución concurrente de sub-agentes independientes

## Recursos Adicionales

- [Documentación de Agentes Secuenciales de ADK](https://google.github.io/adk-docs/agents/workflow-agents/sequential-agents/)
- [Ejemplo Completo de Pipeline de Desarrollo de Código](https://google.github.io/adk-docs/agents/workflow-agents/sequential-agents/#full-example-code-development-pipeline)
