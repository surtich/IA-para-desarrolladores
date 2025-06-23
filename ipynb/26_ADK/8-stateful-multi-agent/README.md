# Sistemas Multiagente con Estado en ADK

Este ejemplo demuestra cómo crear un sistema multiagente con estado en ADK, combinando el poder de la gestión de estado persistente con la delegación de agentes especializados. Este enfoque crea sistemas de agentes inteligentes que recuerdan la información del usuario a través de las interacciones mientras aprovechan la experiencia especializada en el dominio.

## ¿Qué es un Sistema Multiagente con Estado?

Un Sistema Multiagente con Estado combina dos patrones poderosos:

1. **Gestión de Estado**: Persistir información sobre usuarios y conversaciones a través de las interacciones.
2. **Arquitectura Multiagente**: Distribuir tareas entre agentes especializados según su experiencia.

El resultado es un ecosistema de agentes sofisticado que puede:
- Recordar la información del usuario y el historial de interacciones.
- Enrutar consultas al agente especializado más apropiado.
- Proporcionar respuestas personalizadas basadas en interacciones pasadas.
- Mantener el contexto a través de múltiples delegados de agentes.

Este ejemplo implementa un sistema de servicio al cliente para una plataforma de cursos en línea, donde agentes especializados manejan diferentes aspectos del soporte al cliente mientras comparten un estado común.

## Estructura del Proyecto

```
7-stateful-multi-agent/
│
├── customer_service_agent/         # Paquete principal del agente
│   ├── __init__.py                 # Requerido para el descubrimiento de ADK
│   ├── agent.py                    # Definición del agente raíz
│   └── sub_agents/                 # Agentes especializados
│       ├── course_support_agent/   # Maneja preguntas sobre el contenido del curso
│       ├── order_agent/            # Gestiona el historial de pedidos y reembolsos
│       ├── policy_agent/           # Responde preguntas sobre políticas
│       └── sales_agent/            # Maneja las compras de cursos
│
├── main.py                         # Punto de entrada de la aplicación con configuración de sesión
├── utils.py                        # Funciones de ayuda para la gestión de estado
├── .env                            # Variables de entorno
└── README.md                       # Esta documentación
```

## Componentes Clave

### 1. Gestión de Sesiones

El ejemplo utiliza `InMemorySessionService` para almacenar el estado de la sesión:

```python
session_service = InMemorySessionService()

def initialize_state():
    """Inicializa el estado de la sesión con valores predeterminados."""
    return {
        "user_name": "Brandon Hancock",
        "purchased_courses": [""],
        "interaction_history": [],
    }

# Crea una nueva sesión con el estado inicial
session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID,
    state=initialize_state(),
)
```

### 2. Compartir Estado entre Agentes

Todos los agentes del sistema pueden acceder al mismo estado de sesión, lo que permite:
- Que el agente raíz rastree el historial de interacciones.
- Que el agente de ventas actualice los cursos comprados.
- Que el agente de soporte de cursos verifique si el usuario ha comprado cursos específicos.
- Que todos los agentes personalicen las respuestas basándose en la información del usuario.

### 3. Delegación Multiagente

El agente de servicio al cliente enruta las consultas a subagentes especializados:

```python
customer_service_agent = Agent(
    name="customer_service",
    model="gemini-2.0-flash",
    description="Agente de servicio al cliente para la comunidad de AI Developer Accelerator",
    instruction="""
    Eres el agente principal de servicio al cliente para la comunidad de AI Developer Accelerator.
    Tu función es ayudar a los usuarios con sus preguntas y dirigirlos al agente especializado apropiado.
  
    # ... instrucciones detalladas ...
  
    """,
    sub_agents=[policy_agent, sales_agent, course_support_agent, order_agent],
    tools=[get_current_time],
)
```

## Cómo Funciona

1. **Creación de Sesión Inicial**:
   - Se crea una nueva sesión con la información del usuario y el historial de interacciones vacío.
   - El estado de la sesión se inicializa con valores predeterminados.

2. **Seguimiento de Conversaciones**:
   - Cada mensaje del usuario se agrega a `interaction_history` en el estado.
   - Los agentes pueden revisar interacciones pasadas para mantener el contexto.

3. **Enrutamiento de Consultas**:
   - El agente raíz analiza la consulta del usuario y decide qué especialista debe manejarla.
   - Los agentes especializados reciben el contexto de estado completo cuando se les delega.

4. **Actualizaciones de Estado**:
   - Cuando un usuario compra un curso, el agente de ventas actualiza `purchased_courses`.
   - Estas actualizaciones están disponibles para todos los agentes para futuras interacciones.

5. **Respuestas Personalizadas**:
   - Los agentes adaptan las respuestas basándose en el historial de compras y las interacciones anteriores.
   - Se toman diferentes caminos según lo que el usuario ya haya comprado.

## Primeros Pasos

### Configuración

1. Activa el entorno virtual desde el directorio raíz:
```bash
# macOS/Linux:
source ../.venv/bin/activate
# Windows CMD:
..\.venv\Scripts\activate.bat
# Windows PowerShell:
..\.venv\Scripts\Activate.ps1
```

2. Asegúrate de que tu clave de API de Google esté configurada en el archivo `.env`:
```
GOOGLE_API_KEY=tu_clave_api_aquí
```

### Ejecutando el Ejemplo

Para ejecutar el ejemplo multiagente con estado:

```bash
python main.py
```

Esto hará lo siguiente:
1. Inicializará una nueva sesión con el estado predeterminado.
2. Iniciará una conversación interactiva con el agente de servicio al cliente.
3. Rastrea todas las interacciones en el estado de la sesión.
4. Permitirá que los agentes especializados manejen consultas específicas.

### Flujo de Conversación de Ejemplo

Prueba este flujo de conversación para probar el sistema:

1. **Comienza con una consulta general**:
   - "¿Qué cursos ofrecen?"
   - (El agente raíz enrutará al agente de ventas)

2. **Pregunta sobre la compra**:
   - "Quiero comprar el curso de Plataforma de Marketing de IA"
   - (El agente de ventas procesará la compra y actualizará el estado)

3. **Pregunta sobre el contenido del curso**:
   - "¿Puedes hablarme sobre el contenido del curso de Plataforma de Marketing de IA?"
   - (El agente raíz enrutará al agente de soporte de cursos, que ahora tiene acceso)

4. **Pregunta sobre reembolsos**:
   - "¿Cuál es su política de reembolso?"
   - (El agente raíz enrutará al agente de políticas)

¡Observa cómo el sistema recuerda tu compra a través de diferentes agentes especializados!

## Características Avanzadas

### 1. Seguimiento del Historial de Interacciones

El sistema mantiene un historial de interacciones para proporcionar contexto:

```python
# Actualiza el historial de interacciones con la consulta del usuario
add_user_query_to_history(
    session_service, APP_NAME, USER_ID, SESSION_ID, user_input
)
```

### 2. Control de Acceso Dinámico

El sistema implementa acceso condicional a ciertos agentes:

```
3. Agente de Soporte de Cursos
   - Para preguntas sobre el contenido del curso
   - Solo disponible para cursos que el usuario ha comprado
   - Verifica si "ai_marketing_platform" está en los cursos comprados antes de dirigir aquí
```

### 3. Personalización Basada en el Estado

Todos los agentes adaptan las respuestas basándose en el estado de la sesión:

```
Adapta tus respuestas basándote en el historial de compras del usuario y las interacciones anteriores.
Cuando el usuario aún no ha comprado ningún curso, anímalo a explorar la Plataforma de Marketing de IA.
Cuando el usuario ha comprado cursos, ofrece soporte para esos cursos específicos.
```

## Consideraciones de Producción

Para una implementación en producción, considera:

1. **Almacenamiento Persistente**: Reemplaza `InMemorySessionService` con `DatabaseSessionService` para persistir el estado a través de los reinicios de la aplicación.
2. **Autenticación de Usuario**: Implementa una autenticación de usuario adecuada para identificar a los usuarios de forma segura.
3. **Manejo de Errores**: Agrega un manejo de errores robusto para fallas de agentes y corrupción de estado.
4. **Monitoreo**: Implementa registro y monitoreo para rastrear el rendimiento del sistema.

## Recursos Adicionales

- [Documentación de Sesiones de ADK](https://google.github.io/adk-docs/sessions/session/)
- [Documentación de Sistemas Multiagente de ADK](https://google.github.io/adk-docs/agents/multi-agent-systems/)
- [Gestión de Estado en ADK](https://google.github.io/adk-docs/sessions/state/)
