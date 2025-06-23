# Almacenamiento Persistente en ADK

Este ejemplo demuestra cómo implementar almacenamiento persistente para tus agentes ADK, permitiéndoles recordar información y mantener el historial de conversaciones a través de múltiples sesiones, reinicios de aplicaciones e incluso despliegues de servidores.

## ¿Qué es el Almacenamiento Persistente en ADK?

En ejemplos anteriores, usamos `InMemorySessionService` que almacena los datos de la sesión solo en memoria; estos datos se pierden cuando la aplicación se detiene. Para aplicaciones del mundo real, a menudo necesitarás que tus agentes recuerden la información del usuario y el historial de conversaciones a largo plazo. Aquí es donde entra en juego el almacenamiento persistente.

ADK proporciona el `DatabaseSessionService` que te permite almacenar datos de sesión en una base de datos SQL, asegurando:

1. **Memoria a largo plazo**: La información persiste a través de los reinicios de la aplicación.
2. **Experiencias de usuario consistentes**: Los usuarios pueden continuar las conversaciones donde las dejaron.
3. **Soporte multiusuario**: Los datos de diferentes usuarios permanecen separados y seguros.
4. **Escalabilidad**: Funciona con bases de datos de producción para despliegues a gran escala.

Este ejemplo muestra cómo implementar un agente de recordatorios que recuerda tu nombre y tus tareas pendientes a través de diferentes conversaciones usando una base de datos SQLite.

## Estructura del Proyecto

```
5-persistent-storage/
│
├── memory_agent/               # Paquete del agente
│   ├── __init__.py             # Requerido para que ADK descubra el agente
│   └── agent.py                # Definición del agente con herramientas de recordatorio
│
├── main.py                     # Punto de entrada de la aplicación con configuración de sesión de base de datos
├── utils.py                    # Funciones de utilidad para la interfaz de usuario del terminal y la interacción con el agente
├── .env                        # Variables de entorno
├── my_agent_data.db            # Archivo de base de datos SQLite (creado en la primera ejecución)
└── README.md                   # Esta documentación
```

## Componentes Clave

### 1. DatabaseSessionService

El componente central que proporciona persistencia es el `DatabaseSessionService`, que se inicializa con una URL de base de datos:

```python
from google.adk.sessions import DatabaseSessionService

db_url = "sqlite:///./my_agent_data.db"
session_service = DatabaseSessionService(db_url=db_url)
```

Este servicio permite a ADK:
- Almacenar datos de sesión en un archivo de base de datos SQLite
- Recuperar sesiones anteriores para un usuario
- Gestionar automáticamente los esquemas de la base de datos

### 2. Gestión de Sesiones

El ejemplo demuestra una gestión de sesiones adecuada:

```python
# Comprobar si existen sesiones para este usuario
existing_sessions = await session_service.list_sessions(
    app_name=APP_NAME,
    user_id=USER_ID,
)

# Si hay una sesión existente, úsala; de lo contrario, crea una nueva
if existing_sessions and len(existing_sessions.sessions) > 0:
    # Usar la sesión más reciente
    SESSION_ID = existing_sessions.sessions[0].id
    print(f"Continuando sesión existente: {SESSION_ID}")
else:
    # Crear una nueva sesión con estado inicial
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state=initialize_state(),
    )
```

### 3. Gestión de Estado con Herramientas

El agente incluye herramientas que actualizan el estado persistente:

```python
def add_reminder(reminder: str, tool_context: ToolContext) -> dict:
    # Obtener recordatorios actuales del estado
    reminders = tool_context.state.get("reminders", [])
    
    # Añadir el nuevo recordatorio
    reminders.append(reminder)
    
    # Actualizar el estado con la nueva lista de recordatorios
    tool_context.state["reminders"] = reminders
    
    return {
        "action": "add_reminder",
        "reminder": reminder,
        "message": f"Recordatorio añadido: {reminder}",
    }
```

Cada cambio en `tool_context.state` se guarda automáticamente en la base de datos.

## Primeros Pasos

### Prerrequisitos

- Python 3.9+
- Clave API de Google para modelos Gemini
- SQLite (incluido con Python)

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

2. Asegúrate de que tu clave API de Google esté configurada en el archivo `.env`:
```
GOOGLE_API_KEY=tu_clave_api_aquí
```

### Ejecutando el Ejemplo

Para ejecutar el ejemplo de almacenamiento persistente:

```bash
python main.py
```

Esto hará lo siguiente:
1. Conectarse a la base de datos SQLite (o crearla si no existe)
2. Comprobar sesiones anteriores para el usuario
3. Iniciar una conversación con el agente de memoria
4. Guardar todas las interacciones en la base de datos

### Interacciones de Ejemplo

Prueba estas interacciones para probar la memoria persistente del agente:

1. **Primera ejecución:**
   - "¿Cuál es mi nombre?"
   - "Mi nombre es John"
   - "Añade un recordatorio para comprar comestibles"
   - "Añade otro recordatorio para terminar el informe"
   - "¿Cuáles son mis recordatorios?"
   - Salir del programa con "exit"

2. **Segunda ejecución:**
   - "¿Cuál es mi nombre?"
   - "¿Qué recordatorios tengo?"
   - "Actualiza mi segundo recordatorio para entregar el informe antes del viernes"
   - "Elimina el primer recordatorio"
   
¡El agente recordará tu nombre y tus recordatorios entre ejecuciones!

## Uso del Almacenamiento de Base de Datos en Producción

Aunque este ejemplo utiliza SQLite por simplicidad, `DatabaseSessionService` admite varios backends de bases de datos a través de SQLAlchemy:

- PostgreSQL: `postgresql://usuario:contraseña@localhost/nombre_bd`
- MySQL: `mysql://usuario:contraseña@localhost/nombre_bd`
- MS SQL Server: `mssql://usuario:contraseña@localhost/nombre_bd`

Para uso en producción:
1. Elige un sistema de base de datos que satisfaga tus necesidades de escalabilidad
2. Configura la agrupación de conexiones para mayor eficiencia
3. Implementa la seguridad adecuada para las credenciales de la base de datos
4. Considera las copias de seguridad de la base de datos para datos críticos del agente

## Recursos Adicionales

- [Documentación de Sesiones de ADK](https://google.github.io/adk-docs/sessions/session/)
- [Implementaciones del Servicio de Sesiones](https://google.github.io/adk-docs/sessions/session/#sessionservice-implementations)
- [Gestión de Estado en ADK](https://google.github.io/adk-docs/sessions/state/)
- [Documentación de SQLAlchemy](https://docs.sqlalchemy.org/) para configuración avanzada de bases de datos
