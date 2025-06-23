import asyncio

from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from memory_agent.agent import memory_agent
from utils import call_agent_async

load_dotenv()

# ===== PARTE 1: Inicializar el Servicio de Sesión Persistente =====
# Usando la base de datos SQLite para almacenamiento persistente
db_url = "sqlite:///./my_agent_data.db"
session_service = DatabaseSessionService(db_url=db_url)


# ===== PARTE 2: Definir el Estado Inicial =====
# Esto solo se usará al crear una nueva sesión
initial_state = {
    "user_name": "Brandon Hancock",
    "reminders": [],
}


async def main_async():
    # Configurar constantes
    APP_NAME = "Agente de Memoria"
    USER_ID = "aiwithbrandon"

    # ===== PARTE 3: Gestión de Sesiones - Encontrar o Crear =====
    # Buscar sesiones existentes para este usuario
    existing_sessions = await session_service.list_sessions(
        app_name=APP_NAME,
        user_id=USER_ID,
    )

    # Si hay una sesión existente, usarla, de lo contrario crear una nueva
    if existing_sessions and len(existing_sessions.sessions) > 0:
        # Usar la sesión más reciente
        SESSION_ID = existing_sessions.sessions[0].id
        print(f"Continuando sesión existente: {SESSION_ID}")
    else:
        # Crear una nueva sesión con el estado inicial
        new_session = await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            state=initial_state,
        )
        SESSION_ID = new_session.id
        print(f"Nueva sesión creada: {SESSION_ID}")

    # ===== PARTE 4: Configuración del Ejecutor del Agente =====
    # Crear un ejecutor con el agente de memoria
    runner = Runner(
        agent=memory_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    # ===== PARTE 5: Bucle de Conversación Interactiva =====
    print("\n¡Bienvenido al Chat del Agente de Memoria!")
    print("Tus recordatorios serán recordados en todas las conversaciones.")
    print("Escribe 'exit' o 'quit' para finalizar la conversación.\n")

    while True:
        # Obtener la entrada del usuario
        user_input = input("Tú: ")

        # Comprobar si el usuario quiere salir
        if user_input.lower() in ["exit", "quit"]:
            print("Finalizando conversación. Tus datos han sido guardados en la base de datos.")
            break

        # Procesar la consulta del usuario a través del agente
        await call_agent_async(runner, USER_ID, SESSION_ID, user_input)


if __name__ == "__main__":
    asyncio.run(main_async())
