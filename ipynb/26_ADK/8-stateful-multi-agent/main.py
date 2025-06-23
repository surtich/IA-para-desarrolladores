import asyncio

# Importar el agente principal de servicio al cliente
from customer_service_agent.agent import customer_service_agent
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from utils import add_user_query_to_history, call_agent_async

load_dotenv()

# ===== PARTE 1: Inicializar el Servicio de Sesiones en Memoria =====
# Usando almacenamiento en memoria para este ejemplo (no persistente)
session_service = InMemorySessionService()


# ===== PARTE 2: Definir el Estado Inicial =====
# Esto se usará al crear una nueva sesión
initial_state = {
    "user_name": "Brandon Hancock",
    "purchased_courses": [],
    "interaction_history": [],
}


async def main_async():
    # Configurar constantes
    APP_NAME = "Customer Support"
    USER_ID = "aiwithbrandon"

    # ===== PARTE 3: Creación de Sesión =====
    # Crear una nueva sesión con el estado inicial
    new_session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        state=initial_state,
    )
    SESSION_ID = new_session.id
    print(f"Nueva sesión creada: {SESSION_ID}")

    # ===== PARTE 4: Configuración del Ejecutor de Agentes =====
    # Crear un ejecutor con el agente principal de servicio al cliente
    runner = Runner(
        agent=customer_service_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    # ===== PARTE 5: Bucle de Conversación Interactiva =====
    print("\n¡Bienvenido al Chat de Servicio al Cliente!")
    print("Escribe 'exit' o 'quit' para finalizar la conversación.\n")

    while True:
        # Obtener la entrada del usuario
        user_input = input("Tú: ")

        # Verificar si el usuario quiere salir
        if user_input.lower() in ["exit", "quit"]:
            print("Finalizando conversación. ¡Adiós!")
            break

        # Actualizar el historial de interacciones con la consulta del usuario
        await add_user_query_to_history(
            session_service, APP_NAME, USER_ID, SESSION_ID, user_input
        )

        # Procesar la consulta del usuario a través del agente
        await call_agent_async(runner, USER_ID, SESSION_ID, user_input)

    # ===== PARTE 6: Examen del Estado =====
    # Mostrar el estado final de la sesión
    final_session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    print("\nEstado final de la sesión:")
    for key, value in final_session.state.items():
        print(f"{key}: {value}")


def main():
    """Punto de entrada para la aplicación."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
