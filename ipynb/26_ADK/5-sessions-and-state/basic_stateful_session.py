import uuid

from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from question_answering_agent import question_answering_agent

load_dotenv()

async def main():
    # Create a new session service to store state
    session_service_stateful = InMemorySessionService()

    initial_state = {
        "user_name": "Brandon Hancock",
        "user_preferences": """
            Me gusta jugar Pickleball, Disc Golf y Tenis.
            Mi comida favorita es la mexicana.
            Mi programa de televisión favorito es Juego de Tronos.
            Le encanta cuando la gente le da "me gusta" y se suscribe a su canal de YouTube.
        """,
    }

    # Create a NEW session
    APP_NAME = "Brandon Bot"
    USER_ID = "brandon_hancock"
    SESSION_ID = str(uuid.uuid4())
    stateful_session = await session_service_stateful.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state=initial_state,
    )
    print("NUEVA SESIÓN CREADA:")
    print(f"\tID de Sesión: {SESSION_ID}")

    runner = Runner(
        agent=question_answering_agent,
        app_name=APP_NAME,
        session_service=session_service_stateful,
    )

    new_message = types.Content(
        role="user", parts=[types.Part(text="¿Cuál es el programa de televisión favorito de Brandon?")]
    )

    for event in runner.run(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=new_message,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                print(f"Respuesta Final: {event.content.parts[0].text}")

    print("==== Exploración de Eventos de Sesión ====")
    session = await session_service_stateful.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )

    # Log final Session state
    print("=== Estado Final de la Sesión ===")
    for key, value in session.state.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
