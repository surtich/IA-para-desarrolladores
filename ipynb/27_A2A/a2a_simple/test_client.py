import uuid

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    Message,
    MessageSendParams,
    Part,
    Role,
    SendMessageRequest,
    TextPart,
)

PUBLIC_AGENT_CARD_PATH = "/.well-known/agent.json"
BASE_URL = "http://localhost:9999"


async def main() -> None:
    async with httpx.AsyncClient() as httpx_client:
        # Inicializar A2ACardResolver
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=BASE_URL,
        )

        final_agent_card_to_use: AgentCard | None = None

        try:
            print(
                f"Obteniendo tarjeta de agente pública de: {BASE_URL}{PUBLIC_AGENT_CARD_PATH}"
            )
            _public_card = await resolver.get_agent_card()
            print("Tarjeta de agente pública obtenida")
            print(_public_card.model_dump_json(indent=2))

            final_agent_card_to_use = _public_card

        except Exception as e:
            print(f"Error al obtener la tarjeta de agente pública: {e}")
            raise RuntimeError("Fallo al obtener la tarjeta de agente pública")

        client = A2AClient(
            httpx_client=httpx_client, agent_card=final_agent_card_to_use
        )
        print("A2AClient inicializado")

        message_payload = Message(
            role=Role.user,
            messageId=str(uuid.uuid4()),
            parts=[Part(root=TextPart(text="Hola, ¿cómo estás?"))],
        )
        request = SendMessageRequest(
            id=str(uuid.uuid4()),
            params=MessageSendParams(
                message=message_payload,
            ),
        )
        print("Enviando mensaje")

        response = await client.send_message(request)
        print("Respuesta:")
        print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
