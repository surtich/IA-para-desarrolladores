import os
import dotenv
import asyncio  # Importación necesaria
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

# Carga de variables de entorno
dotenv_path = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_path)

async def main():
    client = MultiServerMCPClient(
        {
            "weather": {
                "transport": "streamable_http",
                "url": "http://127.0.0.1:3000/mcp/",
            }
        }
    )

    model = ChatGoogleGenerativeAI(
       model="gemini-2.5-flash", # Asegúrate de usar una versión válida
       api_key=os.getenv("GOOGLE_API_KEY"),
    )
    
    tools = await client.get_tools()
    agent_executor = create_agent(model, tools)
    
    # Ejecución
    response = await agent_executor.ainvoke({"messages": ["¿Cuál es el clima en Madrid?"]})
    
    for message in response["messages"]:
        if isinstance(message, AIMessage):
            if message.tool_calls:
                print(f"AI calls: {message.tool_calls}")
            else:
                print(f"AI: {message.content}")
        elif isinstance(message, HumanMessage):
            print(f"Human: {message.content}")
        elif isinstance(message, ToolMessage):
            print(f"Tool: {message.content}")
        else:
            print(f"Message: {message.content}")

# La forma correcta de invocar la función principal asíncrona
if __name__ == "__main__":
    asyncio.run(main())