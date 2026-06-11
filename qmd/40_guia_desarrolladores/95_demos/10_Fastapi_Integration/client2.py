import asyncio
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

SERVER = "http://127.0.0.1:8000/mcp-server/mcp"

def section(title: str):
    print(f"\n{'=' * 10} {title} {'=' * 10}")


async def main() -> None:
    async with Client(StreamableHttpTransport(SERVER)) as session:

        tools = await session.list_tools()
        section("Available Tools")
        for tool in tools:
            print(f"Tool Name: {tool.name}")

        section("Calling Tool: add")
        result = await session.call_tool(
            "add", 
            arguments={"a": 10, "b": 5}
        )

        print("Resultado de la suma:", result.content[0].text)



if __name__ == "__main__":
    asyncio.run(main())
