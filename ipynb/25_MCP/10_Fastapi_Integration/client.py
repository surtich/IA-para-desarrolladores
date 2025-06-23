import asyncio
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

SERVER = "http://127.0.0.1:8000/mcp/"


def section(title: str):
    print(f"\n{'=' * 10} {title} {'=' * 10}")


async def main() -> None:
    async with Client(StreamableHttpTransport(SERVER)) as session:

        tools = await session.list_tools()
        section("Available Tools")
        for tool in tools:
            print(f"Tool Name: {tool.name}")

        all_products = await session.call_tool(tools[0].name)
        section("All Products (Before)")
        print(all_products)


        create_tool_name = tools[1].name

        section(f"Calling Tool: {create_tool_name}")
        created = await session.call_tool(
            create_tool_name,
            {"name": "Widget", "price": 19.99},
        )
        print("Created product:", created[0].text)

        all_products = await session.call_tool(tools[0].name)
        section("All Products (After)")
        print(all_products)



if __name__ == "__main__":
    asyncio.run(main())
