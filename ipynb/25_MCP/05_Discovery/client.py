import asyncio
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport


async def main():
    async with Client(StreamableHttpTransport("http://127.0.0.1:8000/mcp/")) as c:
        print("Tools BEFORE :", [t.name for t in await c.list_tools()])
        response = await c.call_tool("router", {"text": "please make this upper CASE"})
        print("Response   :", response)
        print("Tools AFTER  :", [t.name for t in await c.list_tools()])


if __name__ == "__main__":
    asyncio.run(main())
