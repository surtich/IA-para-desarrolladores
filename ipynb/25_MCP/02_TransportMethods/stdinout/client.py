import asyncio
import sys

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def main() -> None:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["server.py"],
        env=None,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.call_tool("add", {"a": 7, "b": 5})
            print("7 + 5 =", res.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
