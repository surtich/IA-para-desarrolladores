import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

SERVER = "http://127.0.0.1:8000/mcp/"


async def main() -> None:
    async with streamablehttp_client(SERVER) as (read, write, _):
        async with ClientSession(read, write) as session:
            resources = await session.list_resources()
            print("Resources:", [r.uri for r in resources.resources])

            tools = await session.list_tools()
            print("Tools:", [t.name for t in tools.tools])

            prompts = await session.list_prompts()
            print("Prompts:", prompts.prompts)

            recipe_response = await session.read_resource("recipe://chili_con_carne")
            recipe_text = recipe_response.contents[0].text
            print("\nRecipe:\n", recipe_text)

            doubled_response = await session.call_tool("double", {"n": 21})
            doubled_value = doubled_response.content[0].text
            print(f"\n21 doubled -> {doubled_value}")

            prompt_response = await session.get_prompt(
                "review_recipe",
                {"recipe": recipe_text},
            )
            print("\nPrompt messages:")
            for message in prompt_response.messages:
                print(f"[{message.role}] {message.content.text}")


if __name__ == "__main__":
    asyncio.run(main())
