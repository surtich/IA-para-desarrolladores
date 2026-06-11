from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base

mcp = FastMCP("Recipe-Stateless", stateless_http=True)

_FAKE_DB = {
    "chili_con_carne": "Chili con Carne\n• Beans\n• Ground meat\n• Chili\n…",
    "pancakes": "Pancakes\n• Flour\n• Milk\n• Eggs\n…",
}


@mcp.resource("recipes://list")
def list_recipes() -> str:
    """Returns a comma-separated list of all available recipes."""
    return ", ".join(sorted(_FAKE_DB))


@mcp.resource("recipe://{dish}")
def get_recipe(dish: str) -> str:
    """Returns the recipe for the specified dish."""
    return _FAKE_DB.get(dish, f"No recipe found for {dish!r}.")


@mcp.tool(description="Doubles an integer.")
def double(n: int) -> int:
    return n * 2


@mcp.prompt()
def review_recipe(recipe: str) -> list[base.Message]:
    return [
        base.UserMessage("Please review this recipe:"),
        base.UserMessage(recipe),
    ]


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
