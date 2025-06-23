from fastmcp import FastMCP

mcp = FastMCP(name="WeatherServer", stateless_http=True)


@mcp.tool(
    name="get_weather",
    description="Returns a weather description for a given city",
)
def get_weather(city: str) -> str:
    """
    Args:
        city (str): Name of the city
    Returns:
        str: Description of the current weather (mock data)
    """
    return "Sunny, 22°C"


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=3000)
