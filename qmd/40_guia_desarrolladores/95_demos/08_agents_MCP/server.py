from fastmcp import FastMCP

mcp = FastMCP(name="WeatherServer")

@mcp.tool(
    name="get_weather",
    description="Returns a weather description for a given city",
)
def get_weather(city: str) -> str:
    return "Sunny, 22°C"

if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        stateless=True,
        host="127.0.0.1",
        port=3000
    )