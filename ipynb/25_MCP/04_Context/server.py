import asyncio

from mcp.server.fastmcp import Context, FastMCP

mcp = FastMCP(
    name="ProgressDemoServer",
    stateless_http=False,
)


@mcp.tool(
    name="process_items", description="Processes a list of items with progress updates"
)
async def process_items(items: list[str], ctx: Context) -> list[str]:
    total = len(items)
    results: list[str] = []
    for i, item in enumerate(items, start=1):
        await ctx.info(f"Processing item {i}/{total}: {item}")
        await ctx.report_progress(progress=i, total=total)
        await asyncio.sleep(0.5)
        results.append(item.upper())
    return results


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
