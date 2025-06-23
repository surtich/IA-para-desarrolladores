import uvicorn
from fastapi import FastAPI, HTTPException
from fastmcp import FastMCP
from pydantic import BaseModel

_products: dict[int, dict] = {}

mcp = FastMCP("AddServer", stateless_http=True)
mcp_app = mcp.http_app(path="/mcp")
app = FastAPI(lifespan=mcp_app.router.lifespan_context)
app.mount("/mcp-server", mcp_app)

class Product(BaseModel):
    name: str
    price: float


@app.get("/products")
def list_products():
    """List all products"""
    return list(_products.values())


@app.get("/products/{product_id}")
def get_product(product_id: int):
    """Get a product by its ID"""
    if product_id not in _products:
        raise HTTPException(status_code=404, detail="Product not found")
    return _products[product_id]


@app.post("/products")
def create_product(p: Product):
    """Create a new product"""
    new_id = len(_products) + 1
    _products[new_id] = {"id": new_id, **p.model_dump()}
    return _products[new_id]


@mcp.tool(description="Add two integers")
def add(a: int, b: int) -> int:
    return a + b


if __name__ == "__main__":
    uvicorn.run(app=app, host="127.0.0.1", port=8000)
