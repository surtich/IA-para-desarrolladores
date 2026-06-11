from fastapi import FastAPI, HTTPException
from fastmcp import FastMCP
from pydantic import BaseModel

app = FastAPI(title="Product API")
_products: dict[int, dict] = {}

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

mcp = FastMCP.from_fastapi(app=app, name="ProductMCP")

@mcp.tool(description="Add two integers")
def add(a: int, b: int) -> int:
    return a + b

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8000)