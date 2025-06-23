import os
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider

load_dotenv(find_dotenv())

AUTH0_DOMAIN = os.environ["AUTH0_DOMAIN"]
API_AUDIENCE = os.environ.get("API_AUDIENCE", "http://localhost:8000/mcp")
REQUIRED_SCOPES = ["read:add"]

auth = BearerAuthProvider(
    jwks_uri=f"{AUTH0_DOMAIN.rstrip('/')}/.well-known/jwks.json",
    issuer=AUTH0_DOMAIN.rstrip("/") + "/",
    audience=API_AUDIENCE,
    required_scopes=REQUIRED_SCOPES,
)

mcp = FastMCP(
    name="SecureAddServer",
    stateless_http=True,
    auth=auth,
)

@mcp.tool(description="Add two integers")
def add(a: int, b: int) -> int:
    return a + b

mcp_app = mcp.http_app(path="/")
app = FastAPI(lifespan=mcp_app.lifespan)  # <--- Aquí está la clave

app.mount("/mcp", mcp_app)

@app.get("/.well-known/oauth-protected-resource")
def resource_metadata():
    return {
        "resource": "https://cce7-79-149-242-135.ngrok-free.app/mcp/",
        "authorization_servers": [AUTH0_DOMAIN.rstrip("/") + "/"],
        "scopes_supported": REQUIRED_SCOPES,
        "scope_descriptions": {
            "read:add": "Permite sumar dos enteros"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

