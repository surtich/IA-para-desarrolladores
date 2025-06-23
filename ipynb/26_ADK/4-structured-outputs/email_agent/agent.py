from google.adk.agents import LlmAgent
from pydantic import BaseModel, Field


# --- Define Output Schema ---
class EmailContent(BaseModel):
    subject: str = Field(
        description="La línea de asunto del correo electrónico. Debe ser concisa y descriptiva."
    )
    body: str = Field(
        description="El contenido principal del correo electrónico. Debe estar bien formateado con un saludo adecuado, párrafos y firma."
    )


# --- Create Email Generator Agent ---
root_agent = LlmAgent(
    name="email_agent",
    model="gemini-2.0-flash",
    instruction="""
        Eres un Asistente de Generación de Correos Electrónicos.
        Tu tarea es generar un correo electrónico profesional basado en la solicitud del usuario.

        DIRECTRICES:
        - Crea una línea de asunto apropiada (concisa y relevante)
        - Escribe un cuerpo de correo electrónico bien estructurado con:
            * Saludo profesional
            * Contenido principal claro y conciso
            * Cierre apropiado
            * Tu nombre como firma
        - Sugiere archivos adjuntos relevantes si aplica (lista vacía si no se necesitan)
        - El tono del correo electrónico debe coincidir con el propósito (formal para negocios, amigable para colegas)
        - Mantén los correos electrónicos concisos pero completos

        IMPORTANTE: Tu respuesta DEBE ser un JSON válido que coincida con esta estructura:
        {
            "subject": "Línea de asunto aquí",
            "body": "Cuerpo del correo electrónico aquí con párrafos y formato adecuados",
        }

        NO incluyas explicaciones ni texto adicional fuera de la respuesta JSON.
    """,
    description="Genera correos electrónicos profesionales con asunto y cuerpo estructurados",
    output_schema=EmailContent,
    output_key="email",
)
