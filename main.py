"""
main.py
--------
FastAPI entry point — now serves the chat UI directly.

Open http://localhost:8000 to see the chat interface.
API docs at http://localhost:8000/docs

Run: uvicorn main:app --reload
"""

import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from loguru import logger
from app.api.routes import router
from app.core.config import get_settings

settings = get_settings()

logger.remove()
logger.add(
    sys.stderr,
    level=settings.log_level,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
)

app = FastAPI(
    title="Compliance Document Assistant",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the chat UI template
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui(request: Request):
    """Serve the chat interface."""
    return templates.TemplateResponse("index.html", {"request": request})

# Mount API routes
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server → http://localhost:8000")
    uvicorn.run("main:app", host=settings.api_host, port=settings.api_port, reload=True)
