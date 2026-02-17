"""FWBG REST API server (FastAPI)."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fwbg.api.analyze import router as analyze_router
from fwbg.api.plugins import router as plugins_router
from fwbg.api.strategies import router as strategies_router
from fwbg.api.runs import router as runs_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="FWBG API",
        description="REST API for the FWBG trading strategy framework",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(analyze_router, prefix="/api")
    app.include_router(plugins_router, prefix="/api")
    app.include_router(strategies_router, prefix="/api")
    app.include_router(runs_router, prefix="/api")

    return app


app = create_app()


def run_server(host: str = "0.0.0.0", port: int = 8420):
    """Start the API server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)
