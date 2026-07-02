"""FWBG REST API server (FastAPI)."""
import hmac
import logging
import os

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from fwbg.api.workspace import init_workspace
from fwbg.api.exploration import exit_optimization_router
from fwbg.api.plugins import router as plugins_router, entry_modifiers_router, exit_modifiers_router
from fwbg.api.presets import migrate_presets, migrate_strategy_refs, router as presets_router
from fwbg.api.strategies import router as strategies_router
from fwbg.api.runs import router as runs_router
from fwbg.api.chart import router as chart_router
from fwbg.api.custom_signals import router as custom_signals_router
from fwbg.api.datasources import router as datasources_router
from fwbg.api.assets import router as assets_router
from fwbg.api.data import router as data_router
from fwbg.api.signal_composer import router as signal_composer_router
from fwbg.api.analysis import router as analysis_router
from fwbg.api.discovery import router as discovery_router

log = logging.getLogger(__name__)


# Paths that bypass API-key checks even when auth is enabled.
_AUTH_BYPASS_PATHS = ("/docs", "/redoc", "/openapi.json", "/health")


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Simple X-API-Key check, enabled via FWBG_API_KEY env var.

    When FWBG_API_KEY is unset the middleware is a no-op so existing
    single-user setups keep working. When set, every request to /api/* must
    carry a matching X-API-Key header.
    """

    def __init__(self, app, api_key: str):
        super().__init__(app)
        self._api_key = api_key

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if not path.startswith("/api") or path.startswith(_AUTH_BYPASS_PATHS):
            return await call_next(request)
        provided = request.headers.get("x-api-key", "")
        if not hmac.compare_digest(provided, self._api_key):
            # BaseHTTPMiddleware does not route HTTPException through FastAPI's
            # exception handlers, so we must return the response directly.
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid or missing API key"},
            )
        return await call_next(request)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    ws = init_workspace()
    log.info(f"FWBG workspace: {ws}")
    migrate_presets()
    migrate_strategy_refs()

    app = FastAPI(
        title="FWBG API",
        description="REST API for the FWBG trading strategy framework",
        version="1.0.0",
    )

    # CORS: wildcard origins are incompatible with credentials. Configure
    # FWBG_CORS_ORIGINS as a comma-separated list of trusted origins for
    # browser deployments. Default keeps the dashboard origins.
    cors_env = os.environ.get("FWBG_CORS_ORIGINS", "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173")
    cors_origins = [o.strip() for o in cors_env.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api_key = os.environ.get("FWBG_API_KEY", "").strip()
    if api_key:
        app.add_middleware(APIKeyMiddleware, api_key=api_key)
        log.info("API key authentication enabled")
    else:
        log.warning("FWBG_API_KEY not set — API is unauthenticated. Set FWBG_API_KEY for production.")

    app.include_router(exit_optimization_router, prefix="/api/exploration/exit-optimization")
    app.include_router(plugins_router, prefix="/api")
    app.include_router(presets_router, prefix="/api")
    app.include_router(strategies_router, prefix="/api")
    app.include_router(runs_router, prefix="/api")
    app.include_router(chart_router, prefix="/api")
    app.include_router(custom_signals_router, prefix="/api")
    app.include_router(datasources_router, prefix="/api")
    app.include_router(assets_router, prefix="/api")
    app.include_router(data_router, prefix="/api")
    app.include_router(exit_modifiers_router, prefix="/api")
    app.include_router(entry_modifiers_router, prefix="/api")
    app.include_router(signal_composer_router, prefix="/api")
    app.include_router(analysis_router, prefix="/api")
    app.include_router(discovery_router, prefix="/api")

    return app


app = create_app()


def run_server(host: str = "0.0.0.0", port: int = 8420):
    """Start the API server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port, timeout_graceful_shutdown=5)
