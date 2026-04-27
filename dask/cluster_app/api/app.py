from __future__ import annotations

from pathlib import Path

from cluster_app.config.schema import AppConfig
from cluster_app.services import create_services


def create_app(config: AppConfig | None = None):
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import HTMLResponse
        from fastapi.staticfiles import StaticFiles
        from fastapi.templating import Jinja2Templates
    except ModuleNotFoundError as exc:
        raise RuntimeError("fastapi and jinja2 are required to start the web app.") from exc
    globals()["Request"] = Request

    from cluster_app.api.routes_admin import router as admin_router
    from cluster_app.api.routes_auth import router as auth_router
    from cluster_app.api.routes_filesystem import router as filesystem_router
    from cluster_app.api.routes_jobs import router as jobs_router
    from cluster_app.api.routes_metrics import router as metrics_router
    from cluster_app.api.routes_nodes import router as nodes_router
    from cluster_app.api.websocket import router as websocket_router

    services = create_services(config)
    app = FastAPI(title="Dask Cluster App", version="0.1.0")
    app.state.services = services
    templates = Jinja2Templates(directory=str(Path(__file__).resolve().parents[1] / "ui" / "templates"))
    static_dir = Path(__file__).resolve().parents[1] / "ui" / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    app.include_router(auth_router)
    app.include_router(filesystem_router)
    app.include_router(jobs_router)
    app.include_router(nodes_router)
    app.include_router(admin_router)
    app.include_router(metrics_router)
    app.include_router(websocket_router)

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse(request, "index.html")

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard(request: Request):
        return templates.TemplateResponse(request, "dashboard.html")

    @app.on_event("startup")
    async def startup() -> None:
        if services.presence:
            await services.presence.start()
        services.queue.start()

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await services.queue.stop()
        services.worker.stop()
        services.scheduler.stop()
        if services.presence:
            await services.presence.stop()

    return app


def services_from_request(request):
    from cluster_app.services import services_from_request as shared_services_from_request

    return shared_services_from_request(request)
