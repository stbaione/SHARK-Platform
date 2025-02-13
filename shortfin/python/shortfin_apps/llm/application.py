import logging

from fastapi import FastAPI

from .lifecycle_hooks import lifespan
from .routes import application_router, generation_router

logger = logging.getLogger(__name__)


def add_routes(app: FastAPI):
    app.include_router(application_router)
    app.include_router(generation_router)
    return app


def get_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app = add_routes(app)
    return app
