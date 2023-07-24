from fastapi import FastAPI, UploadFile, status
from fastapi_health import health


from routers import GenerateRouter


def create_app() -> FastAPI:
    app = FastAPI()
    init_router(app)
    return app

def init_settings(app: FastAPI):
    app.add_api_route(
        "/health",
        health([])
    )


def init_router(app: FastAPI):
    app.include_router(GenerateRouter)


app = create_app()