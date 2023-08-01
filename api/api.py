from fastapi import FastAPI
from fastapi_health import health
from fastapi.middleware.cors import CORSMiddleware

from routers import GenerateRouter


def create_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[""],  # 모든 origin에 대해 액세스 허용
        allow_credentials=True,
        allow_methods=[""],  # 모든 HTTP 메서드에 대해 액세스 허용
        allow_headers=["*"],  # 모든 헤더에 대해 액세스 허용
    )
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