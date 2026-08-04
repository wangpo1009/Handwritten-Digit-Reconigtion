"""
Entry point của FastAPI
Chức năng của file main là : Quản lí app

- Khởi tạo FastAPI app
- Đăng kí middleware
- Đăng kí router
- Startup / Shutdown event

"""

from fastapi import FastAPI

from src.api.middleware import register_middleware
from src.api.routes import router
from src.api.dependencies import load_model

app = FastAPI(title="MNIST Inference API", version="1.0.0")

@app.on_event("startup")
def startup_event():
    load_model()
    print("Model loaded successfully.")

register_middleware(app)
app.include_router(router)
