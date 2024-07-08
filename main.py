from fastapi import FastAPI
from routers import upload, retrieve

app = FastAPI()

app.include_router(upload.router)
app.include_router(retrieve.router)