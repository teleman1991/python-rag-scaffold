from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import upload, retrieve

app = FastAPI(
    title="RAG Server API",
    description="APIs for uploading documents and retrieving context",
    version="1.0.0",
        contact={
        "name": "Nik Pash (CEO of Vault)",
        "url": "http://vault77.ai/",
        "email": "nik@vault77.ai",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    }
)

# Set up CORS (if necessary)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this list with allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload.router, tags=["upload"])
app.include_router(retrieve.router, tags=["retrieve"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
