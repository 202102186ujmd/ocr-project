import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from routes.ocr_routes import router as ocr_router
from schemas.schemas import HealthResponse

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle handler: startup y shutdown."""
    logger.info(f"🚀 Iniciando {settings.app_name} v{settings.app_version}")
    logger.info(f"📁 Directorio de uploads: {settings.upload_dir}")
    logger.info(f"🤖 Modelo: {settings.PALIGEMMA_LOCAL_PATH}")  # ✅ CORRECTO
    logger.info(f"💻 Dispositivo: {settings.device}")
    yield
    logger.info("👋 Cerrando aplicación")


# Crear aplicación FastAPI
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="API para OCR usando modelo Paligemma de Google",
    lifespan=lifespan,
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir routers
app.include_router(ocr_router)


@app.get("/", response_model=HealthResponse)
async def root():
    """Endpoint raíz - Información de la API"""
    return HealthResponse(
        status="online",
        version=settings.app_version,
        message="OCR API con Paligemma activa y funcionando",
    )


@app.get("/hello/{name}")
async def say_hello(name: str):
    """Endpoint de prueba"""
    return {"message": f"Hello {name}"}

