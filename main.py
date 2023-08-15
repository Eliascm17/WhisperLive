import warnings

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from whisper_live.utils import ConnectionManager
from whisper_live.websocket_router import router as websocket_router
from whisper_live.transcribe import Transcribe
from whisper_live.logger import get_logger

app = FastAPI()

logger = get_logger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(websocket_router)

ConnectionManager.initialize()

warnings.filterwarnings("ignore", module="whisper_live")