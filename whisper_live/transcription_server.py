from requests import Session
from collections import deque
from dataclasses import dataclass

import pickle, struct, time, pyaudio
import threading
import os, json
import wave
import textwrap
import asyncio
import torch
import numpy as np

from fastapi.responses import HTMLResponse
from fastapi import APIRouter, Depends, Path, WebSocketDisconnect, Query, FastAPI, WebSocket
from whisper_live.transcriber import WhisperModel
from whisper_live.utils import Singleton
from whisper_live.logger import get_logger
from whisper_live.server_client import ServeClient

logger = get_logger(__name__)

class TranscriptionServer(Singleton):
    """
    Represents a transcription server that handles incoming audio from clients.

    Attributes:
        clients (dict): A dictionary to store connected clients.
    """

    def __init__(self):
        self.clients = {}

    async def recv_audio(self, websocket: WebSocket, client_id: int):
        logger.info(f"Initializing connection for client #{client_id}")
        
        options = await websocket.receive_text()
        options = json.loads(options)
        logger.info(f"Options received from client #{client_id}: {options}")

        client = ServeClient(
            websocket,
            multilingual=options["multilingual"],
            language=options["language"],
            task=options["task"],
        )
        logger.info(f"Client #{client_id} initialized successfully")
        await client.initialize()

        self.clients[websocket] = client

        while True:
          try:
              frame_data = await websocket.receive_bytes()
              logger.info(f'Bytes received from client #{client_id}: {len(frame_data)}')
              frame_np = np.frombuffer(frame_data, np.float32)
              self.clients[websocket].add_frames(frame_np)

          except Exception as e:
              logger.error(f"Connection closed for client #{client_id}. Error: {str(e)}")
              self.clients[websocket].cleanup()
              self.clients.pop(websocket)
              break

    async def run(self, websocket: WebSocket, client_id: int):
        await self.recv_audio(websocket, client_id)

def get_transcription_server():
    return TranscriptionServer.get_instance()