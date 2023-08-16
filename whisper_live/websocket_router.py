import asyncio
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Path
from whisper_live.utils import get_connection_manager
from whisper_live.logger import get_logger
from whisper_live.transcribe import Transcribe

logger = get_logger(__name__)
router = APIRouter()
manager = get_connection_manager()

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int = Path(...)):
    logger.info(f"Attempting to connect client #{client_id}")
    transcriber = Transcribe(websocket, client_id)
    await transcriber.initialize()

    try:
        logger.info(f"Client #{client_id} connected successfully")
        main_task = asyncio.create_task(handle_receive(websocket, client_id, transcriber))
        speech_to_text_task = asyncio.create_task(transcriber.speech_to_text())
        await asyncio.gather(main_task, speech_to_text_task)
    except WebSocketDisconnect:
        await handle_disconnect(client_id, websocket, transcriber)
    except Exception as e:
        logger.error(f"An error occurred with client #{client_id}: {str(e)}")

async def handle_receive(websocket: WebSocket, client_id: int, transcriber: Transcribe):
    logger.info(f"Handling receive for client #{client_id}")
    while True:
        data = await websocket.receive()
        logger.info(f"Received data from client #{client_id}: {data}")
        try:
            if "bytes" in data:
                logger.info(f"Processing received bytes from client #{client_id}")
                await process_received_data(data, client_id, transcriber)
        except Exception as e:
            logger.error(f"Connection closed for client #{client_id}. Error: {str(e)}")
            await manager.disconnect(websocket)
            await manager.broadcast_message(f"Client #{client_id} left the chat")
            break

async def process_received_data(data, client_id, transcriber):
    frame_data = data['bytes']
    logger.info(f'Bytes received from client #{client_id}: {len(frame_data)}')
    frame_np = np.frombuffer(frame_data, np.float32)
    await asyncio.to_thread(transcriber.add_frames, frame_np)

async def handle_disconnect(client_id, websocket, transcriber):
    logger.warning(f"Client #{client_id} disconnected")
    await manager.disconnect(websocket)
    await manager.broadcast_message(f"Client #{client_id} left the chat")
    transcriber.cleanup()
