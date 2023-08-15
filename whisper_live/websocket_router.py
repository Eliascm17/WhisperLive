import asyncio
import numpy as np

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Path
from whisper_live.utils import get_connection_manager
from whisper_live.logger import get_logger
from whisper_live.transcribe import Transcribe

logger = get_logger(__name__)

router = APIRouter()

manager = get_connection_manager()

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(
    websocket: WebSocket, 
    client_id: int = Path(...)):
    
    logger.info(f"Attempting to connect client #{client_id}")
    
    transcriber = Transcribe(websocket, client_id)
    
    try:
        await manager.connect(websocket)
        logger.info(f"Client #{client_id} connected successfully")
        
        main_task = asyncio.create_task(handle_receive(websocket, client_id, transcriber))
        await asyncio.gather(main_task)

    except WebSocketDisconnect:
        logger.warning(f"Client #{client_id} disconnected")
        await manager.disconnect(websocket)
        await manager.broadcast_message(f"Client #{client_id} left the chat")
    except Exception as e:
        logger.error(f"An error occurred with client #{client_id}: {str(e)}")

async def handle_receive(
    websocket: WebSocket,
    client_id: int,
    transcriber: Transcribe):

    while True:
        data = await websocket.receive()

        try: 
            if "bytes" in data:
                frame_data = data['bytes']
                logger.info(f'Bytes received from client #{client_id}: {len(frame_data)}')
                frame_np = np.frombuffer(frame_data, np.float32)
                await asyncio.to_thread(transcriber.add_frames, frame_np)

        except Exception as e:
            logger.error(f"Connection closed for client #{client_id}. Error: {str(e)}")
            await manager.disconnect(websocket)
            await manager.broadcast_message(f"Client #{client_id} left the chat")
            break
