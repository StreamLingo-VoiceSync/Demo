"""API routes for the STT system."""

import asyncio
import base64
import json
import numpy as np
import time
import uuid

# Import shared utilities
from common.utils import gen_id
from common.logger import setup_stt_logger

# Import config
from ..core.config import SAMPLE_RATE

# Import schemas
from .schemas import CallParticipant

log = setup_stt_logger()

async def websocket_handler(websocket, gateway):
    client_id = gen_id("client")
    call_id = None
    gateway.client_websockets[client_id] = websocket  # FIX #7: Store websocket
    
    try:
        log.info(f"🔌 Client connected: {client_id}")
        
        async for message in websocket:
            # Text message: usually control (start, join, leave, etc)
            if isinstance(message, str):
                try:
                    data = json.loads(message)
                except Exception:
                    log.debug(f"Malformed JSON message: {message[:80]}...")
                    continue
                
                msg_type = data.get("type")
                
                if msg_type == "start_call":
                    language = data.get("language", "en")
                    speaker_id = data.get("speaker_id", f"speaker_{language}")
                    
                    participant = CallParticipant(
                        client_id=client_id,
                        speaker_id=speaker_id,
                        language=language,
                        websocket=websocket
                    )
                    
                    call_id = gateway.create_call(participant)
                    
                    await websocket.send(json.dumps({
                        "type": "call_started",
                        "call_id": call_id,
                        "client_id": client_id,
                        "speaker_id": speaker_id
                    }))
                
                elif msg_type == "join_call":
                    call_id = data.get("call_id")
                    language = data.get("language", "en")
                    speaker_id = data.get("speaker_id", f"speaker_{language}")
                    
                    participant = CallParticipant(
                        client_id=client_id,
                        speaker_id=speaker_id,
                        language=language,
                        websocket=websocket
                    )
                    
                    success = gateway.join_call(call_id, participant)
                    
                    await websocket.send(json.dumps({
                        "type": "call_joined" if success else "call_join_failed",
                        "call_id": call_id,
                        "client_id": client_id
                    }))
                
                elif msg_type == "audio":
                    audio_bytes = base64.b64decode(data.get("audio", ""))
                    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    src_sr = data.get("sample_rate", SAMPLE_RATE)
                    gateway.push_audio(client_id, audio, src_sr)
                
                elif msg_type == "leave":
                    gateway.leave_call(client_id)
                    await websocket.send(json.dumps({"type": "call_left", "client_id": client_id}))
                    break
            
            elif isinstance(message, bytes):  # FIX #8: Handle binary PCM audio
                # Handle binary PCM int16 audio frames
                try:
                    audio = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
                    gateway.push_audio(client_id, audio, SAMPLE_RATE)
                except Exception as e:
                    log.error(f"Binary audio error: {e}")
                    continue
    
    except Exception as e:
        log.info(f"🔌 Disconnected: {client_id} - {e}")
    
    finally:
        gateway.client_websockets.pop(client_id, None)  # FIX #9: Remove websocket reference
        gateway.leave_call(client_id)