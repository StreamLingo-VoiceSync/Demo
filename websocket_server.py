import asyncio
import websockets
import json
import time
import requests
import base64
from datetime import datetime
from threading import Thread
import numpy as np
import logging
import sys
import subprocess
import os

# Setup logger for websocket server
from common.logger import LOGS_DIR
# Create logs directory for websocket
WS_LOG_DIR = LOGS_DIR / "websocket"
WS_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Configure logger
logger = logging.getLogger("websocket_server")
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler(WS_LOG_DIR / "websocket_server.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# Connected clients
clients = {}
call_sessions = {}

# Service endpoints
STT_ENDPOINT = "ws://localhost:8765"  # STT service WebSocket endpoint
MT_ENDPOINT = "http://localhost:8766/translate"  # MT service HTTP endpoint
TTS_ENDPOINT = "http://localhost:8767/synthesize"  # TTS service HTTP endpoint
WEBSOCKET_PORT = 8000  # WebSocket integration server port

# Translation mappings for demo (fallback)
translations = {
    'en': {
        'es': {
            'hello': 'hola',
            'hi': 'hola',
            'how are you': 'como estas',
            'good morning': 'buenos dias',
            'good afternoon': 'buenas tardes',
            'good evening': 'buenas noches',
            'thank you': 'gracias',
            'please': 'por favor',
            'yes': 'si',
            'no': 'no',
            'goodbye': 'adios',
            'bye': 'adios',
            'okay': 'bien',
            'great': 'genial',
            'nice': 'agradable',
            'good': 'bueno',
            'bad': 'malo',
            'i am fine': 'estoy bien',
            'what is your name': 'como te llamas',
            'my name is': 'me llamo',
            'where are you from': 'de donde eres',
            'i am from': 'soy de',
            'how old are you': 'cuantos anos tienes',
            'i am years old': 'tengo anos',
            'i love you': 'te quiero',
            'i miss you': 'te extrano',
            'see you later': 'hasta luego',
            'see you soon': 'hasta pronto',
            'take care': 'cuidate',
            'have a nice day': 'que tengas un buen dia'
        },
        'fr': {
            'hello': 'bonjour',
            'hi': 'bonjour',
            'how are you': 'comment allez vous',
            'good morning': 'bonjour',
            'good afternoon': 'bonne apres midi',
            'good evening': 'bonne soir',
            'thank you': 'merci',
            'please': 'sil vous plait',
            'yes': 'oui',
            'no': 'non',
            'goodbye': 'au revoir',
            'bye': 'au revoir',
            'okay': 'daccord',
            'great': 'formidable',
            'nice': 'agreable',
            'good': 'bon',
            'bad': 'mauvais',
            'i am fine': 'je vais bien',
            'what is your name': 'comment vous appelez vous',
            'my name is': 'je mappelle',
            'where are you from': 'dou venez vous',
            'i am from': 'je viens de',
            'how old are you': 'quel age avez vous',
            'i am years old': 'jai ans',
            'i love you': 'je taime',
            'i miss you': 'tu me manques',
            'see you later': 'a plus tard',
            'see you soon': 'a bientot',
            'take care': 'prends soin de toi',
            'have a nice day': 'bonne journee'
        },
        'hi': {
            'hello': 'नमस्ते',
            'hi': 'नमस्ते',
            'how are you': 'आप कैसे हैं',
            'good morning': 'सुप्रभात',
            'good afternoon': 'शुभ अपराह्न',
            'good evening': 'शुभ संध्या',
            'thank you': 'धन्यवाद',
            'please': 'कृपया',
            'yes': 'हाँ',
            'no': 'नहीं',
            'goodbye': 'अलविदा',
            'bye': 'अलविदा',
            'okay': 'ठीक है',
            'great': 'बहुत बढ़िया',
            'nice': 'अच्छा',
            'good': 'अच्छा',
            'bad': 'बुरा',
            'i am fine': 'मैं ठीक हूँ',
            'what is your name': 'आपका नाम क्या है',
            'my name is': 'मेरा नाम है',
            'where are you from': 'आप कहाँ से हैं',
            'i am from': 'मैं से हूँ',
            'how old are you': 'आपकी आयु कितनी है',
            'i am years old': 'मेरी आयु वर्ष है',
            'i love you': 'मैं तुमसे प्यार करता हूँ',
            'i miss you': 'आपको याद आ रहा हूँ',
            'see you later': 'बाद में मिलते हैं',
            'see you soon': 'जल्द ही मिलेंगे',
            'take care': 'खुद का ख्याल रखना',
            'have a nice day': 'आपका दिन शुभ हो'
        }
    }
}

def translate_text_fallback(text, source_lang, target_lang):
    """Fallback translation function for demo purposes"""
    text_lower = text.lower().strip()
    
    # Check if we have a direct translation
    if source_lang in translations and target_lang in translations[source_lang]:
        translation_dict = translations[source_lang][target_lang]
        if text_lower in translation_dict:
            return translation_dict[text_lower]
    
    # If no direct translation, return prefixed text
    lang_names = {'es': 'Spanish', 'fr': 'French', 'hi': 'Hindi', 'en': 'English'}
    lang_name = lang_names.get(target_lang, target_lang)
    return f"[{lang_name}] {text}"

async def call_mt_service_with_voice_cloning(text, source_lang, target_lang, speaker_embedding=None):
    """Call the MT service to translate text with voice cloning support"""
    try:
        # Try to call the actual MT service
        payload = {
            "text": text,
            "source_lang": source_lang,
            "target_lang": target_lang
        }
        
        # Add speaker embedding if available
        if speaker_embedding and len(speaker_embedding) > 0:
            payload["speaker_embedding"] = speaker_embedding
        
        # Make synchronous request in async context
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: requests.post(MT_ENDPOINT, json=payload, timeout=10))
        
        if response.status_code == 200:
            result = response.json()
            # Handle different response formats
            if isinstance(result, dict):
                if 'translated_text' in result:
                    return result['translated_text']
                elif 'text' in result:
                    return result['text']
                else:
                    return str(result)
            else:
                return str(result)
        else:
            logger.warning(f"MT service returned status {response.status_code}: {response.text}")
            # Fallback to dictionary translation
            return translate_text_fallback(text, source_lang, target_lang)
    except requests.exceptions.RequestException as e:
        logger.error(f"MT service network error: {e}")
        # Fallback to dictionary translation
        return translate_text_fallback(text, source_lang, target_lang)
    except Exception as e:
        logger.error(f"MT service error: {e}")
        # Fallback to dictionary translation
        return translate_text_fallback(text, source_lang, target_lang)

async def call_tts_service_with_voice_cloning(text, language, speaker_embedding=None, session_id=None):
    """Call the TTS service to synthesize speech with voice cloning"""
    try:
        # Prepare payload for TTS service
        payload = {
            "session_id": session_id or f"tts_{int(time.time())}",
            "tts_text": text,
            "target_language": language,
            "speaker_id": "voice_cloned_speaker",
            "speaker_embedding": speaker_embedding if speaker_embedding else []
        }
        
        # Add voice signature if embedding is provided
        if speaker_embedding and len(speaker_embedding) > 0:
            payload["voice_signature"] = {
                "embedding": speaker_embedding,
                "dimensions": len(speaker_embedding)
            }
        
        logger.info(f"Calling TTS service for language {language} with voice cloning: {bool(speaker_embedding)}")
        
        # Make synchronous request in async context
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: requests.post(TTS_ENDPOINT, json=payload, timeout=30))
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"TTS service successful for {language}")
            return result
        else:
            logger.error(f"TTS service returned status {response.status_code}: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"TTS service network error: {e}")
        return None
    except Exception as e:
        logger.error(f"TTS service error: {e}")
        return None

async def handle_client(websocket, path):
    """Handle WebSocket client connections"""
    client_id = f"client_{int(time.time())}"
    logger.info(f"New client connected: {client_id}")
    
    # Store client info
    clients[client_id] = {
        "websocket": websocket,
        "languages": {"source": "en", "target": "es"},
        "speaker_embedding": None  # Store speaker embedding for voice cloning
    }
    
    try:
        # Connect to STT service
        stt_ws = await websockets.connect(STT_ENDPOINT)
        logger.info(f"Connected to STT service for client {client_id}")
        
        # Send initial configuration to STT
        config_msg = {
            "type": "configure",
            "language": clients[client_id]["languages"]["source"]
        }
        await stt_ws.send(json.dumps(config_msg))
        
        # Forward messages between client and STT service
        async def forward_to_stt():
            async for message in websocket:
                if isinstance(message, str):
                    # Handle control messages
                    try:
                        data = json.loads(message)
                        if data.get("type") == "configure":
                            # Update language configuration
                            clients[client_id]["languages"]["source"] = data.get("source_lang", "en")
                            clients[client_id]["languages"]["target"] = data.get("target_lang", "es")
                            
                            # Send updated configuration to STT
                            config_msg = {
                                "type": "configure",
                                "language": clients[client_id]["languages"]["source"]
                            }
                            await stt_ws.send(json.dumps(config_msg))
                        elif data.get("type") == "set_speaker_embedding":
                            # Store speaker embedding for voice cloning
                            clients[client_id]["speaker_embedding"] = data.get("embedding")
                            logger.info(f"Stored speaker embedding for client {client_id}")
                    except json.JSONDecodeError:
                        pass
                else:
                    # Forward audio data to STT
                    await stt_ws.send(message)
        
        async def forward_from_stt():
            async for message in stt_ws:
                if isinstance(message, str):
                    # Handle STT text messages
                    try:
                        data = json.loads(message)
                        if data.get("type") == "transcript":
                            # Extract transcript and speaker embedding
                            transcript = data.get("text", "")
                            speaker_embedding = data.get("speaker_embedding") or clients[client_id].get("speaker_embedding")
                            
                            logger.info(f"Transcript received: {transcript}")
                            
                            # Translate the text
                            translated_text = await call_mt_service_with_voice_cloning(
                                transcript,
                                clients[client_id]["languages"]["source"],
                                clients[client_id]["languages"]["target"],
                                speaker_embedding
                            )
                            
                            logger.info(f"Translated text: {translated_text}")
                            
                            # Synthesize speech with voice cloning
                            tts_result = await call_tts_service_with_voice_cloning(
                                translated_text,
                                clients[client_id]["languages"]["target"],
                                speaker_embedding,
                                data.get("segment_id")
                            )
                            
                            if tts_result:
                                # Send translated text and audio back to client
                                response = {
                                    "type": "translation_result",
                                    "original_text": transcript,
                                    "translated_text": translated_text,
                                    "audio_bytes": tts_result.get("audio_bytes", ""),
                                    "sample_rate": tts_result.get("sample_rate", 22050),
                                    "language": clients[client_id]["languages"]["target"]
                                }
                                await websocket.send(json.dumps(response))
                            else:
                                # Fallback: send only translated text
                                response = {
                                    "type": "translation_result",
                                    "original_text": transcript,
                                    "translated_text": translated_text,
                                    "audio_bytes": "",
                                    "sample_rate": 22050,
                                    "language": clients[client_id]["languages"]["target"]
                                }
                                await websocket.send(json.dumps(response))
                        else:
                            # Forward other STT messages to client
                            await websocket.send(message)
                    except json.JSONDecodeError:
                        await websocket.send(message)
                else:
                    # Forward binary data to client
                    await websocket.send(message)
        
        # Run both forwarding tasks concurrently
        await asyncio.gather(forward_to_stt(), forward_from_stt())
        
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"Error handling client {client_id}: {e}")
    finally:
        # Clean up
        if client_id in clients:
            del clients[client_id]
        logger.info(f"Client {client_id} connection closed")

# Start the WebSocket server
start_server = websockets.serve(handle_client, "0.0.0.0", WEBSOCKET_PORT)
logger.info(f"WebSocket server started on port {WEBSOCKET_PORT}")

# Run the server
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()