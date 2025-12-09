#!/usr/bin/env python3
"""
WebSocket server for VoiceSync Demo
Integrates with STT, MT, and TTS services
"""

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

async def call_mt_service(text, source_lang, target_lang):
    """Call the MT service to translate text"""
    try:
        # Try to call the actual MT service
        payload = {
            "text": text,
            "source_lang": source_lang,
            "target_lang": target_lang
        }
        
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

async def call_tts_service(text, language, speaker_id="default"):
    """Call the TTS service to synthesize speech"""
    try:
        # Try to call the actual TTS service
        payload = {
            "tts_text": text,
            "target_language": language,
            "speaker_id": speaker_id
        }
        
        # Make synchronous request in async context
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: requests.post(TTS_ENDPOINT, json=payload, timeout=10))
        
        if response.status_code == 200:
            result = response.json()
            # Handle different response formats
            if isinstance(result, dict):
                if 'audio_bytes' in result:
                    return result['audio_bytes']
                elif 'audio' in result:
                    return result['audio']
                else:
                    return base64.b64encode(str(result).encode('utf-8')).decode('utf-8')
            else:
                return base64.b64encode(str(result).encode('utf-8')).decode('utf-8')
        else:
            logger.warning(f"TTS service returned status {response.status_code}: {response.text}")
            # Return a placeholder
            return base64.b64encode(b"placeholder_audio_data").decode('utf-8')
    except requests.exceptions.RequestException as e:
        logger.error(f"TTS service network error: {e}")
        # Return a placeholder
        return base64.b64encode(b"placeholder_audio_data").decode('utf-8')
    except Exception as e:
        logger.error(f"TTS service error: {e}")
        # Return a placeholder
        return base64.b64encode(b"placeholder_audio_data").decode('utf-8')

async def register_client(websocket, data):
    """Register a new client"""
    client_id = data.get('clientId', f'client_{len(clients)+1}')
    target_language = data.get('targetLanguage', 'es')
    
    clients[websocket] = {
        'id': client_id,
        'target_language': target_language,
        'connected_at': datetime.now()
    }
    
    # Notify other clients
    await broadcast_message({
        'type': 'system',
        'message': f'{client_id} joined the call'
    }, exclude=websocket)
    
    # Send welcome message to the client
    await websocket.send(json.dumps({
        'type': 'system',
        'message': f'Welcome to VoiceSync Demo! You are registered as {client_id}'
    }))

async def handle_text_message(websocket, data):
    """Handle text message from client"""
    sender_id = data.get('senderId', 'Unknown')
    text = data.get('text', '')
    target_language = data.get('targetLanguage', 'es')
    
    # Broadcast original message as transcription
    await broadcast_message({
        'type': 'transcription',
        'text': f'Transcribed: {text}',
        'senderId': sender_id
    }, exclude=websocket)
    
    # Simulate processing delay
    await asyncio.sleep(0.5)
    
    # Translate the text using MT service
    translated_text = await call_mt_service(text, 'en', target_language)
    
    # Send translated message to all clients
    await broadcast_message({
        'type': 'translated_text',
        'text': translated_text,
        'senderId': sender_id
    })
    
    # Optionally, synthesize speech using TTS service
    if target_language != 'en':  # Only synthesize if not English
        audio_data = await call_tts_service(translated_text, target_language, sender_id)
        # Send audio data to clients that support it
        await broadcast_message({
            'type': 'audio',
            'audio': audio_data,
            'language': target_language,
            'senderId': sender_id
        })

async def handle_start_recording(websocket, data):
    """Handle start recording event"""
    sender_id = data.get('senderId', 'Unknown')
    
    await websocket.send(json.dumps({
        'type': 'system',
        'message': 'Recording started...'
    }))

async def handle_stop_recording(websocket, data):
    """Handle stop recording event"""
    sender_id = data.get('senderId', 'Unknown')
    target_language = data.get('targetLanguage', 'es')
    
    await websocket.send(json.dumps({
        'type': 'system',
        'message': 'Processing recorded audio...'
    }))
    
    # Simulate processing delay
    await asyncio.sleep(1.0)
    
    # Send simulated transcription
    await websocket.send(json.dumps({
        'type': 'transcription',
        'text': 'Transcribed: [Recorded audio message]',
        'senderId': sender_id
    }))
    
    # Simulate translation
    await asyncio.sleep(0.5)
    
    translated_text = await call_mt_service('[Recorded audio message]', 'en', target_language)
    await websocket.send(json.dumps({
        'type': 'translated_text',
        'text': translated_text,
        'senderId': sender_id
    }))
    
    # Optionally, synthesize speech
    if target_language != 'en':
        audio_data = await call_tts_service(translated_text, target_language, sender_id)
        await websocket.send(json.dumps({
            'type': 'audio',
            'audio': audio_data,
            'language': target_language,
            'senderId': sender_id
        }))

async def broadcast_message(message, exclude=None):
    """Broadcast message to all connected clients except excluded"""
    if clients:
        disconnected = []
        for websocket in clients:
            if websocket != exclude:
                try:
                    await websocket.send(json.dumps(message))
                except websockets.exceptions.ConnectionClosed:
                    disconnected.append(websocket)
        
        # Remove disconnected clients
        for websocket in disconnected:
            if websocket in clients:
                client_info = clients[websocket]
                del clients[websocket]
                logger.info(f"Client {client_info['id']} disconnected")

async def handle_client(websocket):
    """Handle individual client connection"""
    try:
        logger.info(f"New client connected from {websocket.remote_address}")
        
        async for message in websocket:
            try:
                data = json.loads(message)
                message_type = data.get('type')
                
                if message_type == 'register':
                    await register_client(websocket, data)
                elif message_type == 'text_message':
                    await handle_text_message(websocket, data)
                elif message_type == 'start_recording':
                    await handle_start_recording(websocket, data)
                elif message_type == 'stop_recording':
                    await handle_stop_recording(websocket, data)
                else:
                    logger.warning(f"Unknown message type: {message_type}")
                    
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received: {message}")
            except Exception as e:
                logger.error(f"Error handling message: {e}")
                
    except websockets.exceptions.ConnectionClosed:
        logger.info("Client connection closed")
    except Exception as e:
        logger.error(f"Error handling client: {e}")
    finally:
        # Remove client from connected clients
        if websocket in clients:
            client_info = clients[websocket]
            del clients[websocket]
            logger.info(f"Client {client_info['id']} disconnected")
            
            # Notify other clients
            await broadcast_message({
                'type': 'system',
                'message': f'{client_info["id"]} left the call'
            })

async def start_stt_service():
    """Start the STT service"""
    try:
        # Start the actual STT service as a subprocess
        project_root = os.path.dirname(os.path.abspath(__file__))
        stt_main_path = os.path.join(project_root, "stt", "src", "main.py")
        
        # Set PYTHONPATH
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root
        
        # Start STT service
        logger.info("Starting STT service on port 8765")
        subprocess.Popen([
            sys.executable, "-m", "stt.src.main"
        ], cwd=project_root, env=env)
        
        # Give it time to start
        await asyncio.sleep(2)
        logger.info("STT service started")
    except Exception as e:
        logger.error(f"Failed to start STT service: {e}")

async def start_mt_service():
    """Start the MT service"""
    try:
        # Start the actual MT service as a subprocess
        project_root = os.path.dirname(os.path.abspath(__file__))
        mt_main_path = os.path.join(project_root, "mt", "src", "main.py")
        
        # Set PYTHONPATH
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root
        
        # Start MT service
        logger.info("Starting MT service on port 8766")
        subprocess.Popen([
            sys.executable, "-m", "mt.src.main"
        ], cwd=project_root, env=env)
        
        # Give it time to start
        await asyncio.sleep(2)
        logger.info("MT service started")
    except Exception as e:
        logger.error(f"Failed to start MT service: {e}")

async def start_tts_service():
    """Start the TTS service"""
    try:
        # Start the actual TTS service as a subprocess
        project_root = os.path.dirname(os.path.abspath(__file__))
        tts_main_path = os.path.join(project_root, "tts", "src", "main.py")
        
        # Set PYTHONPATH
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root
        
        # Start TTS service
        logger.info("Starting TTS service on port 8767")
        subprocess.Popen([
            sys.executable, "-m", "tts.src.main"
        ], cwd=project_root, env=env)
        
        # Give it time to start
        await asyncio.sleep(2)
        logger.info("TTS service started")
    except Exception as e:
        logger.error(f"Failed to start TTS service: {e}")

async def main():
    """Main function to start the WebSocket server and services"""
    logger.info("Starting VoiceSync WebSocket server...")
    logger.info("Listening on ws://localhost:8000")  # Changed to port 8000
    logger.info("Supported languages: English (en), Spanish (es), French (fr), Hindi (hi)")
    logger.info("Press Ctrl+C to stop the server")
    
    logger.info("")
    logger.info("Service endpoints:")
    logger.info("  STT Service: ws://localhost:8765")
    logger.info("  MT Service: http://localhost:8766")  # Changed to port 8766
    logger.info("  TTS Service: http://localhost:8767")  # Confirmed port 8767
    logger.info("  WebSocket Integration: ws://localhost:8000")  # Changed to port 8000
    logger.info("")
    
    # Start the services
    await start_stt_service()
    await start_mt_service()
    await start_tts_service()
    
    # Start the WebSocket server on port 8000
    server = await websockets.serve(handle_client, "localhost", 8000)  # Changed to port 8000
    
    try:
        await server.wait_closed()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.close()
        await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())