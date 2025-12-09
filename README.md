# VoiceSync Demo Version

## Prerequisites

- Python 3.8 or higher
- Node.js 14.x or higher
- npm 6.x or higher
- Virtual environment tool (venv or conda)

## Project Structure

```
VoiceSync/
├── common/                  # Shared utilities and logger
├── config/                  # Configuration files
├── stt/                     # Speech-to-Text module
├── mt/                      # Machine Translation module
├── tts/                     # Text-to-Speech module
├── voicesync-frontend/     # React frontend application
├── websocket_server.py     # WebSocket server for service integration
├── start.sh                # Script to start all services
├── stop.sh                 # Script to stop all services
├── venv/                   # Python virtual environment
└── README.md               # This file
```

## Installation

### 1. Backend Setup (STT, MT, TTS Services)

1. Navigate to the project root directory:
   ```bash
   cd /home/azureuser/VoiceSync
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### 2. Frontend Setup (React Application)

1. Navigate to the frontend directory:
   ```bash
   cd voicesync-frontend
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

## Running the Application

### Start All Services
```bash
cd /home/azureuser/VoiceSync
./start.sh
```

This will start all services with proper PYTHONPATH settings:
- STT Service: `ws://localhost:8765`
- MT Service: `http://localhost:8766`
- TTS Service: `http://localhost:8767`
- WebSocket Integration: `ws://localhost:8000`

### Stop All Services
```bash
cd /home/azureuser/VoiceSync
./stop.sh
```

## Using the Application

1. Open both Client A (http://localhost:3000) and Client B (http://localhost:3001) in separate browser windows/tabs
2. Click "Start Call" on both clients to establish WebSocket connections
3. Select your target language from the dropdown menu (now includes English)
4. Type messages in the text input field or use the voice recording feature
5. Messages will be translated and displayed in real-time on both clients
6. Audio synthesis will be provided for non-English languages

## Supported Languages

- English (en) - Source language
- Spanish (es)
- French (fr)
- Hindi (hi)

## Translation Dictionary

The system includes an enhanced translation dictionary with common phrases:
- Greetings: hello, hi, goodbye, bye
- Politeness: thank you, please
- Basic responses: yes, no, okay
- Conversation starters: how are you, i am fine
- Questions: what is your name, where are you from, how old are you
- Expressions: i love you, i miss you, take care
- Farewells: see you later, see you soon, have a nice day

For phrases not in the dictionary, the system will prefix the text with "[Language] " to indicate translation status.

## Service Endpoints

- **STT Service**: `ws://localhost:8765` (WebSocket)
- **MT Service**: `http://localhost:8766/translate` (HTTP POST)
- **TTS Service**: `http://localhost:8767/synthesize` (HTTP POST)
- **WebSocket Integration**: `ws://localhost:8000` (WebSocket)## Checking Running Processes

To verify that all components are running correctly:

```bash
# Check all VoiceSync services
pgrep -af "python3.*src.main"

# Check WebSocket server
pgrep -af "python3.*websocket_server"

# Check frontend clients
pgrep -af "node.*start"

# Check specific ports
ss -tulnp | grep -E ":(8765|8766|8767|8000|3000|3001)"
```## Troubleshooting

### WebSocket Connection Issues

If clients fail to connect:
1. Ensure all services are running on their respective ports
2. Check that no firewall is blocking the connections
3. Verify that all terminals are using the same virtual environment

### Python Module Import Errors

If you see "ModuleNotFoundError: No module named 'common'", make sure to set the PYTHONPATH:
```bash
export PYTHONPATH=/home/azureuser/VoiceSync:$PYTHONPATH
```

Or use the provided startup scripts which handle this automatically.

### Port Conflicts

If ports are already in use:
```bash
# Kill processes using specific ports
sudo lsof -i :8765 | grep LISTEN | awk '{print $2}' | xargs kill -9
sudo lsof -i :8766 | grep LISTEN | awk '{print $2}' | xargs kill -9
sudo lsof -i :8767 | grep LISTEN | awk '{print $2}' | xargs kill -9
sudo lsof -i :8000 | grep LISTEN | awk '{print $2}' | xargs kill -9
sudo lsof -i :3000 | grep LISTEN | awk '{print $2}' | xargs kill -9
sudo lsof -i :3001 | grep LISTEN | awk '{print $2}' | xargs kill -9
### Frontend Not Loading

If the React application fails to start:
1. Ensure all dependencies are installed: `npm install`
2. Clear npm cache: `npm cache clean --force`
3. Delete node_modules and reinstall:
   ```bash
   rm -rf node_modules
   npm install
   ```

## System Architecture

```
[Client A] ←→ [WebSocket Server] ←→ [Client B]
                   ↓
           [STT Service] (Port 8765)
                   ↓
           [MT Service]  (Port 8766)
                   ↓
           [TTS Service] (Port 8767)
```

The WebSocket server orchestrates communication between clients and the individual services:
1. **Speech-to-Text (STT)**: Converts voice input to text
2. **Machine Translation (MT)**: Translates text between languages
3. **Text-to-Speech (TTS)**: Converts translated text back to voice

## Demo Version

This is the "Demo Version" of VoiceSync, showcasing the core functionality of real-time voice translation. The interface maintains a minimalist black/grey/white color scheme for professional presentation.

## Stopping the Application

### If using startup scripts:
```bash
cd /home/azureuser/VoiceSync
./stop.sh
```

### Manual stopping:
1. Stop all services: Press `Ctrl+C` in each service terminal
2. Stop the WebSocket server: Press `Ctrl+C` in the server terminal
3. Stop the frontend clients: Press `Ctrl+C` in each client terminal
4. Deactivate the Python virtual environment:
   ```bash
   deactivate
   ```

## License

This is a demonstration project for educational purposes.