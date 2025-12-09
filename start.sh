#!/bin/bash

# VoiceSync Service Startup Script
# Sets up the Python path and starts all services

# Get the current directory (should be the project root)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting VoiceSync services from: $PROJECT_ROOT"

# Ensure logs directory exists
mkdir -p "$PROJECT_ROOT/logs"

# Set PYTHONPATH to include the project root so 'common' module can be found
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "PYTHONPATH set to: $PYTHONPATH"

# Function to start a service
start_service() {
    local service_name=$1
    local service_dir=$2
    local port=$3
    local start_command=$4
    
    echo "Starting $service_name service..."
    cd "$PROJECT_ROOT/$service_dir"
    
    # Use custom start command if provided, otherwise default
    if [ -n "$start_command" ]; then
        PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" $start_command &
    else
        PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python3 -m src.main &
    fi
    
    local pid=$!
    
    # Give it a moment to start
    sleep 2
    
    # Check if the process is still running
    if ps -p $pid > /dev/null; then
        echo "$service_name service started successfully (PID: $pid)"
        echo "$service_name_PID=$pid" >> /tmp/voicesync_pids
    else
        echo "Failed to start $service_name service"
        return 1
    fi
}

# Create PID file
echo "# VoiceSync Service PIDs" > /tmp/voicesync_pids

# Start all services
echo "Starting all VoiceSync services..."

# Start STT service on port 8765
start_service "STT" "stt" "8765"

# Start MT service on port 8766 with the full version
start_service "MT" "mt" "8766"

# Start TTS service on port 8767
start_service "TTS" "tts" "8767"

# Start WebSocket integration server on port 8000
echo "Starting WebSocket integration server..."
cd "$PROJECT_ROOT"
PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python3 websocket_server.py &
websocket_pid=$!

sleep 2

if ps -p $websocket_pid > /dev/null; then
    echo "WebSocket integration server started successfully (PID: $websocket_pid)"
    echo "WEBSOCKET_PID=$websocket_pid" >> /tmp/voicesync_pids
else
    echo "Failed to start WebSocket integration server"
fi

echo "All services started!"
echo "Service endpoints:"
echo "  STT Service: ws://localhost:8765"
echo "  MT Service: http://localhost:8766"
echo "  TTS Service: http://localhost:8767"
echo "  WebSocket Integration: ws://localhost:8000"
echo ""
echo "To stop services, run: ./stop.sh"