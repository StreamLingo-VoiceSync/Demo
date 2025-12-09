#!/bin/bash

# VoiceSync Service Stop Script
# Stops all running services

echo "Stopping VoiceSync services..."

# Check if PID file exists
if [ ! -f /tmp/voicesync_pids ]; then
    echo "No PID file found. Trying to kill services by name..."
    
    # Kill services by name
    pkill -f "python3.*src.main" 2>/dev/null
    pkill -f "python3.*websocket_server" 2>/dev/null
    
    echo "Services stopped (if they were running)"
    exit 0
fi

# Read PIDs from file and kill processes
while IFS= read -r line; do
    if [[ $line == *"="* ]]; then
        # Extract PID
        pid=$(echo $line | cut -d'=' -f2)
        
        # Check if process is running
        if ps -p $pid > /dev/null 2>&1; then
            echo "Stopping process $pid"
            kill $pid
        else
            echo "Process $pid not running"
        fi
    fi
done < /tmp/voicesync_pids

# Remove PID file
rm /tmp/voicesync_pids

echo "All VoiceSync services stopped"