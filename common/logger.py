"""Shared logger configuration"""

import logging
import sys
from pathlib import Path

# Centralized logs directory in the project root
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

def setup_stt_logger():
    """Setup logger for STT system"""
    # Use centralized logs directory
    LOG_DIR = LOGS_DIR / "stt"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler_info = logging.FileHandler(LOG_DIR / "stt_system.log", encoding="utf-8")
    file_handler_error = logging.FileHandler(LOG_DIR / "stt_errors.log", encoding="utf-8")

    stream_handler.setLevel(logging.INFO)
    file_handler_info.setLevel(logging.INFO)
    file_handler_error.setLevel(logging.ERROR)

    log_format = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s')
    stream_handler.setFormatter(log_format)
    file_handler_info.setFormatter(log_format)
    file_handler_error.setFormatter(log_format)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[stream_handler, file_handler_info, file_handler_error]
    )

    return logging.getLogger("stt_system")


def setup_mt_logger():
    """Setup logger for MT system"""
    # Use centralized logs directory
    LOG_DIR = LOGS_DIR / "mt"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger("MT_v9_DualLane_Production")
    log.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(LOG_DIR / "mt_v9_dual_lane_production.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    log.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(console_formatter)
    log.addHandler(console_handler)
    
    return log


def setup_tts_logger():
    """Setup logger for TTS system"""
    # Use centralized logs directory
    LOG_DIR = LOGS_DIR / "tts"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger("tts_system")
    log.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(LOG_DIR / "tts_system.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    log.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(console_formatter)
    log.addHandler(console_handler)
    
    return log