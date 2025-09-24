#!/usr/bin/env python3
"""
Script de prueba minimalista para Azure Voice Live WebSocket
"""
import os
import json
import time
import logging
import uuid
from azure.identity import AzureCliCredential
from dotenv import load_dotenv
import websocket

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

def test_minimal_connection():
    """Test with minimal headers"""
    logger.info("🧪 === TESTING MINIMAL WEBSOCKET CONNECTION ===")
    
    load_dotenv("./.env", override=True)
    
    endpoint = os.environ.get("AZURE_VOICE_LIVE_ENDPOINT")
    api_version = os.environ.get("AZURE_VOICE_LIVE_API_VERSION", "2025-05-01-preview")
    
    # Get token
    try:
        credential = AzureCliCredential()
        token = credential.get_token("https://ai.azure.com/.default")
        logger.info("✅ Token acquired")
    except Exception as e:
        logger.error(f"❌ Token failed: {e}")
        return
        
    # Minimal connection
    ws_endpoint = endpoint.rstrip('/').replace("https://", "wss://")
    url = f"{ws_endpoint}/voice-live/realtime?api-version={api_version}"
    
    logger.info(f"🔗 Testing URL: {url}")
    
    # ONLY essential headers
    headers = [
        f"Authorization: Bearer {token.token}"
    ]
    
    logger.info(f"🔗 Headers: {len(headers)} header(s)")
    
    def on_open(ws):
        logger.info("✅ MINIMAL CONNECTION OPENED!")
        # Don't send anything, just see if connection stays open
        
    def on_message(ws, message):
        logger.info(f"📨 Message: {message[:100]}...")
        
    def on_error(ws, error):
        logger.error(f"❌ Error: {error}")
        
    def on_close(ws, code, msg):
        logger.warning(f"🔌 Closed - Code: {code}, Msg: {msg}")
    
    ws = websocket.WebSocketApp(
        url,
        header=headers,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    
    logger.info("⏳ Starting minimal connection...")
    ws.run_forever(ping_interval=10, ping_timeout=5)

if __name__ == "__main__":
    test_minimal_connection()