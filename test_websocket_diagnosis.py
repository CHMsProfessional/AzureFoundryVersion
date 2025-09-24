#!/usr/bin/env python3
"""
Script de diagnóstico para probar la conectividad WebSocket con Azure Voice Live
"""
import os
import json
import time
import logging
import sys
import uuid
from azure.identity import AzureCliCredential
from dotenv import load_dotenv
import websocket

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s:%(name)s:%(levelname)s:%(message)s'
)
logger = logging.getLogger(__name__)

def test_websocket_connection():
    """Test basic WebSocket connectivity"""
    logger.info("🔍 === DIAGNOSTIC TEST AZURE VOICE LIVE ===")
    
    # Load environment variables
    load_dotenv("./.env", override=True)
    
    endpoint = os.environ.get("AZURE_VOICE_LIVE_ENDPOINT")
    agent_id = os.environ.get("AI_FOUNDRY_AGENT_ID")
    project_name = os.environ.get("AI_FOUNDRY_PROJECT_NAME")
    api_version = os.environ.get("AZURE_VOICE_LIVE_API_VERSION", "2025-05-01-preview")
    
    logger.info(f"🔧 Configuration:")
    logger.info(f"🔧   Endpoint: {endpoint}")
    logger.info(f"🔧   Agent ID: {agent_id[:10]}...{agent_id[-4:] if len(agent_id) > 14 else agent_id}")
    logger.info(f"🔧   Project: {project_name}")
    logger.info(f"🔧   API Version: {api_version}")
    
    if not all([endpoint, agent_id, project_name]):
        logger.error("❌ Missing required configuration")
        return False
        
    # Get Azure token
    try:
        credential = AzureCliCredential()
        scopes = "https://ai.azure.com/.default"
        token = credential.get_token(scopes)
        logger.info("✅ Token acquired successfully")
        logger.debug(f"🔑 Token length: {len(token.token)} characters")
    except Exception as e:
        logger.error(f"❌ Failed to get token: {e}")
        return False
    
    # Test 1: Simple connection without agent-access-token
    logger.info("🧪 === TEST 1: Basic WebSocket Connection (without agent token) ===")
    
    ws_endpoint = endpoint.rstrip('/').replace("https://", "wss://")
    simple_url = f"{ws_endpoint}/voice-live/realtime?api-version={api_version}"
    logger.info(f"🔗 Testing URL: {simple_url}")
    
    headers = [
        f"Authorization: Bearer {token.token}",
        f"x-ms-client-request-id: {uuid.uuid4()}"
    ]
    
    def on_open(ws):
        logger.info("✅ WebSocket opened (Test 1)")
        
    def on_message(ws, message):
        logger.info(f"📨 Message received: {message[:100]}...")
        try:
            data = json.loads(message)
            logger.info(f"📨 Message type: {data.get('type', 'unknown')}")
        except:
            logger.info("📨 Message is not JSON")
            
    def on_error(ws, error):
        logger.error(f"❌ WebSocket error (Test 1): {error}")
        
    def on_close(ws, close_status_code, close_msg):
        logger.warning(f"🔌 Connection closed (Test 1) - Code: {close_status_code}, Message: {close_msg}")
    
    try:
        ws = websocket.WebSocketApp(
            simple_url,
            header=headers,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        logger.info("⏳ Starting Test 1 connection...")
        ws.run_forever()
        time.sleep(3)
        
    except Exception as e:
        logger.error(f"❌ Test 1 failed: {e}")
    
    # Test 2: Full URL with all parameters
    logger.info("🧪 === TEST 2: Full WebSocket Connection (with all parameters) ===")
    
    full_url = f"{ws_endpoint}/voice-live/realtime?api-version={api_version}&agent-project-name={project_name}&agent-id={agent_id}&agent-access-token={token.token}"
    logger.info(f"🔗 Testing URL: {full_url[:100]}...")
    
    def on_open2(ws):
        logger.info("✅ WebSocket opened (Test 2)")
        # Try to send a basic ping
        ping_data = {"type": "ping", "timestamp": time.time()}
        ws.send(json.dumps(ping_data))
        logger.info("📤 Sent ping message")
        
    def on_message2(ws, message):
        logger.info(f"📨 Message received (Test 2): {message[:100]}...")
        try:
            data = json.loads(message)
            logger.info(f"📨 Message type: {data.get('type', 'unknown')}")
            if data.get('type') == 'session.created':
                logger.info("🎉 Session created successfully!")
        except:
            logger.info("📨 Message is not JSON")
            
    def on_error2(ws, error):
        logger.error(f"❌ WebSocket error (Test 2): {error}")
        
    def on_close2(ws, close_status_code, close_msg):
        logger.warning(f"🔌 Connection closed (Test 2) - Code: {close_status_code}, Message: {close_msg}")
    
    try:
        ws2 = websocket.WebSocketApp(
            full_url,
            header=headers,
            on_open=on_open2,
            on_message=on_message2,
            on_error=on_error2,
            on_close=on_close2
        )
        
        logger.info("⏳ Starting Test 2 connection...")
        ws2.run_forever(ping_interval=10, ping_timeout=5)
        
    except Exception as e:
        logger.error(f"❌ Test 2 failed: {e}")

if __name__ == "__main__":
    test_websocket_connection()