#!/usr/bin/env python3
"""
Script de prueba para diferentes endpoints y versiones de API
"""
import os
import logging
import requests
from azure.identity import AzureCliCredential
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

def test_api_endpoints():
    """Test different API versions and endpoints"""
    logger.info("🧪 === TESTING API ENDPOINTS ===")
    
    load_dotenv("./.env", override=True)
    
    endpoint = os.environ.get("AZURE_VOICE_LIVE_ENDPOINT")
    
    # Get token
    try:
        credential = AzureCliCredential()
        token = credential.get_token("https://ai.azure.com/.default")
        logger.info("✅ Token acquired")
    except Exception as e:
        logger.error(f"❌ Token failed: {e}")
        return
        
    headers = {
        "Authorization": f"Bearer {token.token}",
        "Content-Type": "application/json"
    }
    
    # Test different API versions
    api_versions = [
        "2025-05-01-preview",
        "2024-10-01-preview",
        "2024-05-01-preview",
        "2024-02-15-preview"
    ]
    
    # Test different paths
    paths = [
        "/voice-live/realtime",
        "/realtime",
        "/voice-live",
        "/openai/realtime",
        "/openai/voice-live"
    ]
    
    base_url = endpoint.rstrip('/')
    
    for api_version in api_versions:
        for path in paths:
            test_url = f"{base_url}{path}?api-version={api_version}"
            logger.info(f"🧪 Testing: {test_url}")
            
            try:
                # Try HEAD request first to check if endpoint exists
                response = requests.head(test_url, headers=headers, timeout=10)
                logger.info(f"   HEAD response: {response.status_code}")
                
                if response.status_code == 404:
                    logger.warning(f"   ❌ Path not found: {path}")
                elif response.status_code == 401:
                    logger.warning(f"   ❌ Unauthorized")
                elif response.status_code == 403:
                    logger.warning(f"   ❌ Forbidden")
                elif response.status_code in [200, 405]:  # 405 = Method not allowed (expected for WebSocket endpoint)
                    logger.info(f"   ✅ Endpoint exists: {test_url}")
                else:
                    logger.info(f"   ❓ Unexpected status: {response.status_code}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"   ⏰ Timeout")
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"   🔌 Connection error: {e}")
            except Exception as e:
                logger.error(f"   ❌ Error: {e}")
    
    # Test if the base URL is reachable
    logger.info(f"\n🧪 Testing base URL: {base_url}")
    try:
        response = requests.get(base_url, headers=headers, timeout=10)
        logger.info(f"Base URL response: {response.status_code}")
        if response.headers:
            logger.info(f"Response headers: {dict(response.headers)}")
    except Exception as e:
        logger.error(f"Base URL error: {e}")

if __name__ == "__main__":
    test_api_endpoints()