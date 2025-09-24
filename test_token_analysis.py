#!/usr/bin/env python3
"""
Script para analizar el token JWT y verificar sus permisos
"""
import os
import json
import base64
import logging
from azure.identity import AzureCliCredential
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

def decode_jwt_payload(token):
    """Decode JWT payload without verification (for debugging only)"""
    try:
        # JWT structure: header.payload.signature
        parts = token.split('.')
        if len(parts) != 3:
            return None
            
        # Decode payload (add padding if needed)
        payload = parts[1]
        # Add padding if needed
        payload += '=' * (4 - len(payload) % 4)
        
        decoded_bytes = base64.b64decode(payload)
        payload_data = json.loads(decoded_bytes.decode('utf-8'))
        
        return payload_data
    except Exception as e:
        logger.error(f"Failed to decode JWT: {e}")
        return None

def analyze_token():
    """Analyze Azure token for Voice Live API compatibility"""
    logger.info("🔍 === ANALYZING AZURE TOKEN ===")
    
    load_dotenv("./.env", override=True)
    
    try:
        credential = AzureCliCredential()
        
        # Test different scopes
        scopes_to_test = [
            "https://ai.azure.com/.default",
            "https://cognitiveservices.azure.com/.default",
            "https://management.azure.com/.default"
        ]
        
        for scope in scopes_to_test:
            logger.info(f"\n🧪 Testing scope: {scope}")
            try:
                token = credential.get_token(scope)
                logger.info(f"✅ Token acquired for scope: {scope}")
                logger.info(f"🔑 Token length: {len(token.token)} characters")
                logger.info(f"⏰ Token expires: {token.expires_on}")
                
                # Decode and analyze payload
                payload = decode_jwt_payload(token.token)
                if payload:
                    logger.info("📋 Token payload analysis:")
                    logger.info(f"  • Audience (aud): {payload.get('aud', 'Not found')}")
                    logger.info(f"  • Issuer (iss): {payload.get('iss', 'Not found')}")
                    logger.info(f"  • Subject (sub): {payload.get('sub', 'Not found')}")
                    logger.info(f"  • Scopes (scp): {payload.get('scp', 'Not found')}")
                    logger.info(f"  • Application ID (appid): {payload.get('appid', 'Not found')}")
                    logger.info(f"  • Tenant ID (tid): {payload.get('tid', 'Not found')}")
                    logger.info(f"  • Object ID (oid): {payload.get('oid', 'Not found')}")
                    
                    # Check specific claims
                    if 'ai.azure.com' in payload.get('aud', ''):
                        logger.info("✅ Token has correct audience for AI services")
                    else:
                        logger.warning("⚠️ Token audience might not be correct for AI services")
                        
                    if 'user_impersonation' in payload.get('scp', ''):
                        logger.info("✅ Token has user_impersonation scope")
                    else:
                        logger.warning("⚠️ Token missing user_impersonation scope")
                        
                else:
                    logger.error("❌ Could not decode token payload")
                    
            except Exception as e:
                logger.error(f"❌ Failed to get token for scope {scope}: {e}")
                
    except Exception as e:
        logger.error(f"❌ Authentication setup failed: {e}")
        
    # Check environment variables
    logger.info("\n🔧 Environment Variables Check:")
    required_vars = [
        "AZURE_VOICE_LIVE_ENDPOINT",
        "AI_FOUNDRY_AGENT_ID", 
        "AI_FOUNDRY_PROJECT_NAME",
        "AZURE_VOICE_LIVE_API_VERSION"
    ]
    
    for var in required_vars:
        value = os.environ.get(var)
        if value:
            # Show partial value for security
            if len(value) > 20:
                display_value = f"{value[:10]}...{value[-6:]}"
            else:
                display_value = value
            logger.info(f"✅ {var}: {display_value}")
        else:
            logger.error(f"❌ {var}: Not set")

if __name__ == "__main__":
    analyze_token()