#!/usr/bin/env python3
"""
Script de diagnóstico específico para ChatbotOnTerminal.py
Verifica configuración antes de ejecutar el chatbot
"""

import os
import json
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, AzureCliCredential

def test_terminal_chatbot_config():
    """Test configuration for ChatbotOnTerminal.py"""
    
    print("🔍 === DIAGNÓSTICO CHATBOT TERMINAL ===")
    
    # Load environment
    load_dotenv("./.env", override=True)
    
    # Check required variables
    required_vars = {
        "AZURE_VOICE_LIVE_ENDPOINT": os.environ.get("AZURE_VOICE_LIVE_ENDPOINT"),
        "AI_FOUNDRY_AGENT_ID": os.environ.get("AI_FOUNDRY_AGENT_ID"),
        "AI_FOUNDRY_PROJECT_NAME": os.environ.get("AI_FOUNDRY_PROJECT_NAME"),
        "AZURE_VOICE_LIVE_API_VERSION": os.environ.get("AZURE_VOICE_LIVE_API_VERSION")
    }
    
    print("\n📋 Variables de entorno:")
    missing_vars = []
    for var, value in required_vars.items():
        if not value or value.startswith("<"):
            print(f"  ❌ {var}: NO CONFIGURADO")
            missing_vars.append(var)
        else:
            # Mask sensitive values
            if len(value) > 20:
                display_value = f"{value[:10]}...{value[-4:]}"
            else:
                display_value = value
            print(f"  ✅ {var}: {display_value}")
    
    if missing_vars:
        print(f"\n❌ Variables faltantes: {', '.join(missing_vars)}")
        print("💡 Configúralas en tu archivo .env")
        return False
    
    # Test authentication
    print("\n🔑 Test de autenticación:")
    try:
        credential = AzureCliCredential()
        token = credential.get_token("https://ai.azure.com/.default")
        print(f"  ✅ Azure CLI: Token válido (expira: {token.expires_on})")
        return True
    except Exception as cli_error:
        print(f"  ❌ Azure CLI: {cli_error}")
        
        try:
            credential = DefaultAzureCredential()
            token = credential.get_token("https://ai.azure.com/.default")
            print(f"  ✅ Default Credential: Token válido (expira: {token.expires_on})")
            return True
        except Exception as default_error:
            print(f"  ❌ Default Credential: {default_error}")
    
    print("\n❌ === AUTENTICACIÓN FALLÓ ===")
    print("💡 Soluciones:")
    print("   1. Ejecuta 'az login' en tu terminal")
    print("   2. O configura Service Principal en .env:")
    print("      AZURE_TENANT_ID=tu-tenant-id")
    print("      AZURE_CLIENT_ID=tu-client-id") 
    print("      AZURE_CLIENT_SECRET=tu-client-secret")
    return False

def validate_voice_live_config():
    """Validate Voice Live specific configuration"""
    
    load_dotenv("./.env", override=True)
    
    endpoint = os.environ.get("AZURE_VOICE_LIVE_ENDPOINT")
    agent_id = os.environ.get("AI_FOUNDRY_AGENT_ID")
    project_name = os.environ.get("AI_FOUNDRY_PROJECT_NAME")
    
    print("\n🎯 Validación específica de Voice Live:")
    
    # Check endpoint format
    if endpoint:
        if not endpoint.startswith("https://"):
            print(f"  ⚠️  Endpoint should start with https://: {endpoint}")
        if endpoint.endswith("/"):
            print(f"  ⚠️  Endpoint shouldn't end with /: {endpoint}")
        else:
            print(f"  ✅ Endpoint format looks good")
    
    # Check agent_id format (usually a GUID)
    if agent_id and len(agent_id) == 36 and agent_id.count("-") == 4:
        print(f"  ✅ Agent ID format looks like a GUID")
    elif agent_id:
        print(f"  ⚠️  Agent ID doesn't look like a GUID: {agent_id[:20]}...")
    
    # Check project name
    if project_name and not project_name.startswith("<"):
        print(f"  ✅ Project name configured")
    
    return True

if __name__ == "__main__":
    print("🧪 Ejecutando diagnóstico para ChatbotOnTerminal.py...\n")
    
    config_ok = test_terminal_chatbot_config()
    validate_voice_live_config()
    
    if config_ok:
        print("\n🎉 === CONFIGURACIÓN VÁLIDA ===")
        print("✅ Puedes ejecutar ChatbotOnTerminal.py")
        print("\nComandos sugeridos:")
        print("  python ChatbotOnTerminal.py")
        exit(0)
    else:
        print("\n❌ === CONFIGURACIÓN INCOMPLETA ===")
        print("🔧 Corrige los problemas antes de ejecutar el chatbot")
        exit(1)