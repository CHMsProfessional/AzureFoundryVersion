#!/usr/bin/env python3
"""
Script para verificar la configuración de autenticación de Azure
"""

import os
import sys
from dotenv import load_dotenv

def test_azure_auth():
    """Probar diferentes métodos de autenticación de Azure"""
    
    print("🔍 === TEST DE AUTENTICACIÓN AZURE ===")
    
    # Cargar variables de entorno
    load_dotenv("./.env", override=True)
    
    # Verificar variables de entorno básicas
    print("\n📋 Variables de entorno:")
    required_vars = [
        "AZURE_VOICE_LIVE_ENDPOINT",
        "AI_FOUNDRY_AGENT_ID", 
        "AI_FOUNDRY_PROJECT_NAME",
        "AZURE_VOICE_LIVE_API_VERSION"
    ]
    
    auth_vars = [
        "AZURE_TENANT_ID",
        "AZURE_CLIENT_ID", 
        "AZURE_CLIENT_SECRET"
    ]
    
    for var in required_vars:
        value = os.environ.get(var, "❌ NO CONFIGURADO")
        if value != "❌ NO CONFIGURADO" and len(value) > 20:
            display_value = f"{value[:10]}...{value[-4:]}"
        else:
            display_value = value
        print(f"  {var}: {display_value}")
    
    print("\n🔑 Variables de autenticación:")
    sp_configured = True
    for var in auth_vars:
        value = os.environ.get(var, "❌ NO CONFIGURADO")
        if value == "❌ NO CONFIGURADO":
            sp_configured = False
        if value != "❌ NO CONFIGURADO" and len(value) > 10:
            display_value = f"{value[:6]}...***"
        else:
            display_value = value
        print(f"  {var}: {display_value}")
    
    # Test 1: Azure CLI
    print("\n🧪 Test 1: Azure CLI")
    try:
        from azure.identity import AzureCliCredential
        cli_cred = AzureCliCredential()
        token = cli_cred.get_token("https://ai.azure.com/.default")
        print(f"  ✅ Azure CLI: Token obtenido (expira: {token.expires_on})")
        return True
    except Exception as e:
        print(f"  ❌ Azure CLI: {e}")
    
    # Test 2: Service Principal
    if sp_configured:
        print("\n🧪 Test 2: Service Principal")
        try:
            from azure.identity import ClientSecretCredential
            tenant_id = os.environ.get("AZURE_TENANT_ID")
            client_id = os.environ.get("AZURE_CLIENT_ID")
            client_secret = os.environ.get("AZURE_CLIENT_SECRET")
            
            sp_cred = ClientSecretCredential(tenant_id, client_id, client_secret)
            token = sp_cred.get_token("https://ai.azure.com/.default")
            print(f"  ✅ Service Principal: Token obtenido (expira: {token.expires_on})")
            return True
        except Exception as e:
            print(f"  ❌ Service Principal: {e}")
    else:
        print("\n⏭️ Test 2: Service Principal - SALTADO (variables no configuradas)")
    
    # Test 3: DefaultAzureCredential
    print("\n🧪 Test 3: DefaultAzureCredential (excluyendo PowerShell)")
    try:
        from azure.identity import DefaultAzureCredential
        cred = DefaultAzureCredential(
            exclude_powershell_credential=True,
            exclude_visual_studio_code_credential=True,
            exclude_interactive_browser_credential=True
        )
        token = cred.get_token("https://ai.azure.com/.default")
        print(f"  ✅ DefaultAzureCredential: Token obtenido (expira: {token.expires_on})")
        return True
    except Exception as e:
        print(f"  ❌ DefaultAzureCredential: {e}")
    
    print("\n❌ === NINGÚN MÉTODO DE AUTENTICACIÓN FUNCIONÓ ===")
    print("📖 Consulta AUTH_SETUP.md para instrucciones de configuración")
    return False

if __name__ == "__main__":
    try:
        success = test_azure_auth()
        if success:
            print("\n🎉 === AUTENTICACIÓN EXITOSA ===")
            print("✅ Tu aplicación debería funcionar correctamente")
            sys.exit(0)
        else:
            sys.exit(1)
    except ImportError as e:
        print(f"❌ Error importando módulos Azure: {e}")
        print("💡 Ejecuta: pip install azure-identity azure-core")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        sys.exit(1)