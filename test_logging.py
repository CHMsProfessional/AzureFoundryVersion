#!/usr/bin/env python3
"""
Script de prueba para verificar que el logging funciona correctamente
"""
import os
import sys
import logging
from datetime import datetime

def test_logging():
    """Test logging functionality"""
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f'logs/{timestamp}_test.log'
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s:%(name)s:%(levelname)s:%(message)s',
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    
    # Get logger
    logger = logging.getLogger(__name__)
    
    # Test different log levels
    logger.debug("🐛 This is a DEBUG message")
    logger.info("ℹ️ This is an INFO message")
    logger.warning("⚠️ This is a WARNING message")
    logger.error("❌ This is an ERROR message")
    
    print(f"\n📝 Log file created: {log_file}")
    print(f"📂 Working directory: {os.getcwd()}")
    
    # Check if log file was created and has content
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"📄 Log file size: {len(content)} characters")
            if content:
                print("✅ Logging is working correctly!")
                print("\n📄 Log file contents:")
                print("-" * 50)
                print(content)
                print("-" * 50)
            else:
                print("❌ Log file is empty!")
    else:
        print(f"❌ Log file was not created: {log_file}")

if __name__ == "__main__":
    test_logging()