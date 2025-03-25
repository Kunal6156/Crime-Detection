import os
import base64
import hashlib
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

def generate_encryption_key() -> bytes:
    """Generate a secure encryption key"""
    return Fernet.generate_key()

def encrypt_data(data: str, key: bytes) -> str:
    """Encrypt data using Fernet symmetric encryption"""
    try:
        f = Fernet(key)
        encrypted_data = f.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()
    except Exception as e:
        logger.error(f"Encryption error: {e}")
        return data

def decrypt_data(encrypted_data: str, key: bytes) -> str:
    """Decrypt data using Fernet symmetric encryption"""
    try:
        f = Fernet(key)
        decrypted_data = f.decrypt(base64.urlsafe_b64decode(encrypted_data))
        return decrypted_data.decode()
    except Exception as e:
        logger.error(f"Decryption error: {e}")
        return ""

def generate_secure_hash(content: str) -> str:
    """Generate a secure hash for content verification"""
    salt = os.urandom(16)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000
    )
    key = base64.urlsafe_b64encode(kdf.derive(content.encode()))
    return key.decode()

def sanitize_input(input_string: str) -> str:
    """Sanitize input to prevent injection and XSS"""
    return ''.join(char for char in input_string if char.isprintable())