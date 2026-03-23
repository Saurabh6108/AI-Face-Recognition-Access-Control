import os
import pickle
import numpy as np
from cryptography.fernet import Fernet

def generate_key():
    key = Fernet.generate_key()
    with open('data/secret.key', 'wb') as key_file:
        key_file.write(key)
    print("✅ Encryption key generated and saved!")
    return key

def load_key():
    if not os.path.exists('data/secret.key'):
        print("⚠️ No key found - generating new key...")
        return generate_key()
    with open('data/secret.key', 'rb') as key_file:
        key = key_file.read()
    print("✅ Encryption key loaded!")
    return key

def encrypt_encoding(encoding):
    key = load_key()
    fernet = Fernet(key)
    encoding_bytes = pickle.dumps(encoding)
    encrypted = fernet.encrypt(encoding_bytes)
    print("🔒 Encoding encrypted successfully!")
    return encrypted

def decrypt_encoding(encrypted_encoding):
    key = load_key()
    fernet = Fernet(key)
    decrypted_bytes = fernet.decrypt(encrypted_encoding)
    encoding = pickle.loads(decrypted_bytes)
    print("🔓 Encoding decrypted successfully!")
    return encoding

def test_encryption():
    print("\n🔐 Testing Encryption System...")
    print("─" * 40)
    fake_encoding = np.random.rand(128)
    print(f"Original encoding (first 5 values): {fake_encoding[:5]}")
    encrypted = encrypt_encoding(fake_encoding)
    print(f"\n🔒 Encrypted data (first 50 chars): {str(encrypted)[:50]}...")
    decrypted = decrypt_encoding(encrypted)
    print(f"\n🔓 Decrypted encoding (first 5 values): {decrypted[:5]}")
    if np.allclose(fake_encoding, decrypted):
        print("\n✅ Encryption test PASSED!")
        print("Original and decrypted encodings match perfectly!")
    else:
        print("\n❌ Encryption test FAILED!")

if __name__ == "__main__":
    test_encryption()