"""Cryptography example — RSA, ElGamal, Diffie-Hellman, SHA-256."""
from axiompy import Axiom

# ---- RSA ----
print("=== RSA ===")
pub, priv = Axiom.rsa_keygen(512)
msg = 42
ct = Axiom.rsa_encrypt(msg, pub)
pt = Axiom.rsa_decrypt(ct, priv)
print(f"Plaintext:  {msg}")
print(f"Ciphertext: {ct}")
print(f"Decrypted:  {pt}")
print(f"Match:      {msg == pt}")

# ---- ElGamal ----
print("\n=== ElGamal ===")
pub, priv = Axiom.elgamal_keygen(128)
msg = 99
ct = Axiom.elgamal_encrypt(msg, pub)
pt = Axiom.elgamal_decrypt(ct, priv)
print(f"Plaintext:  {msg}")
print(f"Ciphertext: {ct}")
print(f"Decrypted:  {pt}")
print(f"Match:      {msg == pt}")

# ---- Diffie-Hellman ----
print("\n=== Diffie-Hellman ===")
p = 0xFFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A63A3621
pub_a, pub_b, s_a, s_b = Axiom.diffie_hellman_key_exchange(p, 2, 12345, 67890)
print(f"Alice public:   {pub_a}")
print(f"Bob public:     {pub_b}")
print(f"Alice shared:   {s_a}")
print(f"Bob shared:     {s_b}")
print(f"Match:          {s_a == s_b}")

# ---- SHA-256 ----
print("\n=== SHA-256 ===")
print(f"sha256(b''):     {Axiom.sha256(b'')}")
print(f"sha256(b'abc'):  {Axiom.sha256(b'abc')}")
print(f"sha256(b'hello'): {Axiom.sha256(b'hello')}")
