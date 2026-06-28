"""Tests for axiompy.crypto."""
import pytest
from axiompy.crypto import (
    rsa_keygen,
    rsa_encrypt,
    rsa_decrypt,
    elgamal_keygen,
    elgamal_encrypt,
    elgamal_decrypt,
    diffie_hellman_key_exchange,
    sha256,
    _is_prime,
)


class TestHelpers:
    def test_is_prime_small(self):
        assert _is_prime(2)
        assert _is_prime(3)
        assert _is_prime(17)
        assert _is_prime(7919)
        assert not _is_prime(1)
        assert not _is_prime(4)
        assert not _is_prime(100)

    def test_sha256_known(self):
        assert sha256(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        assert sha256(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        assert sha256(b"hello") == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


class TestRSA:
    def test_roundtrip_512(self):
        pub, priv = rsa_keygen(512)
        msg = 42
        ct = rsa_encrypt(msg, pub)
        pt = rsa_decrypt(ct, priv)
        assert pt == msg

    def test_roundtrip_large_message(self):
        pub, priv = rsa_keygen(512)
        msg = 123456789
        ct = rsa_encrypt(msg, pub)
        pt = rsa_decrypt(ct, priv)
        assert pt == msg


class TestElGamal:
    def test_roundtrip(self):
        pub, priv = elgamal_keygen(128)
        msg = 99
        ct = elgamal_encrypt(msg, pub)
        pt = elgamal_decrypt(ct, priv)
        assert pt == msg


class TestDiffieHellman:
    def test_shared_secret(self):
        p = 0xFFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A63A3621
        g = 2
        pub_a, pub_b, s_a, s_b = diffie_hellman_key_exchange(p, g, 12345, 67890)
        assert s_a == s_b

    def test_small_prime(self):
        p = 23
        g = 5
        pub_a, pub_b, s_a, s_b = diffie_hellman_key_exchange(p, g, 6, 15)
        assert s_a == s_b
