import random

# ---- number-theoretic helpers ----------------------------------------------


def _is_prime(n: int, k: int = 20) -> bool:
    """Miller-Rabin primality test."""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    for _ in range(k):
        a = random.randrange(2, n - 2) if n > 4 else 2
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def _random_prime(bits: int) -> int:
    """Generate a random prime of roughly ``bits`` bits."""
    while True:
        n = random.getrandbits(bits)
        n |= (1 << (bits - 1)) | 1  # set top bit and make odd
        if _is_prime(n):
            return n


def _modinv(a: int, m: int) -> int:
    """Modular multiplicative inverse via extended Euclidean."""
    return pow(a, -1, m)


# ---- RSA ------------------------------------------------------------------


def rsa_keygen(bits: int = 1024, e: int = 65537) -> tuple[tuple[int, int], tuple[int, int]]:
    """Generate an RSA key pair.

    Args:
        bits: Bit length of the modulus (default 1024).
        e: Public exponent (default 65537).

    Returns:
        ``(public_key, private_key)`` where each key is ``(n, exponent)``.
    """
    p = _random_prime(bits // 2)
    q = _random_prime(bits // 2)
    n = p * q
    phi = (p - 1) * (q - 1)
    d = _modinv(e, phi)
    return ((n, e), (n, d))


def rsa_encrypt(message: int, public_key: tuple[int, int]) -> int:
    """Encrypt an integer message with an RSA public key.

    Args:
        message: Integer message (must be < n).
        public_key: ``(n, e)``.

    Returns:
        int: Ciphertext.
    """
    n, e = public_key
    return pow(message, e, n)


def rsa_decrypt(ciphertext: int, private_key: tuple[int, int]) -> int:
    """Decrypt an RSA ciphertext with a private key.

    Args:
        ciphertext: Integer ciphertext.
        private_key: ``(n, d)``.

    Returns:
        int: Plaintext.
    """
    n, d = private_key
    return pow(ciphertext, d, n)


# ---- ElGamal --------------------------------------------------------------


def elgamal_keygen(bits: int = 256) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Generate an ElGamal key pair.

    Returns:
        ``(public_key, private_key)`` where public = ``(p, g, h)``
        and private = ``(p, g, x)``.
    """
    p = _random_prime(bits)
    g = random.randrange(2, p - 1)
    x = random.randrange(2, p - 2)
    h = pow(g, x, p)
    return ((p, g, h), (p, g, x))


def elgamal_encrypt(message: int, public_key: tuple[int, int, int]) -> tuple[int, int]:
    """Encrypt an integer message with an ElGamal public key.

    Args:
        message: Integer message (< p).
        public_key: ``(p, g, h)``.

    Returns:
        ``(c1, c2)`` ciphertext pair.
    """
    p, g, h = public_key
    y = random.randrange(2, p - 2)
    c1 = pow(g, y, p)
    s = pow(h, y, p)
    c2 = (message * s) % p
    return (c1, c2)


def elgamal_decrypt(ciphertext: tuple[int, int], private_key: tuple[int, int, int]) -> int:
    """Decrypt an ElGamal ciphertext.

    Args:
        ciphertext: ``(c1, c2)`` pair.
        private_key: ``(p, g, x)``.

    Returns:
        int: Plaintext.
    """
    c1, c2 = ciphertext
    p, _g, x = private_key
    s = pow(c1, x, p)
    s_inv = _modinv(s, p)
    return (c2 * s_inv) % p


# ---- Diffie-Hellman -------------------------------------------------------


def diffie_hellman_key_exchange(p: int, g: int, private_a: int, private_b: int
                                ) -> tuple[int, int, int, int]:
    """Simulate Diffie-Hellman key exchange.

    Args:
        p: Prime modulus.
        g: Generator.
        private_a: Alice's private key.
        private_b: Bob's private key.

    Returns:
        ``(public_a, public_b, shared_a, shared_b)`` — both shared secrets are equal.
    """
    public_a = pow(g, private_a, p)
    public_b = pow(g, private_b, p)
    shared_a = pow(public_b, private_a, p)
    shared_b = pow(public_a, private_b, p)
    return (public_a, public_b, shared_a, shared_b)


# ---- SHA-256 (pure Python) -------------------------------------------------


def sha256(message: bytes) -> str:
    """Compute SHA-256 hash of a byte string (pure Python).

    Args:
        message: Input bytes.

    Returns:
        str: Hex digest (64 characters).
    """
    import struct

    h0 = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ]

    k = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
        0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ]

    def _rr(x, n):
        return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF

    ml = len(message) * 8
    msg = bytearray(message)
    msg.append(0x80)
    while (len(msg) * 8) % 512 != 448:
        msg.append(0x00)
    msg.extend(struct.pack(">Q", ml))

    for i in range(0, len(msg), 64):
        chunk = msg[i:i + 64]
        w = list(struct.unpack(">16L", chunk)) + [0] * 48
        for t in range(16, 64):
            s0 = _rr(w[t - 15], 7) ^ _rr(w[t - 15], 18) ^ (w[t - 15] >> 3)
            s1 = _rr(w[t - 2], 17) ^ _rr(w[t - 2], 19) ^ (w[t - 2] >> 10)
            w[t] = (w[t - 16] + s0 + w[t - 7] + s1) & 0xFFFFFFFF

        a, b, c, d, e, f, g, h_val = h0

        for t in range(64):
            S1 = _rr(e, 6) ^ _rr(e, 11) ^ _rr(e, 25)
            ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
            temp1 = (h_val + S1 + ch + k[t] + w[t]) & 0xFFFFFFFF
            S0 = _rr(a, 2) ^ _rr(a, 13) ^ _rr(a, 22)
            maj = ((a & b) ^ (a & c) ^ (b & c)) & 0xFFFFFFFF
            temp2 = (S0 + maj) & 0xFFFFFFFF

            h_val = g
            g = f
            f = e
            e = (d + temp1) & 0xFFFFFFFF
            d = c
            c = b
            b = a
            a = (temp1 + temp2) & 0xFFFFFFFF

        h0 = [(h0[i] + [a, b, c, d, e, f, g, h_val][i]) & 0xFFFFFFFF for i in range(8)]

    return "".join(f"{x:08x}" for x in h0)
