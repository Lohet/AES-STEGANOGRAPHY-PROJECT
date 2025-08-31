# file: stego_aes_block_lsb.py
import os
import math
import struct
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from skimage.metrics import structural_similarity as ssim


# ------------------------------
# AES helpers
# ------------------------------
def aes_encrypt_text(key: bytes, plaintext: str) -> Tuple[bytes, bytes]:
    """
    AES-CBC with PKCS#7 padding.
    Returns (iv, ciphertext).
    key must be 16/24/32 bytes.
    """
    iv = os.urandom(16)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    ct = cipher.encrypt(pad(plaintext.encode("utf-8"), AES.block_size))
    return iv, ct


def aes_decrypt_text(key: bytes, iv: bytes, ciphertext: bytes) -> str:
    cipher = AES.new(key, AES.MODE_CBC, iv)
    pt = unpad(cipher.decrypt(ciphertext), AES.block_size)
    return pt.decode("utf-8")


# ------------------------------
# Bit utilities
# ------------------------------
def bytes_to_bits(data: bytes) -> List[int]:
    return [(byte >> (7 - i)) & 1 for byte in data for i in range(8)]


def bits_to_bytes(bits: List[int]) -> bytes:
    out = bytearray()
    for i in range(0, len(bits), 8):
        b = 0
        for j in range(8):
            if i + j < len(bits):
                b = (b << 1) | (bits[i + j] & 1)
            else:
                b <<= 1
        out.append(b)
    return bytes(out)


# ------------------------------
# Similarity metrics
# ------------------------------
def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.mean((a - b) ** 2))


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a_mean = a.mean()
    b_mean = b.mean()
    num = np.sum((a - a_mean) * (b - b_mean))
    den = math.sqrt(np.sum((a - a_mean) ** 2) * np.sum((b - b_mean) ** 2)) + 1e-12
    return float(num / den)


# ------------------------------
# Core stego config
# ------------------------------
@dataclass
class StegoConfig:
    block_size: int = 8
    k_pixels_per_block: int = 10      # pixels chosen per block
    max_mse: float = 15.0             # lower is more similar
    min_ssim: float = 0.95            # higher is more similar
    min_ncc: float = 0.90             # higher is more similar


# ------------------------------
# Block helpers
# ------------------------------
def to_grayscale(img: Image.Image) -> Image.Image:
    return img.convert("L")


def image_to_blocks(arr: np.ndarray, bs: int) -> List[Tuple[int, int, np.ndarray]]:
    """
    Returns list of (row_block_index, col_block_index, block_array)
    Only full blocks are used.
    """
    h, w = arr.shape
    blocks = []
    for r in range(0, h - h % bs, bs):
        for c in range(0, w - w % bs, bs):
            blocks.append((r // bs, c // bs, arr[r:r + bs, c:c + bs]))
    return blocks


def block_pair_right_neighbor(blocks: List[Tuple[int, int, np.ndarray]]) -> List[Tuple[Tuple[int,int], np.ndarray, np.ndarray]]:
    """
    Pair each block with its right neighbor (if exists).
    """
    pairs = []
    d = {(rb, cb): blk for (rb, cb, blk) in blocks}
    max_r = max(rb for rb, _, _ in blocks)
    max_c = max(cb for _, cb, _ in blocks)
    for (rb, cb, blk) in blocks:
        if cb < max_c:
            pairs.append(((rb, cb), blk, d[(rb, cb + 1)]))
    return pairs


# ------------------------------
# PRNG for deterministic "random" selection
# ------------------------------
def seeded_rng(seed_bytes: bytes) -> random.Random:
    return random.Random(int.from_bytes(seed_bytes, "big", signed=False) % (2**63 - 1))


def pick_k_positions(block_shape: Tuple[int, int], k: int, rng: random.Random) -> List[Tuple[int, int]]:
    h, w = block_shape
    indices = [(i, j) for i in range(h) for j in range(w)]
    rng.shuffle(indices)
    return indices[:k]


# ------------------------------
# LSB matching
# ------------------------------
def lsb(value: int) -> int:
    return value & 1


def lsb_match(value: int, bit: int, rng: random.Random) -> int:
    if (value & 1) == bit:
        return value
    if value == 0:
        return 1
    if value == 255:
        return 254
    return value + (1 if rng.random() < 0.5 else -1)


# ------------------------------
# Embedding / Extraction
# ------------------------------
def embed_bits_into_image(img_gray: Image.Image, payload_bits: List[int], key: bytes, iv: bytes, cfg: StegoConfig, start_block: int = 0) -> Tuple[Image.Image, int]:
    arr = np.array(img_gray, dtype=np.uint8)
    bs = cfg.block_size

    blocks = image_to_blocks(arr, bs)
    pairs = block_pair_right_neighbor(blocks)

    eligible_blocks = []
    for (rb_cb, A, B) in pairs:
        _mse = mse(A, B)
        _ssim = ssim(A, B, data_range=255)
        _ncc = ncc(A, B)
        if _mse <= cfg.max_mse and _ssim >= cfg.min_ssim and _ncc >= cfg.min_ncc:
            eligible_blocks.append((rb_cb, A.copy()))

    if not eligible_blocks:
        raise RuntimeError("No eligible blocks found. Loosen thresholds or use another image.")

    bit_idx = 0
    total_bits = len(payload_bits)
    blocks_used = 0

    for idx in range(start_block, len(eligible_blocks)):
        (rb, cb), _block = eligible_blocks[idx]
        if bit_idx >= total_bits:
            break
        r0, c0 = rb * bs, cb * bs
        seed_material = key + iv + struct.pack(">II", rb, cb)
        rng = seeded_rng(seed_material)
        positions = pick_k_positions((bs, bs), cfg.k_pixels_per_block, rng)
        for (pr, pc) in positions:
            if bit_idx >= total_bits:
                break
            pixel_val = int(arr[r0 + pr, c0 + pc])
            new_val = lsb_match(pixel_val, payload_bits[bit_idx], rng)
            arr[r0 + pr, c0 + pc] = np.uint8(new_val)
            bit_idx += 1
        blocks_used += 1

    if bit_idx < total_bits:
        raise RuntimeError("Not enough capacity in image to embed all bits.")

    return Image.fromarray(arr).convert("L"), blocks_used


def extract_bits_from_image(stego_gray: Image.Image, num_bits: int, key: bytes, iv: bytes, cfg: StegoConfig, start_block: int = 0) -> Tuple[List[int], int]:
    arr = np.array(stego_gray, dtype=np.uint8)
    bs = cfg.block_size

    blocks = image_to_blocks(arr, bs)
    pairs = block_pair_right_neighbor(blocks)

    eligible_blocks = []
    for (rb_cb, A, B) in pairs:
        _mse = mse(A, B)
        _ssim = ssim(A, B, data_range=255)
        _ncc = ncc(A, B)
        if _mse <= cfg.max_mse and _ssim >= cfg.min_ssim and _ncc >= cfg.min_ncc:
            eligible_blocks.append((rb_cb, A))

    bits: List[int] = []
    blocks_used = 0
    for idx in range(start_block, len(eligible_blocks)):
        (rb, cb), _ = eligible_blocks[idx]
        if len(bits) >= num_bits:
            break
        r0, c0 = rb * bs, cb * bs
        seed_material = key + iv + struct.pack(">II", rb, cb)
        rng = seeded_rng(seed_material)
        positions = pick_k_positions((bs, bs), cfg.k_pixels_per_block, rng)
        for (pr, pc) in positions:
            if len(bits) >= num_bits:
                break
            bits.append(lsb(int(arr[r0 + pr, c0 + pc])))
        blocks_used += 1

    if len(bits) < num_bits:
        raise RuntimeError("Could not extract enough bits from stego image.")

    return bits, blocks_used


# ------------------------------
# Public API
# ------------------------------
def hide_text(cover_path: str, stego_out_path: str, text: str, key: bytes, cfg: StegoConfig = StegoConfig()) -> None:
    img = Image.open(cover_path)
    img = to_grayscale(img)

    iv, ct = aes_encrypt_text(key, text)

    header = struct.pack(">I", len(ct)) + iv  # length + IV
    header_bits = bytes_to_bits(header)
    ct_bits = bytes_to_bits(ct)

    iv_zero = b"\x00" * 16
    stego_after_header, header_blocks_used = embed_bits_into_image(img, header_bits, key, iv_zero, cfg, start_block=0)
    stego_final, _ = embed_bits_into_image(stego_after_header, ct_bits, key, iv, cfg, start_block=header_blocks_used)

    stego_final.save(stego_out_path)


def reveal_text(stego_path: str, key: bytes, cfg: StegoConfig = StegoConfig()) -> str:
    img = Image.open(stego_path)
    img = to_grayscale(img)

    iv_zero = b"\x00" * 16
    header_bits, header_blocks_used = extract_bits_from_image(img, 160, key, iv_zero, cfg, start_block=0)
    header = bits_to_bytes(header_bits)[:20]
    ct_len = struct.unpack(">I", header[:4])[0]
    iv = header[4:20]

    ct_bits, _ = extract_bits_from_image(img, ct_len * 8, key, iv, cfg, start_block=header_blocks_used)
    ciphertext = bits_to_bytes(ct_bits)[:ct_len]

    return aes_decrypt_text(key, iv, ciphertext)


# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    cfg = StegoConfig(
        block_size=8,
        k_pixels_per_block=12,
        max_mse=20.0,
        min_ssim=0.94,
        min_ncc=0.88
    )

    cover_image = "cover.png"   # must exist in same folder
    stego_image = "stego.png"
    secret_text = "Top secret: Meet at 17:30."
    key_32 = b"this_is_a_32byte_key_for_demo!!!!"[:32]  # 32 bytes (AES-256)

    # Hide
    hide_text(cover_image, stego_image, secret_text, key_32, cfg)
    print(f"[+] Stego image saved to: {stego_image}")

    # Reveal
    recovered = reveal_text(stego_image, key_32, cfg)
    print("[+] Recovered text:", recovered)
