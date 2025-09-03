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
    k_pixels_per_block: int = 32      # even more pixels per block for higher capacity
    max_mse: float = 100.0            # much more permissive
    min_ssim: float = 0.80            # much more permissive
    min_ncc: float = 0.70             # much more permissive


# ------------------------------
# Block helpers
# ------------------------------
def to_grayscale(img: Image.Image) -> Image.Image:
    return img.convert("L")


def image_to_blocks(arr: np.ndarray, bs: int) -> List[Tuple[int, int, np.ndarray]]:
    """
    Returns list of (row_block_index, col_block_index, block_array)
    Only full blocks are used; leftover edges are ignored to keep it simple.
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
    # Build a dict for quick access
    d = {(rb, cb): blk for (rb, cb, blk) in blocks}
    max_r = max(rb for rb, _, _ in blocks)
    max_c = max(cb for _, cb, _ in blocks)
    for (rb, cb, blk) in blocks:
        if cb < max_c:
            pairs.append(((rb, cb), blk, d[(rb, cb + 1)]))
    return pairs


# ------------------------------
# PRNG for deterministic "random" selection per block
# ------------------------------
def seeded_rng(seed_bytes: bytes) -> random.Random:
    # Reduce bytes to a reproducible int seed
    return random.Random(int.from_bytes(seed_bytes, "big", signed=False) % (2**63 - 1))


def pick_k_positions(block_shape: Tuple[int, int], k: int, rng: random.Random) -> List[Tuple[int, int]]:
    h, w = block_shape
    indices = [(i, j) for i in range(h) for j in range(w)]
    rng.shuffle(indices)
    return indices[:k]


# ------------------------------
# LSB matching (±1)
# ------------------------------
def lsb(value: int) -> int:
    return value & 1


def lsb_match(value: int, bit: int, rng: random.Random) -> int:
    if (value & 1) == bit:
        return value
    # choose +1 or -1, but keep [0,255]
    if value == 0:
        return 1
    if value == 255:
        return 254
    return value + (1 if rng.random() < 0.5 else -1)


# (Removed duplicate definitions of embed_bits_into_image, extract_bits_from_image, hide_text, and reveal_text. The patched versions below are kept.)

# ------------------------------
# Embedding / Extraction (patched)
# ------------------------------
def embed_bits_into_image(
    img_gray: Image.Image,
    payload_bits: List[int],
    key: bytes,
    iv: bytes,
    cfg: StegoConfig,
    start_block: int = 0
) -> Tuple[Image.Image, int]:
    """
    Embeds payload_bits into img_gray starting at eligible_blocks[start_block].
    Returns (modified_image, blocks_consumed)
    """
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
    print(f"[DEBUG] Eligible blocks for embedding: {len(eligible_blocks)}")
    if not eligible_blocks:
        raise RuntimeError("No eligible blocks found with current thresholds. Loosen thresholds or use a different image.")

    bit_idx = 0
    total_bits = len(payload_bits)
    h, w = arr.shape

    # iterate eligible_blocks starting from start_block
    blocks_used = 0
    for idx in range(start_block, len(eligible_blocks)):
        (rb, cb), _block = eligible_blocks[idx]
        if bit_idx >= total_bits:
            break
        r0, c0 = rb * bs, cb * bs
        seed_pos_material = key + struct.pack(">II", rb, cb)            # positions independent of IV
        seed_dir_material = key + iv + struct.pack(">II", rb, cb)      # direction uses IV + key

        rng_pos = seeded_rng(seed_pos_material)
        rng_dir = seeded_rng(seed_dir_material)

        positions = pick_k_positions((bs, bs), cfg.k_pixels_per_block, rng_pos)
        for (pr, pc) in positions:
            if bit_idx >= total_bits:
                break
            pixel_val = int(arr[r0 + pr, c0 + pc])
            new_val = lsb_match(pixel_val, payload_bits[bit_idx], rng_dir)
            arr[r0 + pr, c0 + pc] = np.uint8(new_val)
            bit_idx += 1

        blocks_used += 1

    if bit_idx < total_bits:
        raise RuntimeError(
            f"Insufficient capacity in eligible blocks. Embedded {bit_idx} / {total_bits} bits. "
            f"Increase k_pixels_per_block, loosen thresholds, or use larger image."
        )

    return Image.fromarray(arr).convert("L"), blocks_used


def extract_bits_from_image(
    stego_gray: Image.Image,
    num_bits: int,
    key: bytes,
    iv: bytes,
    cfg: StegoConfig,
    start_block: int = 0
) -> Tuple[List[int], int]:
    """
    Extract num_bits starting from eligible_blocks[start_block].
    Returns (bits_list, blocks_consumed)
    """
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
    print(f"[DEBUG] Eligible blocks for extraction: {len(eligible_blocks)}")

    bits: List[int] = []
    blocks_used = 0
    for idx in range(start_block, len(eligible_blocks)):
        (rb, cb), _ = eligible_blocks[idx]
        if len(bits) >= num_bits:
            break
        r0, c0 = rb * bs, cb * bs
        seed_pos_material = key + struct.pack(">II", rb, cb)  # positions must match embedding -> independent of IV
        rng_pos = seeded_rng(seed_pos_material)
        positions = pick_k_positions((bs, bs), cfg.k_pixels_per_block, rng_pos)
        for (pr, pc) in positions:
            if len(bits) >= num_bits:
                break
            bits.append(lsb(int(arr[r0 + pr, c0 + pc])))
        blocks_used += 1

    if len(bits) < num_bits:
        raise RuntimeError(f"Could not recover required number of bits ({len(bits)}/{num_bits}).")

    return bits, blocks_used


# ------------------------------
# Public API: hide / reveal text (patched)
# ------------------------------
def embed_bits_directly(arr: np.ndarray, bits: List[int], start_idx: int = 0) -> None:
    """Embed bits directly into the LSB of the array, starting from start_idx."""
    h, w = arr.shape
    total_pixels = h * w
    for i, bit in enumerate(bits):
        if start_idx + i >= total_pixels:
            raise RuntimeError("Not enough space in image for embedding")
        pixel_idx = start_idx + i
        row = pixel_idx // w
        col = pixel_idx % w
        # Clear LSB and set to new bit
        arr[row, col] = (arr[row, col] & 0xFE) | bit

def extract_bits_directly(arr: np.ndarray, num_bits: int, start_idx: int = 0) -> List[int]:
    """Extract bits directly from the LSB of the array, starting from start_idx."""
    h, w = arr.shape
    total_pixels = h * w
    if start_idx + num_bits > total_pixels:
        raise RuntimeError("Not enough bits in image")
    bits = []
    for i in range(num_bits):
        pixel_idx = start_idx + i
        row = pixel_idx // w
        col = pixel_idx % w
        bits.append(arr[row, col] & 1)
    return bits

def hide_text(
    cover_path: str,
    stego_out_path: str,
    text: str,
    key: bytes,
    cfg: StegoConfig = StegoConfig()
) -> None:
    img = Image.open(cover_path)
    img = img.convert("RGB")
    arr = np.array(img)
    blue = arr[:, :, 2]

    iv, ct = aes_encrypt_text(key, text)
    header = struct.pack(">I", len(ct)) + iv
    header_bits = bytes_to_bits(header)
    ct_bits = bytes_to_bits(ct)

    total_bits = len(header_bits) + len(ct_bits)
    h, w = blue.shape
    if total_bits > h * w:
        raise RuntimeError(f"Image too small. Need {total_bits} pixels, but image has only {h * w} pixels.")

    try:
        # Embed header starting at beginning of image
        embed_bits_directly(blue, header_bits, 0)
        # Embed ciphertext after header
        embed_bits_directly(blue, ct_bits, len(header_bits))
    except RuntimeError as e:
        raise RuntimeError(f"Embedding failed: {e}")

    arr[:, :, 2] = blue
    out_img = Image.fromarray(arr)
    out_img.save(stego_out_path)


def reveal_text(
    stego_path: str,
    key: bytes,
    cfg: StegoConfig = StegoConfig()
) -> str:
    img = Image.open(stego_path)
    img = img.convert("RGB")
    arr = np.array(img)
    blue = arr[:, :, 2]

    try:
        # Extract header (160 bits = 20 bytes: 4 for length + 16 for IV)
        header_bits = extract_bits_directly(blue, 160, 0)
        header = bits_to_bytes(header_bits)[:20]
        ct_len = struct.unpack(">I", header[:4])[0]
        iv = header[4:20]

        # Extract ciphertext
        ct_num_bits = ct_len * 8
        ct_bits = extract_bits_directly(blue, ct_num_bits, 160)
        ciphertext = bits_to_bytes(ct_bits)[:ct_len]
    except RuntimeError as e:
        raise RuntimeError(f"Extraction failed: {e}")

    return aes_decrypt_text(key, iv, ciphertext)

# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    """
    Example:
      python stego_aes_block_lsb.py
    Make sure 'cover.png' exists. The script writes 'stego.png' and prints the recovered text.
    """
    cfg = StegoConfig(
        block_size=8,
        k_pixels_per_block=12,  # tune based on capacity
        max_mse=20.0,
        min_ssim=0.94,
        min_ncc=0.88
    )

    cover_image = "cover.png"   # <-- put your image here
    stego_image = "stego.png"
    secret_text = "Top secret: Meet at 17:30."
    key_32 = b"this_is_a_32byte_key_for_demo!!!!"[:32]  # 32 bytes (AES-256)

    # Hide
    hide_text(cover_image, stego_image, secret_text, key_32, cfg)
    print(f"[+] Stego image saved to: {stego_image}")

    # Reveal
    recovered = reveal_text(stego_image, key_32, cfg)
    print("[+] Recovered text:", recovered)
