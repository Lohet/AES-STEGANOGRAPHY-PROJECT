from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
from PIL import Image
import stepic
import base64

# --- AES ENCRYPTION FUNCTIONS ---

def aes_encrypt(message, key):
    # Convert message to bytes (binary form)
    message_bytes = message.encode()
    print("Plaintext bytes:", message_bytes)
    print("Plaintext binary:", ' '.join(format(b, '08b') for b in message_bytes))

    # Pad and encrypt
    cipher = AES.new(key, AES.MODE_CBC)
    padded = pad(message_bytes, AES.block_size)
    print("Padded binary:", ' '.join(format(b, '08b') for b in padded))

    ct_bytes = cipher.encrypt(padded)
    full_cipher = cipher.iv + ct_bytes
    print("Ciphertext (bytes):", full_cipher)
    print("Ciphertext (binary):", ' '.join(format(b, '08b') for b in full_cipher))

    return full_cipher  # IV + ciphertext

def aes_decrypt(ciphertext, key):
    iv = ciphertext[:AES.block_size]
    ct = ciphertext[AES.block_size:]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    pt = unpad(cipher.decrypt(ct), AES.block_size)
    return pt.decode()

# --- STEGANOGRAPHY FUNCTIONS ---

def hide_message_in_image(image_path, message_bytes, output_path):
    img = Image.open(image_path)
    encoded_message = base64.b64encode(message_bytes).decode()
    encoded_img = stepic.encode(img, encoded_message.encode())
    encoded_img.save(output_path)
    print(f"Message hidden in {output_path}")

def extract_message_from_image(image_path):
    img = Image.open(image_path)
    encoded_message = stepic.decode(img)
    message_bytes = base64.b64decode(encoded_message)
    return message_bytes

# --- MAIN FLOW ---

def main():
    key = get_random_bytes(16)  # AES key
    print("AES Key (hex):", key.hex())

    secret_message = "I am the danger"

    # Encrypt
    ciphertext = aes_encrypt(secret_message, key)

    # Hide ciphertext in image
    hide_message_in_image("original.png", ciphertext, "encoded.png")

    # Extract and decrypt
    extracted_ciphertext = extract_message_from_image("encoded.png")
    print("Extracted ciphertext (binary):", ' '.join(format(b, '08b') for b in extracted_ciphertext))

    decrypted_message = aes_decrypt(extracted_ciphertext, key)
    print("Decrypted message:", decrypted_message)

if __name__ == "__main__":
    main()
