from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
from PIL import Image
import stepic

# --- AES ENCRYPTION FUNCTIONS ---

def aes_encrypt(message, key):
    cipher = AES.new(key, AES.MODE_CBC)
    ct_bytes = cipher.encrypt(pad(message.encode(), AES.block_size))
    return cipher.iv + ct_bytes  # prepend IV for use in decryption

def aes_decrypt(ciphertext, key):
    iv = ciphertext[:AES.block_size]
    ct = ciphertext[AES.block_size:]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    pt = unpad(cipher.decrypt(ct), AES.block_size)
    return pt.decode()

# --- STEGANOGRAPHY FUNCTIONS ---

def hide_message_in_image(image_path, message_bytes, output_path):
    img = Image.open(image_path)
    # Encode the message bytes as base64 string (stepic requires string)
    import base64
    encoded_message = base64.b64encode(message_bytes).decode()
    encoded_img = stepic.encode(img, encoded_message.encode())
    encoded_img.save(output_path)
    print(f"Message hidden in {output_path}")

def extract_message_from_image(image_path):
    img = Image.open(image_path)
    import base64
    encoded_message = stepic.decode(img)
    message_bytes = base64.b64decode(encoded_message)
    return message_bytes

# --- MAIN FLOW ---

def main():
    key = get_random_bytes(16)  # AES key (must be shared between sender and receiver)

    secret_message = "I am the danger"

    # Encrypt the secret message
    ciphertext = aes_encrypt(secret_message, key)
    print("Ciphertext (encrypted message):", ciphertext)

    # Hide ciphertext in image
    hide_message_in_image("original.png", ciphertext, "encoded.png")

    # Later, extract and decrypt
    extracted_ciphertext = extract_message_from_image("encoded.png")
    print("Extracted ciphertext:", extracted_ciphertext)

    decrypted_message = aes_decrypt(extracted_ciphertext, key)
    print("Decrypted message:", decrypted_message)

if __name__ == "__main__":
    main()