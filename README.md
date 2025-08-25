# AES-STEGANOGRAPHY-PROJECT
# 🔐 AES Steganography Project  

This project combines **Cryptography (AES Encryption)** and **Steganography (Image Hiding)** to securely transmit secret messages.  
The idea is simple: **encrypt the message with AES** and then **hide the ciphertext inside an image** using steganography.  
Even if someone extracts the hidden data, they can’t read it without the AES key.  

---

## 🚀 Features  
- **AES-256 Encryption** for strong message security.  
- **Image Steganography (LSB method)** to hide encrypted data in images.  
- **Encoding**: Hide ciphertext inside PNG images.  
- **Decoding**: Extract hidden ciphertext and decrypt back to plaintext.  
- Works with custom messages and images.  

---

## 🛠️ Tech Stack  
- **Python 3**  
- [PyCryptodome](https://pypi.org/project/pycryptodome/) → AES Encryption/Decryption  
- [Pillow](https://pypi.org/project/pillow/) → Image handling  
- [Stepic / Custom LSB Algorithm] → Steganography  

---

## 📂 Project Structure  
AES-STEGANOGRAPHY-PROJECT/
│
├── main.py # Main script (encryption + steganography)
├── encoded.png # Example stego image (output)
├── requirements.txt # Required Python packages
└── README.md # Project documentation

---

## ⚡ How It Works  
1. **Encrypt Message** → AES takes plaintext + secret key and produces ciphertext.  
2. **Hide Ciphertext** → Ciphertext is embedded in the pixels of an image.  
3. **Extract Ciphertext** → Retrieve hidden bits from the image.  
4. **Decrypt Ciphertext** → AES decrypts back into the original plaintext.  

---

## ▶️ Usage  

### 1. Install Dependencies  
```bash
pip install -r requirements.txt

2. Run the Project
python main.py

3. Example Output
Ciphertext (encrypted message): b'\xd6\x88\xd0C...'
Message hidden in encoded.png
Extracted ciphertext: b'\xd6\x88\xd0C...'
Decrypted message: Meet me at 5 PM.

###📸 Demo
Input Message: Meet me at 5 PM.

Cover Image: A normal PNG file.

Stego Image: Looks identical, but secretly carries the AES-encrypted message.

###🎯 Applications
Secure communication over open channels.

Digital watermarking.

Covert data transfer.

###⚠️ Limitations
Large messages may require larger images.

JPEG compression may destroy hidden data (use PNG).

Security relies on both the AES key and the stego technique.

###👨‍💻 Authors
Lohet

Built as part of a learning project combining cryptography & steganography.
