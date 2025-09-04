import os
import csv
import base64
import pandas as pd

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

def aes_decrypt(aes_key, aes_iv, enc_text: str) -> str:
    cipher = AES.new(aes_key, AES.MODE_CBC, aes_iv)
    decrypted = cipher.decrypt(base64.b64decode(enc_text))
    unpadded = unpad(decrypted, AES.block_size)
    return unpadded.decode('utf-8')

def decrypt_data(data: pd.DataFrame, aes_key, aes_iv, enc_col: str = 'phone_id') -> pd.DataFrame:
    assert enc_col in data.columns, f"Encrypted column '{enc_col}' does not exist in DataFrame"
    data['id'] = data[enc_col].apply(lambda x: aes_decrypt(aes_key, aes_iv, x))
    data.drop(columns=[enc_col], inplace=True)
    return data

def has_header(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        first_row = next(csv.reader(f))
        return 'phone_id' in first_row and 'message' in first_row
