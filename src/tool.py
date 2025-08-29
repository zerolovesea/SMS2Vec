import os
import pandas as pd
import base64
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