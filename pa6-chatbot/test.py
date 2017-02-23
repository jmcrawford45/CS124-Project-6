import Crypto
import Crypto.Random
from Crypto.Cipher import AES

def pad_data(data):
    if len(data) % 16 == 0:
        return data
    databytes = bytearray(data)
    padding_required = 15 - (len(databytes) % 16)
    databytes.extend(b'\x80')
    databytes.extend(b'\x00' * padding_required)
    return bytes(databytes)

def unpad_data(data):
    if not data:
        return data

    data = data.rstrip(b'\x00')
    if data[-1] == 128: # b'\x80'[0]:
        return data[:-1]
    else:
        return data


def generate_aes_key():
    rnd = Crypto.Random.OSRNG.posix.new().read(AES.block_size)
    return rnd

def encrypt(key, iv, data):
    aes = AES.new(key, AES.MODE_CBC, iv)
    data = pad_data(data)
    return aes.encrypt(data)

def decrypt(key, iv, data):
    aes = AES.new(key, AES.MODE_CBC, iv)
    data = aes.decrypt(data)
    return unpad_data(data)

def test_crypto ():
    key = "passwordpassword"
    iv = "54e2654a8b52038c659360ecd8638532".decode("hex")# get some random value for IV
    msg = "To:bob@gmail.com"
    alt = "To:mel@gmail.com"

    code = encrypt(key, iv, msg)
    print "".join("{:02x}".format(ord(c)) for c in iv)
    print "".join("{:02x}".format(ord(c)) for c in code)
    altIV = b''.join(chr(ord(a) ^ ord(b) ^ ord(c)) for a,b,c in zip(alt, iv, msg))
    
    print "".join("{:02x}".format(ord(c) ^ ord(d)) for c,d in zip(iv[3:],altIV[3:]))
    a = '0f0a0e'.decode('hex')
    print "".join((chr(ord(c) ^ ord(d))) for c,d in zip(a,'mel'))
    decoded = decrypt(key, iv, code)
    altDecoded = decrypt(key, altIV, code)
    print(decoded)
    print(altDecoded)
if __name__ == '__main__':
    test_crypto()