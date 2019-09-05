import struct

with open("/users/Result_1.raw", mode='rb') as f: # b is important -> binary
    fileContent = f.read()

print(fileContent)
print(struct.unpack('f', fileContent))
