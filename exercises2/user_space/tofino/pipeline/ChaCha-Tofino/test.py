import sys
print(sys.version)

from scapy.all import *
import ipaddress
import random
import binascii

CHACHA_CONST = [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574]

def ROL(x, w):
    x &= 0xffffffff
    return ((x << w) | (x >> (32 - w))) & 0xffffffff
    
def QR(state, a, b, c, d):
    state[a] += state[b]
    state[a] &= 0xffffffff
    state[d] ^= state[a]
    state[d] = ROL(state[d], 16)
    
    state[c] += state[d]
    state[c] &= 0xffffffff
    state[b] ^= state[c]
    state[b] = ROL(state[b], 12)
    
    state[a] += state[b]
    state[a] &= 0xffffffff
    state[d] ^= state[a]
    state[d] = ROL(state[d], 8)
    
    state[c] += state[d]
    state[c] &= 0xffffffff
    state[b] ^= state[c]
    state[b] = ROL(state[b], 7)
    
def chacha20_pad(keys, nonces, position, rounds=20):
    state = [0 for i in range(16)]
    state[:4] = CHACHA_CONST
    state[4:12] = keys
    state[12] = position
    state[13] = 0
    state[14:] = nonces
    
    istate = list(state)
    for i in range(1, rounds + 1):
        if i % 2 == 1:
            QR(state, 0, 4, 8, 12)
            QR(state, 1, 5, 9, 13)
            QR(state, 2, 6, 10, 14)
            QR(state, 3, 7, 11, 15)
        else:
            QR(state, 0, 5, 10, 15)
            QR(state, 1, 6, 11, 12)
            QR(state, 2, 7, 8, 13)
            QR(state, 3, 4, 9, 14)
            
    for i in range(len(state)):
        state[i] += istate[i]
        state[i] &= 0xffffffff
    return state



n_block = 6     # Number of blocks encrypted

chacha_pre = binascii.unhexlify('800000')  # encrypt
# chacha_pre = binascii.unhexlify('000000')  # decrypt

chacha_pre += binascii.unhexlify('0000')  # eg_port = 0
chacha_inonce = binascii.unhexlify('0000000000000000')   # nonce = 0
data = binascii.unhexlify('00') * 4 * 16 * n_block

chacha_header = chacha_pre + chacha_inonce + data



pkt = Ether()/Raw(chacha_header)
sniffer = scapy.sendrecv.AsyncSniffer(count=0, iface='veth0')
sniffer.start()
scapy.sendrecv.sendp(pkt, iface='veth2')
time.sleep(10)
pkts = sniffer.stop()

sniffed = False
for pkt in pkts:
    sniffed = True
    print('sniffed_on:', pkt.sniffed_on)
    sniffed_pkt = pkt
    break
if not sniffed:
    raise Exception


rounds = 20
payload = bytes(sniffed_pkt[Raw])
nonces = [int.from_bytes(payload[5 + i * 4:5 + i * 4 + 4], 'big') for i in range(2)]
keys = [0] * 8
#keys = [0x33221100, 0x77665544, 0xBBAA9988, 0xFFEEDDCC, 0xCCDDEEFF, 0x8899AABB, 0x44556677, 0x00112233]

ideal_state = [[] for i in range(n_block)]
for i in range(n_block):
    ideal_state[i] = chacha20_pad(keys, nonces, i, rounds)

actual_state = [[] for i in range(n_block)]
for i in reversed(range(n_block)):
    ciphertext = payload[13 + 64 * i:13 + 64 * (i + 1)]
    actual_state[i] = [int.from_bytes(ciphertext[i * 4:i * 4 + 4], 'big') for i in range(16)]



print('ideal')
for i in range(n_block):
    print(i)
    print(['%x' % i for i in ideal_state[i]])

print('actual')
for i in range(n_block):
    print(i)
    print(['%x' % i for i in actual_state[i]])

print('Correct answer?:', ideal_state == actual_state)
