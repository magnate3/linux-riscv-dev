#import logging as log 
import json
import sys
import binascii
import time
p4 = bfrt.tna_resubmit.pipe
def digest_callback(dev_id, pipe_id, direction, parser_id, session, msg):
        global p4,  Digest
        print('---------------------------------')
        print("Received message from data plane!")
        for dig in msg:
                print(dig)
        try:
                p4.SwitchIngressDeparser.digest.callback_deregister()
        except:
                pass
        finally:
                print("Deregistering old callback function (if any)")
        return 0
          
def bindDigestCallback():
        global digest_callback,  p4
        
        try:
                p4.SwitchIngressDeparser.digest.callback_deregister()
        except:
                pass
        finally:
                print("Deregistering old callback function (if any)")

        #Register as callback for digests (bind to DMA?)
        print("Registering callback...")
        p4.SwitchIngressDeparser.digest.callback_register(digest_callback)

        print("Bound callback to digest")
if __name__ == '__main__':
    bindDigestCallback()
    for i in range(5):
        time.sleep(2)
