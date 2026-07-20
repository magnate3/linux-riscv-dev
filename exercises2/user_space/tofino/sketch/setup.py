import os
os.environ['SDE'] = "/".join(os.environ['PATH'].split(":")[0].split("/"))
os.environ['SDE_INSTALL'] = "/".join([os.environ['SDE'], 'install'])
print("%env SDE         {}".format(os.environ['SDE']))
print("%env SDE_INSTALL {}".format(os.environ['SDE_INSTALL']))

p4 = bfrt.simple_l2.pipe
p4_learn =  bfrt.simple_l2.learn.pipe.IngressDeparser

def my_learning_cb(dev_id, pipe_id, direction, parser_id, session, msg):
    global p4
    
    smac = p4.Ingress.smac
    dmac = p4.Ingress.dmac

    for digest in msg:
        vid  = digest["vid"]
        port = digest["ingress_port"]
        mac_move = digest["mac_move"]
        
        mac      = digest["src_mac"]
        old_port = port ^ mac_move

        print("VID=%d MAC=0x%012X Port=%d" % (vid, mac, port), end="")
        if mac_move != 0:
            print("(Move from port=%d)" % old_port)
        else:
            print("(New)")
            
        smac.entry_with_smac_hit(
            vid=vid, src_addr=mac, port=port, is_static=False,
            entry_ttl=10000).push()
        dmac.entry_with_dmac_unicast(
            vid=vid, dst_addr=mac, port=port).push()
    return 0

try:
    p4_learn.l2_digest.callback_deregister()
except:
    pass
finally:
    print("Deregistering old learning callback (if any)")
          
p4_learn.l2_digest.callback_register(my_learning_cb)
print("Learning callback registered")

def my_aging_cb(dev_id, pipe_id, direction, parser_id, entry):
    global p4

    smac = p4.Ingress.smac
    dmac = p4.Ingress.dmac
    
    mac = entry.key[b'hdr.ethernet.src_addr']
    vid = entry.key[b'meta.vid']

    print("Aging out: VID=%d, MAC=0x%012X"%(vid, mac))
    
    entry.remove() # from smac
    try:
        dmac.delete(vid=vid, dst_addr=mac)
    except:
        print("WARNING: Could not find the matching DMAC entry")

p4.Ingress.smac.idle_table_set_notify(enable=False)
print("Deregistering old aging callback (if any)")

p4.Ingress.smac.idle_table_set_notify(enable=True, callback=my_aging_cb,
                                      interval=10000,
                                      min_ttl=10000, max_ttl=60000)
print("Aging callback registered")
