import sys
import os
import argparse
import time

sde_install = os.environ['SDE_INSTALL']
sys.path.append('%s/lib/python2.7/site-packages/tofino'%(sde_install))
sys.path.append('%s/lib/python2.7/site-packages/p4testutils'%(sde_install))
sys.path.append('%s/lib/python2.7/site-packages'%(sde_install))

import grpc
import time
from pprint import pprint
import bfrt_grpc.client as gc
import bfrt_grpc.bfruntime_pb2 as bfruntime_pb2


def connect():
    # Connect to BfRt Server
    interface = gc.ClientInterface(grpc_addr='localhost:50052', client_id=0, device_id=0)
    target = gc.Target(device_id=0, pipe_id=0xffff)
    # print('Connected to BfRt Server!')

    # Get the information about the running program
    bfrt_info = interface.bfrt_info_get()

    # Establish that you are working with this program
    interface.bind_pipeline_config(bfrt_info.p4_name_get())
    return interface, target, bfrt_info

def load_data(path):
    data = []
    with open(path) as f:
        data = f.read()
    data = data.strip()
    data = data.split('\n')
    print ("!! Length of data to be loaded: ", len(data))
    
    # it must be a power of two
    assert (len(data) & len(data) -1)== 0
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path', required=True,
        help='path'
    )

    args = parser.parse_args()
    path = args.path 

    interface, target, bfrt_info = connect()

    assert os.path.exists(path)
    print ("!! Data to be loaded from", path)

    data = load_data(path)

    REG_STAGE = 4
    data = [data[i:i+len(data)//REG_STAGE] for i in range(0, len(data), len(data)//REG_STAGE)]
    

    keys = []
    vals = []
    for i, d in enumerate(data):
        curr_reg =  'Pipe0SwitchIngress.d' + str(i)
        reg = bfrt_info.table_get(curr_reg)

        key = []
        val = []
        for j, entry in enumerate(d):
            k = reg.make_key([gc.KeyTuple('$REGISTER_INDEX', j)])
            v = reg.make_data([gc.DataTuple(curr_reg + '.f1', int(entry))])
            key.append(k)
            val.append(v)
        keys.append(key)
        vals.append(val)
    
    start = time.time()
    for i in range(len(keys)):
        curr_reg =  'Pipe0SwitchIngress.d' + str(i)
        reg = bfrt_info.table_get(curr_reg)
        print ("!! Populating", curr_reg, "with", len(keys[i]), "entries")
        reg.entry_mod(target, keys[i], vals[i])
    end = time.time()

    print ("!! Zipf distribution loaded successfully! Time taken: ",  end - start)


if __name__ == '__main__':
    main()
