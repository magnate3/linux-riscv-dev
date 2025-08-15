import time
import bfrt_grpc.client as gc

# Connect to the BF Runtime server
for bfrt_client_id in range(10):
    try:
        interface = gc.ClientInterface(
            grpc_addr="localhost:50052",
            client_id=bfrt_client_id,
            device_id=0,
            num_tries=1,
        )
        print("Connected to BF Runtime Server as client", bfrt_client_id)
        break
    except:
        print("Could not connect to BF Runtime Server")
        quit

# Get information about the running program
bfrt_info = interface.bfrt_info_get()
print("The target is running the P4 program: {}".format(bfrt_info.p4_name_get()))

# Establish that you are the "main" client
if bfrt_client_id == 0:
    interface.bind_pipeline_config(bfrt_info.p4_name_get())

# Get the target device, currently setup for all pipes
target = gc.Target(device_id=0, pipe_id=0xffff)

pktgen_app = bfrt_info.table_get("tf1.pktgen.app_cfg")
pktgen_app_key = pktgen_app.make_key([gc.KeyTuple('app_id', 0)])

time.sleep(1) # Sleep for 1 second
pktgen_app_action_data=pktgen_app.make_data([gc.DataTuple('app_enable',bool_val=False)])                                            
pktgen_app.entry_mod(target,[pktgen_app_key],[pktgen_app_action_data])
print("packet gen is stopped")

