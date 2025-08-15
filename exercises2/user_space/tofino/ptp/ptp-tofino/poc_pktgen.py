# To be used with run_pd_rpc.py
# ./run_pd_rpc.py poc_pktgen.py

from scapy.all import *

PPS = 50000000


p = Ether() / IPv6() / UDP(sport=55550,dport=55551) / "This is the payload" / (b'\x00' * 39) # 120 B
pktgen.write_pkt_buffer(0, len(p), str(p))


pktgen.enable(68)

# Create configuration object and set parameters
ac = pktgen.AppCfg_t()
ac.buffer_offset=0
ac.length = len(p)
ac.trigger_type = pktgen.TriggerType_t.TIMER_PERIODIC
ac.timer = 1000000000 / PPS# nanoseconds betweent packets
ac.src_port = 68
pktgen.cfg_app(0, ac) # Apply the configuration to a specific packet generator application
pktgen.app_enable(0) # Enable the application
