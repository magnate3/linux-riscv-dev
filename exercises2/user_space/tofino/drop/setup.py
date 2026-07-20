import logging
import os
import sys
import time

# Shortcuts for pipes
switcharoo_cuckoo_pipe = bfrt.switcharoo.cuckoo
switcharoo_bloom_pipe = bfrt.switcharoo.bloom

# Port defines
OUTPUT_PORT_1 = 188
OUTPUT_PORT_2 = 180
OUTPUT_PORT_3 = 172
OUTPUT_PORT_4 = 164

RECIRCULATE_PORT_INSERT_IP1_TO_CUCKOO = 128
RECIRCULATE_PORT_INSERT_IP2_TO_CUCKOO = 132
RECIRCULATE_PORT_INSERT_IP3_TO_CUCKOO = 136
RECIRCULATE_PORT_INSERT_IP4_TO_CUCKOO = 140
RECIRCULATE_PORT_LOOKUP_IP1_TO_CUCKOO = 144
RECIRCULATE_PORT_LOOKUP_IP2_TO_CUCKOO = 148
RECIRCULATE_PORT_LOOKUP_IP3_TO_CUCKOO = 152
RECIRCULATE_PORT_LOOKUP_IP4_TO_CUCKOO = 156
RECIRCULATE_PORT_SWAP_TO_CUCKOO = 160

RECIRCULATE_PORT_WAIT_IN_BLOOM = 0
RECIRCULATE_PORT_INSERT_IP1_TO_BLOOM = 4
RECIRCULATE_PORT_INSERT_IP2_TO_BLOOM = 8
RECIRCULATE_PORT_INSERT_IP3_TO_BLOOM = 12
RECIRCULATE_PORT_INSERT_IP4_TO_BLOOM = 16
RECIRCULATE_PORT_WAIT_IP1_TO_BLOOM = 20
RECIRCULATE_PORT_WAIT_IP2_TO_BLOOM = 24
RECIRCULATE_PORT_WAIT_IP3_TO_BLOOM = 28
RECIRCULATE_PORT_WAIT_IP4_TO_BLOOM = 32
RECIRCULATE_PORT_NOP_IP1_TO_BLOOM = 36
RECIRCULATE_PORT_NOP_IP2_TO_BLOOM = 40
RECIRCULATE_PORT_NOP_IP3_TO_BLOOM = 44
RECIRCULATE_PORT_NOP_IP4_TO_BLOOM = 48
RECIRCULATE_PORT_SWAPPED_TO_BLOOM = 52


TABLE_SIZE = len(bfrt.switcharoo.cuckoo.CuckooIngress.table_1_4.get(regex=True, print_ents=False))


#################################
########### PORT SETUP ##########
#################################
# In this section, we setup the ports used by SWITCHAROO.
def setup_ports():
    global OUTPUT_PORT_1, OUTPUT_PORT_2, OUTPUT_PORT_3, OUTPUT_PORT_4, \
            RECIRCULATE_PORT_INSERT_IP1_TO_CUCKOO, RECIRCULATE_PORT_INSERT_IP2_TO_CUCKOO, \
            RECIRCULATE_PORT_INSERT_IP3_TO_CUCKOO, RECIRCULATE_PORT_INSERT_IP4_TO_CUCKOO, \
            RECIRCULATE_PORT_LOOKUP_IP1_TO_CUCKOO, RECIRCULATE_PORT_LOOKUP_IP2_TO_CUCKOO, \
            RECIRCULATE_PORT_LOOKUP_IP3_TO_CUCKOO, RECIRCULATE_PORT_LOOKUP_IP4_TO_CUCKOO, \
            RECIRCULATE_PORT_SWAP_TO_CUCKOO, RECIRCULATE_PORT_WAIT_IN_BLOOM, \
            RECIRCULATE_PORT_INSERT_IP1_TO_BLOOM, RECIRCULATE_PORT_INSERT_IP2_TO_BLOOM, \
            RECIRCULATE_PORT_INSERT_IP3_TO_BLOOM, RECIRCULATE_PORT_INSERT_IP4_TO_BLOOM, \
            RECIRCULATE_PORT_WAIT_IP1_TO_BLOOM, RECIRCULATE_PORT_WAIT_IP2_TO_BLOOM, \
            RECIRCULATE_PORT_WAIT_IP3_TO_BLOOM, RECIRCULATE_PORT_WAIT_IP4_TO_BLOOM, \
            RECIRCULATE_PORT_NOP_IP1_TO_BLOOM, RECIRCULATE_PORT_NOP_IP2_TO_BLOOM, \
            RECIRCULATE_PORT_NOP_IP3_TO_BLOOM, RECIRCULATE_PORT_NOP_IP4_TO_BLOOM, \
            RECIRCULATE_PORT_SWAPPED_TO_BLOOM
    output_ports = [OUTPUT_PORT_1, OUTPUT_PORT_2, OUTPUT_PORT_3, OUTPUT_PORT_4]
    loopback_ports = [RECIRCULATE_PORT_INSERT_IP1_TO_CUCKOO, RECIRCULATE_PORT_INSERT_IP2_TO_CUCKOO, 
                      RECIRCULATE_PORT_INSERT_IP3_TO_CUCKOO, RECIRCULATE_PORT_INSERT_IP4_TO_CUCKOO,
                      RECIRCULATE_PORT_LOOKUP_IP1_TO_CUCKOO, RECIRCULATE_PORT_LOOKUP_IP2_TO_CUCKOO, 
                      RECIRCULATE_PORT_LOOKUP_IP3_TO_CUCKOO, RECIRCULATE_PORT_LOOKUP_IP4_TO_CUCKOO,
                      RECIRCULATE_PORT_SWAP_TO_CUCKOO, 
                      RECIRCULATE_PORT_WAIT_IN_BLOOM, 
                      RECIRCULATE_PORT_INSERT_IP1_TO_BLOOM, RECIRCULATE_PORT_INSERT_IP2_TO_BLOOM, 
                      RECIRCULATE_PORT_INSERT_IP3_TO_BLOOM, RECIRCULATE_PORT_INSERT_IP4_TO_BLOOM,
                      RECIRCULATE_PORT_WAIT_IP1_TO_BLOOM, RECIRCULATE_PORT_WAIT_IP2_TO_BLOOM, 
                      RECIRCULATE_PORT_WAIT_IP3_TO_BLOOM, RECIRCULATE_PORT_WAIT_IP4_TO_BLOOM,
                      RECIRCULATE_PORT_NOP_IP1_TO_BLOOM, RECIRCULATE_PORT_NOP_IP2_TO_BLOOM,
                      RECIRCULATE_PORT_NOP_IP3_TO_BLOOM, RECIRCULATE_PORT_NOP_IP4_TO_BLOOM,
                      RECIRCULATE_PORT_SWAPPED_TO_BLOOM]
    for p in output_ports:
        print("Setting Output Port: %d" % p)
        bfrt.port.port.add(DEV_PORT=p, SPEED='BF_SPEED_100G', FEC='BF_FEC_TYP_REED_SOLOMON', PORT_ENABLE=True)

    for p in loopback_ports:
        print("Setting Loopback Port: %d" % p)
        bfrt.port.port.add(DEV_PORT=p, SPEED='BF_SPEED_100G', FEC='BF_FEC_TYP_REED_SOLOMON', PORT_ENABLE=True,
                           LOOPBACK_MODE='BF_LPBK_MAC_NEAR')


#################################
##### MIRROR SESSIONS TABLE #####
#################################
# In this section, we setup the mirror sessions of SWITCHAROO.
# There is only one session, that is used to truncate/send swap operations to the Cuckoo Pipe.
PKT_MIN_LENGTH = 100
SWAP_MIRROR_SESSION = 100


def setup_mirror_session_table():
    global bfrt, SWAP_MIRROR_SESSION, RECIRCULATE_PORT_SWAP_TO_CUCKOO, PKT_MIN_LENGTH

    mirror_cfg = bfrt.mirror.cfg

    mirror_cfg.entry_with_normal(
        sid=SWAP_MIRROR_SESSION,
        direction="BOTH",
        session_enable=True,
        ucast_egress_port=RECIRCULATE_PORT_SWAP_TO_CUCKOO,
        ucast_egress_port_valid=1,
        max_pkt_len=PKT_MIN_LENGTH,
        packet_color="GREEN"
    ).push()


#################################
##### TRAFFIC MANAGER POOLS #####
#################################
# In this section, we enlarge the TM buffer pools to the maximum available.
def setup_tm_pools():
    global bfrt

    tm = bfrt.tf1.tm
    tm.pool.app.mod_with_color_drop_enable(pool='EG_APP_POOL_0', green_limit_cells=20000000 // 80,
                                           yellow_limit_cells=20000000 // 80, red_limit_cells=20000000 // 80)
    tm.pool.app.mod_with_color_drop_enable(pool='IG_APP_POOL_0', green_limit_cells=20000000 // 80,
                                           yellow_limit_cells=20000000 // 80, red_limit_cells=20000000 // 80)
    tm.pool.app.mod_with_color_drop_enable(pool='EG_APP_POOL_1', green_limit_cells=20000000 // 80,
                                           yellow_limit_cells=20000000 // 80, red_limit_cells=20000000 // 80)
    tm.pool.app.mod_with_color_drop_enable(pool='IG_APP_POOL_1', green_limit_cells=20000000 // 80,
                                           yellow_limit_cells=20000000 // 80, red_limit_cells=20000000 // 80)


#########################
##### FORWARD TABLE #####
#########################
# This function setups the entries in the forward table.
# You can add/edit/remove entries to choose where output the packets based on the IPv4 identification.
def setup_forward_table():
    global switcharoo_bloom_pipe, OUTPUT_PORT_1, OUTPUT_PORT_2, OUTPUT_PORT_3, OUTPUT_PORT_4

    forward_table = switcharoo_bloom_pipe.BloomIngress.forward
    forward_table.clear()

    forward_table.add_with_send(identification=0xFFFF, port_number=OUTPUT_PORT_1)
    forward_table.add_with_send(identification=0xEEEE, port_number=OUTPUT_PORT_2)
    forward_table.add_with_send(identification=0xDDDD, port_number=OUTPUT_PORT_3)
    forward_table.add_with_send(identification=0xCCCC, port_number=OUTPUT_PORT_4)


########################
######### STATS ########
########################
# This section creates a timer that calls a callback to dump and print stats.
# In particular, it dumps counters that store all the different packets entering the Cuckoo/Bloom Pipe.
# Timer and previous counters for stats
start_ts = time.time()
previous_insertions = 0
previous_recirc_bps = 0


def get_stats():
    import logging

    global time, start_ts, previous_insertions, previous_recirc_bps, switcharoo_cuckoo_pipe, switcharoo_bloom_pipe, \
        RECIRCULATE_PORT_INSERT_IP1_TO_CUCKOO, RECIRCULATE_PORT_INSERT_IP2_TO_CUCKOO, \
        RECIRCULATE_PORT_INSERT_IP3_TO_CUCKOO, RECIRCULATE_PORT_INSERT_IP4_TO_CUCKOO, \
        RECIRCULATE_PORT_LOOKUP_IP1_TO_CUCKOO, RECIRCULATE_PORT_LOOKUP_IP2_TO_CUCKOO, \
        RECIRCULATE_PORT_LOOKUP_IP3_TO_CUCKOO, RECIRCULATE_PORT_LOOKUP_IP4_TO_CUCKOO, \
        RECIRCULATE_PORT_SWAP_TO_CUCKOO, RECIRCULATE_PORT_WAIT_IN_BLOOM, \
        OUTPUT_PORT_1, OUTPUT_PORT_2, OUTPUT_PORT_3, OUTPUT_PORT_4

    insertions_counter = switcharoo_cuckoo_pipe.insertions_counter.get(
        REGISTER_INDEX=0, from_hw=1, print_ents=False
    )

    wait_counter_on_bloom = switcharoo_bloom_pipe.wait_counter_on_bloom.get(
        REGISTER_INDEX=0, from_hw=1, print_ents=False
    )

    from_nop_to_wait = switcharoo_bloom_pipe.from_nop_to_wait.get(
        REGISTER_INDEX=0, from_hw=1, print_ents=False
    )

    wait_max_loops_on_bloom = switcharoo_bloom_pipe.wait_max_loops_on_bloom.get(
        REGISTER_INDEX=0, from_hw=1, print_ents=False
    )
    insert_max_loops_on_bloom = switcharoo_bloom_pipe.insert_max_loops_on_bloom.get(
        REGISTER_INDEX=0, from_hw=1, print_ents=False
    )

    insert_counter_on_bloom = switcharoo_bloom_pipe.insert_counter_on_bloom.get(
        REGISTER_INDEX=0, from_hw=1, print_ents=False
    )
    swap_counter_on_bloom = switcharoo_bloom_pipe.swap_counter_on_bloom.get(
        REGISTER_INDEX=0, from_hw=1, print_ents=False
    )
    swapped_counter_on_bloom = switcharoo_bloom_pipe.swapped_counter_on_bloom.get(
        REGISTER_INDEX=0, from_hw=1, print_ents=False
    )
    nop_counter_on_bloom = switcharoo_bloom_pipe.nop_counter_on_bloom.get(
        REGISTER_INDEX=0, from_hw=1, print_ents=False
    )

    from_insert_to_lookup_swap = switcharoo_bloom_pipe.from_insert_to_lookup_swap.get(
        REGISTER_INDEX=0, from_hw=1, print_ents=False
    )

    from_insert_to_lookup_bloom = switcharoo_bloom_pipe.from_insert_to_lookup_bloom.get(
        REGISTER_INDEX=0, from_hw=1, print_ents=False
    )

    expired_entry_table_1 = switcharoo_cuckoo_pipe.expired_entry_table_1.get(
        REGISTER_INDEX=0, from_hw=1, print_ents=False
    )

    expired_entry_table_2 = switcharoo_cuckoo_pipe.expired_entry_table_2.get(
        REGISTER_INDEX=0, from_hw=1, print_ents=False
    )

    table_2_match_counter = switcharoo_cuckoo_pipe.table_2_match_counter.get(
        REGISTER_INDEX=0, from_hw=1, print_ents=False
    )

    table_1_match_counter = switcharoo_cuckoo_pipe.table_1_match_counter.get(
        REGISTER_INDEX=0, from_hw=1, print_ents=False
    )

    swap_creation = switcharoo_cuckoo_pipe.swap_creation.get(
        REGISTER_INDEX=0, from_hw=1, print_ents=False
    )

    port_stats = bfrt.port.port_stat.get(regex=True, print_ents=False)
    recirc_ports_stats = filter(
        lambda x: x.key[b'$DEV_PORT'] in [
            RECIRCULATE_PORT_INSERT_IP1_TO_CUCKOO, RECIRCULATE_PORT_INSERT_IP2_TO_CUCKOO, 
            RECIRCULATE_PORT_INSERT_IP3_TO_CUCKOO, RECIRCULATE_PORT_INSERT_IP4_TO_CUCKOO,
            RECIRCULATE_PORT_LOOKUP_IP1_TO_CUCKOO, RECIRCULATE_PORT_LOOKUP_IP2_TO_CUCKOO, 
            RECIRCULATE_PORT_LOOKUP_IP3_TO_CUCKOO, RECIRCULATE_PORT_LOOKUP_IP4_TO_CUCKOO,
            RECIRCULATE_PORT_SWAP_TO_CUCKOO,
            RECIRCULATE_PORT_WAIT_IN_BLOOM
        ],
        port_stats
    )

    output_port_stats = list(filter(
        lambda x: x.key[b'$DEV_PORT'] in [OUTPUT_PORT_1, OUTPUT_PORT_2, OUTPUT_PORT_3, OUTPUT_PORT_4],
        port_stats
    ))

    ig_counters_pipe0 = bfrt.tf1.tm.counter.ig_port.get(regex=True, print_ents=False, pipe=0, from_hw=1)
    ig_counters_pipe1 = bfrt.tf1.tm.counter.ig_port.get(regex=True, print_ents=False, pipe=1, from_hw=1)
    eg_counters_pipe0 = bfrt.tf1.tm.counter.ig_port.get(regex=True, print_ents=False, pipe=0, from_hw=1)
    eg_counters_pipe1 = bfrt.tf1.tm.counter.ig_port.get(regex=True, print_ents=False, pipe=1, from_hw=1)

    ts = time.time() - start_ts
    current_insertions = insertions_counter.data[b'insertions_counter.f1'][0]
    delta_insertions = current_insertions - previous_insertions

    current_recirc_bps = sum(map(lambda x: x.data[b'$OctetsTransmittedTotal'], recirc_ports_stats))
    delta_recirc_bps = current_recirc_bps - previous_recirc_bps

    input_pkts = sum(map(lambda x: x.data[b'$FramesReceivedAll'], output_port_stats))
    output_pkts = sum(map(lambda x: x.data[b'$FramesTransmittedAll'], output_port_stats))

    ig_dropped_pkts_pipe0 = sum(map(lambda x: x.data[b"drop_count_packets"], ig_counters_pipe0))
    ig_dropped_pkts_pipe1 = sum(map(lambda x: x.data[b"drop_count_packets"], ig_counters_pipe1))
    eg_dropped_pkts_pipe0 = sum(map(lambda x: x.data[b"drop_count_packets"], eg_counters_pipe0))
    eg_dropped_pkts_pipe1 = sum(map(lambda x: x.data[b"drop_count_packets"], eg_counters_pipe1))

    logging.info("SWITCHAROO-%f-RESULT-RECIRC_BPS %f bps" % (ts, delta_recirc_bps * 8))
    logging.info("SWITCHAROO-%f-RESULT-RECIRC_BYTES %f bytes" % (ts, current_recirc_bps))
    logging.info("SWITCHAROO-%f-RESULT-IPS %f ips" % (ts, delta_insertions))
    logging.info("SWITCHAROO-%f-RESULT-INSERTIONS %f ins" % (ts, current_insertions))
    logging.info("SWITCHAROO-%f-RESULT-SWAPS %f pkts" % (ts, swap_counter_on_bloom.data[b'swap_counter_on_bloom.f1'][0]))
    logging.info("SWITCHAROO-%f-RESULT-SWAP_CREATED %f pkts" % (ts, swap_creation.data[b'swap_creation.f1'][0]))
    logging.info("SWITCHAROO-%f-RESULT-WAIT %f pkts" % (ts, wait_counter_on_bloom.data[b'wait_counter_on_bloom.f1'][0]))
    logging.info(
        "SWITCHAROO-%f-RESULT-INSERT %f pkts" % (ts, insert_counter_on_bloom.data[b'insert_counter_on_bloom.f1'][0]))
    logging.info(
        "SWITCHAROO-%f-RESULT-SWAPPED %f pkts" % (ts, swapped_counter_on_bloom.data[b'swapped_counter_on_bloom.f1'][0]))
    logging.info("SWITCHAROO-%f-RESULT-NOP %f pkts" % (ts, nop_counter_on_bloom.data[b'nop_counter_on_bloom.f1'][0]))
    logging.info("SWITCHAROO-%f-RESULT-WAIT_MAX_LOOPS %f pkts" % (
        ts, wait_max_loops_on_bloom.data[b'wait_max_loops_on_bloom.f1'][0]))
    logging.info("SWITCHAROO-%f-RESULT-INSERT_MAX_LOOPS %f pkts" % (
        ts, insert_max_loops_on_bloom.data[b'insert_max_loops_on_bloom.f1'][0]))
    logging.info("SWITCHAROO-%f-RESULT-INSERT2LOOKUP_SWAP %f pkts" % (
        ts, from_insert_to_lookup_swap.data[b'from_insert_to_lookup_swap.f1'][0]))
    logging.info("SWITCHAROO-%f-RESULT-INSERT2LOOKUP_BLOOM %f pkts" % (
        ts, from_insert_to_lookup_bloom.data[b'from_insert_to_lookup_bloom.f1'][0]))
    logging.info("SWITCHAROO-%f-RESULT-NOP2WAIT %f pkts" % (ts, from_nop_to_wait.data[b'from_nop_to_wait.f1'][0]))
    logging.info(
        "SWITCHAROO-%f-RESULT-EXPIRED_1 %f pkts" % (ts, expired_entry_table_1.data[b'expired_entry_table_1.f1'][0]))
    logging.info(
        "SWITCHAROO-%f-RESULT-EXPIRED_2 %f pkts" % (ts, expired_entry_table_2.data[b'expired_entry_table_2.f1'][0]))
    logging.info(
        "SWITCHAROO-%f-RESULT-TABLE_1_MATCH %f pkts" % (ts, table_1_match_counter.data[b'table_1_match_counter.f1'][0]))
    logging.info(
        "SWITCHAROO-%f-RESULT-TABLE_2_MATCH %f pkts" % (ts, table_2_match_counter.data[b'table_2_match_counter.f1'][0]))
    logging.info("SWITCHAROO-%f-RESULT-INPUT_PKTS %f pkts" % (ts, input_pkts))
    logging.info("SWITCHAROO-%f-RESULT-OUTPUT_PKTS %f pkts" % (ts, output_pkts))
    logging.info("SWITCHAROO-%f-RESULT-IG_DROP %f pkts" % (ts, ig_dropped_pkts_pipe0 + ig_dropped_pkts_pipe1))
    logging.info("SWITCHAROO-%f-RESULT-EG_DROP %f pkts\n" % (ts, eg_dropped_pkts_pipe0 + eg_dropped_pkts_pipe1))

    previous_insertions = current_insertions
    previous_recirc_bps = current_recirc_bps


def stats_timer():
    import threading

    global port_stats_timer, get_stats
    get_stats()
    threading.Timer(1, stats_timer).start()


lab_path = os.path.join(os.environ['HOME'], "labs/switcharoo")

# Setup Logging
logging.basicConfig(
    format='%(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

(year, month, day, hour, minutes, _, _, _, _) = time.localtime(time.time())
log_path = os.path.join(lab_path, "logs")
log_timestamped_name = '32p-%d-log-%d-%s-%s_%s-%s' % (
    TABLE_SIZE, year, str(month).zfill(2), str(day).zfill(2), str(hour).zfill(2), str(minutes).zfill(2))
os.makedirs(log_path, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(log_path, "%s.log" % log_timestamped_name))
file_handler.setFormatter(logging.Formatter('%(message)s'))
logging.root.addHandler(file_handler)

setup_ports()

setup_mirror_session_table()
setup_tm_pools()

setup_forward_table()

stats_timer()  # Comment out to disable stats

bfrt.complete_operations()
