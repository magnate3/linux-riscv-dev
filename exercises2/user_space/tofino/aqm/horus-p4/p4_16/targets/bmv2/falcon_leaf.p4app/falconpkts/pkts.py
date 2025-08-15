from random import randint
import scapy.all as scapy

MDC_DEFAULT_PKT_SIZE = 64

field_dict = {'pkt_type': 1,
              'cluster_id': 2,
              'local_cluster_id': 1,
              'src_id': 2,
              'dst_id': 2,
              'q_len': 1,
              'seq_num': 2}

field_cls_dict = {1: scapy.XByteField,
                  2: scapy.XShortField,
                  3: scapy.X3BytesField,
                  4: scapy.XIntField,
                  8: scapy.XLongField}

FALCON_PORT = 1234
ETHER_IPV4_TYPE = 0x0800

PKT_TYPE_NEW_TASK = 0x00
PKT_TYPE_NEW_TASK_RANDOM = 0x01
PKT_TYPE_TASK_DONE = 0x02
PKT_TYPE_TASK_DONE_IDLE = 0x03
PKT_TYPE_QUEUE_REMOVE = 0x04
PKT_TYPE_SCAN_QUEUE_SIGNAL = 0x05
PKT_TYPE_IDLE_SIGNAL = 0x06
PKT_TYPE_QUEUE_SIGNAL = 0x07
PKT_TYPE_PROBE_IDLE_QUEUE = 0x08
PKT_TYPE_PROBE_IDLE_RESPONSE = 0x09
PKT_TYPE_IDLE_REMOVE = 0x10

def get_field(name):
    exists = type(name) == str and name.lower() in field_dict
    if exists:
        field_size = field_dict[name.lower()]
        if field_size in field_cls_dict:
            cls = field_cls_dict[field_size]
            return cls(name.lower(), 0)
        raise ValueError('field_size is incorrect')
    raise ValueError('field is not supported')

class FalconPacket(scapy.Packet):
    name = 'falconPacket'
    fields_desc = [
        get_field('pkt_type'),
        get_field('cluster_id'),
        get_field('local_cluster_id'),
        get_field('src_id'),
        get_field('dst_id'),
        get_field('q_len'),
        get_field('seq_num')
    ]

def get_random_ip_addresses():
    ip_list = [('100.168.1.1', '100.132.44.1'),
               ('72.67.48.53', '72.10.30.55'),
               ]
    rand_idx = randint(0, len(ip_list)-1)
    return ip_list[rand_idx]

def generate_load(length):
    load = ''
    for i in range(length):
        load += chr(randint(0, 255))
    return load

def make_eth_hdr(src_mac=None, dst_mac=None, ip_encap=False, **kwargs):
    hdr = scapy.Ether()
    hdr.type = ETHER_IPV4_TYPE
    if src_mac:
        hdr.src = src_mac
    if dst_mac:
        hdr.dst = dst_mac
    return hdr

def make_falcon_task_pkt(dst_ip, cluster_id, local_cluster_id, src_id, dst_id=0, q_len=0, seq_num=1000, pkt_len=128, **kwargs):
    eth_hdr = make_eth_hdr(**kwargs)
    falcon_hdr = FalconPacket(pkt_type=PKT_TYPE_NEW_TASK, cluster_id=cluster_id, local_cluster_id=local_cluster_id, src_id=src_id, dst_id=dst_id, q_len=q_len, seq_num=seq_num)

    data_len = pkt_len - len(eth_hdr) - len(falcon_hdr)
    if data_len <= 0:
        data_len = 1
    payload = generate_load(data_len)
    pkt = scapy.IP(dst=dst_ip) / scapy.UDP(dport=FALCON_PORT) / falcon_hdr / payload
    return pkt

def make_falcon_task_done_pkt(dst_ip, cluster_id, local_cluster_id, src_id, dst_id=0, is_idle=False, q_len=0, seq_num=1000, pkt_len=128, **kwargs):
    eth_hdr = make_eth_hdr(**kwargs)
    if is_idle:
        falcon_hdr = FalconPacket(pkt_type=PKT_TYPE_TASK_DONE_IDLE, cluster_id=cluster_id, local_cluster_id=local_cluster_id, src_id=src_id, dst_id=dst_id, q_len=q_len, seq_num=seq_num)
    else:
        falcon_hdr = FalconPacket(pkt_type=PKT_TYPE_TASK_DONE, cluster_id=cluster_id, local_cluster_id=local_cluster_id, src_id=src_id, dst_id=dst_id, q_len=q_len, seq_num=seq_num)
    pkt = scapy.IP(dst=dst_ip) / scapy.UDP(dport=FALCON_PORT) / falcon_hdr
    return pkt

# def make_mdc_ping_pkt(agent, **kwargs):
#     """
#     Creates a ping mDC control packet

#     Parameters:
#         agent (1 byte): mDC agent ID (1 byte)
#         ip_encap (bool): True if MDC pkt is embedded in IP header
#         src_mac (str): source MAC address
#         dst_mac (str): destination MAC address
#         src_ip (str): has effect iff ip_encap=True. generated randomly if not provided
#         dst_ip (str): has effect iff ip_encap=True. generated randomly if not provided

#     Returns:
#         scapy.Packet: a new packet

#     """

#     eth_hdr = make_eth_hdr(**kwargs)
#     mdc_hdr = mdc_ping_hdr(agent)

#     ip_encap = kwargs.get('ip_encap', False)
#     if ip_encap:
#         mdc_hdr = encapsulate_ip(mdc_hdr, **kwargs)

#     data_len = MDC_DEFAULT_PKT_SIZE - len(eth_hdr) - len(mdc_hdr)
#     if data_len <= 0:
#         data_len = 1
#     payload = generate_load(data_len)

#     pkt = eth_hdr / mdc_hdr / payload
#     return pkt


# def make_mdc_pong_pkt(agent, **kwargs):
#     """
#     Creates a pong mDC control packet

#     Parameters:
#         agent (1 byte): mDC agent ID (1 byte)
#         ip_encap (bool): True if MDC pkt is embedded in IP header
#         src_mac (str): source MAC address
#         dst_mac (str): destination MAC address
#         src_ip (str): has effect iff ip_encap=True. generated randomly if not provided
#         dst_ip (str): has effect iff ip_encap=True. generated randomly if not provided

#     Returns:
#         scapy.Packet: a new packet

#     """

#     eth_hdr = make_eth_hdr(**kwargs)
#     mdc_hdr = mdc_pong_hdr(agent)

#     ip_encap = kwargs.get('ip_encap', False)
#     if ip_encap:
#         mdc_hdr = encapsulate_ip(mdc_hdr, **kwargs)

#     data_len = MDC_DEFAULT_PKT_SIZE - len(eth_hdr) - len(mdc_hdr)
#     if data_len <= 0:
#         data_len = 1
#     payload = generate_load(data_len)

#     pkt = eth_hdr / mdc_hdr / payload
#     return pkt


# def make_mdc_sync_state_pkt(address, label, **kwargs):
#     """
#     Creates a sync-state mDC control packet

#     Parameters:
#         address (2 bytes): session address (2 bytes)
#         label (int): mDC label (4 bytes)
#         ip_encap (bool): True if MDC pkt is embedded in IP header
#         src_mac (str): source MAC address
#         dst_mac (str): destination MAC address
#         src_ip (str): has effect iff ip_encap=True. generated randomly if not provided
#         dst_ip (str): has effect iff ip_encap=True. generated randomly if not provided

#     Returns:
#         scapy.Packet: a new packet

#     """

#     eth_hdr = make_eth_hdr(**kwargs)
#     mdc_hdr = mdc_sync_state_hdr(address=address, label=label)

#     ip_encap = kwargs.get('ip_encap', False)
#     if ip_encap:
#         mdc_hdr = encapsulate_ip(mdc_hdr, **kwargs)

#     data_len = MDC_DEFAULT_PKT_SIZE - len(eth_hdr) - len(mdc_hdr)
#     if data_len <= 0:
#         data_len = 1
#     payload = generate_load(data_len)

#     pkt = eth_hdr / mdc_hdr / payload
#     return pkt


# def make_mdc_sync_state_done_pkt(address, agent, **kwargs):
#     """
#     Creates a sync-state-done mDC control packet

#     Parameters:
#         address (2 bytes): session address (2 bytes)
#         agent (1 byte): mDC agent ID (1 byte)
#         ip_encap (bool): True if MDC pkt is embedded in IP header
#         src_mac (str): source MAC address
#         dst_mac (str): destination MAC address
#         src_ip (str): has effect iff ip_encap=True. generated randomly if not provided
#         dst_ip (str): has effect iff ip_encap=True. generated randomly if not provided

#     Returns:
#         scapy.Packet: a new packet

#     """

#     eth_hdr = make_eth_hdr(**kwargs)
#     mdc_hdr = mdc_sync_state_done_hdr(address=address, agent=agent)

#     ip_encap = kwargs.get('ip_encap', False)
#     if ip_encap:
#         mdc_hdr = encapsulate_ip(mdc_hdr, **kwargs)

#     data_len = MDC_DEFAULT_PKT_SIZE - len(eth_hdr) - len(mdc_hdr)
#     if data_len <= 0:
#         data_len = 1
#     payload = generate_load(data_len)

#     pkt = eth_hdr / mdc_hdr / payload
#     return pkt


# def make_mdc_set_active_agent_pkt(agent, **kwargs):
#     """
#     Creates a set-active-agent mDC control packet

#     Parameters:
#         agent (1 byte): mDC agent ID (1 byte)
#         ip_encap (bool): True if MDC pkt is embedded in IP header
#         src_mac (str): source MAC address
#         dst_mac (str): destination MAC address
#         src_ip (str): has effect iff ip_encap=True. generated randomly if not provided
#         dst_ip (str): has effect iff ip_encap=True. generated randomly if not provided

#     Returns:
#         scapy.Packet: a new packet

#     """

#     eth_hdr = make_eth_hdr(**kwargs)
#     mdc_hdr = mdc_set_active_agent_hdr(agent=agent)

#     ip_encap = kwargs.get('ip_encap', False)
#     if ip_encap:
#         mdc_hdr = encapsulate_ip(mdc_hdr, **kwargs)

#     data_len = MDC_DEFAULT_PKT_SIZE - len(eth_hdr) - len(mdc_hdr)
#     if data_len <= 0:
#         data_len = 1
#     payload = generate_load(data_len)

#     pkt = eth_hdr / mdc_hdr / payload
#     return pkt
