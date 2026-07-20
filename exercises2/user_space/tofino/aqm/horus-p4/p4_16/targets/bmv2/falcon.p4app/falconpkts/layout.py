import scapy.all as scapy


field_dict = {'pkt_type': 1,
              'cluster_id': 2,
              'local_cluster_id': 1,
              'src_id': 2,
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
        get_field('q_len'),
        get_field('seq_num')
    ]

def make_falcon_hdr(pkt_type, cluster_id, local_cluster_id, src_id, q_len=0, seq_num=1000):
    return FalconPacket(pkt_type=pkt_type, cluster_id=cluster_id, local_cluster_id=local_cluster_id, src_id=src_id, q_len=q_len, seq_num=seq_num)

