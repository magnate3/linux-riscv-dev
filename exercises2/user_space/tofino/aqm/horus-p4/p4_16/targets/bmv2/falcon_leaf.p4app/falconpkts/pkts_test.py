from pkts import *

if __name__ == '__main__':
    eth_kwargs = {
        'src_mac': '01:02:03:04:05:06',
        'dst_mac': 'AA:BB:CC:DD:EE:FF'
    }

    _dst_ip = '10.0.2.101'


    new_task_packet = make_falcon_task_pkt(dst_ip=_dst_ip, cluster_id=105, local_cluster_id=5, src_id=8, **eth_kwargs)
    print('>> New Task packet (size = %d bytes):' % len(new_task_packet))
    new_task_packet.show()