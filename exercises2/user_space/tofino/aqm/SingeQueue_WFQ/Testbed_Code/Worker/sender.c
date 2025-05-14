#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <linux/if_ether.h>
#include <linux/if_packet.h>
#include <stdio.h>
#include <pthread.h>
#include <net/if.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include "sender.h"

void construct_ether(struct eth_header *eth_h)
{
    memcpy(eth_h->dmac, DST_MAC, 6);
    memcpy(eth_h->smac, SRC_MAC, 6);
    eth_h->ethType = htons(0x0800);
}

void construct_ip(struct ipv4_hdr *ip_header)
{
    ip_header->version_ihl = (4 << 4) | (5);
    ip_header->type_of_service = 0;
    ip_header->total_length = htons(sizeof(struct ipv4_hdr) + sizeof(struct udp_hdr) + sizeof(struct worker_hdr) + 0); // 0 表示没有应用层数据
    ip_header->packet_id = 0;
    ip_header->time_to_live = 255;
    ip_header->next_proto_id = 17; // 指向udp
    ip_header->src_addr = inet_addr(src_ip);
    ip_header->dst_addr = inet_addr(dest_ip);
    ip_header->fragment_offset = 0;
    ip_header->hdr_checksum = 0; //可能要设置？
}

void construct_worker(struct worker_hdr *worker_h)
{
    worker_h->round_number = htonl(0);
    // worker_h->round_number = 0;
	worker_h -> egress_port = 0;
	worker_h -> qid = 0;
	worker_h -> round_index = 0;
}
void construct_udp(struct udp_hdr *u_h)
{
    u_h->dst_port = htons(7001);
    u_h->src_port = htons(7001);
    u_h->dgram_len = htons(sizeof(struct worker_hdr)+ sizeof(struct udp_hdr));
    u_h->dgram_cksum = 0;
}

size_t construct_normal_packet(void *buf)
{
    struct ipv4_hdr *ip_header = (struct ipv4_hdr*)buf;
    struct udp_hdr *u_h = (struct udp_hdr*)(ip_header + 1);
    struct worker_hdr *worker_h = (struct worker_hdr*)(u_h + 1);
    construct_ip(ip_header);
    construct_udp(u_h);
    construct_worker(worker_h);
    return sizeof(struct ipv4_hdr) + sizeof(struct udp_hdr) + sizeof(struct worker_hdr) + 0; //用户层数据为0
}

void* main_loop(void *arg)
{
    int fd = socket(PF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
    if(fd < 0)
    {
        printf("pthread %ld create socket error\n", pthread_self());
        return NULL;
    }
    char buf[1024];
    memset(buf, 0, sizeof(buf));
    construct_ether((struct eth_header*)buf);
    struct sockaddr_ll sll = {0};
    struct ifreq ethreq;
    strncpy(ethreq.ifr_name, "enp3s0f1", IFNAMSIZ);         //指定网卡名称    
    if(-1 == ioctl(fd, SIOCGIFINDEX, &ethreq))    //获取网络接口    
    {    
        perror("ioctl");    
        close(fd);    
        exit(-1);    
    }
    sll.sll_ifindex = ethreq.ifr_ifindex;
    int p_num = 0;
    while((p_num++) < 1)
    {
        size_t p_len = construct_normal_packet((struct eth_header*)buf + 1) + sizeof(struct eth_header);
        if(sendto(fd, buf, p_len + 60, 0, (struct sockaddr*)&sll, sizeof(sll)) == -1)
        {
            printf("send error\n");
        }
    }
    printf("finish\n");
}
int main(int argc, char const *argv[])
{
    pthread_t id = 0;
    int i;
    for(i = 0; i < 1; i++)
    {
        pthread_create(&id, NULL, main_loop, NULL);
        pthread_join(id, NULL);
    }
    return 0;
}
