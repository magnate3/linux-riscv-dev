#include <stdio.h>
#include <pcap.h>
#include <arpa/inet.h>
#include <linux/if_ether.h>
#include <netinet/ip6.h>

#define SNAP_LEN	1500

int main(int argc, char *argv[]) {

	char *eth_if = argv[1];
	char *errbuf;
	pcap_t *p = pcap_open_live(eth_if, SNAP_LEN, 1, 0, errbuf);
	if (p == NULL) {
		printf("Yikes, something went wrong:\n%s\n", errbuf);
		return 1; // exit
	}

	unsigned char *packet;
	struct pcap_pkthdr header;

	while (packet = (unsigned char *)pcap_next(p, &header)){
		// assuming it's ethernet, we check whether it carries IPv6
		struct ethhdr *eth_h = (struct ethhdr*) packet;	
		if (ntohs(eth_h->h_proto) == ETH_P_IPV6) {
			// now, fill the ip6_hdr struct
			struct ip6_hdr *ipv6_h = (struct ip6_hdr*)(packet + sizeof(struct ethhdr));
			// and extract the flow label. We're only interested in the last 20 bits, so 0xfffff
			uint32_t ipv6_fl = ntohl(ipv6_h->ip6_flow) & 0xfffff;
			printf("Flow label for this packet: 0x%x\n", ipv6_fl);
		}
	}

	return(0);
}