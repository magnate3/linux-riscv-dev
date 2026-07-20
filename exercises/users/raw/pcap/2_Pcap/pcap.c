/**
 * @file pcap.c
 * @brief パケットをキャプチャする関数の実装ファイル
 */

#include <arpa/inet.h>
#include <linux/if.h>
#include <net/ethernet.h>
#include <netinet/if_ether.h>
#include <netinet/ip.h>
#include <netpacket/packet.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

#include "analyze.h"

/**
 * @brief Rawソケットの準備
 * @param device : ネットワークインターフェース名
 * @param promiscFlag : プロミスキャスとモードにするかどうかのフラグ
 * @param ipOnly : IPパケットのみを対象とするかどうかのフラグ
 * @return socketのディスクリプタ
 *         ERRORの場合 -1
 */
int InitRawSocket(char *device, int promiscFlag, int ipOnly)
{
    struct ifreq ifreq;
    struct sockaddr_ll sa;
    int soc = 0;

    // socket(PF_PACKET, SOCK_RAW, プロトコル) でデータリンク層を扱えるディスクリプタを得る
    if (ipOnly) {
        // IPパケットのみ
        if ((soc = socket(PF_PACKET, SOCK_RAW, htons(ETH_P_IP))) < 0) {
            perror("socket");
            return -1;
        }
    }
    else {
        // 全パケット
        if ((soc = socket(PF_PACKET, SOCK_RAW, htons(ETH_P_ALL))) < 0) {
            perror("socket");
            return -1;
        }
    }

    memset(&ifreq, 0, sizeof(struct ifreq));
    strncpy(ifreq.ifr_name, device, sizeof(ifreq.ifr_name) - 1);

    // ioctl でネットワークインターフェース名に対応したインターフェースインデックスを得る
    // SIOCGIFINDEX = system I/O controller get if index
    if (ioctl(soc, SIOCGIFINDEX, &ifreq) < 0) {
        perror("ioctl");
        close(soc);
        return -1;
    }

    memset(&sa, 0, sizeof(struct sockaddr_ll));
    sa.sll_family = PF_PACKET;
    if (ipOnly) {
        sa.sll_protocol = htons(ETH_P_IP);
    }
    else {
        sa.sll_protocol = htons(ETH_P_ALL);
    }
    sa.sll_ifindex = ifreq.ifr_ifindex;

    // bind() で socketのディスクリプタに sa情報をセットする
    if (bind(soc, (struct sockaddr *)&sa, sizeof(sa)) < 0) {
        perror("bind");
        close(soc);
        return -1;
    }

    if (promiscFlag) {
        // デバイスのフラグを取得
        if (ioctl(soc, SIOCGIFFLAGS, &ifreq) < 0) {
            perror("ioctl");
            close(soc);
            return -1;
        }
        // IFF_PROMISCを付加する
        ifreq.ifr_flags = ifreq.ifr_flags | IFF_PROMISC;
        if (ioctl(soc, SIOCSIFFLAGS, &ifreq) < 0) {
            perror("ioctl");
            close(soc);
            return -1;
        }
    }

    return soc;
}

int main(int argc, char *argv[], char *envp[])
{
    int soc = 0;
    int size = 0;
    u_char buf[65535] = {'\0'};

    if (argc <= 1) {
        fprintf(stderr, "usage : pcap device-name\n");
        return 1;
    }

    if ((soc = InitRawSocket(argv[1], 0, 0)) == -1) {
        fprintf(stderr, "InitRawSocket:error : %s\n", argv[1]);
        return -1;
    }

    while (1) {
        if ((size = read(soc, buf, sizeof(buf))) <= 0) {
            perror("read");
        }
        else {
            AnalyzePacket(buf, size);
        }
    }

    close(soc);

    return 0;
}
