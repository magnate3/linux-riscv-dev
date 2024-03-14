#include <stdio.h>
#include <string.h>
#include <time.h>

#define FILE_NAME_LEN 64

char g_pcap_filename[FILE_NAME_LEN] = {0};
FILE *g_pcap_file = NULL;

/*
 * Standard libpcap format.
 */
#define TCPDUMP_MAGIC   0xa1b2c3d4
#define PCAP_VERSION_MAJOR 2
#define PCAP_VERSION_MINOR 4

struct pcap_file_header {
    unsigned int magic;
    unsigned short version_major;
    unsigned short version_minor;
    unsigned int thiszone;          /* gmt to local correction */
    unsigned int sigfigs;           /* accuracy of timestamps */
    unsigned int snaplen;           /* max length saved portion of each pkt */
    unsigned int linktype;          /* data link type (LINKTYPE_*) */
};

struct pcap_timeval {
    int tv_sec;     /* seconds */
    int tv_usec;    /* microseconds */
};

struct pcap_sf_pkthdr {
    struct pcap_timeval ts;     /* time stamp */
    unsigned int caplen;        /* length of portion present */
    unsigned int len;           /* length of this packet (off wire) */
};

int pcap_init(const char *file)
{
    size_t size = 0;
    struct pcap_file_header pcap_filehdr;

    memset(&pcap_filehdr, 0, sizeof(pcap_filehdr));
    pcap_filehdr.magic = TCPDUMP_MAGIC;
    pcap_filehdr.version_major = PCAP_VERSION_MAJOR;
    pcap_filehdr.version_minor = PCAP_VERSION_MINOR;
    pcap_filehdr.thiszone = 0;
    pcap_filehdr.sigfigs  = 0;
    pcap_filehdr.snaplen  = 65535;
    pcap_filehdr.linktype = 1;

    if ('\0' == file[0]) {
        return 0;
    }

    g_pcap_file = fopen(file, "wb");
    if (!g_pcap_file) {
        printf("Open pcap file failed\n");
        return -1;
    }

    size = sizeof(pcap_filehdr);
    if (fwrite(&pcap_filehdr, size, 1, g_pcap_file) != 1) {
        printf("Write pcapfile header failed\n");
        fclose(g_pcap_file);
        return -1;
    }

    return 0;
}

int pcap_write(const char *buf, const unsigned int len)
{
    struct pcap_sf_pkthdr pkthdr;
    clock_t time;

    time = clock();
    memset(&pkthdr, 0, sizeof(pkthdr));
    pkthdr.ts.tv_sec = time / CLOCKS_PER_SEC;
    pkthdr.ts.tv_usec = time % CLOCKS_PER_SEC;
    pkthdr.caplen = len;
    pkthdr.len = pkthdr.caplen;

    if (g_pcap_file) {
        if (fwrite(&pkthdr, sizeof(pkthdr), 1, g_pcap_file) != 1) {
            printf("Fwrite packet header failed.\n");
            return -1;
        }
        if (fwrite(buf, (size_t)len, 1, g_pcap_file) != 1) {
            printf("Fwrite data failed.\n");
            return -1;
        }
    }

    return 0;
}

void pcap_clean(void)
{
    if (g_pcap_file) {
        fclose(g_pcap_file);
        g_pcap_file = NULL;
    }

    return;
}

int main()
{
    char file[FILE_NAME_LEN] = "test.pcap";
    char buf[2048] = {0x00, 0x50, 0x56, 0xe4, 0xa4, 0x69, 0x00, 0x0c, 0x29, 0xde, 0x31, 0xa6, 0x08, 0x00, 0x45, 0x00,
                      0x00, 0x3c, 0xa1, 0x78, 0x40, 0x00, 0x40, 0x06, 0xe7, 0x37, 0xc0, 0xa8, 0x3a, 0x80, 0x5b, 0xbd,
                      0x5b, 0x26, 0xe8, 0xfa, 0x00, 0x50, 0x5b, 0xcb, 0xb3, 0x62, 0x00, 0x00, 0x00, 0x00, 0xa0, 0x02,
                      0xfa, 0xf0, 0xb2, 0x3a, 0x00, 0x00, 0x02, 0x04, 0x05, 0xb4, 0x04, 0x02, 0x08, 0x0a, 0x7c, 0x2b,
                      0xa0, 0x3e, 0x00, 0x00, 0x00, 0x00, 0x01, 0x03, 0x03, 0x07};

    if(0 != pcap_init(file)) {
        printf("Pcap init failed.\n");
        return -1;
    }
    if(0 != pcap_write(buf, 74)) {
        printf("Pcap write failed.\n");
        return -1;
    }
    pcap_clean();

    return 0;
}
