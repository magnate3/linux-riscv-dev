#include "net.h"

int setup_socket (void)
{
    int sock = socket (AF_PACKET, SOCK_RAW, IPPROTO_RAW);
    if (sock < 0)
    {
        perror ("socket");
        return sock;
    }

    return sock;
}

int convert_ip (const char *ip, uint32_t *ip_addr)
{
    if (ip == NULL)
    {
        *ip_addr = 0;
        return 0;
    }

    struct in_addr ip_addr_struct;
    if (inet_aton (ip, &ip_addr_struct) == 0)
    {
        PERROR ("inet_aton");
        return -1;
    }
    *ip_addr = ip_addr_struct.s_addr;
    return 0;
}

int retrieve_local_ip (int ifindex, uint32_t *out_addr)
{
    char ifname[IF_NAMESIZE];
    if (if_indextoname (ifindex, ifname) == NULL)
    {
        PERROR ("if_indextoname");
        return -1;
    }

    int sock = socket (AF_INET, SOCK_DGRAM, 0);
    if (sock < 0)
    {
        PERROR ("socket");
        return -1;
    }

    struct ifreq ifr;
    memset (&ifr, 0, sizeof (struct ifreq));
    strncpy (ifr.ifr_name, ifname, IF_NAMESIZE);

    int ret = ioctl (sock, SIOCGIFADDR, &ifr);
    if (ret < 0)
    {
        PERROR ("ioctl");
        return -1;
    }

    close (sock);

    struct sockaddr_in *addr = (struct sockaddr_in *) &ifr.ifr_addr;
    *out_addr = addr->sin_addr.s_addr;
    return 0;
}

int retrieve_local_mac (int ifindex, uint8_t *out_mac)
{
    char ifname[IF_NAMESIZE + 1];
    if (if_indextoname (ifindex, ifname) == NULL)
    {
        PERROR ("if_indextoname");
        return -1;
    }

    // read /sys/class/net/<ifname>/address
    char path[192];
    snprintf (path, 192, "/sys/class/net/%s/address", ifname);
    FILE *f = fopen (path, "r");
    if (!f)
    {
        PERROR ("fopen");
        return -1;
    }

    char mac_str[18];
    if (fgets (mac_str, 18, f) == NULL)
    {
        PERROR ("fgets");
        return -1;
    }

    if (fclose (f) != 0)
    {
        PERROR ("fclose");
        return -1;
    }

    int ret = sscanf (mac_str, "%hhx:%hhx:%hhx:%hhx:%hhx:%hhx", &out_mac[0], &out_mac[1], &out_mac[2], &out_mac[3], &out_mac[4], &out_mac[5]);
    if (ret != 6)
    {
        PERROR ("sscanf");
        return -1;
    }

    return 0;
}

int exchange_data (const char *server_ip, bool is_server, uint32_t packet_size, uint8_t *buffer, uint8_t *out_buffer)
{
    int sock = socket (AF_INET, SOCK_DGRAM, 0);
    if (sock < 0)
    {
        PERROR ("socket");
        return -1;
    }

    // bind to local address port 1234
    struct sockaddr_in local_addr;
    local_addr.sin_family = AF_INET;
    local_addr.sin_port = htons (1234);
    local_addr.sin_addr.s_addr = INADDR_ANY;
    int ret = bind (sock, (struct sockaddr *) &local_addr, sizeof (struct sockaddr_in));
    if (ret < 0)
    {
        PERROR ("bind");
        return -1;
    }

    if (!is_server)
    {
        struct sockaddr_in server_addr;
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons (1234);
        server_addr.sin_addr.s_addr = inet_addr (server_ip);

        int ret = sendto (sock, buffer, packet_size, 0, (struct sockaddr *) &server_addr, sizeof (struct sockaddr_in));
        if (ret < 0)
        {
            PERROR ("sendto");
            return -1;
        }

        memset (out_buffer, 0, packet_size);
        ret = recvfrom (sock, out_buffer, packet_size, 0, NULL, NULL);
        if (ret < 0)
        {
            PERROR ("recvfrom");
            return -1;
        }
    }
    else
    {
        memset (out_buffer, 0, packet_size);
        struct sockaddr_in client_addr;
        socklen_t client_addr_len = sizeof (struct sockaddr_in);

        ret = recvfrom (sock, out_buffer, packet_size, 0, (struct sockaddr *) &client_addr, &client_addr_len);
        if (ret < 0)
        {
            PERROR ("recvfrom");
            return -1;
        }

        ret = sendto (sock, buffer, packet_size, 0, (struct sockaddr *) &client_addr, sizeof (struct sockaddr_in));
        if (ret < 0)
        {
            PERROR ("sendto");
            return -1;
        }
    }

    close (sock);

    return 0;
}

int exchange_eth_ip_addresses (const int ifindex, const char *restrict server_ip, bool is_server,
                               uint8_t *restrict src_mac, uint8_t *restrict dest_mac,
                               uint32_t *restrict src_ip, uint32_t *restrict dest_ip)
{
    retrieve_local_mac (ifindex, src_mac);
    retrieve_local_ip (ifindex, src_ip);

    uint8_t buffer[ETH_IP_INFO_PACKET_SIZE];
    uint8_t out_buffer[ETH_IP_INFO_PACKET_SIZE];

    memcpy (buffer, src_mac, ETH_ALEN);
    memcpy (buffer + ETH_ALEN, src_ip, sizeof (uint32_t));

    int ret = exchange_data (server_ip, is_server, ETH_IP_INFO_PACKET_SIZE, buffer, out_buffer);
    if (ret < 0)
    {
        LOG (stderr, "ERR: exchange_data\n");
        return -1;
    }

    memcpy (dest_mac, out_buffer, ETH_ALEN);
    memcpy (dest_ip, out_buffer + ETH_ALEN, sizeof (uint32_t));

    return 0;
}

int build_base_packet (char *buf, const uint8_t *src_mac, const uint8_t *dest_mac,
                       const uint32_t src_ip, const uint32_t dest_ip)
{
    struct ethhdr *eth = (struct ethhdr *) buf;
    for (int i = 0; i < ETH_ALEN; ++i)
    {
        eth->h_source[i] = src_mac[i];
        eth->h_dest[i] = dest_mac[i];
    }

    eth->h_proto = __constant_htons (ETH_P_PINGPONG);

    struct iphdr *ip = (struct iphdr *) (eth + 1);
    ip->ihl = 5;
    ip->version = 4;
    ip->tos = 0;
    ip->tot_len = htons (PACKET_SIZE - sizeof (struct ethhdr));
    ip->id = 0;
    ip->frag_off = htons (0);
    ip->ttl = 64;
    ip->protocol = IPPROTO_RAW;
    ip->check = 0;
    ip->saddr = src_ip;
    ip->daddr = dest_ip;

    return 0;
}

inline struct pingpong_payload *packet_payload (const char *buf)
{
    return (struct pingpong_payload *) (buf + sizeof (struct ethhdr) + sizeof (struct iphdr));
}

struct sockaddr_ll build_sockaddr (int ifindex, const unsigned char *dest_mac)
{
    struct sockaddr_ll sock_addr;
    sock_addr.sll_ifindex = ifindex;
    sock_addr.sll_halen = ETH_ALEN;
    for (int i = 0; i < ETH_ALEN; ++i)
        sock_addr.sll_addr[i] = dest_mac[i];

    return sock_addr;
}

inline int send_pingpong_packet (int sock, const char *restrict buf, struct sockaddr_ll *restrict sock_addr)
{
    return sendto (sock, buf, PACKET_SIZE, 0, (struct sockaddr *) sock_addr, sizeof (struct sockaddr_ll));
}

struct sender_data {
    uint64_t iters;
    uint64_t interval;
    char *base_packet;
    struct sockaddr_ll *sock_addr;
    // function to send the packet
    send_packet_t send_packet;
    // auxiliary data for the send function
    void *aux;
};

int stick_this_thread_to_core (int core_id)
{
    printf("Setting thread affinity to core %d\n", core_id);
    int num_cores = sysconf (_SC_NPROCESSORS_ONLN);
    if (core_id < 0 || core_id >= num_cores)
        return -EINVAL;

    cpu_set_t cpuset;
    CPU_ZERO (&cpuset);
    CPU_SET (core_id, &cpuset);

    pthread_t current_thread = pthread_self ();
    return pthread_setaffinity_np (current_thread, sizeof (cpu_set_t), &cpuset);
}

void *thread_send_packets (void *args)
{
    sleep(2);
    cpu_set_t current_mask;
    if (sched_getaffinity (0, sizeof (cpu_set_t), &current_mask) < 0)
    {
        PERROR ("sched_getaffinity");
        return NULL;
    }
    // set the thread affinity to the next thread
    int core_id = 0;
    int cnt = 0;
    for (int i = 0; i < CPU_SETSIZE; ++i)
    {
        if (CPU_ISSET (i, &current_mask))
        {
            core_id = i + 1;
            ++cnt;
        }
    }
    // set the sending thread affinity only if the main thread is pinned to a single core
    // i.e. we are running a test with isolation.
    if (cnt == 1 && stick_this_thread_to_core (core_id) < 0)
    {
        PERROR ("stick_this_thread_to_core");
        return NULL;
    }

    struct sender_data *data = (struct sender_data *) args;

    for (uint64_t id = 1; id <= data->iters; ++id)
    {
        uint64_t __start = get_time_ns ();
        int ret = data->send_packet (data->base_packet, id, data->sock_addr, data->aux);
        if (ret < 0)
        {
            PERROR ("data->send_packet");
            return NULL;
        }
        uint64_t interval = get_time_ns () - __start;
        if (interval < data->interval)
            pp_sleep (data->interval - interval);
    }

    free (data->base_packet);
    free (data->sock_addr);

    return NULL;
}

static pthread_t sender_thread;

int start_sending_packets (uint64_t iters, uint64_t interval, char *base_packet, struct sockaddr_ll *sock_addr, send_packet_t send_packet, void *aux)
{
    struct sender_data *data = malloc (sizeof (struct sender_data));
    if (!data)
    {
        PERROR ("malloc");
        return -1;
    }

    data->iters = iters;
    data->interval = interval;
    data->send_packet = send_packet;
    data->aux = aux;

    data->base_packet = NULL;
    data->sock_addr = NULL;
    if (base_packet)
    {
        data->base_packet = malloc (PACKET_SIZE);
        memcpy (data->base_packet, base_packet, PACKET_SIZE);
    }
    if (sock_addr)
    {
        data->sock_addr = malloc (sizeof (struct sockaddr_ll));
        memcpy (data->sock_addr, sock_addr, sizeof (struct sockaddr_ll));
    }

    int ret = pthread_create (&sender_thread, NULL, thread_send_packets, data);
    if (ret < 0)
    {
        PERROR ("pthread_create");
        return -1;
    }

    return 0;
}

pthread_t get_sender_thread (void)
{
    return sender_thread;
}