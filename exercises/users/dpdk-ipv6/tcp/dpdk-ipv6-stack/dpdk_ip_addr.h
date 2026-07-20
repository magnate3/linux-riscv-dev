
#ifndef IP_ADDR_H
#define IP_ADDR_H
#include <netinet/ip.h>
#include <netinet/tcp.h>
#define IPV4_STR(addr) \
    ((const unsigned char *)&(addr))[0], \
    ((const unsigned char *)&(addr))[1], \
    ((const unsigned char *)&(addr))[2], \
    ((const unsigned char *)&(addr))[3]
#define IPV4_FMT "%u.%u.%u.%u"

#define IPV6_STR(addr) \
    ntohs(((uint16_t*)&(addr))[0]), \
    ntohs(((uint16_t*)&(addr))[1]), \
    ntohs(((uint16_t*)&(addr))[2]), \
    ntohs(((uint16_t*)&(addr))[3]), \
    ntohs(((uint16_t*)&(addr))[4]), \
    ntohs(((uint16_t*)&(addr))[5]), \
    ntohs(((uint16_t*)&(addr))[6]), \
    ntohs(((uint16_t*)&(addr))[7])
#define IPV6_FMT "%04x:%04x:%04x:%04x:%04x:%04x:%04x:%04x"
#endif
