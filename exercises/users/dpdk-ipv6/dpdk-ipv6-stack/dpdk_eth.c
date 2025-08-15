#include "dpdk_eth.h"
struct inet_ifaddr *inet_addr_ifa_get(int af, const struct netif_port *dev,
                                      union inet_addr *addr)
{
    struct inet_ifaddr *ifa=NULL;

    printf("%s not implement \n",__func__);   
    return ifa;
}

void inet_addr_ifa_put(struct inet_ifaddr *ifa)
{
    //ifa_put(ifa);
    printf("%s not implement \n",__func__);   
}

struct netif_port* netif_port_get(portid_t id)
{
      struct netif_port *port = NULL;
      printf("%s not implement \n",__func__);   
      return port;
}
