#ifndef MY_INCLUDE_H
#define MY_INCLUDE_H

#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>

#include <sys/time.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <sys/ioctl.h>
#include <arpa/inet.h>
#include <net/if.h>

#include <asm/types.h>
#include <linux/net_tstamp.h>
#include <linux/errqueue.h>

#define TX_PORT 			6666	//transmitting host port
#define TX_PORT2 			7777
#define RX_PORT 			6666	//receiving host port
#define TX_ADDR 			"10.11.11.82"	//transmitting host ip
#define RX_ADDR 			"10.11.11.80"	//receiving host ip
#define	INTERFACE			"enp0s31f6"	//transmitting host interface


#define TX_DATA_BUFF_LEN 	2048	//max tx payload size
#define TX_PAYLOAD_SIZE 	256		//payload size


#ifndef SO_TIMESTAMPING
# define SO_TIMESTAMPING         37
# define SCM_TIMESTAMPING        SO_TIMESTAMPING
#endif

#ifndef SO_TIMESTAMPNS
# define SO_TIMESTAMPNS 35
#endif

#ifndef SIOCGSTAMPNS
# define SIOCGSTAMPNS 0x8907
#endif

#ifndef SIOCSHWTSTAMP
# define SIOCSHWTSTAMP 0x89b0
#endif



#endif
