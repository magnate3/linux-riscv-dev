#ifndef __HELP_H__

#define __HELP_H__
#define BYTE_N(p, n)			(((const unsigned char*) (p))[n] )
#define MAC_PRINTF_FORMAT		"%x:%x:%x:%x:%x:%x"
#define MAC_PRINTF_VALUES(p)		BYTE_N(p, 0), BYTE_N(p, 1), BYTE_N(p, 2), \
										BYTE_N(p, 3), BYTE_N(p, 4), BYTE_N(p, 5)
#define print_mac_addr(addr) printf("%02"PRIx8":%02"PRIx8":%02"PRIx8":%02"PRIx8":%02"PRIx8":%02"PRIx8, \
  addr.addr_bytes[0], addr.addr_bytes[1], addr.addr_bytes[2], addr.addr_bytes[3], addr.addr_bytes[4], addr.addr_bytes[5])
#endif
