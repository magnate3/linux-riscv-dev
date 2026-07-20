#ifndef _GENLCORE_H
#define _GENLCORE_H

#include <linux/netlink.h>
#include <linux/genetlink.h>

#include "nlcore.h"

int genl_open(struct nl_sock *nlsock);

char *genlmsg_put_hdr(char *p, int cmd);
char *genlmsg_put_msg_specific_hdr(char *p, void *hdr, int len);
char *genlmsg_add_nla(char *p, int type, int len, void *data);

int genl_service_id(struct nl_sock *nlsock, const char *name);

#define GENLMSG_HDR(nlhdr) ((struct genlmsghdr *)((char *)(nlhdr) + NLMSG_HDRLEN))

#define GENLMSG_DATA(nlhdr) ((char *)nlhdr + NLMSG_HDRLEN + GENL_HDRLEN)

#define GENLMSG_DATA_LEN(nlhdr) ((nlhdr)->nlmsg_len - NLMSG_HDRLEN - GENL_HDRLEN)

#define NLA_DATA(nla) ((char *)nla + NLA_HDRLEN)

#define NLA_OK(nla, len) ((nla)->nla_len > NLA_HDRLEN && (nla)->nla_len <= (len))

#define NLA_NEXT(nla, len) (len -= NLA_ALIGN((nla)->nla_len), (struct nlattr *)((char *)(nla) + NLA_ALIGN((nla)->nla_len)))

#endif
