/* Generic Netlink core functions */

#include <string.h>
#include <syslog.h>
#include <stdint.h>
#include <stdio.h>
#include <linux/genetlink.h>

#include "genlcore.h"

int genl_open(struct nl_sock *nlsock)
{
	return nl_open(nlsock, NETLINK_GENERIC);
}

char *genlmsg_put_hdr(char *p, int cmd)
{
	struct genlmsghdr *genlhdr = (struct genlmsghdr *)p;

	memset(genlhdr, 0, GENL_HDRLEN);
	genlhdr->version = 1;
	genlhdr->cmd = cmd;

	return p + GENL_HDRLEN;
}

char *genlmsg_put_msg_specific_hdr(char *p, void *hdr, int len)
{
	memcpy(p, hdr, len);
	memset(p + len, 0, NLMSG_ALIGN(len) - len);

	return p + NLMSG_ALIGN(len);
}

char *genlmsg_add_nla(char *p, int type, int len, void *data)
{
	struct nlattr *nla = (struct nlattr *)p;

	memset(p, 0, NLA_HDRLEN + NLA_ALIGN(len));

	nla->nla_type = type;
	nla->nla_len = NLA_HDRLEN + len;
	memcpy(p + NLA_HDRLEN, data, len);
	return p + NLA_ALIGN(nla->nla_len);
}

struct service_id_cb_priv {
	int id;
};

static int service_id_cb(struct nlmsghdr *nlhdr, void *_priv)
{
	struct nlattr *nla;
	struct service_id_cb_priv *priv = (struct service_id_cb_priv *)_priv;
	int n;

	if (!nlhdr)
		return 0;

	for (nla = (struct nlattr *)GENLMSG_DATA(nlhdr), n = GENLMSG_DATA_LEN(nlhdr);
	     NLA_OK(nla, n); nla = NLA_NEXT(nla, n)) {
		if (nla->nla_type == CTRL_ATTR_FAMILY_ID) {
			priv->id = *(uint16_t *)NLA_DATA(nla);
			break;
		}
	}

	return 0;
}

int genl_service_id(struct nl_sock *nlsock, const char *name)
{
	char buf[128], *p;
	struct service_id_cb_priv priv;

	p = nlmsg_put_hdr(buf, GENL_ID_CTRL, 0);
	p = genlmsg_put_hdr(p, CTRL_CMD_GETFAMILY);
	p = genlmsg_add_nla(p, CTRL_ATTR_FAMILY_NAME, strlen(name) + 1, (void *)name);

	if (nl_send_msg(nlsock, buf, p - buf))
		return -1;

	priv.id = -1;
	if (nl_recv_msg(nlsock, GENL_ID_CTRL, service_id_cb, &priv))
		return -1;

	DEBUG("service %s has id=0x%02x", name, priv.id);
	return priv.id;
}
