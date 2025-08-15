#include <linux/module.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <net/sock.h>
#include <net/genetlink.h>
#include "my_genl_attr.h"

static struct genl_family my_genl_family;

static const struct genl_multicast_group my_genl_mcgrps[] = {
	{ .name = "genl_drv_test", },
};

static const struct nla_policy my_genl_policy[MY_ATTR_MAX + 1] = {
	[MY_ATTR_MSG] = {.type = NLA_NUL_STRING},
};

static int my_genl_fill_hdr(u8 cmd, pid_t pid, struct sk_buff *skb)
{
	genlmsg_put(skb, pid, 0, &my_genl_family, 0, cmd);

	return 0;
}

static int my_genl_fill_payload(struct sk_buff *skb, int type, u8 *data, size_t len)
{
	return nla_put(skb, type, len, data);
}

static int my_genl_send_msg_to_user(u8 *data, size_t len, pid_t pid)
{
	struct sk_buff *skb;
	size_t size;
	int ret;

	pr_info("[%s] start send echo data\n", __func__);

	size = nla_total_size(len);

	skb = genlmsg_new(size, GFP_KERNEL);
	if (skb == NULL) {
		pr_err("[%s] new sk_buff failed\n", __func__);
		return -ENOMEM;
	}

	ret = my_genl_fill_hdr(MY_CMD_ECHO, pid, skb);
	if (ret != 0) {
		pr_err("[%s] fill header failed, ret=%d\n", __func__, ret);
		ret = -1;
		goto free_mem;
	}
	ret = my_genl_fill_payload(skb, MY_ATTR_MSG, data, len);
	if (ret != 0) {
		pr_err("[%s] fill payload failed, ret=%d\n", __func__, ret);
		ret = -1;
		goto free_mem;
	}

	ret = genlmsg_unicast(&init_net, skb, pid);
	pr_info("[%s] unicast msg\n", __func__);
	return 0;

free_mem:
	nlmsg_free(skb);
	return ret;
}

static int my_genl_echo(struct sk_buff *skb, struct genl_info *info)
{
	struct nlmsghdr *nlhdr;
	struct genlmsghdr *genlhdr;
	struct nlattr *nla;
	size_t len;
	char *payload;
	char echo_data[] = "this is from kernel.";

	nlhdr = nlmsg_hdr(skb);
	genlhdr = nlmsg_data(nlhdr);
	nla = genlmsg_data(genlhdr);

	pr_info("[%s] genl received\n", __func__);
	pr_info("[%s] skb->len:%u, nlh->nlmsg_len:%u\n",
		__func__, skb->len, nlhdr->nlmsg_len);
	pr_info("[%s] genl hdr cmd:%u, version:%u\n",
		__func__, genlhdr->cmd, genlhdr->version);
	pr_info("[%s] nlattr type:%u len:%u\n",
		__func__, nla->nla_type, nla->nla_len);

	payload = (char *)nla_data(nla);
	pr_info("[%s] genl payload: %s\n", __func__, payload);

	len = strlen(echo_data);

	return my_genl_send_msg_to_user(echo_data, len, nlhdr->nlmsg_pid);
}

static const struct genl_ops my_genl_ops[] = {
	{
		.cmd = MY_CMD_ECHO,
		.doit = my_genl_echo,
		.dumpit = NULL,
		.done = NULL,
		.policy = my_genl_policy,
	},
};

static struct genl_family my_genl_family = {
#if 0
	.id = GENL_ID_GENERATE,
#endif
	.hdrsize = 0,
	.name = "my_genl",
	.version = 1,
	.maxattr = MY_ATTR_MAX,
	.module = THIS_MODULE,
	.ops = my_genl_ops,
	.n_ops = ARRAY_SIZE(my_genl_ops),
	.mcgrps = my_genl_mcgrps,
	.n_mcgrps = ARRAY_SIZE(my_genl_mcgrps),
};

static int __init my_genl_init(void)
{
	int ret;

#if 0
	ret = genl_register_family_with_ops_groups(&my_genl_family,
						   ops,
						   my_genl_mcgrps);
#endif
	ret = genl_register_family(&my_genl_family);

	pr_info("[%s] family id:%d\n", __func__, my_genl_family.id);


	if (ret)
		return ret;





	return 0;
}

static void __exit my_genl_exit(void)
{
	pr_info("[%s]\n", __func__);
	genl_unregister_family(&my_genl_family);
}

module_init(my_genl_init);
module_exit(my_genl_exit);

MODULE_LICENSE("Dual MPL/GPL");
MODULE_AUTHOR("yanli.qian");
MODULE_DESCRIPTION("generic netlink test driver module");
