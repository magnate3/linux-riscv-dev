#include <linux/module.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/netlink.h>
#include <net/sock.h>

#define NETLINK_TEST 30

static struct sock *nlsk = NULL;

static void nl_send_to_user(char *str, int pid)
{
	struct sk_buff *skb = NULL;
	struct nlmsghdr *nlh = NULL;
	size_t size;

	size = nlmsg_total_size(strlen(str));
	skb = nlmsg_new(size, GFP_KERNEL);
	if (skb == NULL) {
		pr_err("[%s] alloc mem failed\n", __func__);
		return;
	}

	nlh = nlmsg_put(skb, 0, 0, 0, size, 0);
	if (nlh == NULL) {
		kfree_skb(skb);
		pr_err("[%s] put failed\n", __func__);
		return;
	}

	NETLINK_CB(skb).nsid = 0;
	NETLINK_CB(skb).dst_group = 0;

	strncpy(nlmsg_data(nlh), str, strlen(str));
	pr_info("[%s] send msg to user space:%s\n", __func__, str);

	netlink_unicast(nlsk, skb, pid, MSG_DONTWAIT);

	nlmsg_free(skb);
}

static void nl_recv_callback(struct sk_buff *skb)
{
	char str[100] = {0};
	struct nlmsghdr *nlh = nlmsg_hdr(skb);
	int user_pid = 0;
	int i;

	pr_info("[%s] skb->len:%u, nlh->nlmsg_len:%u\n",
		__func__, skb->len, nlh->nlmsg_len);

	if (skb->len < sizeof(struct nlmsghdr) ||
	    nlh->nlmsg_len < sizeof(struct nlmsghdr) ||
	    skb->len < nlh->nlmsg_len) {
		pr_err("[%s] len is error\n", __func__);
		return;
	}

	user_pid = nlh->nlmsg_pid;
	strncpy(str, nlmsg_data(nlh), nlmsg_len(nlh));
	pr_info("[%s] recv str from user space(%d): %s\n", __func__, user_pid, str);

	for (i = 0; i < 5; i++) {
		nl_send_to_user("hello, this is KERNEL", user_pid);
		msleep(1000);
	}
}

static int __init nl_drv_init(void)
{
	struct netlink_kernel_cfg cfg = {
		.input = nl_recv_callback,
	};


	pr_info("[%s]\n", __func__);

	nlsk = netlink_kernel_create(&init_net, NETLINK_TEST, &cfg);
	if (nlsk == NULL) {
		pr_err("[%s] can't create netlink socket\n", __func__);
		return -ENOMEM;
	}

	pr_info("[%s] OK\n", __func__);

	return 0;
}

static void __exit nl_drv_exit(void)
{
	pr_info("[%s]\n", __func__);

	netlink_kernel_release(nlsk);
}

module_init(nl_drv_init);
module_exit(nl_drv_exit);

MODULE_LICENSE("Dual MPL/GPL");
MODULE_AUTHOR("yanli.qian");
MODULE_DESCRIPTION("netlink test driver module");
