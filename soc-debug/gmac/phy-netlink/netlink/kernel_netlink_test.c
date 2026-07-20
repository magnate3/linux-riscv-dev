#include<linux/init.h>
#include<linux/module.h>
#include<linux/types.h>
#include<net/sock.h>
#include<linux/netlink.h>
#include "netlink_test.h"
#include "common.h"
#include "marvel.h"
#include "ti.h"
#define MSG_LEN		125
#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt
MODULE_LICENSE("GPL");
MODULE_AUTHOR("ArielWu");
MODULE_DESCRIPTION("netlink example");

struct sock *nlsk = NULL;
extern struct net init_net;
struct phy_driver_op_t marvel;
struct phy_driver_op_t ti;
struct phy_driver_op_t *g_phy_op = NULL;
//spinlock_t              lock;
static DEFINE_MUTEX(test_phy_mutex);
static LIST_HEAD(phy_op_list);
int phy_probe(const char  *name)
{
     struct phy_driver_op_t *ptr1,*next;
     struct phy_device *  phy = get_phy(name);
     bool match = false;
     if(NULL == phy)
     {
          return PHY_NOT_EXIST ;
     }
     if(NULL != g_phy_op && g_phy_op->drv_match(NULL, phy))
     {
           return 0;
     }
     list_for_each_entry_safe(ptr1,next,&phy_op_list,list){
         if(ptr1->drv_match(NULL, phy))
         {
              g_phy_op = ptr1;
              match = true;
              break;
         } 
 
     }
     if (!match)
     {
          return PHY_NOT_MATCH;
     }
     return 0;
}
#if 0
#define  CONFIG_DEBUG_ATOMIC_SLEEP
#ifdef CONFIG_DEBUG_ATOMIC_SLEEP
static inline int preempt_count_equals(int preempt_offset)
{
       int nested = preempt_count() + rcu_preempt_depth();
       return (nested == preempt_offset);
}
#endif
#endif
int phy_op_do(struct phy_op_t *phy)
{
#if 1
    mutex_lock(&test_phy_mutex);
    if(phy_probe(phy->name))
    {
         pr_err("phy probe err \n");
         return PHY_PROBE_ERR;
    }
    g_phy_op->phy_do(phy);
    mutex_unlock(&test_phy_mutex);
#else
     pr_info("%d,irqs_disabled %d,  preempt_count: %d, in atomic %d , pid %d, name %s\n ", preempt_count_equals(0), irqs_disabled(), preempt_count(), in_atomic(), current->pid, current->comm);
     struct phy_driver_op_t *ptr1,*next;
     list_for_each_entry_safe(ptr1,next,&phy_op_list,list){
          pr_info("phy driver \n"); 
     }
#endif
    return 0;
}
#if 0
int send_usrmsg(char *pbuf,uint16_t len) {
	struct sk_buff *nl_skb;
	struct nlmsghdr *nlh;
	int ret;

	//create sk_buff space
	nl_skb = nlmsg_new(len,GFP_ATOMIC);
	
	//set netlink msg header
	nlh = nlmsg_put(nl_skb,0,0,NETLINK_TEST,len,0);

	//copy data and send
	memcpy(nlmsg_data(nlh),pbuf,len);
	ret = netlink_unicast(nlsk,nl_skb,USER_PORT,MSG_DONTWAIT);

	return ret;

}
#else

int send_usr_str_msg(const struct nlmsghdr *nlhusr, char *pbuf,uint16_t len) {
	struct sk_buff *nl_skb;
	struct nlmsghdr *nlh;
	int ret;

	//create sk_buff space
	nl_skb = nlmsg_new(len+1,GFP_ATOMIC);
	
	//set netlink msg header
	nlh = nlmsg_put(nl_skb,0,0,NETLINK_TEST,len+1,0);

	nlh->nlmsg_type = nlhusr->nlmsg_type;
	nlh->nlmsg_pid = nlhusr->nlmsg_pid;
	nlh->nlmsg_seq = nlhusr->nlmsg_seq;
	//copy data and send
	memcpy(nlmsg_data(nlh),pbuf,len);
        memset(nlmsg_data(nlh) + len, 0, 1);
	ret = netlink_unicast(nlsk,nl_skb,nlhusr->nlmsg_pid,MSG_DONTWAIT);
	//ret = netlink_unicast(nlsk,nl_skb,USER_PORT,MSG_DONTWAIT);

	return ret;

}
int send_usr_msg(const struct nlmsghdr *nlhusr, char *pbuf,uint16_t len) {
	struct sk_buff *nl_skb;
	struct nlmsghdr *nlh;
	int ret;

	//create sk_buff space
	nl_skb = nlmsg_new(len,GFP_ATOMIC);
	
	//set netlink msg header
	nlh = nlmsg_put(nl_skb,0,0,NETLINK_TEST,len,0);

	nlh->nlmsg_type = nlhusr->nlmsg_type;
	nlh->nlmsg_pid = nlhusr->nlmsg_pid;
	nlh->nlmsg_seq = nlhusr->nlmsg_seq;
	//copy data and send
	memcpy(nlmsg_data(nlh),pbuf,len);
	ret = netlink_unicast(nlsk,nl_skb,nlhusr->nlmsg_pid,MSG_DONTWAIT);
	return ret;

}
#endif
#if 0
static void netlink_rcv_msg(struct sk_buff *skb) {
	struct nlmsghdr *nlh = NULL;
	char *umsg=NULL;
	char *kmsg="hello users!!!";
	
	if(skb->len >= nlmsg_total_size(0)) {
		nlh = nlmsg_hdr(skb);
		umsg = NLMSG_DATA(nlh);
		if(umsg) {
			printk("kernel recv from user:%s\n",umsg);
			send_usrmsg(kmsg,strlen(kmsg));
		}
	}
}

#else
static int cool_recv_netlink(struct sk_buff *skb, struct nlmsghdr *nlh,
                             struct netlink_ext_ack *extack)
{
        /* We don't have anything else to do, we get the nlmsghdr directly. We use `nlmsg_type` for our type, and NLMSG_DATA(nlh) for data. */
 
        if (nlh->nlmsg_type >= __MSG_TYPE_MAX) {
                pr_err("cool: invalid message type: %d\n", nlh->nlmsg_type);
                return -EOPNOTSUPP;
        }
 
        switch (nlh->nlmsg_type) {
        case MSG_TYPE_STRING:
                /* Note that we don't need copy_from_user */
                pr_info("Got from usermode (len %d): %s\n", nlh->nlmsg_len, (char *)NLMSG_DATA(nlh));
                break;
        default:
                pr_info("Unhandled message type: %d\n", nlh->nlmsg_len);
                break;
        }
 
        //cool_send_reply(nlh->nlmsg_pid);
 
        return 0;
}
static int process_msg(struct sk_buff *skb, struct nlmsghdr *nlh)
{
     u32			seq;
     char                       *data;
     int			data_len;
     u16			msg_type = nlh->nlmsg_type;
     char *ptr;
     /*
     int			err;
     err = audit_netlink_ok(skb, msg_type);
     if (err)
        	return err;
     */
        /* We don't have anything else to do, we get the nlmsghdr directly. We use `nlmsg_type` for our type, and NLMSG_DATA(nlh) for data. */
 
        if (nlh->nlmsg_type >= __MSG_TYPE_MAX) {
                pr_err("cool: invalid message type: %d\n", nlh->nlmsg_type);
                return -EOPNOTSUPP;
        }
        seq = nlh->nlmsg_seq;
        //pr_info("msg seq %u \n", seq);
        pr_info("nlmsg len %d type %d pid 0x%X seq %d\n", nlh->nlmsg_len, nlh->nlmsg_type, nlh->nlmsg_pid, seq); 
        switch (msg_type) {
        case MSG_TYPE_STRING:
                /* Note that we don't need copy_from_user */
                data_len = nlh->nlmsg_len  -  NLMSG_HDRLEN;
                data = (char *)NLMSG_DATA(nlh);
                ptr = kmalloc(data_len + 1, GFP_KERNEL);
               	if (!ptr) {
		    pr_err("Allocation failed\n");
		    return 1;
	        }
                memset(ptr, 0, data_len);
                // will copy more data than user 
                memcpy(ptr, data, data_len);
                pr_info("Got from usermode (total len %d and data len %d ): %s<! ***********!>\n", nlh->nlmsg_len , data_len, ptr);
                pr_info("Got from usermode (total len %d and data len %d ): %s\n", nlh->nlmsg_len , data_len, data);
                send_usr_str_msg(nlh, "hello world", strlen("hello world"));
                break;
        case MSG_TYPE_PHY: {
                struct phy_op_t *phy=  NLMSG_DATA(nlh);
	        pr_info("dev name %s ,op %d,  page 0x%x, reg: 0x%x, val: 0x%x \n", phy->name,phy->op,phy->page, phy->reg, phy->val);
                //phy_probe(phy->name);
                //marvel.phy_do(phy);
                //pr_info("%d,irqs_disabled %d,  preempt_count: %d, in atomic %d , pid %d, name %s\n ", preempt_count_equals(0), irqs_disabled(), preempt_count(), in_atomic(), current->pid, current->comm);
                phy_op_do(phy);
                send_usr_msg(nlh, (char *)phy, sizeof(struct phy_op_t));
                break;
        }
        default:
                pr_info("Unhandled message type: %d\n", nlh->nlmsg_len);
                break;
        }
 
        //cool_send_reply(nlh->nlmsg_pid);
 
        return 0;
}
static void netlink_rcv_msg(struct sk_buff *skb)
{
        /* netlink_rcv_skb knows when to call our callback in case of multi-part, ack, etc.
 *          * See the implementation for more details. */
#if 0
        netlink_rcv_skb(skb, &cool_recv_netlink);
#else
        struct nlmsghdr *nlh;
        int len;
        int err;
        nlh = nlmsg_hdr(skb);
        len = skb->len;
 #if 0 
        while (nlmsg_ok(nlh, len)) {
           err = process_msg(skb, nlh);
           if (err || (nlh->nlmsg_flags & NLM_F_ACK))
           {
         		netlink_ack(skb, nlh, err, NULL);
           }
         	nlh = nlmsg_next(nlh, &len);
        }
#else
       	while (NLMSG_OK(nlh, len)) {
		err = process_msg(skb, nlh);
		/* if err or if this message says it wants a response */
		if (err || (nlh->nlmsg_flags & NLM_F_ACK))
                 {
			//netlink_ack(skb, nlh, err,NULL);
                 }
		nlh = NLMSG_NEXT(nlh, len);
	}
#endif
#endif
}
#endif

struct netlink_kernel_cfg cfg = {
	.input = netlink_rcv_msg,
};

int test_netlink_init(void) {
	
	//create netlink socket
	nlsk = (struct sock*)netlink_kernel_create(&init_net,NETLINK_TEST,&cfg);
        marvel.drv_match= marvel_drv_match;
        marvel.phy_do = marvel_phy_do;
        ti.drv_match= ti_drv_match;
        ti.phy_do = ti_phy_do;
        list_add_tail(&marvel.list, &phy_op_list);
        list_add_tail(&ti.list, &phy_op_list);
        //spin_lock_init(&lock);
	printk("test_netlink_init\n");

	return 0;
}

void test_netlink_exit(void) {
	if(nlsk) {
		netlink_kernel_release(nlsk);
		nlsk=NULL;
	}
	printk("test_netlink_exit!\n");

}

module_init(test_netlink_init);
module_exit(test_netlink_exit);
