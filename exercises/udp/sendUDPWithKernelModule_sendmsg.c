/*
* Send Udp packet Demo
* Authors: Tingwei Liut <tingw.liu@gmail.com>
* 2011-08-11 08:38:39
*
* Any bugs please report <tingw.liu@gmail.com>
*/
#include <linux/module.h>
#include <net/ip.h>
#include <net/sock.h>
#include <linux/in.h>
#include <linux/udp.h>
#include <linux/net.h>
#include <linux/inet.h>


MODULE_LICENSE("GPL");
MODULE_AUTHOR("tingw.liu@gmail.com");
char *sendstring="pwp test string";
char *dip="10.66.16.141";
unsigned short dport=31900;
module_param(sendstring,charp,0644);
module_param(dip,charp,0644);
module_param(dport,ushort,0644);
struct sockaddr_in recvaddr;
struct socket *sock;

static int send_msg(struct socket *sock, char *buffer, size_t length)
{
        struct msghdr        msg;
        struct kvec        iov;
        int                len;
       
        memset(&msg,0,sizeof(msg));
        msg.msg_flags = MSG_DONTWAIT|MSG_NOSIGNAL;       
        msg.msg_name = (struct sockaddr *)&recvaddr;
        msg.msg_namelen = sizeof(recvaddr);


        iov.iov_base     = (void *)buffer;
        iov.iov_len      = length;

        len = kernel_sendmsg(sock, &msg, &iov, 1, length);
        return len;
}


static void make_daddr(void)
{
        memset(&recvaddr,0,sizeof(recvaddr));
        recvaddr.sin_family = AF_INET;
        recvaddr.sin_port = htons(dport);
        recvaddr.sin_addr.s_addr = in_aton(dip);
}

static void make_socket(void)
{
        if(sock_create_kern(&init_net,PF_INET, SOCK_DGRAM, IPPROTO_UDP, &sock) < 0){
                printk(KERN_ALERT "sock_create_kern error\n");
                sock = NULL;
                return;
        }
        if(sock->ops->connect(sock, (struct sockaddr*)&recvaddr,
                                   sizeof(struct sockaddr), 0) < 0){
                printk(KERN_ALERT "sock connect error\n");
                goto error;
        }
        return;
  error:
        sock_release(sock);
        sock = NULL;
        return;
}

static int __init send_udp_init(void)
{
        make_daddr();
        make_socket();
        if(sock == NULL)
		return -1;

	printk("send_udp_init ok\n");
        send_msg(sock,sendstring,strlen(sendstring));
        return 0;
}

static void __exit send_udp_exit(void)
{
        if(sock)
                sock_release(sock);
	printk("send_udp_exit ok\n");
        return;
}

module_init(send_udp_init);
module_exit(send_udp_exit);
