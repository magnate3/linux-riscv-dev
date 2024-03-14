#include<linux/in.h>
#include<linux/inet.h>
#include<linux/socket.h>
#include<net/sock.h>
#include<linux/init.h>
#include<linux/module.h>

#define BUFFER_SIZE 1024

int connect_send_recv(void){
    struct socket *sock;
    struct sockaddr_in s_addr;
    unsigned short port_num = 8888;
    int ret = 0;
    char *send_buf = NULL;
    char *recv_buf = NULL;
    struct kvec send_vec, recv_vec;
    struct msghdr send_msg, recv_msg;

    /* kmalloc a send buffer*/
    send_buf = kmalloc(BUFFER_SIZE, GFP_KERNEL);
    if (send_buf == NULL) {
        printk("client: send_buf kmalloc error!\n");
        return -1;
    }

    /* kmalloc a receive buffer*/
    recv_buf = kmalloc(BUFFER_SIZE, GFP_KERNEL);
    if(recv_buf == NULL){
        printk("client: recv_buf kmalloc error!\n");
        return -1;
    }

    memset(&s_addr, 0, sizeof(s_addr));
    s_addr.sin_family = AF_INET;
    s_addr.sin_port = htons(port_num);

    s_addr.sin_addr.s_addr = in_aton("10.10.16.82");
    sock = (struct socket *)kmalloc(sizeof(struct socket), GFP_KERNEL);

    // 创建一个sock, &init_net是默认网络命名空间
    ret = sock_create_kern(&init_net, AF_INET, SOCK_STREAM, 0, &sock);
    if (ret < 0) {
        printk("client:socket create error!\n");
        return ret;
    }
    printk("client: socket create ok!\n");

    //连接
    ret = sock->ops->connect(sock, (struct sockaddr *)&s_addr, sizeof(s_addr), 0);
    if (ret != 0) {
        printk("client: connect error!\n");
        return ret;
    }
    printk("client: connect ok!\n");

    memset(send_buf, 'a', BUFFER_SIZE);

    memset(&send_msg, 0, sizeof(send_msg));
    memset(&send_vec, 0, sizeof(send_vec));

    send_vec.iov_base = send_buf;
    send_vec.iov_len = BUFFER_SIZE;

    // 发送数据
    ret = kernel_sendmsg(sock, &send_msg, &send_vec, 1, BUFFER_SIZE);
    if (ret < 0) {
        printk("client: kernel_sendmsg error!\n");
        return ret;
    } else if(ret != BUFFER_SIZE){
        printk("client: ret!=BUFFER_SIZE");
    }
    printk("client: send ok!\n");

    memset(recv_buf, 0, BUFFER_SIZE);

    memset(&recv_vec, 0, sizeof(recv_vec));
    memset(&recv_msg, 0, sizeof(recv_msg));

    recv_vec.iov_base = recv_buf;
    recv_vec.iov_len = BUFFER_SIZE;

    // 接收数据
    ret = kernel_recvmsg(sock, &recv_msg, &recv_vec, 1, BUFFER_SIZE, 0);
    printk("client: received message:\n %s\n", recv_buf);

    // 关闭连接
    kernel_sock_shutdown(sock, SHUT_RDWR);
    sock_release(sock);

    return 0;
}

static int client_example_init(void){
    printk("client: init\n");
    return connect_send_recv();
}

static void client_example_exit(void){
    printk("client: exit!\n");
}

module_init(client_example_init);
module_exit(client_example_exit);
MODULE_LICENSE("GPL");