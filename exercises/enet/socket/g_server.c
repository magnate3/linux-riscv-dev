/*
#    Copyright By Schips, All Rights Reserved
#    https://gitee.com/schips/
#
#    File Name:  group_server.c
#    Created  :  Mon 23 Mar 2020 04:02:12 PM CST
*/

#include <stdio.h>
#include <string.h>
#include <unistd.h>
//#include <linux/in.h>
#include <arpa/inet.h>
#include <sys/types.h>          /* See NOTES */
#include <sys/socket.h>

#define IP_FOUND  "IP_FOUND"
#define IP_FOUND_ACK  "IP_FOUND_ACK"
#define MCAST "239.1.1.1"
#define MY_IP "192.168.137.223"

//说明:设置主机的TTL值，是否允许本地回环，加入多播组，然后服务器向加入多播组的主机发送数据，主机接收数据，并响应服务器。

int main(int argc,char **argv)
{

        int sock_fd,client_fd;
        int ret;
        struct sockaddr_in localaddr;
        struct sockaddr_in recvaddr;
        socklen_t  socklen;
        char recv_buf[20];
        char send_buf[20];
        int ttl = 10;//如果转发的次数等于10,则不再转发
        int loop=0;

        sock_fd = socket(AF_INET, SOCK_DGRAM , 0);
        if(sock_fd == -1)
        {
                perror(" socket !");
        }

        memset(&localaddr,0,sizeof(localaddr));
        localaddr.sin_family = AF_INET;
        localaddr.sin_port = htons(6868);
	// should be INADDR_ANY, or be 192.168.137.223 will not recv
        localaddr.sin_addr.s_addr = htonl(INADDR_ANY);
        //localaddr.sin_addr.s_addr =  inet_addr(MY_IP);
        //localaddr.sin_addr.s_addr = htonl("192.168.137.223");

        ret = bind(sock_fd, (struct sockaddr *)&localaddr,sizeof(localaddr));
        if(ret == -1)
        {
                perror("bind !");
        }

         socklen = sizeof(struct sockaddr);

        //设置多播的TTL值
        if(setsockopt(sock_fd, IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof(ttl))<0){
                perror("IP_MULTICAST_TTL");
                return -1;
        }
        //设置数据是否发送到本地回环接口
        if(setsockopt(sock_fd, IPPROTO_IP, IP_MULTICAST_LOOP, &loop, sizeof(loop))<0){
                perror("IP_MULTICAST_LOOP");
                return -1;
        }
        //加入多播组
        struct ip_mreq mreq;
        mreq.imr_multiaddr.s_addr=inet_addr(MCAST);//多播组的IP
        mreq.imr_interface.s_addr =  inet_addr(MY_IP);
        //mreq.imr_interface.s_addr=htonl(INADDR_ANY);//本机的默认接口IP,本机的随机IP
        if(setsockopt(sock_fd, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) < 0){
                perror("IP_ADD_MEMBERSHIP");
                return -1;
        }

        while(1)
        {
                ret = recvfrom(sock_fd, recv_buf, sizeof(recv_buf), 0,(struct sockaddr *)&recvaddr, &socklen);
                if(ret < 0 )
                {
                        perror(" recv! ");
                }

                printf(" recv client addr : %s \n", (char *)inet_ntoa(recvaddr.sin_addr));
                printf(" recv client port : %d \n",ntohs(recvaddr.sin_port));
                printf(" recv msg :%s \n", recv_buf);

                //if(strstr(recv_buf,IP_FOUND))
                //{
                        //响应客户端请求
                        strncpy(send_buf, IP_FOUND_ACK, strlen(IP_FOUND_ACK) + 1);
                        ret = sendto(sock_fd, send_buf, strlen(IP_FOUND_ACK) + 1, 0, (struct sockaddr*)&recvaddr, socklen);//将数据发送给客户端
                        if(ret < 0 )
                        {
                                perror(" sendto! ");
                        }
                        printf(" send ack  msg to client !\n");
                //}
        }

        // 离开多播组
        ret = setsockopt(sock_fd, IPPROTO_IP, IP_DROP_MEMBERSHIP, &mreq, sizeof(mreq));
        if(ret < 0){
                perror("IP_DROP_MEMBERSHIP");
                return -1;
        }

        close(sock_fd);

        return 0;
}
