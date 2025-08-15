
# udpgw-remote-server-addr

badvpn的tun2socks加上了选 项--socks5-udp，以支持直接将tun设备的udp数据包通过sock5代理直接发送。
以前是封装为tcp数据包，然后发送给--udpgw-remote-server-addr指定的地址。然后udpgw程序再使用udp发向源服务器，并将接收的包转发回tun设备
。


[image](udp.png)


# clash 连接udpgw 

[image](clash.png)