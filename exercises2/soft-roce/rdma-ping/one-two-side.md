非对称环境指的是一个服务器需要和多个客户端进行交互的环境。   

Herd在客户端使用one-sided RDMA write发送消息，这是因为one-sided操作的接收端（in-bound）性能最好。   

Herd使用基于UD的send/recv操作***从服务器端向客户端回复***，这是由于UD的QP的连接数比RC的QP少，从而具有更好的可扩展性和发送端（out-bound）性能。   

FaSST进一步将基于UD的send/recv操作应用到一个***对称***的场景中，利用doorbell batching对UD提升比较大的特性，FaSST RPC对发送方具有更好的性能。  