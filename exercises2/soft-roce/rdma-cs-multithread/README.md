---
## RDMA-CS v1

A basic single client/single server model that lacks the full feature set.

---
## RDMA-CS v2

A multithreaded client/server that allows for multiple client conenction to one server. Basic administration
features are added alongside the increased functionality.


### multiple thread

+ 处于RDMA_CM_EVENT_ESTABLISHED状态创建新的线程   
+ Remake the cm_id   rdma_destroy_id   

```
void *hey_listen(void *cmid){
        // Standard initializing
        struct rdma_cm_id *cm_id = cmid;
        struct rdma_event_channel *ec = cm_id->channel;
        struct pnode tlist;
        struct cnode clist;
        while(1){
                // Listen for connection requests
                fprintf(log_p, "Listening for connection requests...\n");
                if(rdma_listen(cm_id, 1))
                        stop_it("rdma_listen()", errno, log_p);
                // Make an ID specific to the client that connected
                clist.id = cm_event(ec, RDMA_CM_EVENT_CONNECT_REQUEST, log_p);
                clist.length = SERVER_MR_SIZE;
                clist.status = CLOSED;
                idnum++;
                clist.cid = idnum;
                cm_event(ec, RDMA_CM_EVENT_ESTABLISHED, log_p);
                // Spawn an agent thread for the new conenction
                sem_wait(&tlist_sem);
                tlist.type = 1;
                if(pthread_create(&tlist.id, NULL, secret_agent, clist.id)){
                        stop_it("pthread_create()", errno, log_p);
                }
                sem_post(&tlist_sem);
                add_thread(tlist);
                clist.tid = tlist.id;
                add_client(clist);
                // Remake the cm_id
                if(rdma_destroy_id(cm_id))
                        stop_it("rdma_destroy_id()", errno, log_p);
                ec = rdma_create_event_channel();
                if(rdma_create_id(ec, &cm_id, "qwerty", RDMA_PS_TCP))
                        stop_it("rdma_create_id()", errno, log_p);
                // Bind to the port
                binding_of_isaac(cm_id, port);
                // Rinse and repeat
        }
        return 0;
}
```
+ ***子线程***   
   + a 处理rdma_get_send_comp 和 rdma_get_recv_comp   
```
void *secret_agent(void *id){
        struct rdma_cm_id *cm_id = id;
        struct ibv_mr *mr = ibv_reg_mr(cm_id->qp->pd, malloc(SERVER_MR_SIZE),SERVER_MR_SIZE,
         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);
        if(mr == NULL)
                stop_it("ibv_reg_mr()", errno, log_p);
        sem_wait(&clist_sem);
        struct cnode * node = clist_head;
        while(node != NULL){
                if(node->tid == pthread_self()){
                        node->rkey = mr->rkey;
                        node->remote_addr = (uint64_t)mr->addr;
                        break;
                }
                node = node->next;
        }
        sem_post(&clist_sem);
        // Exchange addresses and rkeys with the client
        uint32_t rkey;
        uint64_t remote_addr;
        swap_info(cm_id, mr, &rkey, &remote_addr, NULL, log_p);
        // The real good
        uint32_t opcode;
        while(1){
                rdma_recv(cm_id, mr, log_p);
                opcode = get_completion(cm_id, RECV, 1, log_p);
                if(opcode == DISCONNECT){
                        fprintf(log_p, "Client issued a disconnect.\n");
                        rdma_send_op(cm_id, 0, log_p);
                        get_completion(cm_id, SEND, 1, log_p);
                        break;
                } else if (opcode == OPEN_MR){
                        remote_add(pthread_self());
                        set_status(pthread_self(), OPEN);
                } else if (opcode == CLOSE_MR){
                        remote_remove(pthread_self());
                        set_status(pthread_self(), CLOSED);
                }
        }
        // Disconnect and remove client from lists
        remove_thread(pthread_self());
        remote_remove(pthread_self());
        remove_client(pthread_self());
        obliterate(NULL, cm_id, mr, cm_id->channel, log_p);
        return 0;
}
```

### server

```
[root@centos7 rdma_v2]# ./server 9999
---------------
| RDMA server |
------------------------------------------
1) Shut down                             |
2) View connected clients                |
3) Disconnect a client                   |
4) Shut down when all clients disconnect |
> 2
---Client id: 1
Client MR length: 1024 bytes
Client MR status: open
---------------
| RDMA server |
------------------------------------------
1) Shut down                             |
2) View connected clients                |
3) Disconnect a client                   |
4) Shut down when all clients disconnect |
> 
```

### client

```
root@ubuntu:~/the-geek-in-the-corner# ./client 10.11.11.251 9999
Connecting...
Send completed successfully!
Sent local address: 0x2ad9bc00
Sent local rkey: 0x758
Receive completed successfully!
30 bytes received.
Received remote address: 0xffff900014f0
Received remote rkey: 0x62c
Received remote memory region length: 1024 bytes
----------------------
| RDMA client        |
----------------------
1) Disconnect        |
2) Write inline      |
3) Write             |
4) Read              |
5) Open Server MR    |
6) Close server MR   |
> 4
Would you like to print the data to console or write to a file? (p for print, w for write)
> p
Server memory region is 1024 bytes long. Choosing a relative point (0 - 1023) to start reading from, followed by how many bytes you wish to read (0-512).
> 0 256
> Read completed successfully!
Data: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 
----------------------
| RDMA client        |
----------------------
1) Disconnect        |
2) Write inline      |
3) Write             |
4) Read              |
5) Open Server MR    |
6) Close server MR   |
> 4
Would you like to print the data to console or write to a file? (p for print, w for write)
> p
Server memory region is 1024 bytes long. Choosing a relative point (0 - 1023) to start reading from, followed by how many bytes you wish to read (0-512).
> 0 256
> Read completed successfully!
Data: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 
----------------------
| RDMA client        |
----------------------
1) Disconnect        |
2) Write inline      |
3) Write             |
4) Read              |
5) Open Server MR    |
6) Close server MR   |
> 
```

---
## RDMA kernel module

This kernel module is a very messy test. I was trying to learn to how make kernel modules and interact with the drivers. The main accomplishment to come out of that is my comprehension of how the fast-reg memory system works. It was built on centos 6.8 with Mellanox OFED 3.3 drivers and kernel version 3.10.87. 

[Click here to download my write-up on the fast-reg memory system](https://github.com/pixelrazor/rdma-cs/raw/master/rdma%20kernel%20module%20test/Fast.pdf)

To build it, you must copy the header files ib_verbs.h and rdma_cm.h from /usr/src/ofa_kernel/default/include/rdma to your kernel's build directory (in my case /lib/modules/3.10.87/build/include/rdma/). You also need to change the ip in the init function to the one that your server will have. From there, on two seperate machines, build the module with make. Then, on the server, run 'make server'. Finally, on the client, run 'make client'. Both sides should return -1, but this should be normal. Check the system log with dmesg to see if it ran correctly or ran into an error.
