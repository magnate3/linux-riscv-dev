**A simple example to demonstrate how to write a rdma app over:**

    verbs APIs,
    mlx5dv APIs, and
    mlx5dv APIs over VFIO.

# How to run
<pre>
* In server side run: ./server [ib_devname]
* In client side run: ./client.verb [ib_devname] # Client based on verbs API
  or:
  ./client.dv [ib_devname] # Client based on dv API
  or:
  ./client.vfio [pci_name] # Client side based on vfio driver
</pre>

# Note
+ For server there's only 1 version, which is based on verbs APIs;
+ These are hardcoded:
  client side:
  + SERVER_IP
  + client_sgid_idx (in client.c);
  + dmac (for RoCE; in pp_dv.c);
  + max_wr_num (number of wr posted; in client..c; Default is PP_MAX_WR);

  server side:
  + server_sgid_idx; max_wr_num(max same as the value in client side, need to improve);

  For max_wr_num not have to be changed; e.g., for debugging you may want to change it to 1.

+ There's no wq overflow check.

<pre>
--
Mark Zhang markzhang@nvidia.com
</pre>

