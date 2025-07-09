Test name: 
    rdma_mt_send_lat - rdma multi-thread sample code.
 
Author(s): 
    Changqing Li <licq@mellanox.com>
 
Short description:

Dependencies:
    MLNX_OFED-2.0-3.0.0.0
 
Supported OSes: 
    Linux
 
Description:

USAGE - start the server

USAGE - start the client

Known issues:
	+ memory leak on server side.
	+ 

Todo:
	+ SW Arch redesign to decouple of each modules.
	+ Improve APIs of RDMA operations.
	+ Clean data structure(rdma_buf_t).
	+ Add epoll support.
	+ bind thread to CPU.

Commands to start run the testing:

	Common test:
		Server:
			make clean; make all; echo ""; taskset -c 1 ./rdma_mt_send_lat -d mlx4_0 -i 2 -G 1

		Client:
			scp root@s6://licq/work/rdma_sample/2_rdma_mt_lat/src/* ./; echo ""; taskset -c 1 ./rdma_mt_send_lat -d mlx4_0 -i 2 -G 2 -n 1000 -o 2 -t 2 192.168.2.6

	Client -> Server testing (-D 1)
		Server:
			make clean; make all; echo ""; taskset -c 1 ./rdma_mt_send_lat -d mlx4_0 -i 2 -G 1

		Client:
			scp root@s6://licq/work/rdma_sample/2_rdma_mt_lat/src/* ./; echo ""; taskset -c 1 ./rdma_mt_send_lat -d mlx4_0 -i 2 -G 2 -n 1000 -o 2 -t 2 -D 1 192.168.2.6

	Multi-thread testing (-T n)
		client: 
		    scp root@s6://licq/work/rdma_sample/2_rdma_mt_lat/src/* ./; echo ""; taskset -c 1 ./rdma_mt_send_lat -d mlx4_0 -i 2 -G 2 -n 1000 -o 2 -t 2 -D 0 -T 2 192.168.2.6

Revision Description:
	0.1.01: single thread works. Perfromance is good.
		The number of Lat <    1us is   0 
		The number of Lat <    2us is   0 
		The number of Lat <    4us is 736 
		The number of Lat <    8us is 258 
		The number of Lat <   16us is   5 
		The number of Lat <   32us is   0 
		The number of Lat <   64us is   0 
		The number of Lat <  128us is   0 
		The number of Lat <  256us is   1 
		The number of Lat <  512us is   0 
	