UCX_PATH=/root/rdma-bench/ucx/build/
CUDA_PATH=/usr/local/cuda-11.7/
build: ucp_hello_world uct_hello_world ucp_client_server 
#build: ucp_hello_world uct_hello_world ucp_client_server ucp_example

ucp_hello_world: ucp_hello_world.c
	gcc -L$(UCX_PATH)/lib -I$(UCX_PATH)/include   -o ucp_hello_world ucp_hello_world.c -lucp -lucs -luct

uct_hello_world: uct_hello_world.c
	gcc -L$(UCX_PATH)/lib -I$(UCX_PATH)/include   -o uct_hello_world uct_hello_world.c -lucp -lucs -luct

ucp_client_server: ucp_client_server.c
	gcc -L$(UCX_PATH)/lib -I$(UCX_PATH)/include   -o ucp_client_server ucp_client_server.c -lucp -lucs -luct

ucp_example: ucp_example.cpp
	g++ -L$(UCX_PATH)/lib -I$(UCX_PATH)/include  -o ucp_example ucp_example.cpp -lucp -lucs -luct
clean:
	rm -f *.o ucp_client_server ucp_example uct_hello_world ucp_hello_world
