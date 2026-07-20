all: rc_pingpong.c pingpong.c pingpong.h
	gcc -o mp-rdma rc_pingpong.c pingpong.c -libverbs -D_GNU_SOURCE -w
clean:
	rm mp-rdma

default: all
