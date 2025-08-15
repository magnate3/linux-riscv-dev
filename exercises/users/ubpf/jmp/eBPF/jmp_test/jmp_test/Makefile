# -*- Makefile -*-

XDP_PROG=xdp_kern.o

CLANG=clang
DEV=enx00e04c3662aa

CFLAGS=-g -O3 -I../libbpf/src #-Wall -Werror
LDFLAGS=-L../libbpf/src
LIBS=-lbpf -lelf -lz

all: $(XDP_PROG) xdp_user

$(XDP_PROG): xdp_kern.c
	$(CLANG) -target bpf $(CFLAGS) -c -o $@ $<

xdp_user: xdp_user.c
	$(CLANG) -static $(CFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)


clean: 
	rm -f $(XDP_PROG) xdp_user


dump: $(XDP_PROG)
	llvm-objdump -S $(XDP_PROG)

load: $(XDP_PROG) 
	./xdp_user

unload:
	ip link set dev $(DEV) xdpgeneric off

show:
	ip link show dev $(DEV)
