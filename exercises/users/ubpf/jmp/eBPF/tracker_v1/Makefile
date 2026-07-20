# -*- Makefile -*-

XDP_PROG=xdp_network_tracker.o

CLANG=clang
DEV=ens33

CFLAGS=-g -O3 -I../libbpf/src #-Wall -Werror
LDFLAGS=-L../libbpf/src
LIBS=-lbpf -lelf -lz

all: $(XDP_PROG) xdp_network_tracker_user

$(XDP_PROG): xdp_network_tracker.c
	$(CLANG) -target bpf $(CFLAGS) -c -o $@ $<

xdp_network_tracker_user: xdp_network_tracker_user.c
	$(CLANG) -static $(CFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)


clean: unmount
	rm -f $(XDP_PROG) xdp_network_tracker_user

unmount:
	umount /sys/fs/bpf/ 2>/dev/null || true
	@rm -f sys_fs_bpf_mounted

sys_fs_bpf_mounted:
	mount -t bpf none /sys/fs/bpf
	@touch sys_fs_bpf_mounted

map_illegal_domains: sys_fs_bpf_mounted
	bpftool map create /sys/fs/bpf/map_illegal_domains type hash \
		name map_illegal_domains key 4 value 4 entries 2 flags 1
query: sys_fs_bpf_mounted
	bpftool map create /sys/fs/bpf/query type hash \
		name query key 4 value 56 entries 100 flags 1
map_blocked_requesters: sys_fs_bpf_mounted
	bpftool map create /sys/fs/bpf/map_blocked_requesters type hash \
		name map_blocked_requesters key 4 value 8 entries 1000

dump: $(XDP_PROG)
	llvm-objdump -S $(XDP_PROG)

load: $(XDP_PROG) map_illegal_domains query
	./xdp_network_tracker_user

unload:
	ip link set dev $(DEV) xdpgeneric off

show:
	ip link show dev $(DEV)
