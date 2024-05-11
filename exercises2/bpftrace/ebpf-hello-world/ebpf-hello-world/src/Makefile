CLANG=clang
all: build
.PHONY: all
HELP_DIR=../libbpf/src
CLANG_FLAG=-I$(HELP_DIR) 
LIB_FLAG=-L../libbpf/build/libbpf

build:
	uname -a
	mkdir -p $@
	bpftool btf dump file /sys/kernel/btf/vmlinux format c > $@/vmlinux.h
	$(CLANG) -g -O2 -Wall -Wextra -target bpf -D__TARGET_ARCH_x86_64i -I $@ $(CLANG_FLAG) -c  hello.bpf.c -o $@/hello.bpf.o
	bpftool gen skeleton $@/hello.bpf.o > $@/hello.skel.h
	$(CLANG) -g -O2 -Wall -Wextra -I $@ -c hello.c -o build/hello.o
	$(CLANG) -g -O2 -Wall -Wextra $@/hello.o $(LIB_FLAG) -lbpf -lelf -lz -o $@/hello
.PHONY: build

test:
	sudo ./build/tracepoint -t

clean:
	rm -rf build deps

.PHONY: clean
