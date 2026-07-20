CC = gcc
LLC ?= llc
CLANG = clang
CLANG_FLAGS := -D__KERNEL__ -D__BPF_TRACING__ 
CLANG_FLAGS += -Wno-unused-value -Wno-pointer-sign       -D__TARGET_ARCH_x86 -Wno-compare-distinct-pointer-types \
       	       -Wno-gnu-variable-sized-type-not-at-end   -Wno-address-of-packed-member -Wno-tautological-compare \
       	       -Wno-unknown-warning-option         
CLANG_FLAGS +=  -O2 -emit-llvm  -o -| llc -march=bpf -filetype=obj
#CLANG_FLAGS +=  -O2 -emit-llvm -c samples/bpf/xdp_demo2_kern.c -o -| llc -march=bpf -filetype=obj

MAIN = xdpsock

KERN_TARGETS =  xdpsock_kern
USER_TARGETS := xdpsock_user


KERNEL_SOURCE = /root/linux-4.18.9

BPFTOOLS = $(KERNEL_SOURCE)/samples/bpf
BPFLOADER = $(BPFTOOLS)/bpf_load.c
USER_SRCS = BPFLOADER {KERN_TARGETS}.c
BPFLOADER_OBJ = bpf_load.o
BPFLOADER_LINK_OBJ = $(PWD)/bpf_load.o
DEMO_OBJ = ${USER_TARGETS}.o
KBUILD_CFLAGS := -nostdinc -isystem /usr/lib/gcc/x86_64-linux-gnu/8/include

UBUILD_CFLAGS := -I$(KERNEL_SOURCE)/usr/include
UBUILD_CFLAGS += -I/opt/bpf/usr/local/include/bpf
UBUILD_CFLAGS += -I$(KERNEL_SOURCE)/samples/bpf
UBUILD_CFLAGS += -I$(KERNEL_SOURCE)/tools/lib/bpf/
UBUILD_CFLAGS += -I$(KERNEL_SOURCE)/tools/lib/
UBUILD_CFLAGS += -I$(KERNEL_SOURCE)/tools/include
UBUILD_CFLAGS += -I$(KERNEL_SOURCE)/tools/perf
UBUILD_CFLAGS += -I$(KERNEL_SOURCE)/tools/testing/selftests/bpf/
#UBUILD_CFLAGS += -I./

KBUILD_CFLAGS += -I$(KERNEL_SOURCE)/tools/testing/selftests/bpf/
KBUILD_CFLAGS += -I$(KERNEL_SOURCE)/arch/x86/include 
KBUILD_CFLAGS += -I$(KERNEL_SOURCE)/arch/x86/include/generated  
KBUILD_CFLAGS += -I$(KERNEL_SOURCE)/include 
KBUILD_CFLAGS += -I$(KERNEL_SOURCE)/arch/x86/include/uapi
KBUILD_CFLAGS += -I$(KERNEL_SOURCE)/arch/x86/include/generated/uapi 
KBUILD_CFLAGS += -I$(KERNEL_SOURCE)/include/uapi 
KBUILD_CFLAGS += -I$(KERNEL_SOURCE)/include/generated/uapi -include $(KERNEL_SOURCE)/include/linux/kconfig.h
KBUILD_CFLAGS += -I$(KERNEL_SOURCE)/samples/bpf

#LIBRARY_PATH += -L$(KERNEL_SOURCE)/tools/lib/bpf
BPF_LIBRARY_PATH := -L/opt/bpf/usr/local/lib64
BPF_STATIC_LIB := /opt/bpf/usr/local/lib64/libbpf.a

BPFSO = -lbpf

# Setting -DHAVE_ATTR_TEST=0 for the kernel containing below patch:
# 06f84d1989b7 perf tools: Make usage of test_attr__* optional for perf-sys.h
#
# The patch was included in Linus's tree starting v5.5-rc1, but was also included
# in stable kernel branch linux-5.4.y. So it's hard to determine whether a kernel
# is affected based on the kernel version alone:
# - for a v5.4 kernel from Linus's tree, no;
# - for a v5.4 kernel from the stable tree (used by many distros), yes.
#
# So let's look at the actual kernel source code to decide.
#
# See more context at:
# https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id=06f84d1989b7e58d56fa2e448664585749d41221
# https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id=fce9501aec6bdda45ef3a5e365a5e0de7de7fe2d
CFLAGS += $(shell grep -q "define HAVE_ATTR_TEST 1" $(KERNEL_SOURCE)/tools/perf/perf-sys.h \
                  && echo "-DHAVE_ATTR_TEST=0")

.PHONY: clean $(CLANG) bpfload build 

clean:
	rm -f *.o *.so $(MAIN)

build: ${KERN_TARGETS.c} ${BPFLOADER}
	$(CLANG)   -c $(KERN_TARGETS:=.c) $(KBUILD_CFLAGS) $(CLANG_FLAGS) -o ${KERN_TARGETS:=.o}
bpfload: build ${DEMO_OBJ}
	$(CC)   -o $(MAIN)  ${DEMO_OBJ} ${BPFLOADER_OBJ} -lelf -lpthread $(BPF_STATIC_LIB)
	#$(CC)   -o $(MAIN)  ${DEMO_OBJ} ${BPFLOADER_LINK_OBJ} -lelf $(BPF_STATIC_LIB)
${DEMO_OBJ}: ${BPFLOADER_OBJ}
	$(CC) -c  -o $(DEMO_OBJ)  $(UBUILD_CFLAGS) $(BPF_LIBRARY_PATH)  ${USER_TARGETS}.c
${BPFLOADER_OBJ}: 
	$(CC) -c   -o $(BPFLOADER_OBJ)  $(UBUILD_CFLAGS) $(BPFLOADER)


$(MAIN): bpfload

.DEFAULT_GOAL := $(MAIN)

run: bpfload
	export LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(KERNEL_SOURCE)/tools/lib/bpf; ./hello
