MODULE_NAME1 := rdma_rxe
obj-m := $(MODULE_NAME1).o
#obj-m += $(MODULE_NAME2).o
OBJ_LIST1 := \
	rxe.o \
	rxe_comp.o \
	rxe_req.o \
	rxe_resp.o \
	rxe_recv.o \
	rxe_pool.o \
	rxe_queue.o \
	rxe_verbs.o \
	rxe_av.o \
	rxe_srq.o \
	rxe_qp.o \
	rxe_cq.o \
	rxe_mr.o \
	rxe_opcode.o \
	rxe_mmap.o \
	rxe_icrc.o \
	rxe_mcast.o \
	rxe_task.o \
	rxe_net.o \
	rxe_sysfs.o \
	rxe_hw_counters.o
$(MODULE_NAME1)-y := $(OBJ_LIST1)

ccflags-y := -O2
ccflags-y                               += -I$(src)
KERNELDIR := /lib/modules/$(shell uname -r)/build

all: nvmedriver

nvmedriver:
	make -C $(KERNELDIR) M=$(PWD) modules
clean:
	make -C $(KERNELDIR) M=$(PWD) clean
