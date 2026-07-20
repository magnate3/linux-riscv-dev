objects = rte_ring_main.o rte_ring.o
CC = gcc
platform = x86
#platform = arch64
ifeq ($(platform),x86)
CFLAGS += -I./ -I./common/include/arch/ -I ./common/include/arch/x86 -I./common/include/  -DRTE_ARCH_X86 -D_RTE_ATOMIC_X86_H
else
CFLAGS += -I./ -I./common/include/arch/ -I ./common/include/arch/arm -I./common/include/  -DRTE_ARCH_64 -DRTE_FORCE_INTRINSICS=1
endif
LDFLAGS += -lbsd  -lpthread
MAIN = test
all: $(objects)
	$(CC) -g -o $(MAIN) $(objects) $(LDFLAGS)

$(objects): %.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -f $(MAIN) *.o
