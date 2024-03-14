APP = oceanus

ifeq ($(RTE_SDK),)
$(error "Please define RTE_SDK environment variable")
endif
RTE_TARGET ?=  arm64-armv8a-linuxapp-gcc
DPDK_LIB=$(RTE_SDK)/${RTE_TARGET}
SRCS := $(wildcard *.c)
LIBOCEANUS := oceanus
OCEANUS_DIR := ..
LIBOCEANUS_DIR := $(OCEANUS_DIR)/build
LDFLAGS := -L$(LIBOCEANUS_DIR) -l$(LIBOCEANUS)
CC := gcc
CFLAGS := -O3 -I$(OCEANUS_DIR)/include
CFLAGS +=  -I${DPDK_LIB}/include 
LDFLAGS +=  -L ${DPDK_LIB}/lib -ldpdk -lpthread -lnuma -lrt -lm -ldl 
#-lrte_timer  -lrte_sched -lrte_eal
PHONY := all
all: $(APP)

$(APP): $(SRCS) $(LIBOCEANUS_DIR)/lib$(LIBOCEANUS).so
	$(CC) $(CFLAGS) $(SRCS) -o $@ $(LDFLAGS)

$(LIBOCEANUS_DIR)/lib$(LIBOCEANUS).so: FORCE
	@make -C $(OCEANUS_DIR)

PHONY += FORCE
FORCE:;

PHONY += run
run: $(APP)
	export LD_LIBRARY_PATH=$(shell pwd)/$(LIBOCEANUS_DIR):$(LD_LIBRARY_PATH) && ./oceanus

PHONY += clean
clean:
	rm -f $(APP)

PHONY += distclean
distclean:
	rm -f $(APP)
	make -C $(OCEANUS_DIR) distclean

.PHONY: $(PHONY)
