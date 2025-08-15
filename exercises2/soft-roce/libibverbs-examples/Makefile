.PHONY: clean
CFLAGS := -Wall  -g -O2
LD := gcc
LDFLAGS := ${LDFLAGS} -libverbs
APPS := ibv_devices
APPS += ibv_context
all: ${APPS}
ibv_devices: device_list.o
	${LD} -o $@ $^ ${LDFLAGS}
ibv_context: context_test.o
	${LD} -o $@ $^ ${LDFLAGS}
clean:
	rm -f *.o ${APPS}
