# -fgnu89-inline
CFLAGS  := -O3 -Wall -Werror -Wno-unused-result -Wno-address-of-packed-member
LD      := gcc
LDFLAGS := ${LDFLAGS} -lrdmacm -libverbs -lrt -lpthread

main: common.o conn.o main.o
	${LD} -o $@ $^ ${LDFLAGS}

PHONY: clean
clean:
	rm -f *.o main
