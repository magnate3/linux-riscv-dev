.PHONY: clean
#CFLAGS := -g -O2 -Werror -g
CFLAGS := -Wall  -g -O2
CFLAGS += -I ./ccan  -I ./
LD := gcc
LDFLAGS := ${LDFLAGS} -libverbs
APPS := srq_pingpong
all: ${APPS}
srq_pingpong:  pingpong.o  srq_pingpong.o
	${LD} -o $@ $^ ${LDFLAGS}
clean:
	rm -f *.o ${APPS}
