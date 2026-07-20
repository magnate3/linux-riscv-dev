.PHONY: clean
#CFLAGS := -g -O2 -Werror -g
CFLAGS := -Wall  -g -O2
CFLAGS += -I ./ccan  -I ./
LD := gcc
LDFLAGS := ${LDFLAGS} -libverbs
APPS := rc_pingpong
all: ${APPS}
rc_pingpong:  pingpong.o  rc_pingpong.o
	${LD} -o $@ $^ ${LDFLAGS}
clean:
	rm -f *.o ${APPS}
