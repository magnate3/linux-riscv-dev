.PHONY: clean

CC := g++
CFLAGS  := -Wall -g  -std=c++11
LDLIBS  := ${LDLIBS} -lrdmacm -libverbs -lpthread

APPS    := server client

all: ${APPS}


clean:
	rm -f ${APPS}

