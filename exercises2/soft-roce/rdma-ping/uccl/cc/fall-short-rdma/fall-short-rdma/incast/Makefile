CFLAGS  := -O3 -g 
LD      := gcc
LDFLAGS := ${LDFLAGS} -lrdmacm -libverbs -lrt -lpthread -lm


all: incast_server incast_client 

incast_server: incast_server.o ../util.o csapp.o
	${LD} -o $@ $^ ${LDFLAGS}
incast_client: incast_client.o ../util.o csapp.o 
	${LD} -o $@ $^ ${LDFLAGS}


	
PHONY: clean
clean:
	rm -f *.o ../util.o incast_server incast_client 
