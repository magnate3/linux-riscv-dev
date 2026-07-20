ALL = client server

CC=gcc

CFLAGS= -Wall -g 
LIBS= -libverbs 
EXTRA_LIBS = -lrdmacm -lpthread

all: $(ALL)

client: rdma_cs.c client.c 
	$(CC) -g -o $@ $^ $(CFLAGS) $(LIBS) $(EXTRA_LIBS)

server: rdma_cs.c server.c
	$(CC) -g -o $@ $^ $(CFLAGS) $(LIBS) $(EXTRA_LIBS) 

backup: 
	cp --backup=t server.c server.c.backup
	cp --backup=t client.c client.c.backup
	cp --backup=t rdma_cs.c rdma_cs.c.backup

clean:
	rm -f $(ALL)

clear_backups:
	rm *.c.backup.*

clear_logs:
	rm -f server_logs/*
