LDFLAGS=-libverbs -lrdmacm -lpthread
CFLAGS=-g -Wall -std=gnu99

all: server client

server: server.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

client: client.c
	$(CC) $(CFLAGS) $<  -o $@ $(LDFLAGS)

.PHONY: clean
clean:
	rm -rf *.o server client
