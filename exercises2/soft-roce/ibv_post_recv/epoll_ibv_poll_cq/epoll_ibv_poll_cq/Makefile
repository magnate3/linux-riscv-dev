override CFLAGS += -g

BIN = smbd_server smbd_client
OBJS = utils.o event_loop.o

all:	$(BIN)

smbd_server:	smbd_server.o $(OBJS)
	gcc -g -o smbd_server smbd_server.o $(OBJS) -l rdmacm -l ibverbs

smbd_client:	smbd_client.o $(OBJS)
	gcc -g -o smbd_client smbd_client.o $(OBJS) -l rdmacm -l ibverbs

clean:
	rm -f *.o $(BIN)
