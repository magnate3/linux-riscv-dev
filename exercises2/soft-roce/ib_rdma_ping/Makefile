CFLAGS += -g 
CC = gcc
BIN = rping_test
OBJS = rcommon.o rserver.o rclient.o rping.o

all:	$(BIN)

rping_test:	$(OBJS)
	gcc $(OBJS) -g -o rping_test -lrdmacm -libverbs -lpthread
# Build step for C source
%.c.o: %.c
	$(CC)  $(CFLAGS) -c $< -o $@
#rcommon.o: rcommon.c 
#	$(CC) $(CFLAGS) -c rclient.c 
#rclient.o: rclient.c 
#	$(CC) $(CFLAGS) -c rclient.c 
#rserver.o: rserver.c 
#	$(CC) $(CFLAGS) -c rserver.c 
#rping.o: rping.c 
#	$(CC) $(CFLAGS) -c rserver.c 

clean:
	rm -f *.o $(BIN)
