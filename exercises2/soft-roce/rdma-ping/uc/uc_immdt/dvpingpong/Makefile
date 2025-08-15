CC := gcc
LD := gcc
CFLAGS := -Wall -g -O0 -I.

LIBS :=  -libverbs -lmlx5  -lpthread

HEADERS := pp_common.h
OBJS_VERB := sock.o pp_common.o pp_verb.o
OBJS_DV := sock.o pp_common.o pp_dv.o

all: server client

server: server.o $(OBJS_VERB)
	$(LD) -o $@ $^ $(LIBS)

client: client.o $(OBJS_VERB)
	$(LD) -o $@ $^ $(LIBS)


%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o server client
	#rm -f *.o server client.verb client.dv client.vfio
