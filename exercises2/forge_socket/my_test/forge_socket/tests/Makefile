CFLAGS=-I../

targets=test bind-test nc client_win client_win2 client_win3 server_rcv_win

all: $(targets)

nc: nc.c libforge_socket.o
	$(CC) $(CFLAGS) -o $@ $^
clean:
	rm -rf *.o 
	rm -rf $(targets) 
