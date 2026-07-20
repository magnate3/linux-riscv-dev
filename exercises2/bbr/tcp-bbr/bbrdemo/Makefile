LDLIBS = -lrt
CFLAGS = -g

all: bbr_sender bbr_receiver

bbr_sender: bbr_sender.o tcp_bbr.o

bbr_receiver: bbr_receiver.o
clean:
	rm bbr_sender bbr_receiver
