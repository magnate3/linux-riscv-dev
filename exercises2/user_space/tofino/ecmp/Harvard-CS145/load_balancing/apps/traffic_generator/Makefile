all: traffic_sender traffic_receiver

traffic_sender: traffic_sender.cpp trace.hpp
	g++ -o traffic_sender traffic_sender.cpp --std=c++11 -lgflags -lpthread

traffic_receiver: traffic_receiver.cpp trace.hpp
	g++ -o traffic_receiver traffic_receiver.cpp --std=c++11 -lgflags -lpthread