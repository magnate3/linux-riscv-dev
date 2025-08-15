SRC = 	main.cpp
CC = g++
# -fno-omit-frame-pointer allows for better profiling through `perf`
CFLAGS = -O3 -g -fno-omit-frame-pointer
CPP_FLAGS += '-std=c++11'
EXEC = test
OBJS := $(SRC:%.cpp=%.o)
OBJS := $(OBJS:%.S=%.o)
LIBS = -lm -lpthread

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) ${CPP_FLAGS} $(OBJS) $(LIBS) -o $(EXEC)

%.o : %.S
	$(CC) ${CPP_FLAGS} $< $(LIBS) -c -o $@

.PHONY: clean
clean:
	-rm $(OBJS)
	-rm $(EXEC)
