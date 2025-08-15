SRC = 	rng.c \
        calqueue.c \
        pcs.c \
        reverse-generator.c \
        scheduler.c

# -fno-omit-frame-pointer allows for better profiling through `perf`
CFLAGS = -O3 -g -fno-omit-frame-pointer -D TARGET_ARM64
EXEC = test
OBJS := $(SRC:%.c=%.o)
OBJS := $(OBJS:%.S=%.o)
LIBS = -lm -lpthread

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(OBJS) $(LIBS) -o $(EXEC)

%.o : %.S
	$(CC) $< $(LIBS) -c -o $@

.PHONY: clean
clean:
	-rm $(OBJS)
	-rm $(EXEC)
