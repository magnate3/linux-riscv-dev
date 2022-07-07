CC = gcc
CFLAGS = -Wall -g -O2
CFLAGS += -I../include

SAMPLE = nl_echo genl_echo

all:$(SAMPLE)

nl_echo: nl_echo.c
	$(CC) $(CFLAGS) nl_echo.c -o nl_echo

genl_echo: genl_echo.c
	$(CC) $(CFLAGS) genl_echo.c -o genl_echo

.PHONY:clean
clean:
	rm -f $(SAMPLE)
