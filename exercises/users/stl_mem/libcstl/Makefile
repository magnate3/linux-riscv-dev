# Makefile

RM = rm -rf
CC = gcc
SRC := $(wildcard *.c)
OBJS := $(patsubst %.c,%.o,$(wildcard *.c))
CFLAGS = -pipe -Wall -shared -fPIC -O3
CFLAGS_DBG = -pipe -Wall -shared -fPIC -g
INCLUDE_DIRS = -I. -I./bits
DLL_LINK_FLAGS = -shared -s
DLL_LIB = libcstl.so
SRC_HEADERS := $(wildcard *.h)
BITS_DIR := $(wildcard bits/*.h)
DIST_SOURCES = $(SRC) $(SRC_HEADERS) $(BITS_DIR) Makefile
DIST_TARGET = libcstl.tar.bz2
INSTALL_HDR_DIR = $(HOME)/include/wan/.
INSTALL_LIB_DIR = $(HOME)/lib/.

all : $(OBJS)
	$(CC) $(DLL_LINK_FLAGS) -o $(DLL_LIB) $(OBJS)

%.o : %.c
	$(CC) $(CFLAGS) $(INCLUDE_DIRS) -c -o $@ $<

clean:
	$(RM) *.o

dist:
	$(RM) $(DIST_TARGET)
	tar jcvf $(DIST_TARGET) $(DIST_SOURCES)

install:
	cp -f libcstl.so $(INSTALL_LIB_DIR)
	cp -f *.h $(INSTALL_HDR_DIR)
