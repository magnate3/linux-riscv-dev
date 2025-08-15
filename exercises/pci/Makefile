
TARGET = pci_uio

OUTPATH = ../out/obj

SRCFILES = \
	main.c \
	pci_common.c \
	common_string_fns.c 

OBJFILES := $(patsubst %.c, %.o, $(SRCFILES))

LIBNAME =
LIBPATH =

CFLAGS +=  -g -Wall -O2
LDLIBS +=

%.o:%.c
	$(CC) -c $(CFLAGS) $< ${LDLIBS} -o $@

${TARGET}:$(OBJFILES)
#	$(MAKE) -C $(LIBPATH)
	${CC} $^   ${LDLIBS} -o $@

all:

.PHONY:clean
clean:
#	$(MAKE) -C $(LIBPATH) clean
	rm -rf ${OBJFILES} ${TARGET}

