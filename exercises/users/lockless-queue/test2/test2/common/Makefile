#
# 'make depend' uses makedepend to automatically generate dependencies 
#               (dependencies are added to end of Makefile)
# 'make'        build executable file 'mycc'
# 'make clean'  removes all .o and executable files
#

# define the C compiler to use
CC = gcc

# define any compile-time flags
CFLAGS = -Wall -g

# define any directories containing header files other than /usr/include
#
#INCLUDES = -I/home/newhall/include  -I../include
ROOT  = ..
IDIR1 = $(ROOT)/lfqueue
IDIR2 = $(ROOT)/ngx_slab/head
#IDIR12 = ./ngx_slab
INCLUDES = -I$(IDIR1) -I$(IDIR2)

# define library paths in addition to /usr/lib
#   if I wanted to include libraries not in /usr/lib I'd specify
#   their path using -Lpath, something like:
#LFLAGS = -L/home/newhall/lib  -L../lib

# define any libraries to link into executable:
#   if I want to link in libraries (libx.so or libx.a) I use the -llibname 
#   option, something like (this will link in libmylib.so and libm.so:
#LIBS = -lmylib -lm

# define the C source files
#SRCS = emitter.c error.c init.c lexer.c main.c symbol.c parser.c

SDIR1 = $(ROOT)/lfqueue
SDIR2 = $(ROOT)/ngx_slab/src
#SDIR2 = ./ngx_slab
SRCS = util.c  \
	$(SDIR1)/lfqueue.c\
       	$(SDIR2)/ngx_alloc.c   $(SDIR2)/ngx_palloc.c   $(SDIR2)/ngx_shmem.c   $(SDIR2)/ngx_shmtx.c   $(SDIR2)/ngx_slab.c

#SDIR3 = ./common
#SDIR3 = ./
#SRCS2 =  $(SDIR3)/util.c 
#SRCS2 =  $(SDIR3)/util.c \
#       	$(SDIR2)/ngx_alloc.c   $(SDIR2)/ngx_palloc.c   $(SDIR2)/ngx_shmem.c   $(SDIR2)/ngx_shmtx.c   $(SDIR2)/ngx_slab.c

CFLAGS +=  -D LOG_LEVEL=4 -DNGX_HAVE_ATOMIC_OPS -DDEBUG_MULTIPLE_PROCS=2 -DNGX_PTR_SIZE=8  -fPIC -shared
# define the C object files 
#
# This uses Suffix Replacement within a macro:
#   $(name:string1=string2)
#         For each word in 'name' replace 'string1' with 'string2'
# Below we are replacing the suffix .c of all words in the macro SRCS
# with the .o suffix
#
OBJS = $(SRCS:.c=.o)
#OBJS2 = $(SRCS2:.c=.o)

# define the executable file 
MAIN = mycc
MYLIB = libmy.so
#
# The following part of the makefile is generic; it can be used to 
# build any executable just by changing the definitions above and by
# deleting dependencies appended to the file from 'make depend'
#
LIBS += -lpthread
.PHONY: depend clean

#all:    $(MAIN) 
all:    $(MYLIB)
	@echo  Simple compiler named libmy has been compiled

#$(MAIN): $(OBJS) 
#	$(CC) $(CFLAGS) $(INCLUDES) -o $(MAIN) $(OBJS) $(LFLAGS) $(LIBS)
$(MYLIB): $(OBJS) 
	$(CC) $(CFLAGS) $(INCLUDES) -o $(MYLIB) $(OBJS) $(LFLAGS) $(LIBS)
# this is a suffix replacement rule for building .o's from .c's
# it uses automatic variables $<: the name of the prerequisite of
# the rule(a .c file) and $@: the name of the target of the rule (a .o file) 
# (see the gnu make manual section about automatic variables)
.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $<  -o $@

clean:
	$(RM) *.o *~ $(MYLIB)  $(SDIR1)/*.o  $(SDIR2)/*.o $(SDIR2)/test
	find ./ -name "*.o" | xargs rm -rf
	#rm /var/run/test-ngx
depend: $(SRCS)
	makedepend $(INCLUDES) $^

# DO NOT DELETE THIS LINE -- make depend needs it
