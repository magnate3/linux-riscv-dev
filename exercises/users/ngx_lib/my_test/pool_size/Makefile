#
# 'make depend' uses makedepend to automatically generate dependencies 
#               (dependencies are added to end of Makefile)
# 'make'        build executable file 'mycc'
# 'make clean'  removes all .o and executable files
#

# define the C compiler to use
CC = gcc

# define any compile-time flags
#CFLAGS = -Wall -g
CFLAGS = -Wall -g -DNGX_DEBUG=1

# define any directories containing header files other than /usr/include
#
#INCLUDES = -I/home/newhall/include  -I../include
IDIR = ../src
INCLUDES = -I$(IDIR)/core/  -I$(IDIR)/os/unix/   -I$(IDIR)/auto 

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
SDIR = ../src
SRCS =  $(SDIR)/os/unix/ngx_errno.c\
        $(SDIR)/core/ngx_log.c  $(SDIR)/core/ngx_string.c $(SDIR)/os/unix/ngx_alloc.c $(SDIR)/core/ngx_palloc.c pool_size.c

# define the C object files 
#
# This uses Suffix Replacement within a macro:
#   $(name:string1=string2)
#         For each word in 'name' replace 'string1' with 'string2'
# Below we are replacing the suffix .c of all words in the macro SRCS
# with the .o suffix
#
OBJS = $(SRCS:.c=.o)

# define the executable file 
MAIN = mycc
#
# The following part of the makefile is generic; it can be used to 
# build any executable just by changing the definitions above and by
# deleting dependencies appended to the file from 'make depend'
#

.PHONY: depend clean

all:    $(MAIN)
	@echo  Simple compiler named mycc has been compiled

$(MAIN): $(OBJS) 
	$(CC) $(CFLAGS) $(INCLUDES) -o $(MAIN) $(OBJS) $(LFLAGS) $(LIBS)

# this is a suffix replacement rule for building .o's from .c's
# it uses automatic variables $<: the name of the prerequisite of
# the rule(a .c file) and $@: the name of the target of the rule (a .o file) 
# (see the gnu make manual section about automatic variables)
.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $<  -o $@

clean:
	$(RM) *.o *~ $(MAIN)  $(SDIR)/core/*.o  $(SDIR)/os/unix/*.o
	find ../src -name '*.o' -exec $(RM) '{}' \;
depend: $(SRCS)
	makedepend $(INCLUDES) $^

# DO NOT DELETE THIS LINE -- make depend needs it
