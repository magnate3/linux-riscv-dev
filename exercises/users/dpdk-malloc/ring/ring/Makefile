################################################################# 
# A demo Makefile 
# By xbb (http://xnzaa.github.io) 
# bugs: 
#   1. 需要显式指定各个编译目录的位置； 
################################################################# 
CC=gcc
DEBUG = y
RM = rm -rf
MAKE = make

DIR_INC = ./arm64-armv8a-include
#DIR_SRC_BAR = ./bar
DIR_LIB_EAL = ./lib/librte_eal
DIR_SRC_COMMON = ${DIR_LIB_EAL}/common
DIR_SRC_EAL = ${DIR_LIB_EAL}/linux/eal
DIR_SRC_MEMPOOL = ./lib/librte_mempool
DIR_SRC_RING = ./lib/librte_ring
DIR_SRC_HEAP= ./heap
DIR_SRC_RING_MEMPOOL = mempool_ring
DIR_SRC_CUR = eal.c  
#DIR_SRC_CUR = eal.c   malloc_heap.c
DIR_BIN = ./build

#SRC := $(DIR_SRC_COMMON)
#SRC := $(wildcard (DIR_COMMON)/*.c)
#OBJ := $(SRC:$(DIR_COMMON)/%.c=$(DIR_BIN)/%.o)
#OBJ = $(patsubst %.c,${DIR_BIN}/%.o, ${SRC}) 
#OBJ := $(patsubst %.c,${DIR_BIN}/%.o,$(notdir ${SRC})) 
#OBJ := $(SRC:%.c=%${DIR_BIN}/.o)
#OBJ := $(addprefix $(DIR_BIN)/, $(SRC:.c=.o))
#SRC = $(wildcard ${DIR_SRC_BAR}/*.c)  
#SRC += $(wildcard ${DIR_SRC_FOO}/*.c)  
# sequence  DIR_SRC_EAL <  DIR_SRC_HEAP < DIR_SRC_MEMPOO < DIR_SRC_RING < DIR_SRC_RING_MEMPOOL
SRC := $(wildcard ${DIR_SRC_COMMON}/*.c)
SRC += $(wildcard ${DIR_SRC_EAL}/*.c)  
SRC += $(wildcard ${DIR_SRC_HEAP}/*.c)  
SRC += $(wildcard ${DIR_SRC_MEMPOOL}/*.c)  
SRC += $(wildcard ${DIR_SRC_RING}/*.c)  
SRC += $(wildcard ${DIR_SRC_RING_MEMPOOL}/*.c)  
OBJ := $(patsubst %.c,${DIR_BIN}/%.o,$(notdir ${SRC})) 
CUR_OBJ = eal.o   
OBJ += $(CUR_OBJ)
TARGET = main
BIN_TARGET = ${DIR_BIN}/${TARGET}

#CFLAGS = -I${DIR_INC} -I ./lib/librte_eal/common -I ./heap  -D_GNU_SOURCE 
CFLAGS = -Wall -I${DIR_INC} -I ./lib/librte_eal/common -I ./lib/librte_eal/common/include  -I lib/librte_ring -I lib/librte_mempool  -I ./heap -I ${DIR_SRC_RING_MEMPOOL}  -D_GNU_SOURCE 
ifeq ($(DEBUG), y)
CFLAGS += -g
else
CFLAGS += -O2
endif
LDFLAG = -lpthread

.PHONY:all clean

all:${BIN_TARGET}

#${BIN_TARGET}:${SRC} main.c
#	$(CC) $^ -o $@ -I${DIR_INC} $(LDFLAG)

${BIN_TARGET}:${OBJ} main.o
	$(CC) $^ -o $@  $(LDFLAG)

${DIR_BIN}/%.o:$(DIR_SRC_EAL)/%.c
	$(CC) $(CFLAGS) -c $^ -o $@ 
#${DIR_BIN}/%.o:$(DIR_SRC_CUR)
#	$(CC) $(CFLAGS) -c $^ -o $@ 
${DIR_BIN}/%.o:$(DIR_SRC_COMMON)/%.c
	$(CC) $(CFLAGS) -c $^ -o $@ 
${DIR_BIN}/%.o:$(DIR_SRC_MEMPOOL)/%.c
	$(CC) $(CFLAGS) -c $^ -o $@ 
${DIR_BIN}/%.o:$(DIR_SRC_RING)/%.c
	$(CC) $(CFLAGS) -c $^ -o $@ 
${DIR_BIN}/%.o:$(DIR_SRC_RING_MEMPOOL)/%.c
	$(CC) $(CFLAGS) -c $^ -o $@ 
${DIR_BIN}/%.o:$(DIR_SRC_HEAP)/%.c
	$(CC) $(CFLAGS) -c $^ -o $@ 
main.o:main.c
	$(CC) $(CFLAGS) -c $^ -o $@ 
clean:
	find ${DIR_BIN} . -name '*.o' -exec $(RM) '{}' \;
#	find ${DIR_BIN} -name '*.o' | xargs rm -rf
	$(RM) ${BIN_TARGET}
