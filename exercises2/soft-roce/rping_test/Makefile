EXE = rping

#IBDIR=../
#CFLAGS+= \
#	-I${IBDIR}/include/infiniband \
#	-I${IBDIR}/include \
#	-I${IBDIR}/librdmacm/include \
#	-I${IBDIR}/libibv/include \
#	-I${IBDIR}/libibverbs/include
LDLIBS += -libverbs -lrdmacm -lpthread
CFLAGS += -std=c99 -D_GNU_SOURCE

default: $(EXE)

clean:
	rm -rf $(EXE) *.o *~

