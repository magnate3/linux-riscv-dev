IDIR = .
CC = gcc
ODIR = obj

ifeq ($(USE_CUDA),1)
  CUDAFLAGS = -I/usr/local/cuda-10.1/targets/x86_64-linux/include
  CUDAFLAGS += -I/usr/local/cuda/include
  PRE_CFLAGS1 = -I$(IDIR) $(CUDAFLAGS) -g -DHAVE_CUDA
  LIBS = -Wall -lrdmacm -libverbs -lmlx5 -lcuda
else
  PRE_CFLAGS1 = -I$(IDIR) -g
  LIBS = -Wall -lrdmacm -libverbs -lmlx5
endif

ifeq ($(PRINT_LAT),1)
  CFLAGS = $(PRE_CFLAGS1) -DPRINT_LATENCY
else
  CFLAGS = $(PRE_CFLAGS1)
endif

OEXE_1 = uc_pingpong_sg
OEXE_2 = uc_pingpong
OEXE_3 = uc_pingpong_srv_no_send

#DEPS = gpu_direct_rdma_access.h
#DEPS += ibv_helper.h
#DEPS += khash.h
#DEPS += gpu_mem_util.h
#DEPS += utils.h

OBJS = pingpong.o
#OBJS += gpu_mem_util.o
#OBJS += utils.o

$(ODIR)/%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

all : make_odir $(OEXE_1) $(OEXE_2) $(OEXE_3)

make_odir: $(ODIR)/

$(OEXE_1) : $(patsubst %,$(ODIR)/%,$(OBJS)) $(ODIR)/uc_pingpong_sg.o
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

$(OEXE_2) : $(patsubst %,$(ODIR)/%,$(OBJS)) $(ODIR)/uc_pingpong.o
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)
$(OEXE_3) : $(patsubst %,$(ODIR)/%,$(OBJS)) $(ODIR)/uc_pingpong_srv_no_send.o
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

$(ODIR)/:
	mkdir -p $@

.PHONY: clean

clean :
	rm -rf $(ODIR)
	rm -f $(OEXE_1) $(OEXE_2) $(OEXE_3)  $(ODIR)/*.o *~ core.* $(IDIR)/*~ 

