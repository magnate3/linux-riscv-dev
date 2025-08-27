ifeq ($(RTE_SDK),)
$(error "Please define RTE_SDK environment variable")
endif
CC     = gcc
SRCDIR = $(PWD)
OBJDIR = $(PWD)/build
SRCS   = $(wildcard $(SRCDIR)/*.c)
OBJS   = $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%.o,$(SRCS))
#-Wformat-security -Wsign-conversion -g  -D_GNU_SOURCE         \
CFLAGS = -O3 -Wall -Werror -Wno-deprecated-declarations -fno-strict-aliasing \
	 -fPIC -pipe -std=gnu99 -Wunused -Wcomment -Wchar-subscripts         \
	 -Wuninitialized -Wcast-qual -Wcast-align -Wwrite-strings            \
	 -Wstack-protector -fstack-protector-all -Wno-unknown-pragmas        \
	 -Wformat-security  -g  -D_GNU_SOURCE         \
	 -D_XOPEN_SOURCE=700
CFLAGS += -I$(RTE_SDK)/arm64-armv8a-linuxapp-gcc/include

LDFLAGS =   -L$(RTE_SDK)/arm64-armv8a-linuxapp-gcc/lib -Wl,--as-needed

# DPDK libraries, do not reorder
#RTE_LIBS := dpdk rt m pthread dl
RTE_LIBS :=  rte_acl   rte_bbdev   rte_bitratestats   rte_bpf   rte_bus_dpaa   rte_bus_fslmc   rte_bus_ifpga   rte_bus_pci   rte_bus_vdev   rte_bus_vmbus   rte_cfgfile   rte_cmdline   rte_common_cpt   rte_common_dpaax   rte_compressdev   rte_cryptodev   rte_distributor   rte_eal   rte_efd   rte_ethdev   rte_eventdev   rte_fib   rte_flow_classify   rte_gro   rte_gso   rte_hash   rte_ip_frag   rte_ipsec   rte_jobstats   rte_kni   rte_kvargs   rte_latencystats   rte_lpm   rte_mbuf   rte_member   rte_mempool   rte_mempool_bucket   rte_mempool_dpaa   rte_mempool_dpaa2   rte_mempool_ring   rte_mempool_stack   rte_meter   rte_metrics   rte_net   rte_pci   rte_pdump   rte_pipeline   rte_pmd_af_packet   rte_pmd_ark   rte_pmd_atlantic   rte_pmd_axgbe   rte_pmd_bbdev_fpga_lte_fec   rte_pmd_bbdev_null   rte_pmd_bbdev_turbo_sw   rte_pmd_bnxt   rte_pmd_bond   rte_pmd_caam_jr   rte_pmd_crypto_scheduler   rte_pmd_cxgbe   rte_pmd_dpaa   rte_pmd_dpaa2   rte_pmd_dpaa2_event   rte_pmd_dpaa2_sec   rte_pmd_dpaa_event   rte_pmd_dpaa_sec   rte_pmd_dsw_event   rte_pmd_e1000   rte_pmd_ena   rte_pmd_enetc   rte_pmd_enic   rte_pmd_failsafe   rte_pmd_hinic   rte_pmd_hns3   rte_pmd_i40e   rte_pmd_iavf   rte_pmd_ice   rte_pmd_ifc   rte_pmd_ixgbe   rte_pmd_kni   rte_pmd_lio   rte_pmd_memif   rte_pmd_netvsc   rte_pmd_nfp   rte_pmd_nitrox   rte_pmd_null   rte_pmd_null_crypto   rte_pmd_octeontx_crypto   rte_pmd_octeontx_zip   rte_pmd_opdl_event   rte_pmd_pfe   rte_pmd_qat   rte_pmd_qede   rte_pmd_ring   rte_pmd_skeleton_event   rte_pmd_softnic   rte_pmd_sw_event   rte_pmd_tap   rte_pmd_thunderx_nicvf   rte_pmd_vdev_netvsc   rte_pmd_vhost   rte_pmd_virtio   rte_pmd_virtio_crypto   rte_pmd_vmxnet3_uio   rte_port   rte_power   rte_rawdev   rte_rawdev_dpaa2_cmdif   rte_rawdev_dpaa2_qdma   rte_rawdev_ntb   rte_rawdev_skeleton   rte_rcu   rte_reorder   rte_rib   rte_ring   rte_sched   rte_security   rte_stack   rte_table   rte_timer   rte_vhost  m pthread dl
#RTE_LIBS := rte_distributor rte_kni rte_pipeline rte_table rte_port \
#            rte_timer rte_hash rte_lpm rte_power rte_acl rte_meter  \
#            rte_sched m rt rte_kvargs rte_mbuf rte_ip_frag rte_ethdev  rte_eventdev \
#            rte_eal rte_mempool rte_ring rte_cmdline rte_cryptodev  rte_pmd_crypto_scheduler    \
#            rte_cfgfile rte_pci  rte_bus_pci  rte_net  rte_bus_vdev rte_common_dpaax  rte_bus_dpaa rte_pmd_bond rte_pmd_vmxnet3_uio            \
#            rte_pmd_i40e rte_pmd_ixgbe   rte_pmd_hinic rte_pmd_hns3         \
#            rte_pmd_e1000 rte_pmd_ring rt m pthread dl

LIBS += -L/root/dpdk-stable-19.11.1/arm64-armv8a-linuxapp-gcc/lib/  -Wl,--whole-archive
LIBS += $(foreach rte_lib, $(RTE_LIBS), $(shell echo -Wl,-l$(rte_lib)))
LIBS += -Wl,--no-whole-archive -lnuma

TARGET = helloworld

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) $(LIBS) $(OBJS) -o $(OBJDIR)/$@

$(OBJS): $(SRCS) | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $@

clean:
	rm -rf $(OBJDIR)

over: clean all

.PHONY: clean all over
