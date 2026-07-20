# ==============================================================================
# Makefile for building NCCL profiler plugin and a test executable
# ==============================================================================

# ------------------------------------------------------------------------------
# Configuration Variables
#
# Modify these variables to point to your local installations of CUDA, NCCL,
# and CUPTI. Environment variables like CUDA_HOME and NCCL_HOME can override
# these defaults if set.
# ------------------------------------------------------------------------------

# Path to your CUDA installation root. Defaults to /usr/local/cuda if not set
# in the environment.
CUDA_HOME ?= /usr/local/cuda

# Path to your NCCL build directory. Defaults to the path from the original
# Makefile if not set in the environment.
NCCL_HOME ?= /home/staff/vardas/nccl/build

# Path to the root of your CUPTI installation. This is often part of the
# CUDA installation, or in a separate location like a Spack install.
# This variable should point to the directory *containing* the 'include'
# and 'lib64' subdirectories for CUPTI.
CUPTI_HOME ?= /home/staff/vardas/spack/opt/spack/linux-debian11-zen2/gcc-10.2.1/cuda-11.8.0-6znhkjhz2vquzikacb3hbato2llnj3qi/extras/CUPTI

# Specific include path for the profiler example code (needed for "profiler.h")
PROFILER_EXAMPLE_INC ?= /home/staff/vardas/nccl/ext-profiler/example/nccl


# ------------------------------------------------------------------------------
# Tools and Flags
# ------------------------------------------------------------------------------

# Default C compiler. Can be overridden by the CC environment variable.
CC = mpicc

# Default MPI compiler. Can be overridden by the MPICC environment variable.
MPICC ?= mpicc

# Standard CFLAGS. Add -g for debugging if needed.
# Note: Original had -O3 and -g in the rule; adding -g here applies it
# consistently if you want debugging symbols. If -g is only for specific
# targets, keep it in the rule. Let's follow the original's intent
# and add -g back into the rules, keeping CFLAGS for general options.
CFLAGS = -Wall -Wextra -std=c11 -O3 #-DDEBUG

# ------------------------------------------------------------------------------
# Paths derived from Configuration Variables
# ------------------------------------------------------------------------------

# Include paths for headers
INC = -I$(CUDA_HOME)/include \
      -I$(CUPTI_HOME)/include \
      -I$(NCCL_HOME)/include \
      -I$(PROFILER_EXAMPLE_INC)

# Library paths for linking
LIBS = -L$(CUDA_HOME)/lib64 \
       -L$(CUPTI_HOME)/lib64 \
       -L$(NCCL_HOME)/lib

# Libraries to link against
# COMMON_LIBS are needed by both plugin and test
COMMON_LIBS = -lcudart -lnccl -lcupti
# PLUGIN_LIBS are specific to the plugin
PLUGIN_LIBS = $(COMMON_LIBS) -latomic -pthread

# ------------------------------------------------------------------------------
# Target Files
# ------------------------------------------------------------------------------

PLUGIN_SO := libnccl-profiler.so
TEST_EXEC := prof_ncclallreduce.out # Name for the test executable

# ------------------------------------------------------------------------------
# Phony Targets
# .PHONY tells make that these are not files to be created
# ------------------------------------------------------------------------------

.PHONY: default test clean

# ------------------------------------------------------------------------------
# Default Target: Build the profiler plugin
# ------------------------------------------------------------------------------

default: $(PLUGIN_SO)

$(PLUGIN_SO): ncclsee.c buffer_pool.c output_file.c
	$(CC) $(CFLAGS) $(INC) $(LIBS) -fPIC -shared -o $@ -Wl,-soname,$@ $^ $(PLUGIN_LIBS)

# ------------------------------------------------------------------------------
# Test Target: Build the test executable
# ------------------------------------------------------------------------------

test: $(TEST_EXEC)

# The test executable compilation rule
$(TEST_EXEC): prof_ncclallreduce.c
	$(MPICC) $(CFLAGS) $(INC) $(LIBS) -g -o $@ $^ $(COMMON_LIBS)

# ------------------------------------------------------------------------------
# Clean Target: Remove built files
# ------------------------------------------------------------------------------

clean:
	rm -f $(PLUGIN_SO)

clean-test:
	rm -f $(TEST_EXEC)
