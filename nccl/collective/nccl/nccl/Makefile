MPIPATH =  /usr
MPICC = $(MPIPATH)/bin/mpicc
MPIINC = $(MPIPATH)/include
LDFLAGS = -lm 
TARGET := buildBinaryTree
TARGET +=   buildDTree  double-binary-tree halving-Doubling-AllReduce  doubleBinaryTreeOnLeftOnly
#all: dbtree  allReduce_doubleBinaryTree
all: $(TARGET)
buildBinaryTree: buildBinaryTree.c
	gcc  -O0 -g -I$(MPIINC) -o $@ $+  $(LDFLAGS)
halving-Doubling-AllReduce: halving-Doubling-AllReduce.c
	gcc  -O0 -g -I$(MPIINC) -o $@ $+  $(LDFLAGS)
buildDTree: buildBinaryTree.c
	gcc  -O0 -g -I$(MPIINC) -o $@ $+  $(LDFLAGS)
doubleBinaryTreeOnLeftOnly: doubleBinaryTreeOnLeftOnly.c
	gcc  -O0 -g -I$(MPIINC) -o $@ $+  $(LDFLAGS)
double-binary-tree: double-binary-tree.cc
	g++  -O0 -g -I$(MPIINC) -o $@ $+  $(LDFLAGS)
clean:
	rm $(TARGET)
.PHONY: all clean 

