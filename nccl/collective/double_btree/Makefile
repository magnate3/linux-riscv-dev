MPIPATH =  /usr
MPICC = $(MPIPATH)/bin/mpicc
MPIINC = $(MPIPATH)/include
LDFLAGS = -lm 
TARGET := dbtree
TARGET +=  allReduce_doubleBinaryTree
#all: dbtree  allReduce_doubleBinaryTree
all: $(TARGET)
dbtree: dbtree.c
	$(MPICC)  -O0 -g -I$(MPIINC) -o $@ $+  $(LDFLAGS)
allReduce_doubleBinaryTree: allReduce_doubleBinaryTree.c
	$(MPICC)  -O0 -g -I$(MPIINC) -o $@ $+  $(LDFLAGS)
clean:
	rm $(TARGET)
.PHONY: all clean 

