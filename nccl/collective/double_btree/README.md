```
mpirun -np 4 --allow-run-as-root  ./dbtree  -d -s 1024 -c 8
0: 1 [up=-1, dl=1, dr=2], 2 [up=2, dl=-1, dr=-1]
1: 1 [up=0, dl=3, dr=-1], 2 [up=3, dl=-1, dr=-1]
2: 1 [up=0, dl=-1, dr=-1], 2 [up=3, dl=0, dr=-1]
3: 1 [up=1, dl=-1, dr=-1], 2 [up=-1, dl=2, dr=1]
[  0] Reduce using one (first) tree:
Result buffer (first 10): 0 0 0 0 0 0 0 0 0 0 
[  2] Mismatch in the reduce buffer at position 0. Expect 10 got 0!
Result buffer (first 10): 0 0 0 0 0 0 0 0 0 0 
Result buffer (first 10): 10 10 10 10 10 10 10 10 10 10 
[  0] Reduce using one (second) tree:
Result buffer (first 10): 0 0 0 0 0 0 0 0 0 0 
[  1] Mismatch in the reduce buffer at position 0. Expect 10 got 0!
[  1] Mismatch between cnt (896) and size (1024)
[  3] Mismatch in the reduce buffer at position 0. Expect 10 got 0!
Result buffer (first 10): 0 0 0 0 0 0 0 0 0 0 
[  1] Mismatch in the reduce buffer at position 0. Expect 10 got 0!
[  0] Mismatch in the reduce buffer at position 0. Expect 10 got 0!
Result buffer (first 10): 0 0 0 0 0 0 0 0 0 0 
Result buffer (first 10): 0 0 0 0 0 0 0 0 0 0 
Result buffer (first 10): 10 10 10 10 10 10 10 10 10 10 
[  2] Mismatch in the reduce buffer at position 0. Expect 10 got 0!
[  2] Mismatch between cnt (896) and size (1024)
[  0] Reduce using 2 trees sequentially (one after another):
[  2] Mismatch in the reduce buffer at position 0. Expect 10 got 0!
Result buffer (first 10): 0 0 0 0 0 0 0 0 0 0 
Result buffer (first 10): 0 0 0 0 0 0 0 0 0 0 
Result buffer (first 10): 0 0 0 0 0 0 0 0 0 0 
[  1] Mismatch in the reduce buffer at position 0. Expect 10 got 0!
[  1] Mismatch between cnt (896) and size (1024)
[  3] Mismatch in the reduce buffer at position 0. Expect 10 got 0!
Result buffer (first 10): 10 10 10 10 10 10 10 10 10 10 
```

```
root@f565c3af4ab8:/workspace/double_btree# mpirun -np 4 --allow-run-as-root  ./allReduce_doubleBinaryTree
MPI_Allreduce (4 nodes, 4096 bytes message size) Time taken : 92.000000 (usecs)
Rank : 1 failed correctness
Rank : 2 failed correctness
Rank : 3 failed correctness
Double Binary Tree (4 nodes, 4096 bytes message size) Time taken : 9572.000000 (usecs)
Rank : 0 failed correctness
```
