
#  sample_tree


```
ctx_params.kv_unified =  true;
```


```

./build/sample_tree   -m /workspace/qwen/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -p "Hello my name is" -np 4 
```

```
main: n_predict = 32, n_ctx = 16384, n_batch = 512, n_parallel = 4, n_kv_req = 113

<|begin_of_text|>Hello my name is cells seq count 4      cells pos[0] pos_in 0-5         cells seq[0] has stream 0  ,1  ,2  ,3 
 cells seq count 4       cells pos[1] pos_in 0-5         cells seq[1] has stream 0  ,1  ,2  ,3 
 cells seq count 4       cells pos[2] pos_in 0-5         cells seq[2] has stream 0  ,1  ,2  ,3 
 cells seq count 4       cells pos[3] pos_in 0-5         cells seq[3] has stream 0  ,1  ,2  ,3 
 cells seq count 4       cells pos[4] pos_in 0-5         cells seq[4] has stream 0  ,1  ,2  ,3 
min[0] =     0, max[0] =     4
min[1] =     0, max[1] =     4
min[2] =     0, max[2] =     4
min[3] =     0, max[3] =     4


main: generating 4 sequences ...
```

```
main: stream 3 finished at n_cur = 32
main: stream 3 finished at n_cur = 32 ,2  ,3 
 cells seq count 4       cells pos[1] pos_in 0-32        cells seq[1] has stream 0  ,1  ,2  ,3 
 cells seq count 4       cells pos[2] pos_in 0-32        cells seq[2] has stream 0  ,1  ,2  ,3 
 cells seq count 4       cells pos[3] pos_in 0-32        cells seq[3] has stream 0  ,1  ,2  ,3 
 cells seq count 4       cells pos[4] pos_in 0-32        cells seq[4] has stream 0  ,1  ,2  ,3 
 cells seq count 1       cells pos[5] pos_in 0-32        cells seq[5] has stream 0 
 cells seq count 1       cells pos[6] pos_in 0-32        cells seq[6] has stream 0 
 cells seq count 1       cells pos[7] pos_in 0-32        cells seq[7] has stream 0 
 cells seq count 3       cells pos[8] pos_in 0-32        ,1  ,2  ,3 
 cells seq count 1       cells pos[9] pos_in 0-32        cells seq[9] has stream 0 
 cells seq count 1       cells pos[10] pos_in 0-32       cells seq[10] has stream 0 
 cells seq count 1       cells pos[11] pos_in 0-32       cells seq[11] has stream 0 
 cells seq count 3       cells pos[12] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 1       cells pos[13] pos_in 0-32       cells seq[13] has stream 0 
 cells seq count 3       cells pos[14] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 1       cells pos[15] pos_in 0-32       cells seq[15] has stream 0 
 cells seq count 3       cells pos[16] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 3       cells pos[17] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 1       cells pos[18] pos_in 0-32       cells seq[18] has stream 0 
 cells seq count 1       cells pos[19] pos_in 0-32       cells seq[19] has stream 0 
 cells seq count 1       cells pos[20] pos_in 0-32       cells seq[20] has stream 0 
 cells seq count 3       cells pos[21] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 3       cells pos[22] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 3       cells pos[23] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 1       cells pos[24] pos_in 0-32       cells seq[24] has stream 0 
 cells seq count 1       cells pos[25] pos_in 0-32       cells seq[25] has stream 0 
 cells seq count 1       cells pos[26] pos_in 0-32       cells seq[26] has stream 0 
 cells seq count 3       cells pos[27] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 3       cells pos[28] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 1       cells pos[29] pos_in 0-32       cells seq[29] has stream 0 
 cells seq count 1       cells pos[30] pos_in 0-32       cells seq[30] has stream 0 
 cells seq count 3       cells pos[31] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 3       cells pos[32] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 1       cells pos[33] pos_in 0-32       cells seq[33] has stream 0 
 cells seq count 3       cells pos[34] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 3       cells pos[35] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 1       cells pos[36] pos_in 0-32       cells seq[36] has stream 0 
 cells seq count 1       cells pos[37] pos_in 0-32       cells seq[37] has stream 0 
 cells seq count 3       cells pos[39] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 1       cells pos[40] pos_in 0-32       cells seq[40] has stream 0 
 cells seq count 1       cells pos[41] pos_in 0-32       cells seq[41] has stream 0 
 cells seq count 3       cells pos[42] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 3       cells pos[43] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 3       cells pos[44] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 1       cells pos[45] pos_in 0-32       cells seq[45] has stream 0 
 cells seq count 1       cells pos[46] pos_in 0-32       cells seq[46] has stream 0 
 cells seq count 1       cells pos[47] pos_in 0-32       ,2 
 cells seq count 3       cells pos[48] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 1       cells pos[49] pos_in 0-32       cells seq[49] has stream 0 
 cells seq count 1       cells pos[50] pos_in 0-32       ,2 
 cells seq count 2       cells pos[51] pos_in 0-32       ,1  ,3 
 cells seq count 3       cells pos[52] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 3       cells pos[53] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 1       cells pos[54] pos_in 0-32       cells seq[54] has stream 0 
 cells seq count 1       cells pos[55] pos_in 0-32       ,2 
 cells seq count 3       cells pos[56] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 1       cells pos[57] pos_in 0-32       cells seq[57] has stream 0 
 cells seq count 1       cells pos[58] pos_in 0-32       cells seq[58] has stream 0 
 cells seq count 2       cells pos[60] pos_in 0-32       ,1  ,3 
 cells seq count 3       cells pos[61] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 3       cells pos[62] pos_in 0-32       ,1  ,2  ,3 
 cells seq count 3       cells pos[65] pos_in 0-32       ,1  ,2  ,3 

```