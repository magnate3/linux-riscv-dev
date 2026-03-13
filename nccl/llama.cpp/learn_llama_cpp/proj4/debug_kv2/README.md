



+ 1

···
ubatch.n_tokens == sinfo.n_stream()*sinfo.size()
···

+ 2

```
     for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            const auto seq_id = ubatch.seq_id_unq[s];
            const auto stream_id = seq_to_stream[seq_id];
            const auto & cells = v_cells[stream_id];
            const uint32_t head_cur = v_heads[stream_id];
            LLAMA_LOG_DEBUG("%s: seq_id[%d],stream[%d], n = %5d, used = %5d, head = %5d, size = %5d, n_swa = %5d\n",
                    __func__, seq_id,stream_id, cells.used_max_p1(), cells.get_used(), head_cur, get_size(), n_swa);
     }
```

+ 3

```
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {

        for (uint32_t ii = 0; ii < sinfo.size(); ++ii) {
            const uint32_t i = s*sinfo.size() + ii;

            auto & cells = v_cells[sinfo.strm[s]];

            const auto idx = sinfo.idxs[s][ii];

            if (!cells.is_empty(idx)) {
                assert(cells.seq_count(idx) == 1);

                const llama_seq_id seq_id = cells.seq_get(idx);
                const llama_pos    pos    = cells.pos_get(idx);
                const auto stream_id = seq_to_stream[seq_id];
                std::cout<< "stream_id " << stream_id << " seq_id " << seq_id << " pos " << pos << std::endl;
           }

       }
    }
```

+ 4

```
    std::string prompt1 = "What is the capital of Sweden?";
    std::string prompt2 = "What is the capital of France? the the capital of France is the largetst city of France";
```

+ run 
```
prompt1: What is the capital of Sweden?
prompt2: What is the capital of France? the the capital of France is the largetst city of France
Creating new llama_batch with 2 sequences
Token id [3838] in common at positions:
  Sequence 1, Index 0
  Sequence 0, Index 0
Token id [315] in common at positions:
  Sequence 1, Index 4
  Sequence 0, Index 4
Token id [374] in common at positions:
  Sequence 1, Index 1
  Sequence 0, Index 1
Token id [279] in common at positions:
  Sequence 1, Index 2
  Sequence 0, Index 2
Token id [6722] in common at positions:
  Sequence 1, Index 3
  Sequence 0, Index 3
Token id [30] in common at positions:
  Sequence 1, Index 6
  Sequence 0, Index 6

Processing prompt 0, nr tokens: 7 (batch_n_tokens: 0)
  idx: 0, token_id: 3838 
  idx: 1, token_id: 374 
  idx: 2, token_id: 279 
  idx: 3, token_id: 6722 
  idx: 4, token_id: 315 
  idx: 5, token_id: 23190 
  idx: 6, token_id: 30 

Processing prompt 1, nr tokens: 20 (batch_n_tokens: 7)
  idx: 7, token_id: 3838 
  idx: 8, token_id: 374 
  idx: 9, token_id: 279 
  idx: 10, token_id: 6722 
  idx: 11, token_id: 315 
  idx: 12, token_id: 9625 
  idx: 13, token_id: 30 
  idx: 14, token_id: 279 
  idx: 15, token_id: 279 
  idx: 16, token_id: 6722 
  idx: 17, token_id: 315 
  idx: 18, token_id: 9625 
  idx: 19, token_id: 374 
  idx: 20, token_id: 279 
  idx: 21, token_id: 326 
  idx: 22, token_id: 1284 
  idx: 23, token_id: 267 
  idx: 24, token_id: 3283 
  idx: 25, token_id: 315 
  idx: 26, token_id: 9625 

batch.n_tokens: 27
batch.tokens: [3838, 374, 279, 6722, 315, 23190, 30, 3838, 374, 279, 6722, 315, 9625, 30, 279, 279, 6722, 315, 9625, 374, 279, 326, 1284, 267, 3283, 315, 9625, ]
[ 
0, token 'What', pos 0, n_seq_id 1, seq_id 0, logits 0, 
1, token ' is', pos 1, n_seq_id 1, seq_id 0, logits 0, 
2, token ' the', pos 2, n_seq_id 1, seq_id 0, logits 0, 
3, token ' capital', pos 3, n_seq_id 1, seq_id 0, logits 0, 
4, token ' of', pos 4, n_seq_id 1, seq_id 0, logits 0, 
5, token ' Sweden', pos 5, n_seq_id 1, seq_id 0, logits 0, 
6, token '?', pos 6, n_seq_id 1, seq_id 0, logits 1, 
7, token 'What', pos 0, n_seq_id 1, seq_id 1, logits 0, 
8, token ' is', pos 1, n_seq_id 1, seq_id 1, logits 0, 
9, token ' the', pos 2, n_seq_id 1, seq_id 1, logits 0, 
10, token ' capital', pos 3, n_seq_id 1, seq_id 1, logits 0, 
11, token ' of', pos 4, n_seq_id 1, seq_id 1, logits 0, 
12, token ' France', pos 5, n_seq_id 1, seq_id 1, logits 0, 
13, token '?', pos 6, n_seq_id 1, seq_id 1, logits 0, 
14, token ' the', pos 7, n_seq_id 1, seq_id 1, logits 0, 
15, token ' the', pos 8, n_seq_id 1, seq_id 1, logits 0, 
16, token ' capital', pos 9, n_seq_id 1, seq_id 1, logits 0, 
17, token ' of', pos 10, n_seq_id 1, seq_id 1, logits 0, 
18, token ' France', pos 11, n_seq_id 1, seq_id 1, logits 0, 
19, token ' is', pos 12, n_seq_id 1, seq_id 1, logits 0, 
20, token ' the', pos 13, n_seq_id 1, seq_id 1, logits 0, 
21, token ' l', pos 14, n_seq_id 1, seq_id 1, logits 0, 
22, token 'arget', pos 15, n_seq_id 1, seq_id 1, logits 0, 
23, token 'st', pos 16, n_seq_id 1, seq_id 1, logits 0, 
24, token ' city', pos 17, n_seq_id 1, seq_id 1, logits 0, 
25, token ' of', pos 18, n_seq_id 1, seq_id 1, logits 0, 
26, token ' France', pos 19, n_seq_id 1, seq_id 1, logits 1 ]
stream index 0 k data 0(0) 1(1) 2(2) 3(3) 4(4) 5(5) 6(6) 
stream index 1 k data 512(0) 513(1) 514(2) 515(3) 516(4) 517(5) 518(6) 
stream index 0 v data 0(0) 1(1) 2(2) 3(3) 4(4) 5(5) 6(6) 
stream index 1 v data 512(0) 513(1) 514(2) 515(3) 516(4) 517(5) 518(6) 
ubatch.n_tokens: 14 sinfo.n_stream: 2 sinfo.size: 7


ubatch.n_tokens: 14 sinfo.n_stream: 2 sinfo.size: 7
debug_cells: seq_id[0],stream[0], n =     7, used =     7, head =     7, size =   512, n_swa =     0

  seq data :0000000......................................................................................................................................................................................................................................................... *
................................................................................................................................................................................................................................................................ *


 pos data: 0    1    2    3    4    5    6    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    
.    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    
.    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    
.    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .     *
.    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    
.    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    
.    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    
.    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .     *

debug_cells: stream[0] min[0] =     0, max[0] =     6
debug_cells: seq_id[1],stream[1], n =     7, used =     7, head =     7, size =   512, n_swa =     0

  seq data :1111111......................................................................................................................................................................................................................................................... *
................................................................................................................................................................................................................................................................ *


 pos data: 0    1    2    3    4    5    6    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    
.    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    
.    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    
.    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .     *
.    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    
.    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    
.    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    
.    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .     *

debug_cells: stream[1] min[1] =     0, max[1] =     6
stream_id 0 seq_id 0 pos 0
stream_id 0 seq_id 0 pos 1
stream_id 0 seq_id 0 pos 2
stream_id 0 seq_id 0 pos 3
stream_id 0 seq_id 0 pos 4
stream_id 0 seq_id 0 pos 5
stream_id 0 seq_id 0 pos 6
stream_id 1 seq_id 1 pos 0
stream_id 1 seq_id 1 pos 1
stream_id 1 seq_id 1 pos 2
stream_id 1 seq_id 1 pos 3
stream_id 1 seq_id 1 pos 4
stream_id 1 seq_id 1 pos 5
stream_id 1 seq_id 1 pos 6
stream index 0 k data 519(7) 520(8) 521(9) 522(10) 523(11) 524(12) 525(13) 526(14) 527(15) 528(16) 529(17) 530(18) 531(19) 
stream index 0 v data 519(7) 520(8) 521(9) 522(10) 523(11) 524(12) 525(13) 526(14) 527(15) 528(16) 529(17) 530(18) 531(19) 
ubatch.n_tokens: 13 sinfo.n_stream: 1 sinfo.size: 13

ubatch.n_tokens: 13 sinfo.n_stream: 1 sinfo.size: 13
debug_cells: seq_id[1],stream[1], n =    20, used =    20, head =    20, size =   512, n_swa =     0

  seq data :11111111111111111111............................................................................................................................................................................................................................................ *
................................................................................................................................................................................................................................................................ *


 pos data: 0    1    2    3    4    5    6    7    8    9    10   11   12   13   14   15   16   17   18   19   .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    
.    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    
.    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    
.    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .     *
.    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    
.    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    
.    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    
.    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .     *

debug_cells: stream[1] min[1] =     0, max[1] =    19
stream_id 1 seq_id 1 pos 7
stream_id 1 seq_id 1 pos 8
stream_id 1 seq_id 1 pos 9
stream_id 1 seq_id 1 pos 10
stream_id 1 seq_id 1 pos 11
stream_id 1 seq_id 1 pos 12
stream_id 1 seq_id 1 pos 13
stream_id 1 seq_id 1 pos 14
stream_id 1 seq_id 1 pos 15
stream_id 1 seq_id 1 pos 16
stream_id 1 seq_id 1 pos 17
stream_id 1 seq_id 1 pos 18
stream_id 1 seq_id 1 pos 19
new_token_seq1: 576 : token_str1 [ The]
new_token_seq2: 11 : token_str2 [,]
~llama_context:        CPU compute buffer size is  18.7970 MiB, matches expectation of  18.7970 MiB
```