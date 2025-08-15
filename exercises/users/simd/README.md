
# os

```
ubuntu@ubuntux86:/work/sse$ uname -a
Linux ubuntux86 5.13.0-39-generic #44~20.04.1-Ubuntu SMP Thu Mar 24 16:43:35 UTC 2022 x86_64 x86_64 x86_64 GNU/Linux
ubuntu@ubuntux86:/work/sse$ 
```

# gcc -o intrin -msse4 test2.c

```
ubuntu@ubuntux86:/work/sse$ gcc -o intrin -msse4 test2.c 
ubuntu@ubuntux86:/work/sse$ ./intrin 
8.000000,6.000000,4.000000,2.000000 
```

# gcc -o intrin -msse4 test3.c 

```
ubuntu@ubuntux86:/work/sse$ gcc -o intrin -msse4 test3.c 
test3.c: In function ‘main’:
test3.c:7:7: warning: AVX vector return without AVX enabled changes the ABI [-Wpsabi]
    7 |      a=_mm256_set_epi32(1,2,3,4,5,6,7,8);
      |      ~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In file included from /usr/lib/gcc/x86_64-linux-gnu/9/include/immintrin.h:51,
                 from test3.c:2:
/usr/lib/gcc/x86_64-linux-gnu/9/include/avxintrin.h:1258:1: error: inlining failed in call to always_inline ‘_mm256_set_epi32’: target specific option mismatch
 1258 | _mm256_set_epi32 (int __A, int __B, int __C, int __D,
      | ^~~~~~~~~~~~~~~~
test3.c:7:8: note: called from here
    7 |      a=_mm256_set_epi32(1,2,3,4,5,6,7,8);
      |        ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ubuntu@ubuntux86:/work/sse$ 
```

```
ubuntu@ubuntux86:/work/sse$ gcc -o intrin -msse4 -march=native -mtune=native  test3.c 
ubuntu@ubuntux86:/work/sse$ ./intrin 
300000004 
ubuntu@ubuntux86:/work/sse$ 
```

# gcc -O2 -mavx2 test4.c  -o intrin

```
ubuntu@ubuntux86:/work/sse$ gcc -O2 -mavx2 test4.c  -o intrin
ubuntu@ubuntux86:/work/sse$ ./intrin 
1075679296 
 22042 
 15775231 
 0 
 194 
 0 
 -175346105 
 32764 
 ***************************** 
1075679296 
 22042 
 15775231 
 0 
 194 
 0 
 -175346105 
 32764 
```


# gcc -O2 -mavx2 test5.c  -o intrin

```
ubuntu@ubuntux86:/work/sse$ gcc -O2 -mavx2 test5.c  -o intrin
ubuntu@ubuntux86:/work/sse$ ./intrin 
Result[0]=0
Result[1]=2
Result[2]=4
Result[3]=6
Result[4]=8
Result[5]=10
Result[6]=12
Result[7]=14
```
#  gcc -O2 -mavx2 test6.c  -o intrin

```
buntu@ubuntux86:/work/sse$ gcc -O2 -mavx2 test6.c  -o intrin
ubuntu@ubuntux86:/work/sse$ ./intrin 
int:            39, 37, 35, 33, 31, 29, 27, 25
long long:      25, 25, 25, 25
ubuntu@ubuntux86:/work/sse$ 
```
# dpdk

##  i40e_xmit_pkts_vec_avx2

```
uint16_t
i40e_xmit_pkts_vec_avx2(void *tx_queue, struct rte_mbuf **tx_pkts,
           uint16_t nb_pkts)
{
    uint16_t nb_tx = 0;
    struct i40e_tx_queue *txq = (struct i40e_tx_queue *)tx_queue;

    while (nb_pkts) {
        uint16_t ret, num;

        num = (uint16_t)RTE_MIN(nb_pkts, txq->tx_rs_thresh);
        ret = i40e_xmit_fixed_burst_vec_avx2(tx_queue, &tx_pkts[nb_tx],
                        num);
        nb_tx += ret;
        nb_pkts -= ret;
        if (ret < num)
            break;
    }

    return nb_tx;
}

static inline uint16_t
i40e_xmit_fixed_burst_vec_avx2(void *tx_queue, struct rte_mbuf **tx_pkts,
              uint16_t nb_pkts)
{
    struct i40e_tx_queue *txq = (struct i40e_tx_queue *)tx_queue;
    volatile struct i40e_tx_desc *txdp;
    struct i40e_tx_entry *txep;
    uint16_t n, nb_commit, tx_id;
    uint64_t flags = I40E_TD_CMD;
    uint64_t rs = I40E_TX_DESC_CMD_RS | I40E_TD_CMD;

    /* cross rx_thresh boundary is not allowed */
    nb_pkts = RTE_MIN(nb_pkts, txq->tx_rs_thresh);

    if (txq->nb_tx_free < txq->tx_free_thresh)
        i40e_tx_free_bufs(txq);

    nb_commit = nb_pkts = (uint16_t)RTE_MIN(txq->nb_tx_free, nb_pkts);
    if (unlikely(nb_pkts == 0))
        return 0;

    tx_id = txq->tx_tail;
    txdp = &txq->tx_ring[tx_id];
    txep = &txq->sw_ring[tx_id];

    txq->nb_tx_free = (uint16_t)(txq->nb_tx_free - nb_pkts);

    n = (uint16_t)(txq->nb_tx_desc - tx_id);
    if (nb_commit >= n) {
        tx_backlog_entry(txep, tx_pkts, n);

        vtx(txdp, tx_pkts, n - 1, flags);
        tx_pkts += (n - 1);
        txdp += (n - 1);

        vtx1(txdp, *tx_pkts++, rs);

        nb_commit = (uint16_t)(nb_commit - n);

        tx_id = 0;
        txq->tx_next_rs = (uint16_t)(txq->tx_rs_thresh - 1);

        /* avoid reach the end of ring */
        txdp = &txq->tx_ring[tx_id];
        txep = &txq->sw_ring[tx_id];
    }

    tx_backlog_entry(txep, tx_pkts, nb_commit);

    vtx(txdp, tx_pkts, nb_commit, flags); ///////////////////

    tx_id = (uint16_t)(tx_id + nb_commit);
    if (tx_id > txq->tx_next_rs) {
        txq->tx_ring[txq->tx_next_rs].cmd_type_offset_bsz |=
            rte_cpu_to_le_64(((uint64_t)I40E_TX_DESC_CMD_RS) <<
                        I40E_TXD_QW1_CMD_SHIFT);
        txq->tx_next_rs =
            (uint16_t)(txq->tx_next_rs + txq->tx_rs_thresh);
    }

    txq->tx_tail = tx_id;

    I40E_PCI_REG_WRITE(txq->qtx_tail, txq->tx_tail);

    return nb_pkts;
}
```
##   vtx
```
static inline void
vtx1(volatile struct i40e_tx_desc *txdp,
        struct rte_mbuf *pkt, uint64_t flags)
{
    uint64_t high_qw = (I40E_TX_DESC_DTYPE_DATA |
            ((uint64_t)flags  << I40E_TXD_QW1_CMD_SHIFT) |
            ((uint64_t)pkt->data_len << I40E_TXD_QW1_TX_BUF_SZ_SHIFT));

    __m128i descriptor = _mm_set_epi64x(high_qw,
                pkt->buf_physaddr + pkt->data_off);
    _mm_store_si128((__m128i *)txdp, descriptor);
}

static inline void
vtx(volatile struct i40e_tx_desc *txdp,
        struct rte_mbuf **pkt, uint16_t nb_pkts,  uint64_t flags)
{
    const uint64_t hi_qw_tmpl = (I40E_TX_DESC_DTYPE_DATA |
            ((uint64_t)flags  << I40E_TXD_QW1_CMD_SHIFT));

    /* if unaligned on 32-bit boundary, do one to align */
    if (((uintptr_t)txdp & 0x1F) != 0 && nb_pkts != 0) {
        vtx1(txdp, *pkt, flags);
        nb_pkts--, txdp++, pkt++;
    }

    /* do two at a time while possible, in bursts */
    for (; nb_pkts > 3; txdp += 4, pkt += 4, nb_pkts -= 4) {
        uint64_t hi_qw3 = hi_qw_tmpl |
                ((uint64_t)pkt[3]->data_len << I40E_TXD_QW1_TX_BUF_SZ_SHIFT);
        uint64_t hi_qw2 = hi_qw_tmpl |
                ((uint64_t)pkt[2]->data_len << I40E_TXD_QW1_TX_BUF_SZ_SHIFT);
        uint64_t hi_qw1 = hi_qw_tmpl |
                ((uint64_t)pkt[1]->data_len << I40E_TXD_QW1_TX_BUF_SZ_SHIFT);
        uint64_t hi_qw0 = hi_qw_tmpl |
                ((uint64_t)pkt[0]->data_len << I40E_TXD_QW1_TX_BUF_SZ_SHIFT);

        __m256i desc2_3 = _mm256_set_epi64x(
                hi_qw3, pkt[3]->buf_physaddr + pkt[3]->data_off,
                hi_qw2, pkt[2]->buf_physaddr + pkt[2]->data_off);
        __m256i desc0_1 = _mm256_set_epi64x(
                hi_qw1, pkt[1]->buf_physaddr + pkt[1]->data_off,
                hi_qw0, pkt[0]->buf_physaddr + pkt[0]->data_off);
        _mm256_store_si256((void *)(txdp + 2), desc2_3);
        _mm256_store_si256((void *)txdp, desc0_1);
    }

    /* do any last ones */
    while (nb_pkts) {
        vtx1(txdp, *pkt, flags);
        txdp++, pkt++, nb_pkts--;
    }
}
```

### mm256_set_epi64x

```
ubuntu@ubuntux86:/work/sse$ cat test9.c 
#include <immintrin.h>
#include <stdio.h>

int main(int argc, char const *argv[]) {

            // 32-bit integer addition (AVX2)
            __m256i epi32_vec_0 = _mm256_set_epi32(10, 20, 30, 40, 50, 60, 70, 80);
            __m256i epi32_vec_1 = _mm256_set_epi32(5, 5, 5, 5, 5, 5, 5, 5);
            __m256i epi32_result = _mm256_add_epi32(epi32_vec_0, epi32_vec_1);
            int* i = (int*) &epi32_result;
            printf("int:\t\t%d, %d, %d, %d, %d, %d, %d, %d\n", i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]);
            
            // 64-bit integer addition (AVX2)
            __m256i epi64_vec_0 = _mm256_set_epi64x(8,9,10,11);
            __m256i epi64_vec_1 = _mm256_set_epi64x(17,18,19,20);
            __m256i epi64_result = _mm256_add_epi64(epi64_vec_0, epi64_vec_1);
            long long int* lo = (long long int*) &epi64_result;
            printf("long long:\t%lld, %lld, %lld, %lld\n", lo[0], lo[1], lo[2], lo[3]);
    return 0;
}

ubuntu@ubuntux86:/work/sse$ gcc -O2 -mavx2 test9.c  -o intrin
ubuntu@ubuntux86:/work/sse$ ./intrin 
int:            85, 75, 65, 55, 45, 35, 25, 15
long long:      31, 29, 27, 25
```

###  _mm256_store_si256

```
ubuntu@ubuntux86:/work/sse$ cat test10.c 
// gcc -mavx2 parallel_faster.c -o parallel_faster
#include <stdio.h>
#include <immintrin.h> // Include our intrinsics header
//#define BIG_DATA_SIZE 1000000
#define BIG_DATA_SIZE 8

// Create some arrays array
int BigData1[BIG_DATA_SIZE];
int BigData2[BIG_DATA_SIZE];
int Result[BIG_DATA_SIZE];

int main(){
    // Initialize array data
    int i=0;
    for(i =0; i < BIG_DATA_SIZE; ++i){
        BigData1[i] = i;
        BigData2[i] = i;
        Result[i] = 0;
    } 
    // Perform an operation on our data.
    // i.e. do some meaningful work
    for(i =0; i < BIG_DATA_SIZE; i=i+8){
        // Create two registers for signed integers('si')
        __m256i reg1 = _mm256_load_si256((__m256i*)&BigData1[i]);
        __m256i reg2 = _mm256_load_si256((__m256i*)&BigData2[i]);
        // Store the result
        __m256i reg_result = _mm256_add_epi32(reg1,reg2); 
        // Point to our data
//        int* data = (int*)&reg_result[0];
//        Result[i] = data[0];
//        Result[i+1] = data[1];
//        Result[i+2] = data[2];
//        Result[i+3] = data[3];
//        Result[i+4] = data[4];
//        Result[i+5] = data[5];
//        Result[i+6] = data[6];
//        Result[i+7] = data[7];
        // Rather then do all of the work above, we can use a 'store'
        // instruction to more quickly move our result back into the array.
       _mm256_store_si256((__m256i*)&Result[i],reg_result);
    } 
    // Print out the result;
    for(i =0; i < BIG_DATA_SIZE; ++i){
        printf("Result[%d]=%d\n",i,Result[i]);
    } 
    

    return 0;
}

ubuntu@ubuntux86:/work/sse$ gcc -O2 -mavx2 test10.c  -o intrin
ubuntu@ubuntux86:/work/sse$ ./intrin 
Result[0]=0
Result[1]=2
Result[2]=4
Result[3]=6
Result[4]=8
Result[5]=10
Result[6]=12
Result[7]=14
```

# reference

https://github.com/Triple-Z/AVX-AVX2-Example-Code
