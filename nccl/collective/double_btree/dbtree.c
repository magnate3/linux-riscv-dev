#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <assert.h>

#include <mpi.h>

#define MPI_PRINT(rank, stream, fmt, ...) \
        do { fprintf(stream, "[%3d] " fmt, rank, ##__VA_ARGS__); } while (0)

#define MPI_ERROR(rank, fmt, ...) MPI_PRINT(rank, stderr, fmt, ##__VA_ARGS__)

#define MPI_PRINT_ONCE(rank, fmt, ...) \
if (rank == 0) { MPI_PRINT(rank, stdout, fmt, ##__VA_ARGS__); fflush(stdout); }

int rank, size = 16;
int debug, buf_size, chunk_size;

typedef struct {
    int up_rank;
    int down_rank[2];
} tree_t;

int set_down_rank(int size, int rank)
{
    return (rank >= size ? -1 : rank);
}

int build_btree(int size, int rank, tree_t *t1)
{
    /* straight forward tree */
    if (rank == 0) {
        /* top of the tree */
        t1->up_rank = -1;
    } else {
        t1->up_rank = (rank - 1) / 2;
    }

    t1->down_rank[0] = set_down_rank(size, rank * 2 + 1);
    t1->down_rank[1] = set_down_rank(size, rank * 2 + 2);

    return 0;
}

int inverse_rank(int size, int rank)
{
    return (size-1) - rank;
}

int build_dbtree(int size, int rank, tree_t *t1, tree_t *t2)
{
    int i;
    /* straight forward tree */
    build_btree(size, rank, t1);
    build_btree(size, inverse_rank(size, rank), t2);
    t2->up_rank = t2->up_rank >= 0 ? inverse_rank(size, t2->up_rank) : t2->up_rank;
    for (i = 0; i < 2; i++) {
        t2->down_rank[i] = set_down_rank(size, inverse_rank(size, t2->down_rank[i]));
    }
    return 0;
}

void usage(char *cmd)
{
    fprintf(stderr, "Options: %s\n", cmd);
    fprintf(stderr, "\t-h        Display this help\n");
    fprintf(stderr, "\t-d        Print extended debug\n");
    fprintf(stderr, "Test description:\n");
    fprintf(stderr, "\t-s <arg>  Perform reduction on specific buffer size\n");
    fprintf(stderr, "\t-c <arg>  Chunk size\n");
}

void args_process(int argc, char **argv)
{
    char *tmp;
    int c;

    while((c = getopt(argc, argv, "hds:c:")) != -1) {
        switch (c) {
        case 'h':
            usage(argv[0]);
            exit(0);
            break;
        case 'd':
            debug = 1;
            break;
        case 's': {
            int64_t tmp = atoll(optarg);
            if (tmp > 0) {
                buf_size = tmp;
            }
            break;
        }
        case 'c': {
            int64_t tmp = atoll(optarg);
            if (tmp > 0) {
                chunk_size = tmp;
            }
            break;
        }
        default:
            c = -1;
            goto error;
        }
    }
    return;
error:
    if (rank == 0) {
        fprintf(stderr, "Bad argument of '-%c' option\n", (char)c);
        usage(argv[0]);
        MPI_Abort(MPI_COMM_WORLD, -1);
    } else {
        while(1) {
            sleep(1);
        }
    }
}


#if 0
typedef struct {
    int **buf;
    MPI_Request *rcv_reqs;
    int head, tail, size;
} ring_buf_t;


int sg_ring_inc(u8 ind,int mask){
	return ((ind + 1) & mask) ;
}

static inline u8
sg_ring_dec(u8 ind,int mask){
	return ((ind - 1) & mask) ;
}

void ring_buf_init(int size, int esize, ring_buf_t *b)
{
    int i;
    b->head = b->tail = 0;
    b->size = size;
    b->buf = calloc(size, sizeof(b-≥buf[0]));
    b->rcv_reqs = calloc(size, sizeof(b-≥buf[0]));
    for(i = 0; i < size; i++) {

    }
}
#endif

typedef struct {
    int **pool;
    int pool_size, chunk_size;
    int pool_cnt;
} pool_t;

void chunk_pool_init(int size, int chunk_size, pool_t *p)
{
    int i;
    p->chunk_size = chunk_size;
    p->pool_size = p->pool_cnt = size;
    p->pool = calloc(size, sizeof(p->pool[0]));
    for(i = 0; i < size; i++) {
        p->pool[i] = calloc(size, sizeof(p->pool[i][0]));
    }
}

void chunk_pool_fini(pool_t *p)
{
    int i;

    if(p->pool_cnt != p->pool_size) {
        MPI_ERROR(rank, "Mismatch between cnt (%d) and size (%d)\n",
                    p->pool_cnt, p->pool_size);
    }


    for(i = 0; i < size; i++) {
        free(p->pool[i]);
    }
    free(p->pool);
}

int *chunk_pool_get(pool_t *p)
{
    int idx = p->pool_cnt - 1;
    assert(idx >=0);

    p->pool_cnt = idx;
    return p->pool[idx];
}

void chunk_pool_put(pool_t *p, int *ptr)
{
    if(p->pool_cnt >= p->pool_size) {
        MPI_ERROR(rank, "Pool overflow: cnt (%d) and size (%d)\n",
                    p->pool_cnt, p->pool_size);
    }
    p->pool[p->pool_cnt] = ptr;
    p->pool_cnt++;
}

typedef struct {
    tree_t *t;
    int tag_base;
    int *lbuf, *obuf;
    int lbuf_size;
    int chunk_size;
    int chunk_cnt;

    /* buffers */
    pool_t chunk_pool;
    int *chunk_accum;
    int *chunk_rcvd[2];
    int *chunk_comp[2];
    MPI_Request rcv_reqs[2], snd_req;


    /* Algorithm state */

    int contrib_per_chunk;
    int contrib_init;
    int contrib_rcvd;
    int contrib_comp;
    int contrib_total;
    
    int forward_per_chunk;
    int forward_init;
    int forward_cmpl;
    int forward_total;



    /* temp buffers */
    int *acc_chunk;
    int *in_chunks[2];
    int *cur_chunk;
    int *nxt_chunk;
    int *fwd_chunk;
} tree_reduce_t;

#define TAG_BASE_T0 0
#define TAG_BASE_T1 2
#define TAG_REDUCE(x) (x)
#define TAG_BCAST(x) (x + 1)

int contrib_cur_chunk(tree_reduce_t *rdata, int contrib_idx)
{
    return contrib_idx % rdata->contrib_per_chunk;
}

int contribs_for_chunk(tree_reduce_t *rdata, int chunk_idx)
{
    return (chunk_idx + 1) * rdata->contrib_per_chunk;
}

void init_tree_reduce(tree_t *t, int tag_base, int *lbuf, int *obuf, int lbuf_size, int chunk_size, tree_reduce_t *rdata)
{
    int i;

    rdata->t = t;
    rdata->lbuf = lbuf;
    rdata->obuf = obuf;
    rdata->lbuf_size = lbuf_size;
    rdata->chunk_size = chunk_size;
    /* For simplicity assume that buffer is dividable to chunks */
    assert(!(rdata->lbuf_size % rdata->chunk_size));
    rdata->chunk_cnt = rdata->lbuf_size / rdata->chunk_size;

    rdata->contrib_per_chunk = 0;
    for(i = 0; i < 2; i++) {
        if (t->down_rank[i] >= 0) {
            rdata->contrib_per_chunk++;
        }
    }
    rdata->contrib_total = rdata->chunk_cnt * rdata->contrib_per_chunk;
    rdata->contrib_init = rdata->contrib_rcvd = rdata->contrib_comp = 0;

    rdata->forward_per_chunk = (t->up_rank >= 0);
    rdata->forward_total = rdata->chunk_cnt * rdata->forward_per_chunk;
    rdata->forward_init = rdata->forward_cmpl = 0;

    chunk_pool_init(1024, rdata->chunk_size, &rdata->chunk_pool);
}

void fini_tree_reduce(tree_reduce_t *rdata)
{
    chunk_pool_fini(&rdata->chunk_pool);
}

void reduce_initiate_rcvs(tree_reduce_t *rdata, int chunk_idx)
{
    tree_t *t = rdata->t;
    int i;

    if (rdata->contrib_rcvd != contribs_for_chunk(rdata, chunk_idx - 1)) {
        MPI_ERROR(rank, "Not ready for chunk idx = %d, current is %d, expect %d\n",
                chunk_idx, rdata->contrib_rcvd, contribs_for_chunk(rdata, chunk_idx - 1));
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    for (i = 0; i < 2; i++) {
        if (t->down_rank[i] >= 0) {
            rdata->chunk_rcvd[i] = chunk_pool_get(&rdata->chunk_pool);
            MPI_Irecv(rdata->chunk_rcvd[i], rdata->chunk_size, MPI_INT,
                        t->down_rank[i], TAG_REDUCE(rdata->tag_base), MPI_COMM_WORLD,
                        &rdata->rcv_reqs[i]);
            rdata->contrib_init++;
        }
    }
}


void reduce_complete_rcvs(tree_reduce_t *rdata, int chunk_idx)
{
    tree_t *t = rdata->t;
    int i;

    assert(rdata->contrib_init == contribs_for_chunk(rdata, chunk_idx));

    for (i = 0; i < 2; i++) {
        if (t->down_rank[i] >= 0) {
            MPI_Wait(&rdata->rcv_reqs[i], MPI_STATUS_IGNORE);
            rdata->chunk_comp[i] = rdata->chunk_rcvd[i];
            rdata->chunk_rcvd[i] = NULL;
            rdata->contrib_rcvd++;
        }
    }
}

void reduce_complete_fwd(tree_reduce_t *rdata)
{
    if (rdata->forward_cmpl < rdata->forward_init) {
        MPI_Wait(&rdata->snd_req, MPI_STATUS_IGNORE);
        rdata->forward_cmpl++;
        chunk_pool_put(&rdata->chunk_pool, rdata->chunk_accum);
        rdata->chunk_accum = NULL;
    }
}

void reduce_complete_compute(tree_reduce_t *rdata, int chunk_idx)
{
    int inc = 0;
    tree_t *t = rdata->t;
    int i;

    assert(rdata->contrib_rcvd == contribs_for_chunk(rdata, chunk_idx));

    /* Make sure that the previous forward is done */
    reduce_complete_fwd(rdata);

    for (i = 0; i < 2; i++) {
        int j;
        if (t->down_rank[i] >= 0) {
            if (!inc) {
                int *lbuf = &rdata->lbuf[chunk_idx * rdata->chunk_size];


                inc = 1;
                rdata->chunk_accum = rdata->chunk_comp[i];
                rdata->chunk_comp[i] = NULL;
                for (j = 0; j < rdata->chunk_size; j++) {
                    rdata->chunk_accum[j] += lbuf[j];
                }
            } else {
                for (j = 0; j < rdata->chunk_size; j++) {
                    rdata->chunk_accum[j] += rdata->chunk_comp[i][j];
                }
                chunk_pool_put(&rdata->chunk_pool, rdata->chunk_comp[i]);
                rdata->chunk_comp[i] = NULL;
            }
            rdata->contrib_comp++;
        }
    }
}


void reduce_initiate_fwd(tree_reduce_t *rdata, int chunk_idx)
{
    int inc = 0;
    tree_t *t = rdata->t;

    assert(rdata->contrib_comp == contribs_for_chunk(rdata, chunk_idx));

    if (rdata->forward_per_chunk) {
        if (rdata->contrib_per_chunk) {
            /* a node with children */
            MPI_Isend(rdata->chunk_accum, rdata->chunk_size, MPI_INT,
                        t->up_rank, TAG_REDUCE(rdata->tag_base), MPI_COMM_WORLD,
                        &rdata->snd_req);
        } else {
            /* this is a leaf node */
            int *lbuf = &rdata->lbuf[chunk_idx * rdata->chunk_size];
            MPI_Isend(lbuf, rdata->chunk_size, MPI_INT,
                        t->up_rank, TAG_REDUCE(rdata->tag_base), MPI_COMM_WORLD,
                        &rdata->snd_req);
        }
    } else {
        /* Reduce is done */
        memcpy(&rdata->obuf[chunk_idx * rdata->chunk_size], rdata->chunk_accum, rdata->chunk_size * sizeof(rdata->obuf[0]));
        chunk_pool_put(&rdata->chunk_pool, rdata->chunk_accum);

    }
}

void verify_result(int r, int *obuf)
{
    int k;
    if (rank == r) {
        printf("Result buffer (first 10): ");
        for(k = 0; k < 10; k++) {
            printf("%d ", obuf[k]);
        }
        printf("\n");

        /* Calculate the value expected in the buffer */
        int expect = 0;
        for(k = 0; k < size; k++) {
            expect += (k + 1);
        }
        /* verify that the value is correct */
        for(k = 0; k < buf_size; k++) {
            if (obuf[k] != expect) {
                MPI_ERROR(rank, "Mismatch in the reduce buffer at position %d. Expect %d got %d!\n",
                         k, expect, obuf[k]);
                break;
            }
        }

    }
}

int main(int argc, char **argv)
{
    int i;
    tree_t t[2];
    tree_reduce_t rdata[2];
    int nchunks;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0){
        int delay = 0;
        while(delay) {
            sleep(1);
        }
    }

    args_process(argc, argv);

    if(debug && (rank == 0)) {
        for (i=0; i < size; i++) {
            build_dbtree(size, i, &t[0], &t[1]);
            printf("%d: 1 [up=%d, dl=%d, dr=%d], 2 [up=%d, dl=%d, dr=%d]\n", 
                    i, 
                    t[0].up_rank, t[0].down_rank[0], t[0].down_rank[1],
                    t[1].up_rank, t[1].down_rank[0], t[1].down_rank[1]);
        }
    }

    build_dbtree(size, rank, &t[0], &t[1]);

    int *lbuf, *obuf;

    buf_size = ((buf_size / chunk_size) + !!(buf_size % chunk_size)) * chunk_size;
    nchunks = buf_size / chunk_size;
    lbuf = calloc(buf_size, sizeof(lbuf[0]));
    obuf = calloc(buf_size, sizeof(obuf[0]));
    for (i = 0; i < buf_size; i++) {
        lbuf[i] = rank + 1;
    } 


    /* Reduction algorithm on a single tree */
    
    MPI_PRINT_ONCE(rank, "Reduce using one (first) tree:\n");
    init_tree_reduce(&t[0], TAG_BASE_T0, lbuf, obuf, buf_size, chunk_size, &rdata[0]);

    for(i = 0; i < nchunks; i++) {
        reduce_initiate_rcvs(&rdata[0], i);
        reduce_complete_rcvs(&rdata[0], i);
        reduce_complete_compute(&rdata[0], i);
        reduce_initiate_fwd(&rdata[0], i);
    }
    reduce_complete_fwd(&rdata[0]);
    verify_result(rank,obuf);
    fini_tree_reduce(&rdata[0]);

    MPI_Barrier(MPI_COMM_WORLD);

    memset(obuf, 0, buf_size * sizeof(obuf[0]));

    /* Reduction algorithm on a single tree */
    MPI_PRINT_ONCE(rank, "Reduce using one (second) tree:\n");
    init_tree_reduce(&t[1], TAG_BASE_T1, lbuf, obuf, buf_size, chunk_size, &rdata[0]);

    for(i = 0; i < nchunks; i++) {
        reduce_initiate_rcvs(&rdata[0], i);
        reduce_complete_rcvs(&rdata[0], i);
        reduce_complete_compute(&rdata[0], i);
        reduce_initiate_fwd(&rdata[0], i);
    }
    reduce_complete_fwd(&rdata[0]);
    verify_result(rank,obuf);
    fini_tree_reduce(&rdata[0]);

    MPI_Barrier(MPI_COMM_WORLD);

    memset(obuf, 0, buf_size * sizeof(obuf[0]));

    /* Reduction algorithm on 2 trees sequenmtially */
    int offs = buf_size / 2;
    MPI_PRINT_ONCE(rank, "Reduce using 2 trees sequentially (one after another):\n");
    init_tree_reduce(&t[0], TAG_BASE_T0, lbuf, obuf, buf_size / 2, chunk_size, &rdata[0]);
    init_tree_reduce(&t[1], TAG_BASE_T1, lbuf + offs, obuf + offs, buf_size / 2, chunk_size, &rdata[1]);

    for(i = 0; i < nchunks; i++) {
        reduce_initiate_rcvs(&rdata[0], i);
        reduce_complete_rcvs(&rdata[0], i);
        reduce_complete_compute(&rdata[0], i);
        reduce_initiate_fwd(&rdata[0], i);
    }
    reduce_complete_fwd(&rdata[0]);
/*
    for(i = 0; i < nchunks; i++) {
        reduce_initiate_rcvs(&rdata[1], i);
        reduce_complete_rcvs(&rdata[1], i);
        reduce_complete_compute(&rdata[1], i);
        reduce_initiate_fwd(&rdata[1], i);
    }
    reduce_complete_fwd(&rdata[1]);
*/
    verify_result(rank,obuf);
    fini_tree_reduce(&rdata[0]);
    fini_tree_reduce(&rdata[1]);

    MPI_Barrier(MPI_COMM_WORLD);

#if 0
    /* Reduction algorithm on a single tree */
    buf_size = ((buf_size / chunk_size) + !!(buf_size % chunk_size)) * chunk_size;
    nchunks = buf_size / chunk_size;

    lbuf = calloc(buf_size, sizeof(lbuf[0]));
    obuf = calloc(buf_size, sizeof(obuf[0]));
    for (i = 0; i < buf_size; i++) {
        lbuf[i] = rank + 1;
    } 
    init_tree_reduce(&t[0], lbuf, obuf, buf_size / 2, chunk_size, &rdata[0]);
    init_tree_reduce(&t[1], lbuf + buf_size / 2, obuf + buf_size / 2, buf_size / 2, chunk_size, &rdata[1]);

    for(i = 0; i < nchunks; i++) {
        reduce_initiate_rcvs(&rdata[0], i);
        reduce_initiate_rcvs(&rdata[1], i);
        reduce_complete_rcvs(&rdata[0], i);
        reduce_complete_rcvs(&rdata[1], i);
        reduce_complete_compute(&rdata[0], i);
        reduce_complete_compute(&rdata[1], i);
        reduce_initiate_fwd(&rdata[0], i);
        reduce_initiate_fwd(&rdata[1], i);
    }
    verify_result(obuf);
#endif

    MPI_Finalize();

    return 0;
}
