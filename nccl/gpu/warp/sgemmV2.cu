#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <cuda_runtime.h>

#include <cublas_v2.h>
// #include <mma.h>
/*
    sgemmV2: C = A*B 相当于α是1,β是0
        含分块(内有打包+向量化取数) + 双缓冲
    
        V2是针对smem bank conflict进行解决
        并考虑由于是向量化访存,对quarter warp进行合并,使得1个warp的访存请求只需要2个memory transaction

        通过将8x8的矩阵乘转换为4个4x4的矩阵乘来实现,之所以这样转换,是因为要实现消去smem bank conflict且合并quarter warp,
        需要将原先一个warp内部的线程进行重新排列,以z-order的方式排列0-31号线程,排成8x4,如下:
         0  1 | 16 17
         2  3 | 18 19
         4  5 | 20 21
         6  7 | 22 23
        --------------  
         8  9 | 24 25
        10 11 | 26 27
        12 13 | 28 29
        14 15 | 30 31
        这样读A或读B通过广播机制避免了bank conflict,同时符合quarter warp的合并规则,因此如此排列
        我们知道内部一共有256个线程,即8个warp,将其排列成:
        warp0 | warp1 | warp2 | warp3
        warp4 | warp5 | warp6 | warp7

        因此此时需要算出warpId和landId来从smem中取数,同时T0取0,T1取4,T16取8,T17取12,如此一行则只取了64个元素,而一行有128个元素
        因此第2次向量化访存要在原来的基础上+64的偏移

        最终如T0就处理这么4个4x4矩阵:
    一个|或-表示4个元素
                0 1 2 3 4 5 6 7 8 9 A B C D E F 16
             B: - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        A:      T0                              T0
        0|T0
        1|
        2|
        3|
        4|
        5|
        6|
        7|
        8|
        9|
       10|
       11|
       12|
       13|
       14|
       15|
       16|T0
         |
        ...
*/

#define OFFSET(row,col,ld) ((row)*(ld)+(col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#define checkCudaError(func) {\
    cudaError_t e = (func);   \
    if(e != cudaSuccess)      \
        printf("%s %d cudaError : %s\n",__FILE__,__LINE__,cudaGetErrorString(e));   \
}


template<unsigned int BLOCK_SIZE_M,unsigned int BLOCK_SIZE_N,unsigned int BLOCK_SIZE_K,
         unsigned int THREAD_SIZE_M,unsigned int THREAD_SIZE_N>
__global__ void sgemmV2(float* __restrict__ A,float* __restrict__ B,float* __restrict__ C,int M,int N,int K){
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_N;
    const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_M;
    const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;

    const int tid = ty * THREAD_X_PER_BLOCK + tx;
    
    // 分块后的数据放哪,所需的空间
    // 共享内存,含预取
    __shared__ float smemA[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float smemB[2][BLOCK_SIZE_K][BLOCK_SIZE_N];
    // 寄存器空间,含预取
    float regA[2][THREAD_SIZE_M];
    float regB[2][THREAD_SIZE_N];
    // 临时计算C的空间
    float cAux[THREAD_SIZE_M][THREAD_SIZE_N] = {0.0f};
    // gmem->reg->smem,需要临时寄存器把gmem拿到的东西暂存,然后给它放回smem
    // 每个线程需要取数的次数
    const int ld_num_a = BLOCK_SIZE_M * BLOCK_SIZE_K / (THREAD_NUM_PER_BLOCK *4);
    const int ld_num_b = BLOCK_SIZE_K * BLOCK_SIZE_N / (THREAD_NUM_PER_BLOCK *4);
    // 临时寄存器的大小
    float ld_gmem2reg_a[4*ld_num_a];
    float ld_gmem2reg_b[4*ld_num_b];
    
    // 每个线程怎么取数->每个线程取对应行列的数
    // 计算A,B块每行所需的线程数
    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;
    // 线程取A的具体行列
    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4;
    // 线程取B的具体行列
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;
    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;

    // 当前线程数目情况下,行的步距,即一次运算可以取多少行,搭配每个线程需要取数的次数使用
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW; 
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    // 在做循环前的预取先要把每个block的矩阵A,B的位置定好
    float* tmpA = &A[by*BLOCK_SIZE_M*K];
    float* tmpB = &B[bx*BLOCK_SIZE_N];
    // 开始迭代前的预取
    // 共享内存,对AB
    #pragma unroll
    for(int i=0;i<BLOCK_SIZE_M;i+=A_TILE_ROW_STRIDE){
        int ld_idx = i/A_TILE_ROW_STRIDE*4;
        FLOAT4(ld_gmem2reg_a[ld_idx]) = FLOAT4(tmpA[OFFSET(
            A_TILE_ROW_START + i,
            A_TILE_COL,
            K
        )]);
        // 手动转置,先给它放0号位,后面迭代一开始就从0号位读,1号位写下一个时候的数据
        smemA[0][A_TILE_COL][A_TILE_ROW_START+i] = ld_gmem2reg_a[ld_idx];
        smemA[0][A_TILE_COL+1][A_TILE_ROW_START+i] = ld_gmem2reg_a[ld_idx+1];
        smemA[0][A_TILE_COL+2][A_TILE_ROW_START+i] = ld_gmem2reg_a[ld_idx+2];
        smemA[0][A_TILE_COL+3][A_TILE_ROW_START+i] = ld_gmem2reg_a[ld_idx+3];
    }
    #pragma unroll
    for(int i=0;i<BLOCK_SIZE_K;i+=B_TILE_ROW_STRIDE){
        FLOAT4(smemB[0][B_TILE_ROW_START+i][B_TILE_COL]) = FLOAT4(tmpB[OFFSET(
            B_TILE_ROW_START+i,
            B_TILE_COL,
            N
        )]);
    }
    // 以上就把gmem->借助reg->smem,为了让同一block的线程看到一样的smem,同步一下
    __syncthreads();

    int warpId = tid >> 5;
    int laneId = tid % 32;
    int a_tile_idx = /*((warpId>>2)<<5) + (((laneId&15)>>1)<<2);*/ (warpId/4)*32 + ((laneId%16)/2)*4;
    int b_tile_idx = /*((warpId&3)<<4)  + ((laneId>>4)<<3) + ((laneId&1)<<2);*/ (warpId%4)*16 + (laneId/16)*8 + (laneId%2)*4;

    // 寄存器,对AB
    FLOAT4(regA[0][0]) = FLOAT4(smemA[0][0][a_tile_idx]);
    FLOAT4(regA[0][4]) = FLOAT4(smemA[0][0][a_tile_idx+64]);

    FLOAT4(regB[0][0]) = FLOAT4(smemB[0][0][b_tile_idx]);
    FLOAT4(regB[0][4]) = FLOAT4(smemB[0][0][b_tile_idx+64]);
    // 这里我想是写到寄存器,不同步也没事,不影响

    // 大迭代,需要把tile_idx跑满,就是每次加个BLOCK_SIZE_K,要滑到K
    // write_idx: 下一轮数据放的索引 read_idx: 当前轮计算读的索引
    int write_idx = 1;
    int tile_idx = 0;
    do{
        tile_idx += BLOCK_SIZE_K;
        // 预取下一次的数据
        // 还可以取
        if(tile_idx<K){
            // A,先放到临时寄存器中
            #pragma unroll
            for(int i=0;i<BLOCK_SIZE_M;i+=A_TILE_ROW_STRIDE){
                int ld_idx = i/A_TILE_ROW_STRIDE;
                FLOAT4(ld_gmem2reg_a[ld_idx*4]) = FLOAT4(tmpA[OFFSET(
                    A_TILE_ROW_START + i,
                    A_TILE_COL + tile_idx,
                    K
                )]);   
            }
            // B,先放到临时寄存器中
            #pragma unroll
            for(int i=0;i<BLOCK_SIZE_K;i+=B_TILE_ROW_STRIDE){
                int ld_idx = i/B_TILE_ROW_STRIDE;
                FLOAT4(ld_gmem2reg_b[ld_idx*4]) = FLOAT4(tmpB[OFFSET(
                    B_TILE_ROW_START+i+tile_idx,
                    B_TILE_COL,
                    N
                )]);
            }

            //这里感觉也可以不用同步
        }
        // 从read_idx取数
        int read_idx = write_idx^1;
        // 计算,留最后一次计算去把临时寄存器的值写到共享内存,预取到reg中去
        #pragma unroll
        for(int i=0;i<BLOCK_SIZE_K-1;i++){
            // 预取下一次的数据
            FLOAT4(regA[(i+1)%2][0]) = FLOAT4(smemA[read_idx][i+1][a_tile_idx]);
            FLOAT4(regA[(i+1)%2][4]) = FLOAT4(smemA[read_idx][i+1][a_tile_idx+64]);

            FLOAT4(regB[(i+1)%2][0]) = FLOAT4(smemB[read_idx][i+1][b_tile_idx]);
            FLOAT4(regB[(i+1)%2][4]) = FLOAT4(smemB[read_idx][i+1][b_tile_idx+64]);

            // 计算
            #pragma unroll
            for(int p=0;p<THREAD_SIZE_M;p++){
                #pragma unroll
                for(int q=0;q<THREAD_SIZE_N;q++){
                    cAux[p][q] += regA[i%2][p] * regB[i%2][q];
                }
            }
        }

        // 共享内存预取写回
        if(tile_idx<K){
            #pragma unroll
            for(int i=0;i<BLOCK_SIZE_M;i+=A_TILE_ROW_STRIDE){
                int lg_idx = i/A_TILE_ROW_STRIDE*4;
                smemA[write_idx][A_TILE_COL][A_TILE_ROW_START+i] = ld_gmem2reg_a[lg_idx];
                smemA[write_idx][A_TILE_COL+1][A_TILE_ROW_START+i] = ld_gmem2reg_a[lg_idx+1];
                smemA[write_idx][A_TILE_COL+2][A_TILE_ROW_START+i] = ld_gmem2reg_a[lg_idx+2];
                smemA[write_idx][A_TILE_COL+3][A_TILE_ROW_START+i] = ld_gmem2reg_a[lg_idx+3];
            }
            #pragma unroll
            for(int i=0;i<BLOCK_SIZE_K;i+=B_TILE_ROW_STRIDE){
                int lg_idx = i/B_TILE_ROW_STRIDE*4;
                FLOAT4(smemB[write_idx][B_TILE_ROW_START+i][B_TILE_COL]) = FLOAT4(ld_gmem2reg_b[lg_idx]);
            }
            __syncthreads();
            write_idx ^=1;
        }
        //下一次的寄存器预取完成+完成最后一次计算
        // 预取下一次的数据
        FLOAT4(regA[0][0]) = FLOAT4(smemA[read_idx^1][0][a_tile_idx]);
        FLOAT4(regA[0][4]) = FLOAT4(smemA[read_idx^1][0][a_tile_idx+64]);
      
        FLOAT4(regB[0][0]) = FLOAT4(smemB[read_idx^1][0][b_tile_idx]);
        FLOAT4(regB[0][4]) = FLOAT4(smemB[read_idx^1][0][b_tile_idx+64]);

        #pragma unroll
        for(int p=0;p<THREAD_SIZE_M;p++){
            #pragma unroll
            for(int q=0;q<THREAD_SIZE_N;q++){
                cAux[p][q] += regA[1][p] * regB[1][q];
            }
        }
    }while(tile_idx<K);
    // 算完写回,把cAux写到对应的位置
    float* tmpC = &C[by*BLOCK_SIZE_M*N + bx*BLOCK_SIZE_N];
    
    #pragma unroll
    for(int p=0;p<4;p++){
        FLOAT4(tmpC[OFFSET(
            a_tile_idx+p,
            b_tile_idx,
            N
        )]) = FLOAT4(cAux[p][0]);
    }

    #pragma unroll
    for(int p=0;p<4;p++){
        FLOAT4(tmpC[OFFSET(
            a_tile_idx+p,
            b_tile_idx+64,
            N
        )]) = FLOAT4(cAux[p][4]);
    }

    #pragma unroll
    for(int p=0;p<4;p++){
        FLOAT4(tmpC[OFFSET(
            a_tile_idx+64+p,
            b_tile_idx,
            N
        )]) = FLOAT4(cAux[p+4][0]);
    }

    #pragma unroll
    for(int p=0;p<4;p++){
        FLOAT4(tmpC[OFFSET(
            a_tile_idx+64+p,
            b_tile_idx+64,
            N
        )]) = FLOAT4(cAux[p+4][4]);
    }
}

int main(int argc,char** argv){
    if(argc!=4){
        printf("Please input 4 elements,like: ./file M N K\n");
        exit(-1);
    }
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);

    // 注意,由于微内核是采用8x8进行计算的,所以m,n要是八的倍数;
    // 同时我们的bk是8,k要能被bk整除,所以k要是八的倍数
    assert(m%8 == 0);
    assert(n%8 == 0);
    assert(k%8 == 0);


    int gpu_id = 0; // 指定使用的GPU设备ID,例如这里使用第一个GPU
 
    cudaDeviceProp device_prop;
    if (cudaGetDeviceProperties(&device_prop, gpu_id) == cudaSuccess) {
        printf("Using GPU device %d:%s\n",gpu_id,device_prop.name);
    }
 
    cudaSetDevice(gpu_id); // 设置当前线程使用指定的GPU设备
 

    size_t sizeA = sizeof(float) * m * k;
    size_t sizeB = sizeof(float) * k * n;
    size_t sizeC = sizeof(float) * m * n;

    float* hA = (float*)malloc(sizeA);
    float* hB = (float*)malloc(sizeB);
    float* hC = (float*)malloc(sizeC);
    float* hCblas = (float*)malloc(sizeC);
    
    float* dA,*dB,*dC,*dCblas;
    checkCudaError(cudaMalloc(&dA,sizeA));
    checkCudaError(cudaMalloc(&dB,sizeB));
    checkCudaError(cudaMalloc(&dC,sizeC));
    checkCudaError(cudaMalloc(&dCblas,sizeC));

    srand(time(0));
    // 初始化A阵
    for(int i=0;i<m*k;i++){
        hA[i] = drand48()*10;
    }
    // 初始化B阵
    for(int i=0;i<k*n;i++){
        hB[i] = drand48()*10;
    }
    // 初始化C阵
    // for(int i=0;i<m*n;i++){
    //     hC[i] = drand48()*10;
    // }

    checkCudaError(cudaMemcpy(dA,hA,sizeA,cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(dB,hB,sizeB,cudaMemcpyHostToDevice));
    // checkCudaError(cudaMemcpy(dC,hC,sizeC,cudaMemcpyHostToDevice));

    const int bm = 128;
    const int bn = 128;
    const int bk = 8;
    const int tm = 8;
    const int tn = 8;

    int nRepeats = 1000;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 计算量
    float gflops = 2.0*k*m*n*10e-9;
    float elapsed_time;
    cudaEventRecord(start);
    // 预取所需的共享内存大小
    int smemSize = 2 * bm*bk + 2* bk*bn;
    for(int i=0;i<nRepeats;i++){
        dim3 block(bn/tn,bm/tm);
        dim3 grid(n/bn,m/bm);
        sgemmV2<bm,bn,bk,tm,tn><<<grid,block/*,sizeof(float)*smemSize*/>>>(dA,dB,dC,m,n,k);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time,start,stop);
    // 平均一次耗时
    elapsed_time = elapsed_time / nRepeats;
    // ms->s
    elapsed_time = elapsed_time / 1000.0f;
    // 算力:
    float GFLOPS = gflops / elapsed_time;
    printf("SELF: M[%4d] N[%4d] K[%4d] GFLOPS=[%8.4f]\n",m,n,k,GFLOPS);

    // 需要补充cublas的结果校验即耗时比较
    cublasHandle_t blasHandle;
    cublasCreate(&blasHandle);
    float alpha = 1.0f;
    float beta = 0.0f;
    
    checkCudaError(cudaEventRecord(start));
    for(int i=0;i<nRepeats;i++){
        cublasSgemm(blasHandle,CUBLAS_OP_T,CUBLAS_OP_T,
        m,n,k,&alpha,dA,k,dB,n,&beta,dCblas,m);
    }
    checkCudaError(cudaEventRecord(stop));
    checkCudaError(cudaEventSynchronize(stop));
    checkCudaError(cudaEventElapsedTime(&elapsed_time,start,stop));
    // 平均一次耗时
    elapsed_time = elapsed_time / nRepeats;
    // ms->s
    elapsed_time = elapsed_time / 1000.0f;
    // 算力:
    float cuGFLOPS = gflops / elapsed_time;
    printf("CUBLAS: M[%4d] N[%4d] K[%4d] GFLOPS=[%8.4f]\n",m,n,k,cuGFLOPS);
    printf("%.6lf%\n",GFLOPS/cuGFLOPS *100);


    // 检验错误
    checkCudaError(cudaMemcpy(hC,dC,sizeC,cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(hCblas,dCblas,sizeC,cudaMemcpyDeviceToHost));
    

    double eps = 1.e-6;  // machine zero
    bool correct = true;
    for (int i = 0; i < m * n; i++) {
        int row = i / n;
        int col = i % n;
        double abs_err = fabs(hC[i] - hCblas[col * m + row]);
        double dot_length = m;
        double abs_val = fabs(hC[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    i, hC[i], hCblas[col * m + row], eps);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
    // printf("ratio= %f\n", gigaFlops[0] / gigaFlops[1]);

    checkCudaError(cudaEventDestroy(start));
    checkCudaError(cudaEventDestroy(stop));
    cublasDestroy(blasHandle);
    free(hA);
    free(hB);
    free(hC);
    free(hCblas);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dCblas);
    cudaDeviceReset(); // 释放GPU设备,结束使用
    return 0;
}