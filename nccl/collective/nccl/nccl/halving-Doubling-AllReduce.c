//#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// gcc halving-Doubling-AllReduce.c  -o halving-Doubling-AllReduce -lm
// 示例 2: Halving-Doubling AllReduce CPU 模拟
void halvingDoublingAllReduce(
    float* sendbuf,      // 输入数据
    float* recvbuf,      // 输出数据
    int count,           // 元素数量
    int rank,            // 当前 GPU rank
    int nranks           // 总 GPU 数
) {
    int steps = (int)log2(nranks);
    int offset = 0;
    int curr_count = count;
    int i, k;
    float* tmpbuf = (float*)malloc(count * sizeof(float));
    // 初始化接收缓冲区
    for (i = 0; i < count; i++) {
        recvbuf[i] = sendbuf[i];
    }
    
    printf("[Rank %d] 开始 Halving-Doubling AllReduce\n", rank);
    
    // ============ 阶段 1: Reduce-Scatter (Halving) ============
    
    for (k = steps - 1; k >= 0; k--) {
        int distance = 1 << k;  // 2^k
        int partner = rank ^ distance;
        
        if (partner >= nranks) continue;
        
        int send_offset, recv_offset;
        if (rank < partner) {
            // 发送上半，接收下半
            send_offset = offset + curr_count / 2;
            recv_offset = offset;
        } else {
            // 发送下半，接收上半
            send_offset = offset;
            recv_offset = offset + curr_count / 2;
        }
        
        int exchange_count = curr_count / 2;
        
        printf("  [RS Step %d] Rank %d ↔ %d: send[%d:%d], recv[%d:%d]\n",
               steps - k, rank, partner,
               send_offset, send_offset + exchange_count,
               recv_offset, recv_offset + exchange_count);
        
        // 模拟通信（实际需要 MPI/NCCL）
        // MPI_Sendrecv(...)
        
        // 归约接收到的数据
        for (i = 0; i < exchange_count; i++) {
            recvbuf[recv_offset + i] += tmpbuf[i];  // 模拟接收的数据
        }
        
        // 更新参数
        if (rank < partner) {
            offset = recv_offset;
        } else {
            offset = send_offset;
        }
        curr_count = exchange_count;
    }
    
    // ============ 阶段 2: AllGather (Doubling) ============
    for (k = 0; k < steps; k++) {
        int distance = 1 << k;  // 2^k
        int partner = rank ^ distance;
        
        if (partner >= nranks) continue;
        
        int send_offset, recv_offset;
        if (rank < partner) {
            // 发送下半，接收上半
            send_offset = offset;
            recv_offset = offset + curr_count;
        } else {
            // 发送上半，接收下半
            send_offset = offset + curr_count;
            recv_offset = offset;
        }
        
        printf("  [AG Step %d] Rank %d ↔ %d: send[%d:%d], recv[%d:%d]\n",
               k + 1, rank, partner,
               send_offset, send_offset + curr_count,
               recv_offset, recv_offset + curr_count);
        
        // 模拟通信
        // MPI_Sendrecv(...)
        
        // 更新参数
        if (rank > partner) {
            offset = recv_offset;
        }
        curr_count *= 2;
    }
    
    free(tmpbuf);
    printf("[Rank %d] AllReduce 完成\n", rank);
}

int main() {
    int nranks = 8;
    int count = 1024;
    int rank; 
    int i;
    for (rank = 0; rank < nranks; rank++) {
        float* sendbuf = (float*)malloc(count * sizeof(float));
        float* recvbuf = (float*)malloc(count * sizeof(float));
        
        // 初始化数据
        for (i = 0; i < count; i++) {
            sendbuf[i] = rank * 1.0f;
        }
        
        halvingDoublingAllReduce(sendbuf, recvbuf, count, rank, nranks);
        
        free(sendbuf);
        free(recvbuf);
    }
    return 0;
}
