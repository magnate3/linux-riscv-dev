#include <stdio.h>
#include <stdlib.h>

// 示例 1: NCCL Btree 算法实现
typedef struct {
    int rank;
    int parent;
    int child0;
    int child1;
    int parentChildType;  // 0=左孩子, 1=右孩子
} BtreeNode;

void buildBinaryTree(int nranks, BtreeNode* nodes) {
    int rank = 0; 
    for (rank = 0; rank < nranks; rank++) {
        nodes[rank].rank = rank;
        
        int bit;
        for (bit = 1; bit < nranks; bit <<= 1) {
            if (bit & rank) break;
        }
        
        if (rank == 0) {
            // Root
            nodes[rank].parent = -1;
            nodes[rank].child0 = -1;
            nodes[rank].child1 = nranks > 1 ? bit >> 1 : -1;
            nodes[rank].parentChildType = -1;
        } else {
            // 计算父节点
            int up = (rank ^ bit) | (bit << 1);
            if (up >= nranks) up = (rank ^ bit);
            nodes[rank].parent = up;
            nodes[rank].parentChildType = (rank < up) ? 0 : 1;
            
            // 计算子节点
            int lowbit = bit >> 1;
            nodes[rank].child0 = lowbit == 0 ? -1 : rank - lowbit;
            
            int down1 = lowbit == 0 ? -1 : rank + lowbit;
            while (down1 >= nranks) {
                lowbit >>= 1;
                down1 = lowbit == 0 ? -1 : rank + lowbit;
            }
            nodes[rank].child1 = down1;
        }
    }
}

void printBinaryTree(int nranks, BtreeNode* nodes) {
    int i= 0; 
    printf("Binary Tree 拓扑 (%d GPUs):\n", nranks);
    printf("──────────────────────────────────────\n");
    printf("Rank | Parent | Child0 | Child1 | Type\n");
    printf("──────────────────────────────────────\n");
    for (i = 0; i < nranks; i++) {
        printf("%4d | %6d | %6d | %6d | %s\n", 
               i, 
               nodes[i].parent, 
               nodes[i].child0, 
               nodes[i].child1,
               nodes[i].parentChildType == -1 ? "Root" :
               nodes[i].parentChildType == 0 ? "Left" : "Right");
    }
}

int main() {
    int nranks = 8;
    BtreeNode* nodes = (BtreeNode*)malloc(nranks * sizeof(BtreeNode));
    
    buildBinaryTree(nranks, nodes);
    printBinaryTree(nranks, nodes);
    
    free(nodes);
    return 0;
}
