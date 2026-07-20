#include <stdio.h>
#include <stdlib.h>

// 示例 1: NCCL Btree 算法实现
// 示例 4: NCCL Double Binary Tree (Dtree) 实现
typedef struct {
    int rank;
    // Tree 0
    int t0_parent;
    int t0_child0;
    int t0_child1;
    int t0_type;
    // Tree 1
    int t1_parent;
    int t1_child0;
    int t1_child1;
    int t1_type;
} DtreeNode;

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

void buildDoubleBinaryTree(int nranks, DtreeNode* nodes) {
    int rank;
    BtreeNode* bnodes = NULL;
    for (rank = 0; rank < nranks; rank++) {
        nodes[rank].rank = rank;
        
        // Tree 0: 标准 Btree
        BtreeNode tree0;
        // (复用前面的 buildBinaryTree 逻辑)
        
        buildBinaryTree(rank,bnodes); 
        // Tree 1: 镜像或移位
        BtreeNode tree1;
        if (nranks % 2 == 1) {
            // 奇数: 移位
            int shiftrank = (rank - 1 + nranks) % nranks;
            
            buildBinaryTree(shiftrank,bnodes); 
            // 然后将结果映射回原 rank
        } else {
            // 偶数: 镜像
            int mirrorrank = nranks - 1 - rank;
            // buildBinaryTree(mirrorrank, ...)
            buildBinaryTree(mirrorrank,bnodes); 
            // 然后镜像映射
        }
        
        // 保存结果
        nodes[rank].t0_parent = tree0.parent;
        nodes[rank].t0_child0 = tree0.child0;
        nodes[rank].t0_child1 = tree0.child1;
        nodes[rank].t0_type = tree0.parentChildType;
        
        nodes[rank].t1_parent = tree1.parent;
        nodes[rank].t1_child0 = tree1.child0;
        nodes[rank].t1_child1 = tree1.child1;
        nodes[rank].t1_type = tree1.parentChildType;
    }
}

void printDoubleBinaryTree(int nranks, DtreeNode* nodes) {
    int i =0;
    printf("Double Binary Tree 拓扑:\n");
    printf("Rank | Tree0: P/C0/C1 | Tree1: P/C0/C1\n");
    printf("─────────────────────────────────────────\n");
    for (i = 0; i < nranks; i++) {
        printf("%4d | %2d/%2d/%2d       | %2d/%2d/%2d\n",
               i,
               nodes[i].t0_parent, nodes[i].t0_child0, nodes[i].t0_child1,
               nodes[i].t1_parent, nodes[i].t1_child0, nodes[i].t1_child1);
    }
}
int main() {
    int nranks = 8;
    DtreeNode* nodes = (DtreeNode*)malloc(nranks * sizeof(DtreeNode));
    //BtreeNode* nodes = (BtreeNode*)malloc(nranks * sizeof(BtreeNode));
    
    buildDoubleBinaryTree(nranks, nodes);
    printDoubleBinaryTree(nranks, nodes);
    
    free(nodes);
    return 0;
}
