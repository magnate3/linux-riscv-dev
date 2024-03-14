#include <stdio.h>
#include <ngx_rbtree.h>
#include <config.h>
#include <sys/wait.h>
#include <ngx_shmtx.h>
#include <ngx_shmem.h>
#include <stdlib.h>

struct ngx_rbtree_s tree;
ngx_rbtree_node_t sentinel;
int incr_num = 1000;

void left_foreach(ngx_rbtree_node_t * root, ngx_rbtree_node_t *sentinel);

void left_foreach(ngx_rbtree_node_t * root, ngx_rbtree_node_t *sentinel){
    if(root == sentinel){
        return ;
    }
    left_foreach(root->left, sentinel);
    printf("%ld,",root->key);
    left_foreach(root->right, sentinel);
}

int main(void){

    ngx_rbtree_init(&tree, &sentinel, ngx_rbtree_insert_value);
    ngx_rbtree_node_t * node;

//    for(int i = 0; i < incr_num; ++i){
//        node = malloc(sizeof(ngx_rbtree_node_t));
//        node->key = i;
//        ngx_rbtree_insert(&tree, node);
//    }
//
//    //遍历树
//    left_foreach(tree.root, tree.sentinel);

    return 0;
}
