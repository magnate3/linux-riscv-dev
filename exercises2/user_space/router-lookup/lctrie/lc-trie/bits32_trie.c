
#include <stdio.h>

/* header of bitmap */
#include "bits32_trie.h"
#include "bits32_trie_debug.h"
#include "bits32_trie_insert.h"
#include "bits32_trie_lookup.h"
#include "bits32_trie_remove.h"
#include "bits32_trie_new.h"

/*  pls keep this default value for demo, you can add new after them  */
b32t_key g_tkey_list[] = {
    0x12345678,
    0x87654321,
    0x11111111,
    0x22222222,
};
struct b32t_trie *gp_trie = NULL;
int main(void)
{
    int index;
    struct b32t_fib_alias *pfa_new;
    struct b32t_key_vector *leaf;
    gp_trie = b32t_trie_new();
    if(gp_trie == NULL){
        printf("null trie\n");
        return -1;
    }
    printf("%s:[%d - %d]", __func__, (unsigned int)sizeof(g_tkey_list), (unsigned int)sizeof(b32t_key));
    printf("~~~~~~~~  start insert  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
    for(index = 0; index < sizeof(g_tkey_list)/sizeof(b32t_key); index++){
        pfa_new = malloc(sizeof(struct b32t_fib_alias) + sizeof(char));
        pfa_new->fa_data[0] = index;
        printf("%s insert leaf[0x%x] data[0x%x]\n", __func__, g_tkey_list[index], index);
        b32t_leaf_insert(gp_trie, g_tkey_list[index], pfa_new);
    }
    printf("~~~~~~~~   end insert   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
    printf("--------------------------------------------------------------\n");
    printf("~~~~~~~~   now let's look the trie view ~~~~~~~~~~~~~~~~~\n");
    b32t_print_trie(gp_trie);
    printf("~~~~~~~~   trie view end, it's as what you think??  ~~~~~\n");
    printf("--------------------------------------------------------------\n");

    printf("~~~~~~~~   test lookup  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
    printf("%s lookup[0x%x]index[0x%x]\n", __func__, g_tkey_list[index/2], index/2);
    leaf = b32t_find_leaf(gp_trie, g_tkey_list[index/2], false);
    if(leaf){
        printf("%s it seems ok, found leaf->key[0x%x]\n", __func__, leaf->key);
    }else{
        printf("%s null leaf\n", __func__);
    }
    
    printf("~~~~~~~~  test remove  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
    printf("%s remove[0x%x]index[0x%x]\n", __func__, g_tkey_list[2], 2);
	b32t_leaf_delete(gp_trie, g_tkey_list[2]);
	leaf = b32t_find_leaf(gp_trie, g_tkey_list[2], false);
    if(leaf){
        printf("%s it seems wrong, found leaf->key[0x%x] after del\n", __func__, leaf->key);
    }else{
        printf("%s it seems ok, found no leaf->key[0x%x] after del\n", __func__, g_tkey_list[2]);
    }
    printf("%s remove[%x]index[%x]\n", __func__, g_tkey_list[0], 0);
    b32t_leaf_delete(gp_trie, g_tkey_list[0]);
    leaf = b32t_find_leaf(gp_trie, g_tkey_list[0], false);
    if(leaf){
        printf("%s it seems wrong, found leaf->key[0x%x] after del\n", __func__, leaf->key);
    }else{
        printf("%s it seems ok, found no leaf->key[0x%x] after del\n", __func__, g_tkey_list[0]);
    }
    printf("~~~~~~~~  now let's check the trie if it same as what you think  ~~~~\n");
    b32t_print_trie(gp_trie);
    printf("%s finish\n", __func__);
    return 0;
}

