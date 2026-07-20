
#include "bits32_trie_new.h"
#include<string.h>

struct b32t_key_vector *b32t_leaf_new(b32t_key key, struct b32t_fib_alias *fa){
	struct b32t_key_vector *l;
	struct b32t_tnode *kv;

	kv = malloc(B32T_LEAF_SIZE);
	if (!kv)
		return NULL;

	/* initialize key vector */
	l = kv->kv;
	l->key = key;
	l->pos = 0;
	l->bits = 0;

	/* link leaf to fib alias */
	INIT_HLIST_HEAD(&l->leaf);
    if(fa){
        hlist_add_head(&fa->fa_list, &l->leaf);
    }
	return l;
}
static struct b32t_tnode *b32t_tnode_alloc(int bits){
	size_t size;

	/* determine size and verify it is non-zero and didn't overflow */
	size = B32T_TNODE_SIZE(1ul << bits);
	return malloc(size);
}

struct b32t_key_vector *b32t_tnode_new(b32t_key key, int pos, int bits){
	unsigned int shift = pos + bits;
	struct b32t_key_vector *tn;
	struct b32t_tnode *tnode;

	/* verify bits and pos their msb bits clear and values are valid */
	if(!bits || (shift > B32T_KEYLENGTH)){
		printf("BUG: failure at %s:%d/%s()!\n", __FILE__, __LINE__, __func__);
		//panic("BUG!");
	}

	tnode = b32t_tnode_alloc(bits);
	if (!tnode){
        printf("alloc failed:%d\n", bits);
		return NULL;
    }

	tn = tnode->kv;
	tn->key = (shift < B32T_KEYLENGTH) ? (key >> shift) << shift : 0;
	tn->pos = pos;
	tn->bits = bits;

	return tn;
}

struct b32t_trie *b32t_trie_new(void){
    struct b32t_trie *ptrie;

    ptrie = malloc(sizeof(struct b32t_trie) + sizeof(struct b32t_key_vector) );
    if(ptrie == NULL){
        return NULL;
    }
    memset(ptrie, 0, sizeof(struct b32t_trie) + sizeof(struct b32t_key_vector));
    ptrie->kv[0].pos = B32T_KEYLENGTH;
    return ptrie;
}
