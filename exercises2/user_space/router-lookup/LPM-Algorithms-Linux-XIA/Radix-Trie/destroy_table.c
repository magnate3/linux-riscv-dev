#include "radix.h"
#include "lpm_radix.h"

int free_trie(struct node_patric *n)
{
	if (NULL == n->left) {
		free(n);
		return 0;
	} else {
		free_trie(n->left);
		free_trie(n->right);
		free(n);
		return 0;
	}
}
