
#include <stdio.h>
#include <stdlib.h>
 
struct node {
    int data;
    struct node* left;
    struct node* right;
};

struct node* newNode (int data) {
    struct node* node = calloc(1, sizeof(*node));
    if (node) node->data = data;
    return node;
}

void inOrder (struct node *node) {
    if (!node) return;
    inOrder(node->left);
    printf("%d ", node->data);
    inOrder(node->right);
}

/*

the tree... 
    2 
   / \ 
  1   3

 is changed to double tree on left ... 
       2 
      / \ 
     2   3 
    /   / 
   1   3 
  / 
 1

*/

/* Use Post order notation. 
   Excellent for things like changing nodes or freeing them.
   */
void doubleTree (struct node * node) {
    if (!node) return;
    doubleTree(node->left);
    doubleTree(node->right);
    struct node *left = node->left;
    struct node *double_node = newNode(node->data);
    node->left = double_node;
    double_node->left = left;

    return;
}
 
int main() {
 
  /* Constructed binary tree is
            4
          /   \
        2      5
      /  \
    1     3
  */
  struct node *root = newNode(4);
  root->left        = newNode(2);
  root->right       = newNode(5);
  root->left->left  = newNode(1);
  root->left->right = newNode(3);
 

  printf("\n Inorder traversal of the constructed tree is \n");
  inOrder(root);

  doubleTree(root);
 
  printf("\n Inorder traversal of the mirror tree is \n");
  inOrder(root);

  getchar();
  return 0;
}
