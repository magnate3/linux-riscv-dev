#include <stdio.h>
#include <stdlib.h>
#include "Event.c"

//int sizeTree = 0;
//int maxSize = 0;

typedef struct node {
    enum TypesOfEvent type; //type of event

    int idElementInGroup;//id of element in group of hosts or switches
    int portID;
    unsigned long endTime;
    struct node *left;
    struct node *right;
    struct node *parent;
}node;


typedef struct splay_tree {
  struct node *root;
}splay_tree;

node* new_node(int type, int idElementInGroup,
                int portID,
                unsigned long endTime) {
  node *n = (node *)malloc(sizeof(node));
  //int sizeOfNode = sizeof(node);
  //printf("-------------sizeofNode = %d----------\n", sizeOfNode);
  n->type = type;
  n->idElementInGroup = idElementInGroup;
  n->portID = portID;
  n->endTime = endTime;
  n->parent = NULL;
  n->right = NULL;
  n->left = NULL;

  return n;
}

node* minimum(splay_tree *t) {
    node *x = t->root;
    if(x == NULL)
        return x;
    while(x->left != NULL)
        x = x->left;

    return x;
}


splay_tree* new_splay_tree() {
  splay_tree *t = (splay_tree*)malloc(sizeof(splay_tree));
  t->root = NULL;

  return t;
}

void left_rotate(splay_tree *t, node *x) {
  node *y = x->right;
  x->right = y->left;
  if(y->left != NULL) {
    y->left->parent = x;
  }
  y->parent = x->parent;
  if(x->parent == NULL) { //x is root
    t->root = y;
  }
  else if(x == x->parent->left) { //x is left child
    x->parent->left = y;
  }
  else { //x is right child
    x->parent->right = y;
  }
  y->left = x;
  x->parent = y;
}

void right_rotate(splay_tree *t, node *x) {
  node *y = x->left;
  x->left = y->right;
  if(y->right != NULL) {
    y->right->parent = x;
  }
  y->parent = x->parent;
  if(x->parent == NULL) { //x is root
    t->root = y;
  }
  else if(x == x->parent->right) { //x is left child
    x->parent->right = y;
  }
  else { //x is right child
    x->parent->left = y;
  }
  y->right = x;
  x->parent = y;
}

void splay(splay_tree *t, node *n) {
  while(n->parent != NULL) { //node is not root
    if(n->parent == t->root) { //node is child of root, one rotation
      if(n == n->parent->left) {
        right_rotate(t, n->parent);
      }
      else {
        left_rotate(t, n->parent);
      }
    }
    else {
      node *p = n->parent;
      node *g = p->parent; //grandparent

      if(g == NULL)
      {
        if(n->parent->left == n)
        {
          p->parent = n;
          p->left = n->right;
          if(p->left != NULL)
          {
            p->left->parent = p;
          }
          n->right = p;
          n->parent = NULL;
          if(t->root->right == n)
          {
            t->root->right = NULL;
          }
          n->left = t->root;
        }
        else{
          if(n->parent->right == n)
          {
            p->parent = n;
            p->right = n->left;

            if(p->right != NULL)
            {
              p->right->parent = p;
            }
            n->left = p;
            n->parent = NULL;
            if(t->root->left == n)
            {
              t->root->left = NULL;
            }
            n->right = t->root;
          }
        }
        t->root = n;
      }
      else{
        if(n->parent->left == n && p->parent->left == p) { //both are left children
          right_rotate(t, g);
          right_rotate(t, p);
        }
        else if(n->parent->right == n && p->parent->right == p) { //both are right children
          left_rotate(t, g);
          left_rotate(t, p);
        }
        else if(n->parent->right == n && p->parent->left == p) {
          left_rotate(t, p);
          right_rotate(t, g);
        }
        else if(n->parent->left == n && p->parent->right == p) {
          right_rotate(t, p);
          left_rotate(t, g);
        }
      }
    }
  }


}

void enqueue(splay_tree *t, node *n) {
  //sizeTree++;
  //if(maxSize < sizeTree) maxSize = sizeTree;

  node *y = NULL;
  node *temp = t->root;
  while(temp != NULL) {
    y = temp;
    if(n->endTime < temp->endTime)
      temp = temp->left;
    else
      temp = temp->right;
  }
  n->parent = y;

  if(y == NULL) //newly added node is root
    t->root = n;
  else if(n->endTime < y->endTime)
    y->left = n;
  else
    y->right = n;

  splay(t, n);
}

node* maximum(splay_tree *t, node *x) {
  while(x->right != NULL)
    x = x->right;
  return x;
}

node* find(node *t, int key)
{
  if(t == NULL) return NULL;
  if(t->endTime < key)
  {
    return find(t->right, key);
  }
  if(t->endTime > key)
  {
    return find(t->left, key);
  }
  if(t->endTime == key)
    return t;
}


void delete(splay_tree *t, node *n) {
  splay(t, n);

  splay_tree *left_subtree = new_splay_tree();
  left_subtree->root = t->root->left;
  if(left_subtree->root != NULL)
    left_subtree->root->parent = NULL;

  splay_tree *right_subtree = new_splay_tree();
  right_subtree->root = t->root->right;
  if(right_subtree->root != NULL)
    right_subtree->root->parent = NULL;

  free(n);

  if(left_subtree->root != NULL) {
    node *m = maximum(left_subtree, left_subtree->root);
    splay(left_subtree, m);
    left_subtree->root->right = right_subtree->root;
    if(left_subtree->root->right != NULL)
      left_subtree->root->right->parent = left_subtree->root;
    t->root = left_subtree->root;
  }
  else {
    t->root = right_subtree->root;
  }
}

void delete_key(splay_tree *t, int key)
{
  node *temp = find(t->root, key);
  if(temp == NULL)
    return;
  delete(t, temp);
}

node* dequeue(splay_tree *t)
{
    //sizeTree--;
    node *x = minimum(t);
    node *temp = new_node(A, 0, 0, 0);

    temp->type = x->type;
    temp->idElementInGroup = x->idElementInGroup;
    temp->portID = x->portID;
    temp->endTime = x->endTime;

    delete(t, x);
    return temp;
}

void inorder(splay_tree *t, node *n) {
  if(n != NULL) {
    inorder(t, n->left);
    //printf("%d\n", n->data);
    inorder(t, n->right);
  }
}

void leaf(node *t, enum Side side)
{
    printf("===========> ");
    if(side == LEFT)
    {
        printf("left ");
    }
    else{
        printf("right ");
    }
    if(t != NULL)
    {
        printf("for event type = %d at end = %ld in %d\n",
                t->type, t->endTime, t->
                            //idNode
                            idElementInGroup
                    );
        leaf(t->left, LEFT);
        leaf(t->right, RIGHT);
        printf("\n");
    }
    else{
        printf("NULL\n");
    }
}

void show(splay_tree * t)
{
    if(t != NULL)
    {
        printf("\n===========> for event type = %d at end = %ld in %d\n",
                t->root->type, t->root->endTime, t->root->
                                        //idNode
                                        idElementInGroup
                                    );
        leaf(t->root->left, LEFT);
        leaf(t->root->right, RIGHT);
        printf("\n");
    }
    else{
        printf("===========> left NULL\n");
    }
}

