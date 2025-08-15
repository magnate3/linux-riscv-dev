#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include "Event.c"

typedef struct node {
    enum TypesOfEvent type; //type of event
    
    int idElementInGroup;//id of element in group of hosts or switches
    int portID;
    unsigned long endTime;
    struct node * next;
} node;

node* new_node(int type, int idElementInGroup,
                int portID, 
                unsigned long endTime) {
  node *n = (node *)malloc(sizeof(node));
  if (!n) {
    return NULL;
  }
  n->type = type;
  n->idElementInGroup = idElementInGroup;
  n->portID = portID;
  n->endTime = endTime;
  n->next = NULL;

  return n;
}