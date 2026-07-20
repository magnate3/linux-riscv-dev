#include<stdio.h>
#include<stdlib.h>
#include "Event.c"

typedef struct node {
    enum TypesOfEvent type; //type of event
    int idElementInGroup;//id of element in group of hosts or switches
    int portID;
    double endTime;
    struct node* next;
    struct node* parent;
} node;

node* new_node(int type, int idElementInGroup, int portID, double priority){
    node* tmp = malloc(sizeof(node));
    tmp->type = type;
    tmp->idElementInGroup = idElementInGroup;
    tmp->portID = portID;
    tmp->endTime = priority;
    tmp->next = NULL;
    tmp->parent = NULL;
    return tmp;
}
