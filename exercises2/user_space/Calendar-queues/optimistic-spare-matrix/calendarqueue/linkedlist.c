#include "Node.c"

typedef struct node* List;

List * makelist();
void add(struct node* entry, List * list);
struct node* delete(struct node* entry, List * list);
void display(List * list, int num);//num is the number of nodes on one line
void reverse(List * list);
void destroy(List * list);

List * makelist(){
  List * list = malloc(sizeof(List));
  if (!list) {
    return NULL;
  }
  *list = NULL;
  return list;
}

void display(List * list, int length) {
  node * current = *list;
  if(*list == NULL) 
    return;
  int i = 1;
  for(; current != NULL; current = current->next) {
    printf("%ld", current->endTime);
    if(i >= length) {
      printf("\n->");
      i = 1;
    }
    else {
      printf("->");
      i++;
    }
  }
}

int compare_nodes(node *n1, node *n2) {
	return (n1->endTime > n2->endTime);
}

void add(node* entry, List * list) {
  node *previous = NULL;
  node *current = NULL;
  if (entry->next != NULL) entry->next = NULL;
	if(*list == NULL){
    *list = entry;
  }
	else {
		previous = NULL;
		current = *list;
		while (current && compare_nodes(entry,current) > 0) {
			previous = current;
			current = current->next;
		}
		if (current == NULL) {
			previous->next = entry;
		}
		else {
			if (previous != NULL) {
				entry->next = previous->next;
				previous->next = entry;
			}
			else {
				entry->next = *list;
				*list = entry;
			}
		}
	}
}

node* delete(node* entry, List * list){
  node * current = *list;            
  node * previous = current;
  if(entry == *list && entry != NULL) {
    *list = current->next;
	current->next = NULL;
    return current;
  }
  while(current != NULL){   
    if(current->next == entry){      
		previous->next = current->next;
		if(current == *list)
			*list = current->next;
		current->next = NULL;
		break;
    }                               
    previous = current;             
    current = current->next;        
  }
  return current;                                 
}

void destroy(List * list){
  node * current = *list;
  node * next = current;
  while(current != NULL){
    next = current->next;
    free(current);
    current = next;
  }
  free(list);
}
