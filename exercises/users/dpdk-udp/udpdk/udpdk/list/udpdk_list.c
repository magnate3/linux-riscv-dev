
//
// list.c
//
// Copyright (c) 2010 TJ Holowaychuk <tj@vision-media.ca>
//

#include "udpdk_list.h"
#include "udpdk_shmalloc.h"

extern const void *udpdk_list_t_alloc;
extern const void *udpdk_list_node_t_alloc;

/*
 * Allocate a new udpdk_list_t. NULL on failure.
 */

udpdk_list_t *
list_new(void) {
  udpdk_list_t *self;
  if (!(self = udpdk_shmalloc(udpdk_list_t_alloc)))
    return NULL;
  self->head = NULL;
  self->tail = NULL;
  self->free = NULL;
  self->match = NULL;
  self->len = 0;
  return self;
}

/*
 * Free the list.
 */

void
list_destroy(udpdk_list_t *self) {
  unsigned int len = self->len;
  udpdk_list_node_t *next;
  udpdk_list_node_t *curr = self->head;

  while (len--) {
    next = curr->next;
    if (self->free) self->free(curr->val);
    udpdk_shfree(udpdk_list_node_t_alloc, curr);
    curr = next;
  }

  udpdk_shfree(udpdk_list_t_alloc, self);
}

/*
 * Append the given node to the list
 * and return the node, NULL on failure.
 */

udpdk_list_node_t *
list_rpush(udpdk_list_t *self, udpdk_list_node_t *node) {
  if (!node) return NULL;

  if (self->len) {
    node->prev = self->tail;
    node->next = NULL;
    self->tail->next = node;
    self->tail = node;
  } else {
    self->head = self->tail = node;
    node->prev = node->next = NULL;
  }

  ++self->len;
  return node;
}

/*
 * Return / detach the last node in the list, or NULL.
 */

udpdk_list_node_t *
list_rpop(udpdk_list_t *self) {
  if (!self->len) return NULL;

  udpdk_list_node_t *node = self->tail;

  if (--self->len) {
    (self->tail = node->prev)->next = NULL;
  } else {
    self->tail = self->head = NULL;
  }

  node->next = node->prev = NULL;
  return node;
}

/*
 * Return / detach the first node in the list, or NULL.
 */

udpdk_list_node_t *
list_lpop(udpdk_list_t *self) {
  if (!self->len) return NULL;

  udpdk_list_node_t *node = self->head;

  if (--self->len) {
    (self->head = node->next)->prev = NULL;
  } else {
    self->head = self->tail = NULL;
  }

  node->next = node->prev = NULL;
  return node;
}

/*
 * Prepend the given node to the list
 * and return the node, NULL on failure.
 */

udpdk_list_node_t *
list_lpush(udpdk_list_t *self, udpdk_list_node_t *node) {
  if (!node) return NULL;

  if (self->len) {
    node->next = self->head;
    node->prev = NULL;
    self->head->prev = node;
    self->head = node;
  } else {
    self->head = self->tail = node;
    node->prev = node->next = NULL;
  }

  ++self->len;
  return node;
}

/*
 * Insert the given node in the list to 2nd position
 * (or first if the only element)
 * and return the node, NULL or failure
 */
udpdk_list_node_t *
list_spush(udpdk_list_t *self, udpdk_list_node_t *node) {
  if (!node) return NULL;

  if (self->len) {
    node->next = self->head->next;
    node->prev = self->head;
    self->head->next = node;
    if (node->next) {
        node->next->prev = node;
    }
    ++self->len;
    return node;
  } else {
    return list_lpush(self, node);
  }
}

/*
 * Return the node associated to val or NULL.
 */

udpdk_list_node_t *
list_find(udpdk_list_t *self, void *val) {
  udpdk_list_iterator_t *it = list_iterator_new(self, LIST_HEAD);
  udpdk_list_node_t *node;

  while ((node = list_iterator_next(it))) {
    if (self->match) {
      if (self->match(val, node->val)) {
        list_iterator_destroy(it);
        return node;
      }
    } else {
      if (val == node->val) {
        list_iterator_destroy(it);
        return node;
      }
    }
  }

  list_iterator_destroy(it);
  return NULL;
}

/*
 * Return the node at the given index or NULL.
 */

udpdk_list_node_t *
list_at(udpdk_list_t *self, int index) {
  udpdk_list_direction_t direction = LIST_HEAD;

  if (index < 0) {
    direction = LIST_TAIL;
    index = ~index;
  }

  if ((unsigned)index < self->len) {
    udpdk_list_iterator_t *it = list_iterator_new(self, direction);
    udpdk_list_node_t *node = list_iterator_next(it);
    while (index--) node = list_iterator_next(it);
    list_iterator_destroy(it);
    return node;
  }

  return NULL;
}

/*
 * Remove the given node from the list, freeing it and it's value.
 */

void
list_remove(udpdk_list_t *self, udpdk_list_node_t *node) {
  node->prev
    ? (node->prev->next = node->next)
    : (self->head = node->next);

  node->next
    ? (node->next->prev = node->prev)
    : (self->tail = node->prev);

  if (self->free) self->free(node->val);

  udpdk_shfree(udpdk_list_node_t_alloc, node);
  --self->len;
}
