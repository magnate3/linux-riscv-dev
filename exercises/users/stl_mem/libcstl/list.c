/* $Id: list.c,v 1.4 2003/09/15 12:47:46 jfwan Exp $ */
#include <string.h>
#include "list.h"
#include "allocator.h"

/**
 * @author jfwan@vip.sina.com
 * @date   09/15/2003
 */

typedef struct list_struct
{
    list_node node;
    size_t    bytes;
} list_;

static comp_fun  s_node_comp = NULL;
static list_node s_get_node(list l);
static void      s_put_node(list l, list_node node);
static list_node s_create_node(list l, const void *x);
static size_t    s_distance(list_node b, list_node e);
static void      s_transfer(list l, list_node position, list_node b, list_node e);
static int       s_node_compare(const void* lhs, const void* rhs);

list list_new_aux(size_t sz)
{
    list l = (list) allocate(sizeof(list_));
    l->bytes = sz;
    l->node = s_get_node(l);
    l->node->prev = l->node;
    l->node->next = l->node;

    return l;
}

void list_delete_aux(list *l)
{
    list_clear(*l);
    s_put_node(*l, (*l)->node);
    deallocate(*l, sizeof(list_));
    *l = NULL;
}

list_node list_begin(list l)
{
    return l->node->next;
}

list_node list_end(list l)
{
    return l->node;
}

list_node list_rbegin(list l)
{
    return l->node->prev;
}

list_node list_rend(list l)
{
    return l->node;
}

boolean list_empty(const list l)
{
    return (boolean) (l->node->next == l->node);
}

size_t list_size(const list l)
{
    return s_distance(list_begin(l), list_end(l));
}

size_t list_max_size(const_list l)
{
    return (size_t) -1;
}

void list_resize(list l, size_t new_sz)
{
    list_node iter = list_begin(l);
    list_node end  = list_end(l);
    size_t    len  = 0;

    for ( ; iter != end && len < new_sz; iter = iter->next, ++len)
        ;
    if (len == new_sz)
        list_erase_range(l, iter, end);

    list_insert_n(l, end, new_sz - len, 0); /* iter == end */
}

void list_front_aux(list l, void *x)
{
    memcpy(x, list_begin(l)->data, l->bytes);
}

void list_back_aux(list l, void *x)
{
    memcpy(x, list_end(l)->prev->data, l->bytes);
}

void list_push_front_aux(list l, void *x)
{
    list_insert_aux(l, list_begin(l), x);
}

void list_pop_front(list l)
{
    list_erase(l, list_begin(l));
}

void list_push_back_aux(list l, void *x)
{
    list_insert_aux(l, list_end(l), x);
}

void list_pop_back(list l)
{
    list_erase(l, list_end(l)->prev);
}

list_node list_insert_aux(list l, list_node position, void *x)
{
    list_node new_node = s_create_node(l, (void*) &x);
    new_node->prev = position->prev;
    new_node->next = position;
    position->prev->next = new_node;
    position->prev = new_node;

    return new_node;
}

void list_insert_n_aux(list l, list_node position, size_t n, void *x)
{
    for ( ; n > 0; --n)
        list_insert_aux(l, position, x);
}

list_node list_erase(list l, list_node node)
{
    list_node ret = node->next;
    node->prev->next = node->next;
    node->next->prev = node->prev;
    s_put_node(l, node);

    return ret;
}

list_node list_erase_range(list l, list_node b, list_node e)
{
    while ((b = list_erase(l, b)) != e)
        ;
    return e;
}

/* XXX: I'm wondering whether bytes need to be swaped,
 * because I suppose the 2 lists have the same data type.
 */
void list_swap(list x, list y)
{
    list_node node  = x->node;
    /* size_t    bytes = x->bytes; */
    x->node  = y->node;
    /* x->bytes = y->bytes; */
    y->node  = node;
    /* y->bytes = bytes; */
}

void list_clear(list l)
{
    list_erase_range(l, list_begin(l), list_end(l));
}

void list_splice(list l, list_node position, list_node b, list_node e)
{
    if (b != e)
        s_transfer(l, position, b, e);
}

void list_remove_aux(list l, void *x)
{
    list_node b = list_begin(l);
    list_node e = list_end(l);

    while (b != e) {
        list_node tmp = b->next;
        if (memcmp(b->data, (void*) &x, l->bytes) == 0)
            list_erase(l, b);
        b = tmp;
    }
}

/* Remove equal and adjacent elements, and keep one */
void list_unique(list l)
{
    list_node b = list_begin(l);
    list_node e = list_end(l);
    list_node nt;

    if (b == e) return;

    nt = b;
    while ((nt = nt->next) != e) {
        if (memcmp(b->data, nt->data, l->bytes) == 0)
            list_erase(l, nt);
        else
            b = nt;
        nt = b;
    }
}

/* NOTE: these 2 lists should be in increasing order,
 * and the 2 lists must have the same data type
 */
void list_merge(list dest, list src)
{
    list_node b1 = list_begin(dest), e1 = list_end(dest);
    list_node b2 = list_begin(src),  e2 = list_end(src);

    while (b1 != e1 && b2 != e2) {
/*        if (primitive_compare(b1->data, b2->data, dest->bytes) == 1) { */
        if (s_node_compare(&b1->data, &b2->data) < 0) {
            list_node tmp = b2->next;
            s_transfer(dest, b1, b2, tmp);
            b2 = tmp;
        } else {
            b1 = b1->next;
        }
    }

    if (b2 != e2) s_transfer(dest, e1, b2, e2);
}

#ifdef LIST_DEBUG 
/* XXX: temporary use for test */
#include <stdio.h>
static void s_print_list(list l)
{
    list_node b = list_begin(l);
    list_node e = list_end(l);

    for ( ; b != e; b = b->next)
        printf("%d ", *(int*) b->data);
    printf("\n");
}
#endif

void list_sort(list l)
{
    list carry = list_new_aux(l->bytes);
    list counter[64];
    int i, fill = 0;
    /* if length is 0 or 1, do nothing! */
    if (l->node->next == l->node || l->node->next->next == l->node)
        return;
    for (i = 0; i < 64; ++i)
        counter[i] = list_new_aux(l->bytes);

    while (!list_empty(l)) {
        list_splice(carry, list_begin(carry), list_begin(l), list_begin(l)->next);
        i = 0;
        while (i < fill && !list_empty(counter[i])) {
            list_merge(counter[i], carry);
            list_swap(carry, counter[i++]);
        }
        list_swap(carry, counter[i]);
        if (i == fill) ++fill;
    }

#ifdef LIST_DEBUG
    for (i = 0; i < fill; ++i) {
        printf("counter[%d]:\n", i);
        s_print_list(counter[i]);
        printf("\n");
    }
#endif

    for (i = 1; i < fill; ++i)
        list_merge(counter[i], counter[i-1]);
    list_swap(l, counter[fill-1]);

    for (i = 0; i < 64; ++i)
        list_delete(counter[i]);
    list_delete(carry);
}

void list_reverse(list l)
{
    list_node b = list_begin(l);
    list_node e = list_end(l);
    if (b == e || b->next == e) return;

    b = b->next;
    while (b != e) {
        list_node tmp = b;
        b = b->next;
        s_transfer(l, list_begin(l), tmp, b);
    }
}

comp_fun list_set_comparer(comp_fun func)
{
    comp_fun old_fun = s_node_comp;
    s_node_comp = func;
    return old_fun;
}

static list_node s_get_node(list l)
{
    return (list_node) allocate(sizeof(list_node_));
}

static void s_put_node(list l, list_node node)
{
    deallocate(node, l->bytes);
}

static list_node s_create_node(list l, const void *x)
{
    void *tmp = allocate(l->bytes);
    list_node node = s_get_node(l);
    memcpy(tmp, x, l->bytes);
    node->data = tmp;
    return node;
}

static size_t s_distance(list_node b, list_node e)
{
    size_t count = 0;

    while (b != e)
    {
        b = b->next;
        ++count;
    }

    return count;
}

static void s_transfer(list l, list_node position, list_node b, list_node e)
{
    list_node tmp        = e->prev;
    b->prev->next        = e;
    e->prev              = b->prev;
    position->prev->next = b;
    b->prev              = position->prev;
    position->prev       = tmp;
    tmp->next            = position;
}

static int       s_node_compare(const void* lhs, const void* rhs)
{
    if (NULL == s_node_comp)
        error("Comparer of list elements should be set first!\n");
    return (*s_node_comp)(lhs, rhs);
}
