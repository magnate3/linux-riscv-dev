#include <stdio.h>
#include <assert.h>
#include "list.h"

static void print_list(list __l)
{
    list_node first, last;
    first = list_begin(__l);
    last  = list_end(__l);
    printf("\n[ ");
    for ( ; first != last; first = first->next)
        printf("%d ", *(int*) first->data);
    printf("]\n");
}

int int_compare(const void* __l, const void* __r)
{
    int lhs = *(int*) __l;
    int rhs = *(int*) __r;
    return lhs < rhs ? -1 : (lhs == rhs ? 0 : 1);
}

int main()
{
    list_node first, last;
    int value;
    list li = list_new(int);
    list_set_comparer(int_compare);
    list_push_front(li, 1);
    list_push_back(li, 2);
    list_front(li, value);
    assert(value == 1);
    list_back(li, value);
    assert(value == 2);
    list_pop_back(li);
    list_pop_back(li);

    for (value = 500; value >= 0; --value)
        list_push_front(li, value);

    first = list_begin(li);
    last  = list_end(li);
    for (value = 0; first != last; first = first->next, ++value)
        assert(value == *(int*) first->data);

    list_reverse(li);
    first = list_begin(li);
    last  = list_end(li);
    print_list(li);
    for (value = 500; first != last; first = first->next, --value)
        assert(value == *(int*) first->data);

    list_sort(li);
    first = list_begin(li);
    last  = list_end(li);
    print_list(li);
    for (value = 0; first != last; first = first->next, ++value)
        assert(value == *(int*) first->data);

    list_clear(li);

    for (value = 5; value > 0; --value)
        list_push_back(li, value);

    print_list(li);
    list_sort(li);
    print_list(li);

    list_delete(li);

    return 0;
}
