/***************************************************************************
                          hash_test.c  -  description
                             -------------------
    begin                : Fri Jul 4 2003
    copyright            : (C) 2003 by jfwan
    email                : jfwan@vip.sina.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include <stdio.h>
#include <string.h>
#include "hashtable.h"
#include "cmdline.h"
#include "vector.h"
#include "list.h"

void process_c_opt(int ac, char **av)
{
    int i;
    printf("got -c option: ");
    for (i = 0; i < ac; ++i)
        printf("%s ", av[i]);
    printf("\n");
}

void process_h_opt(int ac, char **av)
{
    int i;
    printf("got --help option: ");
    for (i = 0; i < ac; ++i)
        printf("%s ", av[i]);
    printf("\n");
}

int main(int ac, char **av)
{
    hash_table table = hash_new(0);
    void *result;
    vector v = vector_new(int);
    int value;
    int *p;

    hash_entry entry1 = hash_make_entry("jfwan", "Junfeng Wan");
    hash_entry entry2 = hash_make_entry("xhliang", "Xihua Liang");

    hash_insert(table, entry1);
    hash_insert(table, entry2);

    result = hash_lookup(table, "jfwan");

    printf("%s\n", (char*) result);

    register_option("-c", process_c_opt, "just for test");
    register_option("--help", process_h_opt, "just for test");
    process_cmdline(ac, av);

    vector_push_back(v, 1);
    vector_push_back(v, 2);
    vector_pop_back(v, value);
    printf("%d\n", value);
    vector_pop_back(v, value);
    printf("%d\n", value);

    for (value = 0; value < 1000; ++value)
        vector_push_back(v, value);
    for (p = vector_begin(v); p < (int*) vector_end(v); ++p)
        printf("%d ", *p);
    putchar('\n');

    vector_delete(v);
    printf("vector unit test block end\n\n\n");

    /* list unit test block */
    {
        list_node first, last;
        int value;
        list li = list_new(int);
        list_push_front(li, 1);
        list_push_back(li, 2);
        list_front(li, value);
        printf("%d\n", value);
        list_back(li, value);
        printf("%d\n", value);
        list_pop_back(li);
        list_pop_back(li);

        for (value = 256; value >= 0; --value)
            list_push_front(li, value);
#if 0
        for (value = 501; value < 1000; ++value)
            list_push_back(li, value);
#endif
        first = list_begin(li);
        last  = list_end(li);
        for ( ; first != last; first = first->next)
            printf("%d ", *(int*) first->data);
        printf("\n\n\n");

        list_reverse(li);
        first = list_begin(li);
        last  = list_end(li);
        for ( ; first != last; first = first->next)
            printf("%d ", *(int*) first->data);
        printf("\n\n\n");

        list_sort(li);
        first = list_begin(li);
        last  = list_end(li);
        for ( ; first != last; first = first->next)
            printf("%d ", *(int*) first->data);
        printf("\n\n\n");

        list_clear(li);
        for (value = 5; value > 0; --value)
            list_push_back(li, value);
        list_sort(li);
        first = list_begin(li);
        last  = list_end(li);
        for ( ; first != last; first = first->next)
            printf("%d ", *(int*) first->data);
        printf("\n\n\n");

        list_delete(li);
    }

    return 0;
}
