/* $Id: cmdline.c,v 1.3 2003/09/15 12:47:46 jfwan Exp $ */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cmdline.h"

/**
 * @author jfwan@vip.sina.com
 * @date   09/15/2003
 */

typedef struct option_entry_struct
{
    char           *name;
    option_handler  func;
    char           *desc;
} option_entry_;

static hash_table s_option_table = NULL;
static unknown_handler s_unknown_dealer = &unknown_option;

static void cmdline_destroy();

option_entry option_entry_new(const char *name, option_handler func, const char *desc)
{
    option_entry ent;
    if (s_option_table == NULL)
        s_option_table = hash_new(0);
    ent = (option_entry) pmalloc(hash_get_pool(), sizeof(option_entry_));
    ent->name = pstrdup(hash_get_pool(), name);
    ent->func = func;
    ent->desc = pstrdup(hash_get_pool(), desc);
    return ent;
}

void register_option(const char *name, option_handler func, const char *desc)
{
    option_entry ent;
    hash_entry   he;

    if (s_option_table == NULL)
        s_option_table = hash_new(0);

    ent = option_entry_new(name, func, desc);
    he  = hash_make_entry(ent->name, ent);
    hash_insert(s_option_table, he);
}

void register_unknown_handler(unknown_handler func)
{
    s_unknown_dealer = func;
}

void process_cmdline(int ac, char **av)
{
    int i, count;
    char **tmp;
    void *value;

    for (i = 1; i < ac; ++i) {
        if (av[i][0] != '-')
            continue;
        tmp = &av[i+1];
        count = 0;
        while (*tmp != NULL && (*tmp++)[0] != '-')
            ++count;
        value = hash_lookup(s_option_table, av[i]);
        if (value == NULL)
            (*s_unknown_dealer)(av[i]);
        else
            ((option_entry)value)->func(count, &av[i+1]);
        i += count;
    }
    cmdline_destroy();
}

void unknown_option(char *opt)
{
    fprintf(stderr, "option %s unknown: ignored\n", opt);
}

static void cmdline_destroy()
{
    hash_destroy(&s_option_table);
}
