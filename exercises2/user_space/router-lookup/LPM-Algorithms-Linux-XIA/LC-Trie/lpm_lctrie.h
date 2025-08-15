/* 
 * This is a free software and provides no guarantee of any kind.
 * The distribution and changes to the software is as provided by the LICENSE.
 * View the LICENSE file in github.com/sumefsp/LPM-Algorithms-Linux-XIA and
 * any usage of code from this file must include this declaration.
 *
 * 2015, LPM Algorithms for Linux XIA
 * Garnaik Sumeet, Michel Machado
 */

#ifndef __LPM_LCTRIE_H__
#define __LPM_LCTRIE_H__

#include "lc_trie.h"
#include "xidsort.h"
#include "generate_fibs.h"

#define MINLENGTH 20
#define MAXLENGTH 159

struct routtablerec *lctrie_create_fib(struct nextcreate *table,
		unsigned long size);
unsigned int lookup_bloom(const xid *id, routtable_t table);
int lctrie_destroy_fib(struct routtablerec *rtable);

#endif
