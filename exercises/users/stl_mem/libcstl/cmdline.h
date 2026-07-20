/* $Id: cmdline.h,v 1.3 2003/09/15 12:47:46 jfwan Exp $ */
#ifndef WAN_CMDLINE_H
#define WAN_CMDLINE_H

/**
 * @author jfwan@vip.sina.com
 * @date   09/15/2003
 */

#include "general.h"
#include "hashtable.h"

typedef void (*option_handler)(int, char**);
typedef void (*unknown_handler)(char*);

EXTERN_TYPE(option_entry);

option_entry option_entry_new(const char *name, option_handler func, const char *desc);
void         register_option(const char *name, option_handler func, const char *desc);
void         register_unknown_handler(unknown_handler func);
void         process_cmdline(int ac, char **av);
void         unknown_option(char *opt);

#endif // WAN_CMDLINE_H
