#ifndef _LOG_H_
# define _LOG_H_

# define LOG_INFO 5
# define DEBUG

# ifndef LOG_LEVEL
#  ifdef DEBUG
#   define LOG_LEVEL 10
#  else // !DEBUG
#   define LOG_LEVEL 1
#  endif // DEBUG
# endif // LOG_LEVEL



//# define DEBUG_TTY
//# define DEBUG_PRINTK

# ifdef DEBUG_TTY
#  include <linux/spinlock.h>
int print_tty(const char* file, const char* func, const int line, const char *fmt,...);
#  define LOG_KERN(LEVEL, FMT, ...) do {\
        if ((LEVEL) < LOG_LEVEL) {\
            print_tty(NULL,__FUNCTION__,__LINE__,FMT,##__VA_ARGS__); \
        } \
   } while(0);
# else
#  ifdef DEBUG_PRINTK
int print_dmesg(const char* file, const char* func, const int line, const char *fmt,...);
#   define LOG_KERN(LEVEL, FMT, ...) do {\
        if ((LEVEL) < LOG_LEVEL) {\
            print_dmesg(NULL,__FUNCTION__,__LINE__,FMT,##__VA_ARGS__); \
        }\
    } while(0);
#  else
#   define LOG_KERN(LEVEL, FMT, ...) ;
#  endif
# endif

#endif // _LOG_H_

