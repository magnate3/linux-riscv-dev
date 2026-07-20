#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/sched.h>
#include <linux/tty.h>
#include "log.h"

DEFINE_SPINLOCK(tty_log_lock);
unsigned long tty_log_lock_flags;

int print_tty(const char* file, const char* func, const int line, const char *fmt,...)
{
  struct tty_struct *tty;
  va_list ap;
  
  char buf0[256];
  char buf[256];
  char* end_str = "\033[0m\015\012";
  int n;

  if(file)
    snprintf(buf0, 256, "\033[34m%s,%s:%d\t", file, func, line);
  else
    snprintf(buf0, 256, "\033[34m%s:%d\t", func, line);

  va_start(ap, fmt);
  n = vsnprintf(buf, 256, fmt, ap);
  va_end(ap);
  tty = get_current_tty();

  if (tty != NULL) {
    spin_lock_irqsave(&tty_log_lock, tty_log_lock_flags);
    (tty->driver)->ops->write (tty, buf0, strlen(buf0));
    (tty->driver)->ops->write (tty, buf, strlen(buf));
    (tty->driver)->ops->write (tty, end_str, strlen(end_str));
    (tty->driver)->ops->flush_buffer (tty);
    spin_unlock_irqrestore(&tty_log_lock, tty_log_lock_flags);
    return n;
  }
  return -1;
}


int print_dmesg(const char* file, const char* func, const int line, const char *fmt,...)
{
  va_list ap;
  
  char buf0[256];
  char buf[256];
  int n;

  if(file)
    snprintf(buf0, 256, "%s,%s:%d", file, func, line);
  else
    snprintf(buf0, 256, "%s:%d", func, line);

  va_start(ap, fmt);
  n = vsnprintf(buf, 256, fmt, ap);
  va_end(ap);

  printk(KERN_ERR "%s\t%s\n", buf0, buf);
  return n;
}
