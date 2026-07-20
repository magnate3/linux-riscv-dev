#ifndef __DLMALLOC_CONFIG_H__
#define __DLMALLOC_CONFIG_H__

#define INSECURE		0
#define ONLY_MSPACES    	1
#define HAVE_MORECORE		0
#define HAVE_MMAP		0
#define MSPACES			1
#define LACKS_SYS_MMAN_H 	1
#define LACKS_SCHED_H		1
#define FOOTERS			0
#define NO_MALLOC_STATS		1
#define PROCEED_ON_ERROR	1
#define malloc_getpagesize	4096

#if !defined(__STANDALONE__) && defined(__KERNEL__)
#include <linux/kernel.h>       // We're doing kernel work
#include <linux/module.h>       // Specifically, a module
#include <linux/moduleparam.h>  // for parameter use
#include <linux/init.h>

#include <linux/fs.h>
#include <linux/cdev.h>         // Char device structure
#include <asm/uaccess.h>        // for get_user and put_user
#define LACKS_SYS_TYPES_H
#define LACKS_ERRNO_H
#define LACKS_TIME_H
#define LACKS_STDLIB_H
#define  LACKS_STRING_H
#define LACKS_UNISTD_H 
#define ABORT  
extern int errno;
#else // #if !defined(__STANDALONE__) && defined(__KERNEL__)
 #include <string.h>
 //#include "sysc_kernel_api.h"

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#endif

#endif
