#ifndef __UTILS_H__
#define __UTILS_H__

#define START \
	pr_info("%s: START\n", __func__)

#define END\
	pr_info("%s: END\n", __func__)

#define START_THREAD \
	pr_info("%s-%d: START\n", __func__, current->pid)

#define END_THREAD\
	pr_info("%s-%d: END\n", __func__, current->pid)

#endif
