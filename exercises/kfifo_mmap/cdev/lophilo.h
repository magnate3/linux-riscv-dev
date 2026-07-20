#ifndef __LOPHILO_H__
#define __LOPHILO_H__

#ifdef __KERNEL__
#include <linux/types.h>
#else
#include <stdint.h>
#endif

/*
enum lophilo_commands {
	LOPHILO_CMD_UNSPEC,
	LOPHILO_CMD_REGISTER,
	LOPHILO_CMD_UPDATE,
	__LOPHILO_CMD_MAX,
};
#define LOPHILO_CMD_MAX (__LOPHILO_CMD_MAX - 1)

enum request_attributes {
	LOPHILO_R_UNSPEC,
	LOPHILO_R_TYPE,
	LOPHILO_R_PERIOD_MS,
	LOPHILO_R_SOURCE,
        __LOPHILO_R_MAX,
};
#define LOPHILO_R_MAX (__LOPHILO_R_MAX - 1)

enum lophilo_sources_types {
	LOPHILO_TYPE_UNSPEC,
	LOPHILO_TYPE_TIME,
	LOPHILO_TYPE_ADC,
        __LOPHILO_TYPE_MAX,
};
*/

enum lophilo_sources {
	LOPHILO_SOURCE_UNSPEC,
	LOPHILO_LAST_UPDATE_SEC,
	LOPHILO_LAST_UPDATE_USEC,
	LOPHILO_PIN_XA0,
        __LOPHILO_SOURCE_MAX,
};
#define LOPHILO_SOURCE_MAX (__LOPHILO_SOURCE_MAX - 1)

enum update_attributes {
	LOPHILO_U_UNSPEC,
	LOPHILO_U_SOURCE,
	LOPHILO_U_VALUE,
        __LOPHILO_U_MAX,
};
#define LOPHILO_U_MAX (__LOPHILO_U_MAX - 1)

struct lophilo_update {
	uint32_t source;
	uint32_t value;
};
typedef struct lophilo_update lophilo_update_t;

struct lophilo_data {
	lophilo_update_t updates[LOPHILO_SOURCE_MAX+1];
};

#define LOPHILO_FIFO_SIZE       32

#endif
