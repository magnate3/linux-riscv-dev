#ifndef __ZLOG_RING_BUF_H__
#define __ZLOG_RING_BUF_H__


// zero copy log
typedef struct zlog_ring_s {
	uint8_t magic;
	uint8_t ver;
	uint16_t head;
	uint16_t tail;
	uint16_t slot_cnt;
	uint16_t slot_size;
	uint16_t _dummy; // be careful to be aligned
    uint32_t ring_mem_size;

	char data[0];
} zlog_ring_t;

typedef struct zlog_read_s {
    uint32_t owner;
	uint16_t start;
	uint16_t cnt;
} zlog_read_t;

// per zlog slot
typedef struct zlog_s {
	uint32_t    owner;
	char        msg[252];
} zlog_t;


#define MAX_ZLOG_MEM_ORDER 16   // 64KB
#define ZLOG_SLOT_SIZE 256      // power of 2


inline char* zlog_get_slot_addr(zlog_ring_t *zring, uint32_t idx) 
{
    return (char*)zring->data + ((idx % zring->slot_cnt) * zring->slot_size);
}

#endif
