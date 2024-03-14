/* IO PORT USED*/
#include<unistd.h>
#include<sys/types.h>
#define HYPERCALL 0X80      
#define PRINT_STRING_PORT 0xE5
#define NUM_EXIT_PORT 0XE6
#define PRINT_VAL_PORT 0XE7
#define CHAR_PORT 0xE9
#define HC_OPEN (HYPERCALL | 1)
#define HC_READ (HYPERCALL | 2)
#define HC_WRITE (HYPERCALL | 3)
#define HC_CLOSE (HYPERCALL | 4)
#define HC_LSEEK (HYPERCALL | 5)
#define HC_EXIT (HYPERCALL | 6 )
#define HC_PRINT_STRING (HYPERCALL | 7)
#define HC_PRINT_VALUE (HYPERCALL | 8)
#define HC_NUM_EXIT (HYPERCALL | 9)
#define HC_O (HYPERCALL | 10)

/*struct used for file system calls */
struct open_data{
    uint32_t fileoffset;
    int flags;
	mode_t mode;
	int fd;
	int errno_hc;
};
struct close_data{
	int close_fd;
	int result;
	int errno_hc;
};
struct read_write_data{
	int fd;
	uint32_t buffer_offset;
	size_t count;
	ssize_t result;
	int errno_hc;
};
struct lseek_data{
	int fd;
	off_t offset;
	int whence;
	off_t result;
	int errno_hc;
};
