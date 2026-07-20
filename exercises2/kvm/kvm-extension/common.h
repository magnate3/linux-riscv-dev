#define GET_EXITS 0xF3 
#define PRINT_STRING 0xF4 
#define OPEN 0x81 
#define READ 0x82 
#define WRITE 0x83 
#define SEEK 0x84 

typedef struct open_file
{
	char data[50];
	uint32_t point;
}open_file_t;

typedef struct write_file
{
	char data[50];
	uint32_t len;
	uint32_t fd;
}write_file_t;

typedef struct read_file
{
	char data[50];
	uint32_t len;
	uint32_t fd;
}read_file_t;

typedef struct seek_file
{
	uint32_t fd;
    int pos;
    uint32_t whence;
}seek_file_t;