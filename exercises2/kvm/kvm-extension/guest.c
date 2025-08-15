#include <stddef.h>
#include <stdint.h>
#include "./common-guest.h"





// static void outb(uint16_t port, uint8_t value) {
// 	asm("outb %0,%1" : /* empty */ : "a" (value), "Nd" (port) : "memory");
// }
static inline void outb(uint16_t port, uint32_t value) {
  	asm("out %0,%1" : /* empty */ : "a" (value), "Nd" (port) : "memory");
}

static inline uint32_t inb(uint16_t port) {
  uint32_t ret;
  asm("in %1, %0" : "=a"(ret) : "Nd"(port) : "memory" );
  return ret;
}
static void display(char *str){
	uintptr_t uptr= (uintptr_t) str;
	outb(PRINT_STRING,uptr);
}

static uint32_t getNumExits(){
	return inb(GET_EXITS);
}
static void printVal(uint32_t val){
	outb(0xE5, val);
}

static uint32_t open_file( char * path){
	open_file_t of ;
	char * p;
	int i=0;
	for (p = path; *p; ++p){
		of.data[i] = path[i];
		i++;
	}
	of.point = 76;
	// display("Below data of open address");
	// printVal((uintptr_t)of);
	// display("Below data of open string address");
	// printVal((uintptr_t)of->data);
	outb(OPEN,(uintptr_t)&of);
	return inb(OPEN);
}



static uint32_t write_to_file(uint32_t fd,char* data , uint32_t len){

	write_file_t wf ; // preallocating 20 bytes
	wf.len = len;
	wf.fd = fd;

	uint32_t i = 0;
	for (i=0;i<len;i++){
		wf.data[i] = data[i] ;
	}
	// display("Below data wf address");
	// printVal((uintptr_t)wf);
	// display("Below data wf string address");
	// printVal((uintptr_t)wf->data);
	// printVal((uint32_t) (*(wf->data+1)));
	// printVal((uintptr_t)(*(wf->data+2)));
	// printVal((uintptr_t)(*(wf->data+3)));
	outb(WRITE,(uintptr_t)&wf);
	return inb(WRITE);
}

static uint32_t read_to_file(uint32_t fd, char * buf, uint32_t len){
	read_file_t rf ;
	rf.len = len;
	uint32_t i=0;
	for (i=0;i<len;i++){
		rf.data[i] = buf[i];
	}
	rf.fd = fd;
	outb(READ,(uintptr_t)&rf);
	uint32_t read_bytes = inb(READ);
	for (i = 0; i < read_bytes; i++)
	{
		buf[i]=rf.data[i];
	}
	return read_bytes;
	
}
//Set to SEEK_SET by default
static uint32_t seek_file(uint32_t fd , int offset ){
	seek_file_t sft;
	sft.fd = fd;
	sft.pos = offset;
	outb(SEEK,(uintptr_t)&sft);
	return inb(SEEK);
}

void
__attribute__((noreturn))
__attribute__((section(".start")))
_start(void) {
	// const char *p;

	// for (p = "Hello, world! from guest"; *p; ++p)
	// 	outb(0xE9, *p);
	// // outb(0xE5, (uint32_t) 2000);
	// printVal(12345678);

	uint32_t op = getNumExits();
	printVal(op);
	char* p1="hello fromm guest";
	display(p1);
	op = getNumExits();
	printVal(op);

	// display(p1);
	int fd = open_file("./thisnewfile.txt");	
	if (fd == -1){
		char arr[30] = "Not able to open file";
		printVal((uintptr_t)arr);
		display(arr);
	}else{
		// printVal(12312);
		char* text_to_write = "Sample text to be written";
		display(text_to_write);
		// printVal(123121);
		write_to_file(fd,text_to_write,25);
		// char t2[20] = "Below is fd";
		// display(t2);
		printVal(fd);

		seek_file(fd,10);
		char read_buf[50];
		int ret = read_to_file(fd,read_buf,20);
		if (ret == 0){
			display("Nothing to read");
		}
		display(read_buf);	
	}
	// // printVal(1);
	// char *fname2 ="./secondfile.txt";
	// // printVal(2);
	// int fd2 = open_file(fname2);	
	// if (fd2 == -1){
	// 	char arr[30] = "Not able to open file";
	// 	printVal((uintptr_t)arr);
	// 	display(arr);
	// }else{
	// 	// printVal(12312);
	// 	char* text_to_write = "Sample text to be written";
	// 	display(text_to_write);
	// 	// printVal(123121);
	// 	write_to_file(fd2,text_to_write,25);
	// 	// char t2[20] = "Below is fd";
	// 	// display(t2);
	// 	printVal(fd2);

	// 	seek_file(fd2,10);
	// 	char read_buf[50];
	// 	int ret = read_to_file(fd2,read_buf,20);
	// 	if (ret == 0){
	// 		display("Nothing to read");
	// 	}
	// 	display(read_buf);
	// }

	
	*(long *) 0x400 = 42;

	for (;;)
		asm("hlt" : /* empty */ : "a" (42) : "memory");
}
