#include <stddef.h>
#include <stdint.h>
#include"filesystem.h"
#include<fcntl.h>
#include<errno.h>
#include<stdarg.h>
int errno_hc;
static void outb8(uint16_t port, uint8_t value) {
	asm("outb %0,%1" : /* empty */ : "a" (value), "Nd" (port) : "memory");
}
/*
__attribut__ are used to allow compiler  optimization in code generated
here noreturn means function will not return any value
In function for loop will travesrse trough each character and output to file
*/
static inline void printVal(uint32_t value) {
	asm("out %0,%1" : /* empty */ : "a" (value), "Nd" (HC_PRINT_VALUE) : "memory");
}

void display(char *str){
	int offset=str-(char*)0X0;
	asm("out %0, %1": :"a"(offset),"Nd"(HC_PRINT_STRING) :"memory");
}
uint32_t getNumExits(){
	uint32_t exits=0;
	asm("in %1,%0" : "=a" (exits): "Nd" (HC_NUM_EXIT) : "memory");
	return exits;
}
int open_hc( char *pathname,int flags,...){
	va_list ap;
	va_start(ap,flags);
	struct open_data data[1];
	data[0].flags=flags;
	data[0].mode=va_arg(ap,mode_t);
	printVal(data[0].mode);
	data[0].fileoffset=pathname-(char*)0X0;
	data[0].fd=-1;
	va_end(ap);
	int offset=(char*)&data-(char*)0X0;
	asm("outl %0, %1" : : "a"(offset),"Nd"(HC_OPEN) : "memory" );
	errno_hc=data[0].errno_hc;
	return data[0].fd;
}
int close_hc(int close_fd){
	struct close_data data[1];
	data[0].close_fd=close_fd;
	uint32_t offset=(char*)data-(char*)0X0;
	asm("outl %0, %1" : : "a"(offset),"Nd"(HC_CLOSE) : "memory" );
	errno_hc=data[0].errno_hc;
	return data[0].result;
}
ssize_t read_hc(int fd,void* buf,int count){
	struct read_write_data data[1];
	data[0].fd=fd;
	data[0].count=count;
	data[0].buffer_offset=(char*)buf-(char*)0X0;
	uint32_t offset=(char*)data-(char*)0X0;
	asm("outl %0, %1" : : "a"(offset),"Nd"(HC_READ) : "memory" );
	errno_hc=data[0].errno_hc;
	return data[0].result;
}
ssize_t write_hc(int fd,void* buf,int count){
	struct read_write_data data[1];
	data[0].fd=fd;
	data[0].count=count;
	data[0].buffer_offset=(char*)buf-(char*)0X0;
	uint32_t offset=(char*)data-(char*)0X0;
	asm("outl %0, %1" : : "a"(offset),"Nd"(HC_WRITE) : "memory" );
	errno_hc=data[0].errno_hc;
	return data[0].result;
}
off_t lseek_hc(int fd,off_t offset,int whence){
	struct lseek_data data[1];
	data[0].fd=fd;
	data[0].offset=offset;
	data[0].whence=whence;
	uint32_t _offset=(char*)data-(char*)0X0;
	asm("outl %0, %1" : : "a"(_offset),"Nd"(HC_LSEEK) : "memory" );
	errno_hc=data[0].errno_hc;
	return data[0].result;
}
void
__attribute__((noreturn))
__attribute__((section(".start")))
_start(void) {
	display("*******************************GUEST OS BOOTED********************************\n");
	uint32_t numExits = getNumExits();
	char *str="Number of exits required to print this string \0";
	display(str);
	numExits = getNumExits()-numExits;
	printVal(numExits);
	display("\n");
	//char str1[]={'t','e','s','t','.','t','x','t','\0'};
	display("checking open \n");
	int fd=open_hc("test.txt",O_RDWR|O_APPEND,S_IRWXU);
	//int fd=open_hc("test4.txt",O_CREAT|O_RDWR|O_TRUNC,S_IRWXG);
	write_hc(1,"Paras Garg",11);
	//	int fd=open_hc("test2.txt",O_RDWR|O_APPEND,S_IRWXU);
	if(fd!=-1){
		lseek_hc(fd,1,SEEK_CUR);
		char read_array[10];
		int n_read=read_hc(fd,read_array,10);
		if(n_read!=-1){
			read_array[n_read-1]='\0';
			display(read_array);
			display("\n");	
		}
		char write_array[]={'h','c','r','m','n'};
		int n_write=write_hc(fd,write_array,5);
		display("redff \n");
		n_write=write_hc(fd,write_array,5);
		if(n_write!=-1){
			write_array[n_write-1]='\0';
			display("written : ");
			display(write_array);
			display("\n");	
		}
		fd=close_hc(fd);
	}


	const char *p= "Hello, world!\n";
	for (; *p; ++p)
		outb8(CHAR_PORT, *p);

	
display("********************************SENDING HALT**********************************\n");
	*(long *) 0x400 = 42;
	for (;;)
		asm("hlt" : /* empty */ : "a" (42) : "memory");
}
