#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

static inline void outb(uint16_t port, uint32_t value) {
	asm("out %0,%1" : /* empty */ : "a" (value), "Nd" (port) : "memory");
}

static inline uint32_t inb(uint16_t port) {
	uint32_t ret;
	asm("in %1, %0" : "=a"(ret) : "Nd"(port) : "memory" );
	return ret;
}

void print(const char *str) {
	outb(0xEA, (uintptr_t) str);
}

int open(const char *path) {
	outb(0xEC, (uintptr_t) path);
	return inb(0xEC);
}

struct write_action {
	uintptr_t buf_addr;
	int len;
};

int write(void *buf, int len) {
	struct write_action wa = {
		buf_addr: (uintptr_t)buf,
		len: len
	};
	outb(0xED, (uintptr_t)&wa);
	return 0;
}
int close() {
	outb(0xEE, 0);
	return 0;
}

int exits(void) {
	return inb(0xEB);
}

void
__attribute__((noreturn))
__attribute__((section(".start")))
_start(void) {
	char *p = (char*) malloc(16*sizeof(char));
	p = "Hello, world!\n";

	char *path = (char*) malloc(61*sizeof(char));
	path = "file.txt";
	open(path);
	print(p);

	int num_exits = exits();
	// num to ascii (for simplicity assume single digit)
	p[0] = '0'+num_exits;
	// set null char
	p[1] = '\n';
	p[2] = (char)0;
	print(p);
	char *fdata = "This vm had a total of \0 exits\n";
	fdata[23] = '0'+num_exits;
	write(fdata, 32);

	close();
	print("success\n");

	*(long *) 0x400 = 42;

	for (;;)
		asm("hlt" : /* empty */ : "a" (42) : "memory");
}
