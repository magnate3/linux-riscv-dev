#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

int main(void)
{
        void *  addr;
	uint64_t num = 0x0000080007b00000;
	printf("%lu\n", num);             //十进制输出
	printf("0x%"PRIx64"\n", num);     //十六进制输出
	printf("0x%016lx\n", num);        //十六进制输出
        //no long
	printf("0x%016x\n", num);        //十六进制输出
        //no zero 
	printf("0x%16lx\n", num);        //十六进制输出
	printf("addr size %d\n", sizeof(addr));        //十六进制输出
}
