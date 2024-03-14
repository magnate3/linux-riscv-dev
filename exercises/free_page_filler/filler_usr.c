#include <stdio.h>
#include <stdlib.h>

int main(int argc, const char *argv[])
{
	char buf[0x2000+1] = { 0 };
	int i;
	char *ret = 0;
	FILE *fp;

	fp = fopen("/proc/filler_start", "r"); // read mode
	if (fp == NULL) {
		printf("Cannot open file\n");
		return -1;
	}

	ret = fgets( buf, 0x2000, fp);
	if(ret == NULL) {
		printf("There are no buffer\n");
		return -1;
	}

	for(i=0; i < 0x2000; i++)
		printf("[%x]", buf[i]);

	fclose(fp);
	return 0;
}
