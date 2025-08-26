#include<stdio.h>
#include<stdlib.h>
#include <math.h>
#include <inttypes.h>
#include <time.h>

#define BILLION 1000000000
# define do_div(n,base) ({                                      \
		uint32_t __base = (base);                               \
		uint32_t __rem;                                         \
		__rem = ((uint64_t)(n)) % __base;                       \
		(n) = ((uint64_t)(n)) / __base;                         \
		__rem;                                                  \
		})

struct sex{
	int gender;
};
struct student{
	int age;
	int id;
	struct sex s;
};

unsigned createMask(unsigned a, unsigned b)
{
   unsigned r = 0;
   for (unsigned i=a; i<=b; i++)
       r |= 1 << i;

   return r;
}


int main(){
	//struct student a[3];
/*	uint32_t a=12;
	uint32_t b=7;
	char str[64];
	uint64_t aInt=82406267882654;
	sprintf(str, "%"PRIu64"", aInt);
	printf("%s, %"PRIu64"\n", str,aInt);	
	printf("uc_nic_ts [ %"PRIu32" ], %"PRIu32" \n", a-b,(a-b)/3);	
	uint64_t bInt = strtoll(str, NULL, 10);
	printf("%s, %"PRIu64"\n", str,bInt);	
	//printf("%lu, %lu\n", sizeof(struct student), sizeof(a));
	
	int e = 1000000;
	int f = 1024;
	long long unsigned c= (long long unsigned)e*f*8;
	double d=49111698;
	printf("%llu %f\n", c, c/d);

	do_div(e,f);
	printf("%d\n",e);


	 char input[] = "A bird came down the walk";
    	printf("Parsing the input string '%s'\n", input);
    	char *token = strtok(input, " ");
	uint32_t timeout = 52;
	int new = (int)ceil(log2( ((double)timeout/4.096)));
	printf("new timeout %d\n", new);
	
	char g='a';
	char h[10];
	memset(h, 0,10);
	if(h[0]!=0)	
		printf("%s\n", h);
*/
/*	uint32_t a=598;
	uint32_t b = a<<4;
	b = b+9;	
	//int r = createMask(12,16);
	printf("r %d, %"PRIu64"\n", b&0xf, (b&0xfffffff0)>>4);
	int i;
	int k=1;
	for(i=0;i<5;i++){
		k = pow(2,i);
		printf("%d\n",k);
	}		*/

          struct timespec start, end;
          clock_gettime(CLOCK_MONOTONIC, &start);
          uint64_t starttime = start.tv_sec*BILLION + start.tv_nsec;
	  long long unsigned diff = (long long unsigned)start.tv_sec*BILLION + start.tv_nsec;
	 printf("%"PRIu64", %llu\n",starttime, diff );
	int a = 8192;
	int b =1048576;
	uint64_t bytes = (uint64_t)a*b;
	printf("%"PRIu64"\n", bytes);
}
