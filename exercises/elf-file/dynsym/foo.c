#include <stdio.h>
int global = 10;
void foo(void)
{
	  printf("foo\n");
}
void bar(void)
{
	  printf("bar %d \n", global);
}
