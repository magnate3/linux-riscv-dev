#include <string.h>
#include <stdio.h>
#include <bsd/string.h>
#include <stdint.h>
int main()
{
	    char buf[5];
	        char src[10] = "12345678";
		    strlcpy(buf, src, sizeof(buf));
		        printf("%s\n",buf);//输出1234
    printf("%lu\n",sizeof(uintptr_t));//输出1234
			    return 0;
}
