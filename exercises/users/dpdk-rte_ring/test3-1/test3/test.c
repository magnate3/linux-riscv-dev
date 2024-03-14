#include <string.h>
#include <stdio.h>
#include <bsd/string.h>
#include <stdint.h>
int main()
{
    // power of 2
    unsigned char in = 1, out = 255;
    unsigned char diff = in - out; 
    unsigned char next,in2end,out2end;
    next = out +1;
    printf("in %u, out %u,diff %u,  next %u\n",in, out,diff, next);
    in = 255, out = 1;
    diff = in - out; 
    printf("diff %u \n",diff);
    return 0;
}
