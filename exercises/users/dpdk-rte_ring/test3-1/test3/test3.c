#include <string.h>
#include <stdio.h>
#include <bsd/string.h>
#include <stdint.h>
int main()
{
    // power of 2
    unsigned char size = 128;
    unsigned char in = 0, out = 127;
    unsigned char diff = in - out; 
    unsigned char next,in2end,out2end;
    next = out +1;
    in2end = size - (in &(size -1)); 
    out2end = size - (out &(size -1)); 
    printf("in %u, out %u,diff %u, diff&size %u,  next %u\n",in, out,diff, diff&size,next);
    printf("in2end %u, out2end %u\n",in2end, out2end);
    in = 127, out = 0;
    diff = in - out; 
    printf("in %u, out %u,diff %u, diff&size %u\n",in, out,diff, diff&size);
    in2end = size - (in &(size -1)); 
    out2end = size - (out &(size -1)); 
    printf("in2end %u, out2end %u\n",in2end, out2end);
    return 0;
}
