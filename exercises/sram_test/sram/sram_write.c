
#include<sys/types.h>
#include<sys/stat.h>
#include<fcntl.h>
#include<stdio.h>
#include<linux/fs.h>
#include<stdlib.h>
#include<unistd.h>
#define SRAM_DEV "/dev/SRAM0"
int main()
{
  int fd;
   char * data="ctrboss";
   fd=open(SRAM_DEV,O_WRONLY);
if(fd<0)
{
  printf("Unable to open sram device %s\n",SRAM_DEV);
  return 0;
}
  write(fd,data,sizeof(data));
  close(fd);
  return 1;
}
 
