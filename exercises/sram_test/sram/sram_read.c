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
  int i,fd;
  char data[6];
  fd=open(SRAM_DEV,O_RDONLY);
  if(fd<0)
  {
   printf("Unable to open sram device %s\n",SRAM_DEV);
   return 0;
  }
  read(fd,data,sizeof(data));
  printf("data is %x\n",data[0]);
  printf("data is %x\n",data[1]);
  printf("data is %x\n",data[2]);
  printf("data is %x\n",data[3]);
  printf("data is %x\n",data[4]);
  printf("data is %x\n",data[5]);
  
  printf("data is %c\n",data[0]);
  printf("data is %c\n",data[1]);
  printf("data is %c\n",data[2]);
  printf("data is %c\n",data[3]);
  printf("data is %c\n",data[4]);
  printf("data is %c\n",data[5]);	
 
  close(fd);
  return 1;
}   

 
