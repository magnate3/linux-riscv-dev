#include "lophilo.h"x

#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
 
int main(void)
{
  lophilo_update_t * data;
  fd_set rfds;
  struct timeval tv;
  int fd = open("/dev/lophilo", O_RDWR);
  int count = 0 , index = 0, bytes = 0;
  data = (struct lophilo_data_t*)malloc(sizeof(lophilo_update_t)*(LOPHILO_SOURCE_MAX+1));
  if(data == (struct lophilo_data*)-1) {
  	perror("Error allocating memory");
        return 0;
  }
  while(1)
  {
	FD_ZERO(&rfds);
	FD_SET(fd,&rfds);
	tv.tv_sec = 5;
	tv.tv_usec = 0;
	select(fd+1,&rfds,NULL,NULL,&tv);//最后一个参数设为NULL，将会永远等待，即阻塞！
	if(FD_ISSET(fd,&rfds))
        { 
            bytes = read(fd,(char *)data, sizeof(lophilo_update_t)*(LOPHILO_SOURCE_MAX+1));
            
            index = 0;
            count = bytes/sizeof(lophilo_update_t);
            printf("***********bytes %d,  count %d ****************\n",bytes , count);
            while(--count > 0)
            {
                  data = (struct lophilo_data_t*)(data + index);                  
                  printf("index %d , data ->value %d \n", index++ , data->value);
            }
	}
        else
        {
	    printf("No data within 5s,please wait.. \n",&tv.tv_sec);
	}
	//usleep(3);	
	sleep(3);	
  }
  close(fd);
  return 0;
}
