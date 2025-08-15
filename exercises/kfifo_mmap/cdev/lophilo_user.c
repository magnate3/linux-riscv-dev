#include "lophilo.h"

#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
 
int main(void)
{
  struct lophilo_data* data;
  int fd = open("/dev/lophilo", O_RDWR);
  data = mmap(0, sizeof(struct lophilo_data), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0x0);
  if(data == (struct lophilo_data*)-1) {
  	perror("Error allocating memory");
  }
  close(fd);
  if(data != (struct lophilo_data*)-1) {
	  printf("reading from %p\n", data);
	  printf("%d\n", data->updates[LOPHILO_PIN_XA0].value);
	  munmap(data, sizeof(struct lophilo_data));
  }
  return 0;
}
