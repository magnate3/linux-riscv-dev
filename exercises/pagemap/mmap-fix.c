
#include <sys/mman.h>

#include <stdio.h>
#include <stdlib.h>
int test(void * goal_addr)
{
  size_t length = getpagesize();
  printf("%p addr test begin**************** \n");

  // MAP_FIXED means "map at exactly the goal address". otherwise, the
  // given address is used only as a "hint". this also means that if
  // for whatever reason it's not possible to map at exactly the goal
  // address, mmap will fail.

  // MAP_ANONYMOUS means "don't memory map a file, just give me a
  // bunch of memory". you can then pass -1 as the fd.

  void* addr = mmap(goal_addr, length, 
                    PROT_READ | PROT_WRITE, 
                    MAP_PRIVATE | MAP_FIXED | MAP_ANONYMOUS, 
                    -1, 0);

  // Be careful to check for MAP_FAILED instead of NULL
  if(addr == MAP_FAILED)
  {
      printf("mmap failed \n");
      return 0;
  }
  // At this point, we're guaranteed to have memory available at
  // addr. Just to be extra sure, we can assert that the memory is
  // where we wanted it to be.
  addr == goal_addr? printf("mmap succeeded,  with same address!") : printf("mmap succeeded, but with wrong address!");

  printf("mapped addres %p\n", addr);

  // Now we can fiddle with the memory however we want.

  char* c;
  for (c = addr; c <  ((char*)addr) + 10; c++) {
    *c = ((long long)c & 0xff);
  }

  for (c = addr; c <  ((char*)addr) + 10; c++) {
    printf("wrote byte %d to location %p\n", *c, c);
  }
  munmap(addr, length);
  return 0;
}
int main() {
  void* goal_addr = (void*)0x10000;
  test(goal_addr);
  goal_addr = (void *)malloc(getpagesize());   
  test(goal_addr);
  free(goal_addr);
  exit(EXIT_FAILURE);
}
