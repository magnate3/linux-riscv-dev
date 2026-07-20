
#include <sys/mman.h>

#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

int main() {
  void* goal_addr = (void*)0x10000;
  size_t length = 1 << 10;
  void *addr2, *addr3;


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
  //check(addr != MAP_FAILED, "mmap failed");

  // At this point, we're guaranteed to have memory available at
  // addr. Just to be extra sure, we can assert that the memory is
  // where we wanted it to be.
  //check(addr == goal_addr, "mmap succeeded, but with wrong address!");

  printf("mapped addres %p and goal_addr %p\n", addr, goal_addr);
  addr3 = malloc(1024*1024); 
  addr2 = mmap(goal_addr, length, 
                    PROT_READ | PROT_WRITE, 
                    MAP_PRIVATE | MAP_FIXED | MAP_ANONYMOUS, 
                    -1, 0);
  printf("mapped addres %p and goal_addr %p\n", addr2, goal_addr);
  // Now we can fiddle with the memory however we want.

  char* c;
  for (c = addr; c <  ((char*)addr) + 10; c++) {
    *c = ((long long)c & 0xff);
  }

  for (c = addr; c <  ((char*)addr) + 10; c++) {
    printf("read byte %d to location %p\n", *c, c);
  }
  free(addr3);
  munmap(addr, length);
  munmap(addr2, length);
  return 0;
 error:
  exit(EXIT_FAILURE);
}
