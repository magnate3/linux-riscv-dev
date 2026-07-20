// author : Hou Jie (侯捷)
// date : 2015/11/11
// compiler : DevC++ 5.61 (MinGW with GNU 4.9.2)
//
// 說明：這是侯捷 E-learning video "C++內存管理" 的實例程式.
//
// filename : allocc.h
// 取材自 SGI STL 2.91 <stl_alloc.h>, 移植至 C language.


#include <stdlib.h>  //for malloc(),realloc()
#include <stddef.h>  //for size_t
#include <memory.h>  //for memcpy()
#include <stdio.h>  //for memcpy()

//#define __THROW_BAD_ALLOC   cerr << "out of memory" << endl; exit(1)
#define __THROW_BAD_ALLOC   exit(1)

//----------------------------------------------
// 第1級配置器。
//----------------------------------------------

void (*oom_handler)() = 0;

void* oom_malloc(size_t n)
{
  void (*my_malloc_handler)();
  void* result;

  for (;;) {    //不斷嘗試釋放、配置、再釋放、再配置…
    my_malloc_handler = oom_handler;
    if (0 == my_malloc_handler) { __THROW_BAD_ALLOC; }
    (*my_malloc_handler)();    //呼叫處理常式，企圖釋放記憶體
    result = malloc(n);        //再次嘗試配置記憶體
    if (result) return(result);
  }
}

void* oom_realloc(void *p, size_t n)
{
  void (*my_malloc_handler)();
  void* result;

  for (;;) {    //不斷嘗試釋放、配置、再釋放、再配置…
    my_malloc_handler = oom_handler;
    if (0 == my_malloc_handler) { __THROW_BAD_ALLOC; }
    (*my_malloc_handler)();    //呼叫處理常式，企圖釋放記憶體。
    result = realloc(p, n);    //再次嘗試配置記憶體。
    if (result) return(result);
  }
}

void* malloc_allocate(size_t n)
{
  void *result = malloc(n);   //直接使用 malloc()
  if (0 == result) result = oom_malloc(n);
  return result;
}

void malloc_deallocate(void* p, size_t n)
{
  free(p);  //直接使用 free()
}

void* malloc_reallocate(void *p, size_t old_sz, size_t new_sz)
{
  void* result = realloc(p, new_sz); //直接使用 realloc()
  if (0 == result) result = oom_realloc(p, new_sz);
  return result;
}

void (*set_malloc_handler(void (*f)()))()
{ //類似 C++ 的 set_new_handler().
  void (*old)() = oom_handler;
  oom_handler = f;
  return(old);
}

//----------------------------------------------
//第二級配置器
//----------------------------------------------

enum {__ALIGN = 8};                        //小區塊的上調邊界
enum {__MAX_BYTES = 128};                  //小區塊的上限
enum {__NFREELISTS = __MAX_BYTES/__ALIGN}; //free-lists 個數

// union obj {                   //G291[o],CB5[x],VC6[x]
//   union obj* free_list_link;  //這麼寫在 VC6 和 CB5 中也可以，
// };                            //但以後就得使用 "union obj" 而不能只寫 "obj"
typedef struct __obj {
  struct __obj* free_list_link;
} obj;

char*   start_free = 0;
char*   end_free = 0;
size_t  heap_size = 0;
obj* free_list[__NFREELISTS]
     = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

size_t ROUND_UP(size_t bytes) {
    return (((bytes) + __ALIGN-1) & ~(__ALIGN - 1));
}
size_t FREELIST_INDEX(size_t bytes) {
    return (((bytes) + __ALIGN-1)/__ALIGN - 1);
}

//----------------------------------------------
// We allocate memory in large chunks in order to
// avoid fragmentingthe malloc heap too much.
// We assume that size is properly aligned.
//
// Allocates a chunk for nobjs of size "size".
// nobjs may be reduced if it is inconvenient to
// allocate the requested number.
//----------------------------------------------
//char* chunk_alloc(size_t size, int& nobjs)  //G291[o],VC6[x],CB5[x]
char* chunk_alloc(size_t size, int* nobjs)
{
  char* result;
  size_t total_bytes = size * (*nobjs);   //原 nobjs 改為 (*nobjs)
  size_t bytes_left = end_free - start_free;

  if (bytes_left >= total_bytes) {
      result = start_free;
      start_free += total_bytes;
      return(result);
  } else if (bytes_left >= size) {
      *nobjs = bytes_left / size;         //原 nobjs 改為 (*nobjs)
      total_bytes = size * (*nobjs);      //原 nobjs 改為 (*nobjs)
      result = start_free;
      start_free += total_bytes;
      return(result);
  } else {
      size_t bytes_to_get =
                 2 * total_bytes + ROUND_UP(heap_size >> 4);
      // Try to make use of the left-over piece.
      if (bytes_left > 0) {
          obj* volatile *my_free_list =
                 free_list + FREELIST_INDEX(bytes_left);

          ((obj*)start_free)->free_list_link = *my_free_list;
          *my_free_list = (obj*)start_free;
      }
      start_free = (char*)malloc(bytes_to_get);
      if (0 == start_free) {
          int i;
          obj* volatile *my_free_list, *p;

          //Try to make do with what we have. That can't
          //hurt. We do not try smaller requests, since that tends
          //to result in disaster on multi-process machines.
          for (i = size; i <= __MAX_BYTES; i += __ALIGN) {
              my_free_list = free_list + FREELIST_INDEX(i);
              p = *my_free_list;
              if (0 != p) {
                  *my_free_list = p -> free_list_link;
                  start_free = (char*)p;
                  end_free = start_free + i;
                  return(chunk_alloc(size, nobjs));
                  //Any leftover piece will eventually make it to the
                  //right free list.
              }
          }
          end_free = 0;       //In case of exception.
          start_free = (char*)malloc_allocate(bytes_to_get);
          //This should either throw an exception or
          //remedy the situation. Thus we assume it
          //succeeded.
      }
      heap_size += bytes_to_get;
      end_free = start_free + bytes_to_get;
      return(chunk_alloc(size, nobjs));
  }
}
//----------------------------------------------
// Returns an object of size n, and optionally adds
// to size n free list. We assume that n is properly aligned.
// We hold the allocation lock.
//----------------------------------------------
void* refill(size_t n)
{
    int nobjs = 20;
    char* chunk = chunk_alloc(n,&nobjs);
    obj* volatile *my_free_list;   //obj** my_free_list;
    obj* result;
    obj* current_obj;
    obj* next_obj;
    int i;

    if (1 == nobjs) return(chunk);
    my_free_list = free_list + FREELIST_INDEX(n);

    //Build free list in chunk
    result = (obj*)chunk;
    *my_free_list = next_obj = (obj*)(chunk + n);
    for (i=1;  ; ++i) {
      current_obj = next_obj;
      next_obj = (obj*)((char*)next_obj + n);
      if (nobjs-1 == i) {
          current_obj->free_list_link = 0;
          break;
      } else {
          current_obj->free_list_link = next_obj;
      }
    }
    return(result);
}
//----------------------------------------------
//
//----------------------------------------------
void* allocate(size_t n)  //n must be > 0
{
  obj* volatile *my_free_list;    //obj** my_free_list;
  obj* result;

  if (n > (size_t)__MAX_BYTES) {
      return(malloc_allocate(n));
  }

  my_free_list = free_list + FREELIST_INDEX(n);
  result = *my_free_list;
  if (result == 0) {
      void* r = refill(ROUND_UP(n));
      return r;
  }

  *my_free_list = result->free_list_link;
  return (result);
}
//----------------------------------------------
//
//----------------------------------------------
void deallocate(void *p, size_t n)  //p may not be 0
{
  obj* q = (obj*)p;
  obj* volatile *my_free_list;   //obj** my_free_list;

  if (n > (size_t) __MAX_BYTES) {
      malloc_deallocate(p, n);
      return;
  }
  my_free_list = free_list + FREELIST_INDEX(n);
  q->free_list_link = *my_free_list;
  *my_free_list = q;
}
//----------------------------------------------
//
//----------------------------------------------
void* reallocate(void *p, size_t old_sz, size_t new_sz)
{
  void * result;
  size_t copy_sz;

  if (old_sz > (size_t) __MAX_BYTES && new_sz > (size_t) __MAX_BYTES) {
      return(realloc(p, new_sz));
  }
  if (ROUND_UP(old_sz) == ROUND_UP(new_sz)) return(p);
  result = allocate(new_sz);
  copy_sz = new_sz > old_sz? old_sz : new_sz;
  memcpy(result, p, copy_sz);
  deallocate(p, old_sz);
  return(result);
}
//----------------------------------------------
int fun(int i, int j)
{
    printf("%s, i =  %d, and j = %d  \n", __func__, i, j);
    return 0;
}
int (*test(int a))(int i, int j)
{
    printf("%s, a =  %d \n", __func__, a);
    return fun;
}
void oom_func()
{
     printf("oom happends");
}
int main()
{
#if 1
     char * dst = "dirk say hello world";
     int len = strlen(dst) + 1;
     set_malloc_handler(oom_func);
     char * str = allocate(len);
     memset(str,0, len);
     memcpy(str, dst, strlen(dst));
     printf("str：%s\n",str);
#else
    test(3)(4,5);
#endif
    return 0;
}
