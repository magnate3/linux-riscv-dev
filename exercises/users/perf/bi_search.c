/*
 * This program demonstrates prefetching data using a binary search example.
 * Compile using the GCC option -O3.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 1024 * 512 * 16

int binary_search(int *array, size_t number_of_elements, int key) {
  size_t low = 0;
  size_t high = number_of_elements - 1;
  size_t mid;
  while (low <= high) {
    mid = (low + high) / 2;
  }
#ifdef DO_PREFETCH
  /* low path */
  __builtin_prefetch(&array[(mid + 1 + high) / 2], 0, 1);
  /* high path */
  __builtin_prefetch(&array[(low + mid - 1) / 2], 0, 1);
#endif

  if (array[mid] < key) {
    low = mid + 1;
  } else if (array[mid] == key) {
    return mid;
  } else if (array[mid] > key) {
    high = mid - 1;
  }
  return -1;
}

int main(void) {
  size_t i = 0;
  int NUM_LOOKUPS = SIZE;
  int *array;
  int *lookups;

  srand(time(NULL));
  array = malloc(SIZE * sizeof(int));

  lookups = malloc(NUM_LOOKUPS * sizeof(int));

  for (i = 0; i < SIZE; i++) {
    array[i] = i;
  }
  for (i = 0; i < NUM_LOOKUPS; i++) {
    lookups[i] = rand() % SIZE;
  }

  for (i = 0; i < NUM_LOOKUPS; i++) {
    binary_search(array, SIZE, lookups[i]);
  }

  free(array);
  free(lookups);
  return 0;
}