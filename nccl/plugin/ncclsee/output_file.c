#include <errno.h>    // Required for checking errno after mkdir
#include <inttypes.h> // Required for PRIu64
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>  // Required for mkdir mode constants
#include <sys/types.h> // Required for mode_t, getpid, mkdir
#include <time.h>      // Required for timestamp
#include <unistd.h>    // Required for getpid(), mkdir()

// Define a reasonable maximum path length if PATH_MAX is not readily available
#ifndef PATH_MAX
#define PATH_MAX 1024
#endif

/**
 * @brief Creates a timestamped directory and opens a PID-specific CSV file
 * within it.
 *
 * Creates a directory named "ncclsee_YYYYMMDD_HHMMSS". Inside this directory,
 * it opens (or creates) a file named "<pid>.csv" for writing.
 * Handles the case where multiple processes might try to create the directory
 * concurrently.
 *
 * @return FILE* A file pointer to the opened "<pid>.csv" file in write mode,
 * or NULL on failure (e.g., cannot create directory, cannot open file).
 * Prints error messages to stderr on failure.
 */
FILE *create_profile_file(char *timestamp_str) {
  char dir_path[PATH_MAX];
  char file_path[PATH_MAX];
  pid_t pid;


  // 2. Construct directory path
  // Using snprintf for safety against buffer overflows
  int chars_written =
      snprintf(dir_path, sizeof(dir_path), "ncclsee_%s", timestamp_str);
  if (chars_written < 0 || (long unsigned int)chars_written >= sizeof(dir_path)) {
    fprintf(stderr, "ncclsee Error: Failed to construct directory path (snprintf "
                    "failed or truncated)\n");
    return NULL;
  }

  // 3. Create directory (mode 0755: rwxr-xr-x)
  // This needs to be safe for multiple processes calling it concurrently.
  if (mkdir(dir_path, 0755) == -1) {
    // Check if the error is because the directory already exists.
    // This is expected and okay if another process created it first.
    if (errno != EEXIST) {
      // It's a different error (e.g., permissions, path invalid)
      perror("Failed to create directory");
      fprintf(stderr, "ncclsee Error occurred trying to create directory: %s\n",
              dir_path);
      return NULL;
    }
    // If errno is EEXIST, the directory exists, which is fine. Proceed
    // normally.
    /* fprintf(stderr, */
    /*         "ncclsee: Directory %s already exists (likely created by another " */
    /*         "process).\n", */
    /*         dir_path); */
  } else {
    // Optional: Log if this process was the one that created it
    fprintf(stderr, "ncclsee: Process %d created directory %s\n", getpid(),
            dir_path);
  }

  // 4. Get process ID
  pid = getpid();

  // 5. Construct full file path
  chars_written =
      snprintf(file_path, sizeof(file_path), "%s/%d.csv", dir_path, pid);
  if (chars_written < 0 || (long unsigned int)chars_written >= sizeof(file_path)) {
    fprintf(stderr, "ncclsee Error: Failed to construct file path (snprintf failed or "
                    "truncated)\n");
    return NULL;
  }

  // 6. Open the CSV file for writing
  FILE *fp = fopen(file_path, "w");
  if (fp == NULL) {
    perror("Failed to open CSV file for writing");
    fprintf(stderr, "ncclsee Error: occurred trying to open file: %s\n", file_path);
    return NULL;
  }

  return fp;
}
