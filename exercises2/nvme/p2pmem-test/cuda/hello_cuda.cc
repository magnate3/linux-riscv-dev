#include <iostream>
#include <unistd.h>
#include <fcntl.h>
#include <chrono>

// needed by cuFile
#include "cufile.h"

// needed by check_cudaruntimecall
#include "cufile_sample_utils.h"

using namespace std;

#define MAX_BUF_SIZE (1024 * 1024UL)

int gpu_to_storage(const char *file_name, void *gpumem_buf, int mb){
  const size_t size = MAX_BUF_SIZE * mb;
  int fd = open(file_name, O_CREAT | O_RDWR | O_DIRECT, 0664);
if (fd < 0) {
  std::cerr << "write file open error : " << std::strerror(errno)
	  << std::endl;
  cudaFree(gpumem_buf);
  return -1;
}

  size_t ret = -1;
  CUfileError_t status;
  CUfileHandle_t fh;
  CUfileDescr_t desc;
  memset((void *)&desc, 0, sizeof(CUfileDescr_t));
  desc.handle.fd = fd;
  desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  status = cuFileHandleRegister(&fh, &desc);
  if (status.err != CU_FILE_SUCCESS) {
	  std::cerr << "file register error: "
      << cuFileGetErrorString(status) << std::endl;
	  close(fd);
  	cudaFree(gpumem_buf);
	return -1;
  }
      
  ret = cuFileWrite(fh, gpumem_buf, size, 0, 0);
  if (ret < 0) {
      std::cerr << "write failed : "
          << cuFileGetErrorString(errno) << std::endl;
  	cudaFree(gpumem_buf);
      return -1;
   }

  int idx;
  check_cudaruntimecall(cudaGetDevice(&idx));
  std::cout << "Writing memory of size :"
		<< size << " gpu id: " << idx << std::endl;
  return 0;
}

void *storage_to_gpu(const char *file_name, int mb)
{
  int device_id = 1;
  const size_t size = MAX_BUF_SIZE * mb;
  check_cudaruntimecall(cudaSetDevice(device_id));
  int fd = open(file_name, O_RDONLY | O_DIRECT);
  if (fd < 0) {
    std::cerr << "read file open error : " << file_name << " "
        << std::strerror(errno) << std::endl;
    return NULL;
  }
  
  size_t ret = -1;
  CUfileError_t status;
  CUfileHandle_t fh;
  CUfileDescr_t desc;
  memset((void *)&desc, 0, sizeof(CUfileDescr_t));

  desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  desc.handle.fd = fd;
  status = cuFileHandleRegister(&fh, &desc);
  if (status.err != CU_FILE_SUCCESS) {
    std::cerr << "file register error: "
        << cuFileGetErrorString(status) << std::endl;
    close(fd);
    return NULL;
  }

  void *gpumem_buf;

  cudaMalloc(&gpumem_buf, size);
  cudaMemset(gpumem_buf, 0, size);
  ret = cuFileRead(fh, gpumem_buf, size, 0, 0);
  if (ret < 0) {
    std::cerr << "read failed : "
	<< cuFileGetErrorString(errno) << std::endl;
    cuFileHandleDeregister(fh);
    close(fd);
    cudaFree(gpumem_buf);
    return NULL;
  }

  int idx;
  check_cudaruntimecall(cudaGetDevice(&idx));
  std::cout << "Allocating and reading memory of size :"
		<< size << " gpu id: " << idx << std::endl;
  cuFileHandleDeregister(fh);
  close (fd);
  return gpumem_buf;
}

// Write data from CPU memory to a file
int cpu_to_storage(const char *file_name, void *cpumem_buf, int mb)
{
  const size_t size = MAX_BUF_SIZE * mb;
  std::ofstream outfs(file_name);
  outfs.write((char *) cpumem_buf, size);
  cout << "CPU Writing memory of size :" << size << std::endl;
  return 0;
}

int main(int argc, char*argv[])
{
  int mb = 31;
  // read data to GPU memory
  const char *readf = "/mnt/nvme/read.dat";
  const char *writef = "/mnt/nvme/write.dat";
  const char *writef_cpu = "/mnt/nvme/write_cpu.dat";
  const size_t size = MAX_BUF_SIZE * mb;
  void *devPtr = storage_to_gpu(readf, mb);
  
  // measure time to write data from GPU directly to storage
  auto start = std::chrono::steady_clock::now(); 
  gpu_to_storage(writef, devPtr, mb);
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "GDS write time: " << elapsed_seconds.count() << " s\n";

  // measure time to transfer data to CPU then write it to storage
  void *hostPtr = malloc(size);
  start = std::chrono::steady_clock::now(); 
  cudaMemcpy(hostPtr, devPtr, size, cudaMemcpyDeviceToHost);
  cpu_to_storage(writef_cpu, hostPtr, mb);
  end = std::chrono::steady_clock::now();
  elapsed_seconds = end-start;
  std::cout << "Copy to CPU and write time: " << elapsed_seconds.count() << " s\n";

    // Compare file signatures
  unsigned char iDigest[SHA256_DIGEST_LENGTH];
  unsigned char oDigest[SHA256_DIGEST_LENGTH], coDigest[SHA256_DIGEST_LENGTH];
  int ret;
  SHASUM256(readf, iDigest, size);
  DumpSHASUM(iDigest);

  SHASUM256(writef, oDigest, size);
  DumpSHASUM(oDigest);

  SHASUM256(writef_cpu, coDigest, size);
  DumpSHASUM(coDigest);

  if ((memcmp(iDigest, oDigest, SHA256_DIGEST_LENGTH) != 0) ||
      (memcmp(iDigest, coDigest, SHA256_DIGEST_LENGTH) != 0)) {
	std::cerr << "SHA SUM Mismatch" << std::endl;
	ret = -1;
  } else {
	std::cout << "SHA SUM Match" << std::endl;
	ret = 0;
  }
  return ret;
}
