#include <stdio.h>

__global__ 
void func1() {
    printf("cuda examples demo!\n");
}

int main(){
    func1<<<1, 1>>>();
    func1<<<1, 1>>>();
    func1<<<1, 1>>>();
    printf("Not sure about this print sequence!\n");
    
    cudaDeviceSynchronize();
    printf("back on CPU!\n");

    // run 10 threads, this will print string 10 times as it is running in parallel
    func1<<<1, 10>>>();
    cudaDeviceSynchronize();

    return 0;
}