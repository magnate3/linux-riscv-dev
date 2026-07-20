#include <stdio.h>
#include <immintrin.h>
#define N 20000000
int main(){
    double *x,*y,*z,*px,*py,*pz;
    x=(double*)_mm_malloc(sizeof(double)*N,16);//申请内存并且按照2的4次方对齐地址
    y=(double*)_mm_malloc(sizeof(double)*N,16);
    z=(double*)_mm_malloc(sizeof(double)*N,16);
    px=x;py=y;pz=z;
    __m128d vx,vy,vz;// __m128d是SSE指令集中操作双精度浮点数对应的数据类型
    for(int i=0;i<N/2;i++){
        vx=_mm_load_pd(px);//从px指向的内存中取出两个数，放入入vx
        vy=_mm_load_pd(py);//从py指向的内存中取出两个数，放入入vy
	    vz=vx+vy;//计算vx+vy并将结果放入vz，这一行也有对应的函数，不过GCC编译的话直接这样写没有什么问题，测试发现VS的编译器和Intel的编译器都不支持这种写法
	    _mm_store_pd(pz,vz);//将vz中的结果放入pz指向的内存
        px+=2;//由于前面取出了两个数据，所以指针后移两位
        py+=2;
        pz+=2;
    }
    _mm_free(x);//释放_mm_malloc申请的内存
    _mm_free(y);
    _mm_free(z);
    return 1;
}
 
