# Project 334 技术报告 - llama.cpp的龙芯平台移植与优化

> 项目成员：  
> 毕昊阳，中国科学技术大学  
> 胡冰玉，中国科学技术大学  
> 
> 指导教师：  
> 王皓，中国科学技术大学



> **UPDATE**：决赛阶段更新内容提要：
>
> 1. 扩展量化格式：除Q4_0外，增加对Q2_K、Q4_1、Q5_0、Q5_1、Q8_0五种量化格式的推理加速支持；
> 2. 测试多种模型：除7B和13B参数模型外，增加对1B、30B两种不同参数数量模型的测试；
> 3. 工程优化：为支持以上扩展，对代码进行重构优化，并引入自动化test/benchmark；
> 4. 报告更新：以下报告正文内容已与上述内容同步进行更新。
>
> 具体修改内容可直接通过 git diff 或 Web UI 与 commit `95c67b74d1c0426b785e762502ea98f44553b60c` 进行对比。





## 摘要

* **项目目标**：将llama.cpp移植至龙芯处理器3A6000，并进行软硬件协同优化，加速模型的CPU推理速度，使得以Meta LLaMA为代表的流行的大语言模型能够以可接受的速度运行于龙芯平台；
* **完成情况**：本项目的规划和进展情况可见[dev.md](dev.md)。截至本阶段，实现了从2bit到32bit共**7种**数据格式的推理优化加速，并在从1B到30B共**4种**参数规模的LLaMA模型上进行标准测试。较于未经优化的代码，在矩阵乘法和模型推理两项标准任务上均实现可观的性能加速。
* **主要创新**：定位和分析了大语言模型推理的主要性能瓶颈；针对龙芯平台进行了**SIMD**和**Cache**两个方向的计算优化；同时支持**浮点**参数和**量化**参数的运算加速；在3A6000处理器上进行了正确性和性能的标准测试。

本技术报告是对本项目的阶段性总结，也希望为后续工作及其他相关工作提供一些启发，具体包含以下章节：
1. 关于 llama.cpp 的背景介绍；
2. 针对龙芯平台的移植工作介绍；
3. 针对龙芯平台的软硬件协同优化工作介绍；
4. 项目的工程实现；
5. 标准测试结果；
6. 相关工作；
7. 未来工作与收获总结。




## 1. llama.cpp 背景介绍

### 1.1 什么是llama.cpp
llama.cpp是一个开源的大语言模型(Large Language Model, LLM)推理程序，支持包括Meta LLaMA等众多知名模型。所谓推理是指载入训练好的模型参数并运行模型，得到输出。LLM巨大的参数量，给推理过程的计算速度和内存占用都带来挑战。llama.cpp所解决的核心问题，就是在用户级设备，尤其是CPU上，进行高效的LLM推理。其解决该问题的主要手段包括：
1. 基于纯C/C++，无GC，面向底层，充分利用硬件特性，相比Python等语言天然具有性能优势；
2. 引入模型量化技术，显著减小内存占用，也间接提升运算性能。

针对以上两方面，我们分别进行简要介绍。

### 1.2 GGML
整个llama.cpp项目可以分为两部分：底层的张量库GGML(C语言)，和应用层的模型推理代码(C++语言)。严格来说，GGML是一个[独立的项目](https://github.com/ggerganov/ggml)，但在实际开发中，GGML被完整包含在llama.cpp项目中(工程目录下的ggml*文件)一起开发，并反馈合并给上游的原仓库。  
GGML是一个纯C语言的张量库，类似PyTorch，包含以下功能：

1. 张量构建：多维数组及相关基本操作（如拷贝、赋值等），相当于PyTorch中的tensor；
2. 算子实现：加法、矩阵乘法等张量算子，包含在各平台的优化；
3. 计算图调度：由张量和算子构成计算图，输入数据后执行真正的计算，包含多线程调度等。

和许多C语言库一样，GGML通过高效的内存管理，以及SIMD等面向平台特性的优化，来支持上层的高效推理。  
llama.cpp中的模型推理代码的本质，是利用GGML构建不同模型对应的计算图，载入模型参数，最后运行计算图。  本项目对llama.cpp的性能优化，实际是发生在GGML相关的代码中：我们通过在关键性能瓶颈处插入自己实现的算子替换原有的算子，来实现整体性能的大幅提升。

### 1.3 模型量化
除了C/C++带来的性能提升外，llama.cpp高度依赖于一种称为模型量化(model quantization)的推理优化技术，即通过更低精度的数据类型来存储模型参数，以牺牲参数精度为代价，显著降低参数的内存占用。例如，一个4字节的单精度浮点参数，可以存储为一个2字节半精度浮点，或一个1字节甚至4比特的整数。此外，一般会将多个参数（如32个）打包成一个block，同一个block内的参数共享某些额外信息，这些额外信息造成的存储和计算成本被block的大小所均摊。

llama.cpp支持多种量化方法，在项目中命名为Q2_K, Q4_0, Q5_1等。Q代表Quantization，Q后的数字代表平均每个参数被量化后的比特数，末尾的数字或字母代表量化方法的"版本号"，不同版本的方法在具体的内存存储结构以及block共享信息上会有所不同。以Q4_1为例，在该方法中，每个参数被量化为一个4比特整数，而每32个参数被打包成一个block，结构如下：

```C
// ggml-common.h
#define QK4_1 32
typedef struct {
    union {
        struct {
            ggml_half d; // delta
            ggml_half m; // min
        } GGML_COMMON_AGGR;
        ggml_half2 dm;
    };
    uint8_t qs[QK4_1 / 2]; // nibbles / quants
} block_q4_1;
```

举例说明，要想从一个`blockq4_1` 中还原第一个参数，只需 `blk.m + (blk.qs[0] & 0xF) * blk.d` ，其中，`m` 和 `d` 则是整个block所共享的信息，虽然一共占了32比特，但由32个参数所均摊，平均每个参数只多用1比特。与Q4_1不同，Q4_0方法则没有`m` 这一共享信息。

依次类推地，每一种量化方法有不同的数据存储方式，背后对应着推理性能和模型效果的权衡，一般来说，量化的"压缩率"越高，推理性能越高，但因精度损失，模型效果越差。本项目在优化过程中同时考虑了普通的单精度浮点运算和量化运算。我们实现了Q2_K、Q4_0、Q4_1、Q5_0、Q5_1、Q8_0共六种量化方法，涉及不同比特数和存储方法的量化。在每一种量化方法的性能优化中，我们充分利用其存储结构的性质，比如每个block中的量化比特在内存中连续存储等等。




## 2. 针对龙芯平台的移植工作

针对龙芯平台的移植工作分为两个部分：平台无关移植与平台相关移植。下面分别进行介绍。

### 2.1 平台无关移植
llama.cpp基于C/C++，开发过程保持了良好的跨平台规范，所有针对特定平台（如x86、ARM、RISC-V等）的代码均由条件编译（基于宏）和构建系统（基于Make、CMake等工具）控制处理。举例来说，项目中随处可见以下形式的代码：
```C
#if defined(__AVX__)
... // 针对x86平台的优化
#elif defined(__ARM_NEON)
... // 针对ARM平台的优化
#else
... //平台无关的代码
```
在构建系统中，也须加入对应的编译器选项。例如原项目的Makefile中有：
```Makefile
# llama.cpp-b2430/Makefile
ifdef RISCV
	MK_CFLAGS   += -march=rv64gcv -mabi=lp64d
	MK_CXXFLAGS += -march=rv64gcv -mabi=lp64d
else
ifeq ($(UNAME_M),$(filter $(UNAME_M),x86_64 i686 amd64))
...
```

另一方面，龙芯平台上有完整的GNU工具链。因此，直接在LoongArch架构上编译llama.cpp项目是无痛的。针对上面的代码片段，在3A6000处理器则默认会编译出 `#else` 部分的代码。

### 2.2 平台相关移植
llama.cpp中平台无关的代码对应的性能是未经优化且无法接受的。因此，开发过程不可避免地须在项目代码中加入LoongArch的平台相关代码。在本项目中，所涉及的平台相关特性为LASX/LSX扩展向量指令集（注：本项目主要针对3A6000处理器，该处理器同时支持LASX/LSX）。

因此，我们仿照项目中相应的做法，在代码中插入条件编译以保持原有的跨平台特性：
```C
...
#elif defined(__loongarch_lasx__)
... // 针对龙芯平台的优化
...
```
对应地，在Makefile中，我们插入：
```Makefile
# llama.cpp-b2430/Makefile
ifneq ($(filter loongarch64%,$(UNAME_M)),)
	MK_CFLAGS   += -mlasx
	MK_CXXFLAGS += -mlasx
endif
```

至此，针对龙芯平台的移植工作完成。



## 3. 针对龙芯平台的软硬件协同优化
针对龙芯平台的大模型推理速度优化是本项目的重点工作，相对移植来说，占据了主要的工作量。我们的优化流程总体如下：
1. 通过profile工具，定位性能瓶颈为GEMM操作；
2. 针对GEMM优化进行调研，阅读和理解llama.cpp中相应的代码，确定从SIMD和Cache两个方向进行优化；
3. 借助龙芯的LASX向量指令集进行SIMD优化（这一步恰巧与龙芯团队的工作重合）；
4. 在SIMD优化基础上，进一步针对Cache进行优化。

下面进行详细介绍。

### 3.1 定位性能瓶颈
在有限的时间内，我们无法对llama.cpp这样的大型项目进行全面的优化。而想要以最高的效率获得性能提升，应先定位程序真正的性能瓶颈或热点代码。我们通过Linux平台的perf工具对llama.cpp的模型推理进行profile，发现90%以上的CPU计算用于位于`ggml.c` 的 `ggml_compute_forward_mul_mat()` 函数。该函数的作用是对两个张量的前两维进行矩阵乘法运算，也即所谓的GEMM。究其原因，是因为当前大语言模型均基于Transformer架构，而Transformer中绝大部分计算为Attention，后者本质就是在做矩阵乘法。总之，这对本项目来而言是利好的，我们后续只需针对 `ggml_compute_forward_mul_mat()` 函数进行性能优化即可。


### 3.2 确定优化方案
GEMM是高性能计算中的经典课题。本项目团队并非相关专业出身，谨根据有限的调研结果，确定从两个方向进行GEMM优化。

需要指出的是，这并不意味着本项目简单地规约成了一个GEMM优化，因为：
1. llama.cpp重度依赖模型量化技术，量化后的矩阵并非存储为连续的浮点数，GEMM优化必须考虑量化后参数的存储结构；
2. 需要理解项目中张量的存储逻辑、算子的实现逻辑和计算图的构造逻辑。

### 3.3 SIMD优化

Single Instruction Multiple Data (SIMD) 是现代处理器的重要并行手段，一般通过向量扩展指令集实现，如x86 SSE/AVX指令集，和本项目涉及的LASX。LASX包含32个256位向量寄存器，每个寄存器可以并行处理8个单精度浮点数（用于优化浮点矩阵乘法）或32个8位整数（用之优化量化矩阵乘法）。

为清楚说明优化过程，考虑两个相乘的矩阵$A_{M,K} \times B_{K,N}\rightarrow C_{M,N}$。在大模型中，$K$通常为模型隐向量长度，一般在512到4096之间。在llama.cpp中，矩阵可以写成如下伪代码：

```
for i = 0 to M-1:
  for j = 0 to N-1:
      C[i, j] = 0
      for k = 0 to K-1: // A的第i行点乘B的第j列，逐元素处理
        C[i, j] += A[i, k] * B[k, j]
```

点乘操作是对两个向量中两两对应元素的乘积的累加，符合典型的SIMD模式。以LASX指令集支持的256位向量寄存器为例，对于F32浮点运算，我们可同时对`256/32=8` 个浮点数进行并行计算；而对于量化运算，我们可同时对 `256/8=32` 个8bit量化数进行并行运算，对于绝大多数量化方法，32刚好是一个block的参数数量（注：虽然大部分量化方法的量化比特数小于8，但由于字节是CPU指令集所支持的相关运算的最小单位，我们仍需先处理成8位整数后再进行计算）。该逻辑可以写成如下伪代码：

```
for i = 0 to M-1:
  for j = 0 to N-1:
      C[i, j] = 0
      for k = 0 to K-1 with step T: // 每次可SIMD同步处理T个元素
        C[i, j] += SIMD_product(A[i, k:k+T], B[k:k+T, j]) 
```

### 3.4 Cache优化

SIMD仅仅利用了处理器中的向量计算单元，而影响GEMM性能的另一大因素则是访存。根据手册，龙芯3A6000处理器拥有每核64KB L1缓存和256KB L2缓存。合理利用缓存，即在相同计算量下，提升缓存命中，降低内存访问次数，可以显著提升性能。在llama.cpp原有代码以及前述SIMD优化代码中，矩阵乘法的计算没有充分利用缓存。

注意，llama.cpp中已经对矩阵A的内存分布做了优化，此处矩阵A实质上已经做了转置。进一步地，考虑这个过程中的缓存行为。根据处理器缓存大小可以估算，缓存大于单个隐向量而小于整个矩阵。考虑最外层$i$变量循环，当$i=0$时，内层循环使得$A$的第0行与$B$的每一列一一点乘，这个过程中，$A$的行向量除第一次点乘外一直在缓存中，而$B$的列向量则在遍历过程中不断装入缓存，最终因缓存无法容纳所有列向量，而使得前面的列向量被替换。如此，当$i=1$的时候，$B$的所有列向量将须重新一一装入缓存。也就是说，真正被有效利用的缓存只有$A$的行向量所对应的部分，远小于处理器全部缓存大小。

因此，我们考虑通过分块处理来提高缓存命中率，一次性计算出$C$矩阵中的一个$B0\times B1$块，其本质是循环展开。为讨论简单起见，假设$M$和$N$分别整除$B0$和$B1$。为展示Cache优化与SIMD优化的正交性，我们保留SIMD优化部分，伪代码如下：
```
for i = 0 to M-1 with step B0：
  for j = 0 to N-1 with step B1:
    for ib = 0 to B0-1:
      for jb = 0 to B1-1:
          for k = 0 to K-1 with step T:  // 可同时运用SIMD优化
            C[i+ib,j+jb] = SIMD_product(A[i+ib, k:k+T], B[k:k+T, j+jb])
```
当最内的两层循环涉及到的行/列向量可以合理容纳进缓存时，缓存的利用率可以大大提升。另一方面，SIMD和Cache的优化本身是正交关系，应结合起来达到更好的优化效果。我们注意到，通过循环展开时合理的计算排布，不仅可以使数据尽可能留在缓存内，也能够使数据尽可能留在向量寄存器内。而分块大小$B0\times B1$的设计，则同时与缓存大小和向量寄存器的数量相关。
在工程实现中，为尝试不同大小的分块带来的优化效果，同时方便处理$M,N$不整除$B0,B1$时的剩余情况，我们采用基于C++函数模板和`if constexpr`特性给予静态化的参数实现。



## 4. 工程实现

### 4.1 工程目录结构
本项目的总体目录结构如下所示：
- `llama.cpp-b2430/`：Tag为 `b2430` 的 llama.cpp 的原始代码，是本项目开始时的最新release版本。
- `src/`：本项目代码，以及一份修改自 `llama.cpp-b2430/examples/benchmark/benchmark-matmult.cpp`的Benchmark测试代码，这意味着性能测量与社区先前报告的结果完全可比。
- `test/`：基于 `pytest` 的自动化测试与benchmark代码，用于对各种量化格式以及各种模型进行批量化自动测试，也是本报告的数据来源。
- `model_weights/`：存放模型参数文件。由于参数文件过大，我们没有直接将文件上传至代码仓库，而是在此目录下提供了下载文件的Python脚本。

### 4.2 工程实现概览
在开发过程中，我们尽量保持plug-in的原则，在原项目目录（`llama.cpp-b2430/`）内只对构建系统（Makefile）和一些包含条件编译的代码（用于插入我们的工作）进行必要的更改，大部分真正的开发工作都在 `src/` 目录中进行，其中声明的两个函数 `lamm_can_mul_mat()` 和 `lamm_mul_mat()` 被插入至 `llama.cpp-b2430/ggml.c` 中的GEMM执行调度函数 `ggml_compute_forward_mul_mat()` 来达到优化的目的。  
此外，我们在编译过程中加入 `LAMM_OPT_LEVEL` 宏来控制优化水平(LAMM表示LoongArch Matrix Multiplication)，便于测试比较：

- `LAMM_OPT_LEVEL=1`: 性能等于直接移植llama.cpp，不做任何平台优化，可见 `src/loongarch_matmul.cpp` 中的 `gemm_naive()`；
- `LAMM_OPT_LEVEL=2`: SIMD优化代码，可见`src/loongarch_matmul.cpp` 中的 `gemm_simd()`；
- `LAMM_OPT_LEVEL=3`: SIMD+Cache优化代码，可见 `src/loongarch_matmul.cpp` 中的 `gemm_block_simd()`.

### 4.3 软件工程抽象

为保持代码的整洁性和可扩展性，尤其在决赛第一阶段需要面向多种量化方法进行加速优化，我们做了两层软件工程抽象：

1. 平台层抽象：将计算逻辑与底层繁琐的汇编及intrinsic解耦；
2. 逻辑层抽象：将3.3、3.4节中伪代码所示的整体优化方法逻辑，与具体的实现逻辑解耦。

#### 4.3.1 SIMD平台层抽象

LASX上的向量指令繁多，但本项目中真正用到的操作极其有限，并且需要一些基本操作的组合，因此我们对其进行了平台层抽象，核心代码如下：

```C++
// src/lamm_simd_loongarch.cpp
#ifndef LAMM_SIMD_LOONGARCH_H
#define LAMM_SIMD_LOONGARCH_H
...
#include <lasxintrin.h>
#include <lsxintrin.h>

// abstraction for loongarch_asx SIMD intrinsics
namespace simd {

constexpr int kNumVecReg = 32;
constexpr int kVecWidth = 256;
constexpr int kF32PerVec = kVecWidth / 32;

using vreg_t = __m256;    // vector register type
using ivreg_t = __m256i;  // integer vector register type
using hvreg_t = __m128;   // half vector register type
using hivreg_t = __m128i; // half integer register type

...

LA_INLINE vreg_t vset(const float val) {
  FloatInt fi_tmpval = {.f = val};
  return (__m256)__lasx_xvreplgr2vr_w(fi_tmpval.i);
}

LA_INLINE ivreg_t ivset(const char i) { return __lasx_xvreplgr2vr_b(i); }
LA_INLINE hivreg_t hivset(const char i) { return __lsx_vreplgr2vr_b(i); }

LA_INLINE ivreg_t extend(hivreg_t h) {
  __m128i sign = __lsx_vslti_b(h, 0);
  __m128i vlo = __lsx_vilvl_b(sign, h);
  __m128i vhi = __lsx_vilvh_b(sign, h);
  return lasx_set_q(vhi, vlo);
}

LA_INLINE hivreg_t trunc(ivreg_t i, int select) {
  return (select) ? lasx_extracti128_hi(i) : lasx_extracti128_lo(i);
}

LA_INLINE vreg_t to_float(ivreg_t i) { return __lasx_xvffint_s_w(i); }

// x + y: f32
LA_INLINE vreg_t add(vreg_t x, vreg_t y) { return __lasx_xvfadd_s(x, y); }

// x + y: int32
LA_INLINE ivreg_t add(ivreg_t x, ivreg_t y) { return __lasx_xvadd_w(x, y); }

// x * y + z: f32
LA_INLINE vreg_t madd(vreg_t x, vreg_t y, vreg_t z) {
  return __lasx_xvfmadd_s(x, y, z);
}

...

} // namespace simd

#endif // LAMM_SIMD_LOONGARCH_H
```

其中部分代码借鉴了龙芯团队的[相关工作](https://github.com/ggerganov/llama.cpp/pull/6454)。巧合的是，该工作出现在团队成员正在学习LASX指令集的过程中。事实证明，龙芯团队对于LASX的运用比我们要精到得多，我们学到不少技巧的同时，也省去了大量的工作量，为后续进行更深入的优化提供可能性。在此也十分感谢张福新老师及时将相关工作进展同步于我们。

在实现中，我们还针对AVX2实现了同样的接口（`src/lamm_simd_avx2.h`），因为其具有和LASX一样的256位向量寄存器，方便在其他平台同步开发测试，同时也展现了平台层抽象的优势。

#### 4.3.2 算子逻辑层抽象

为了扩展支持多种量化格式的优化，我们对算子逻辑进行了抽象，具体来说，在`src/lamm_impl.hpp` 中，我们用一个 `LAMMImpl` 类包装了不同优化级别对应的总体优化逻辑：

```C++
// src/lamm_impl.hpp
#ifndef LAMM_IMPL_HPP
#define LAMM_IMPL_HPP

#include "lamm_common.h"
#include "lamm_kernel_f32.hpp"
#include "lamm_kernel_q2_k.hpp"
#include "lamm_kernel_q4_0.hpp"
#include "lamm_kernel_q4_1.hpp"
#include "lamm_kernel_q5_0.hpp"
#include "lamm_kernel_q5_1.hpp"
#include "lamm_kernel_q8_0.hpp"
#include <cassert>

template <ggml_type GGMLType> class LAMMImpl {
public:
  using dtype = typename ggml_type_trait<GGMLType>::dtype;
  using vec_dot_dtype = typename ggml_type_trait<GGMLType>::vec_dot_dtype;
  static ggml_type vec_dot_ggml_type;

  static void matmul(const Matrix &A, const Matrix &B, const Matrix &C, int ith,
                     int nth) {
    if constexpr (kOptLevel == 1) {
      matmul_naive(A, B, C, ith, nth);
    } else if constexpr (kOptLevel == 2) {
      matmul_simd(A, B, C, ith, nth);
    } else {
      matmul_simd_block(A, B, C, ith, nth);
    }
  }
...
```

对于无任何优化的代码，调用 `matmul_naive` ，对应3.3节中普通矩阵乘法的伪代码，逻辑如下：

```C++
  LA_INLINE static void matmul_naive(const Matrix &A, const Matrix &B,
                                     const Matrix &C, int ith, int nth) {
    int M = C.row, N = C.col, K = A.col;
    int64_t lda{A.ld}, ldb{B.ld}, ldc{C.ld};
    assert(M == A.row && N == B.col && K == B.row);
    assert(nth > 0);
    // split thread-local job by M
    int job_size = M / nth;
    int job_start = ith * job_size;
    int job_end = job_start + job_size;
    if (job_end > M) {
      job_end = M;
    }

    assert(C.type == GGML_TYPE_F32);
    assert(B.type == vec_dot_ggml_type);
    dtype *a = (dtype *)(A.data);
    vec_dot_dtype *b = (vec_dot_dtype *)(B.data);
    float *c = (float *)(C.data);
    for (int i = job_start; i < job_end; i++) {
      for (int j = 0; j < N; j++) {
        lamm_naive_kernel(a, b, c, lda, ldb, ldc, i, j, K);
      }
    }
    return;
  }
```

对于包含SIMD优化的代码，调用 `matmul_simd` ，对应3.3节中SIMD加速的矩阵乘法伪代码，逻辑如下：

```C++
  LA_INLINE static void matmul_simd(const Matrix &A, const Matrix &B,
                                    const Matrix &C, int ith, int nth) {
    int64_t lda{A.ld}, ldb{B.ld}, ldc{C.ld};
    int M = C.row, N = C.col, K = A.col;
    assert(M == A.row && N == B.col && K == B.row);
    assert(A.type != GGML_TYPE_F32 || K % simd::kF32PerVec == 0);
    assert(nth > 0);
    // split thread-local job by M
    int job_size = M / nth;
    int job_start = ith * job_size;
    int job_end = job_start + job_size;
    if (job_end > M) {
      job_end = M;
    }

    dtype *a = (dtype *)(A.data);
    vec_dot_dtype *b = (vec_dot_dtype *)(B.data);
    float *c = (float *)(C.data);
    for (int i = job_start; i < job_end; i++) {
      for (int j = 0; j < N; j++) {
        lamm_simd_kernel(a, b, c, lda, ldb, ldc, i, j, K);
      }
    }
  }
```

对于包含SIMD+Cache优化的代码，调用 `matmul_simd_block` ，对应3.4节中Cache优化的矩阵乘法伪代码，逻辑如下，同时包含了块大小非整除情况的处理：

```C++
  LA_INLINE static void matmul_simd_block(const Matrix &A, const Matrix &B,
                                          const Matrix &C, int ith, int nth) {
    int M = C.row, N = C.col, K = A.col;
    if (!(M == A.row && N == B.col && K == B.row)) {
      std::cout << "Assertion error" << std::endl;
      std::abort();
    }
    // assert(M == A.row && N == B.col && K == B.row);
    // assert(nth > 0);
    // split thread-local job by M
    int job_size = M / nth;
    int job_start = ith * job_size;
    int job_end = job_start + job_size;
    if (job_end > M) {
      job_end = M;
    }

    // assert ((job_end - job_start) % kBlockSize == 0);

    // first use KxK block
    constexpr int kBlockSize = 4;
    int L0 = job_end - job_start, L1 = N;
    int ii = (L0 / kBlockSize * kBlockSize) + job_start;
    int jj = (L1 / kBlockSize * kBlockSize);
    int64_t lda{A.ld}, ldb{B.ld}, ldc{C.ld};

    if (A.type == GGML_TYPE_F32 && (K % simd::kF32PerVec) != 0) {
      std::cout << "K= " << K << std::endl;
      std::abort();
    }
    dtype *a = (dtype *)(A.data);
    vec_dot_dtype *b = (vec_dot_dtype *)(B.data);
    float *c = (float *)(C.data);
    for (int i = job_start; i < ii; i += kBlockSize) {
      for (int j = 0; j < jj; j += kBlockSize) {
        lamm_simd_block_kernel<kBlockSize, kBlockSize>(a, b, c, lda, ldb, ldc,
                                                       i, j, K);
      }
      for (int j = jj; j < N; j++) {
        lamm_simd_block_kernel<kBlockSize, 1>(a, b, c, lda, ldb, ldc, i, j, K);
      }
    }
    for (int i = ii; i < job_end; i++) {
      for (int j = 0; j < jj; j += kBlockSize) {
        lamm_simd_block_kernel<1, kBlockSize>(a, b, c, lda, ldb, ldc, i, j, K);
      }
      for (int j = jj; j < N; j++) {
        lamm_simd_block_kernel<1, 1>(a, b, c, lda, ldb, ldc, i, j, K);
      }
    }
  }
};
```

利用C++模板对不同的量化类型对应的具体实现则包含在各个 `lamm_kernel_xx.hpp` 文件中，其本质是利用C++模板和函数重载的静态分配。在每个具体实现中，我们只需要集中实现核心的计算逻辑，省去了很多重复代码。

### 4.4 工程实现举例

利用上述工程抽象，我们实现了Q2_K、Q4_0、Q4_1、Q5_0、Q5_1、Q8_0和F32格式的推理优化，代码分别在 `lamm_kernel_q2_k.hpp`、 `lamm_kernel_q4_0.hpp`、 `lamm_kernel_q4_1.hpp`、 `lamm_kernel_q5_0.hpp`、 `lamm_kernel_q5_1.hpp`、 `lamm_kernel_q8_0.hpp`、 `lamm_kernel_f32.hpp` 中。简洁起见，我们以F32为例进行讲解。

未经任何优化的核心代码如下，正与3.3节伪代码中最内层循环所对应地，我们每次只处理一个元素：

```C++
LA_INLINE void lamm_naive_kernel(const float *a, const float *b, float *c,
                                 int64_t lda, int64_t ldb, int64_t ldc, int i,
                                 int j, int K) {
  float sum = 0;
  for (int k = 0; k < K; k++) {
    sum += a[i * lda + k] * b[j * ldb + k];
  }
  c[j * ldc + i] = sum;
}
```

经SIMD优化的代码实现如下，正与3.3节伪代码中最内层循环所对应地，我们通过SIMD指令每次同步处理 `simd::kF32PerVec == 8` 个元素：

```C++
LA_INLINE void lamm_simd_kernel(const float *a, const float *b, float *c,
                                int64_t lda, int64_t ldb, int64_t ldc, int i,
                                int j, int K) {
  simd::vreg_t vc = {0}, va = {0}, vb = {0};
  for (int k = 0; k < K; k += simd::kF32PerVec) {
    va = simd::load(a + i * lda + k);
    vb = simd::load(b + j * ldb + k);
    vc = simd::madd(va, vb, vc);
  }
  c[j * ldc + i] = simd::reduce_sum(vc);
}
```

经Cache+SIMD优化的代码实现如下，正与3.4节伪代码所对应地，我们一次处理A矩阵（起始地址在代码中的 `a`）的 `B0` 行和 B矩阵（起始地址在代码中的 `b`）`B1` 列 。我们通过模板来实现不同块大小的组合，以方便外部逻辑灵活地进行分块操作。同时，对块的大小做了一定的静态限制，这一方面是避免模板引起的代码膨胀，另一方面过大的块本身会因为使用过多的向量寄存器而收效变低。注意到我们在代码中用到了 `LOOP`、`INNER_LOOP` 等宏，这些宏的作用是对手动循环展开的代码进行元编程，定义在 `src/lamm_common.h` 中。结合C++的 `if constexpr` 特性，在最大程度上减少代码重复和运行时性能下降。从开发的角度，虽然牺牲了一定的可读性，但是代码重复减少所带来的复杂性降低也是显著的。

```C++
template <int B0, int B1>
void lamm_simd_block_kernel(const float *a, const float *b, float *c,
                            int64_t lda, int64_t ldb, int64_t ldc, int i, int j,
                            int K) {

  static_assert(B0 > 0 && B0 <= 5);
  static_assert(B1 > 0 && B1 <= 5);

  using namespace simd;
  [[maybe_unused]] vreg_t vc00 = {0}, vc01 = {0}, vc02 = {0}, vc03 = {0}, vc04 = {0};
  [[maybe_unused]] vreg_t vc10 = {0}, vc11 = {0}, vc12 = {0}, vc13 = {0}, vc14 = {0};
  [[maybe_unused]] vreg_t vc20 = {0}, vc21 = {0}, vc22 = {0}, vc23 = {0}, vc24 = {0};
  [[maybe_unused]] vreg_t vc30 = {0}, vc31 = {0}, vc32 = {0}, vc33 = {0}, vc34 = {0};
  [[maybe_unused]] vreg_t vc40 = {0}, vc41 = {0}, vc42 = {0}, vc43 = {0}, vc44 = {0};
  [[maybe_unused]] vreg_t vb0 = {0}, vb1 = {0}, vb2 = {0}, vb3 = {0}, vb4 = {0};
  vreg_t va = {0};

  for (int l = 0; l < K; l += kF32PerVec) {

#define FN(N1)                                                                 \
  if constexpr (B1 > N1) {                                                     \
    vb##N1 = load(b + ldb * (j + N1) + l);                                     \
  }
    LOOP(FN, 5);
#undef FN

#define INNER_FN(N0, N1)                                                       \
  if constexpr (B1 > N1) {                                                     \
    vc##N0##N1 = madd(va, vb##N1, vc##N0##N1);                                 \
  }
#define OUTER_FN(N0)                                                           \
  if constexpr (B0 > N0) {                                                     \
    va = load(a + lda * (i + N0) + l);                                         \
    LOOP_INNER(INNER_FN, N0, 5);                                               \
  }
    LOOP(OUTER_FN, 5);
#undef INNER_FN
#undef OUTER_FN
  } // end for

#define INNER_FN(N1, N0)                                                       \
  if constexpr (B0 > N0) {                                                     \
    c[ldc * (j + N1) + (i + N0)] = reduce_sum(vc##N0##N1);                     \
  }
#define OUTER_FN(N1)                                                           \
  if constexpr (B1 > N1) {                                                     \
    LOOP_INNER(INNER_FN, N1, 5);                                               \
  }
  LOOP(OUTER_FN, 5)
#undef INNER_FN
#undef OUTER_FN
}
```

### 4.5 编译运行

本项目在根目录提供了 `Makefile` 来完成编译(会递归调用 `llama.cpp-b2430/Makefile` )，包含两个target：
1. `benchmark`: 会在 `src/` 下编译出可执行文件 `la-benchmark-matmult`，用于测试矩阵乘法的FLOPS；
2. `main`：会在 `src/` 下编译出可执行文件 `main`，用于测试模型推理速度。

默认会编译以上两个target，每次编译前强烈建议先进行 `make clean` ：

```
make clean && make LAMM_OPT_LEVEL=[1|2|3]
```

要测试矩阵乘法性能，在编译后运行以下指令：
```bash
./src/la-benchmark-matmult -d [q2_k|q4_0|q4_1|q5_0|q5_1|q8_0|f32]
```
要测试模型推理性能（以Meta-Llama-2-7b为例讲解），须先下载模型文件，我们在 `model_weights/` 目录下提供了一个Python脚本，会自动从Huggingface下载模型（依赖`huggingface_hub`库），但注意，LLaMA的下载须申请授权，并获得相应的Token：
```bash
cd model_weights/
pip install huggingface_hub
HF_TOKEN=[YOUR_TOKEN] python llama_weights_download.py
```
然后，用llama.cpp提供的 `convert.py` Python脚本将下载的模型文件转成F32格式的GGUF文件：
```bash
# install required python libraries
pip install -r requirements/requirements-convert.txt
# convert HF model weights to GGUF file
python convert.py ../model_weights/Meta-Llama-2-7b --outfile ../model_weights/Meta-Llama-2-7b.f32.gguf --outtype f32
```
再用llama.cpp提供的 `quantize` 程序将F32格式转化成各种量化格式（以Q4_1为例）：

```bash
# compile the `quantize` executable
cd llama.cpp-b2430/
make quantize -j8 LLAMA_LOONGARCH=1
# quantize F32 GGUF file to Q4_1
./quantize ../model_weights/Meta-Llama-2-7b.f32.gguf ../model_weights/Meta-Llama-2-7b.q4_1.gguf Q4_1
```

最后，运行相应的GGUF文件进行推理：

```bash
# in project root directory
# you can run `./src/main -h` for all available options
./src/main -m ./model_weights/Meta-Llama-2-7b.[f32|q4_1|...].gguf -t 4 -n 256 -p "Building a website can be done in the following simple steps:\nStep 1:"
```



## 5. 标准测试

为了对本项目的性能优化进行定量的检验，我们分别在**矩阵乘法**和**模型推理**两个任务上进行标准测试。  

对每个任务，都进行如下三组对比：
1. 直接移植：无任何龙芯平台特定优化 (Unoptimized)，等价于 `LAMM_OPT_LEVEL=1` 的编译结果;
2. SIMD优化：包含SIMD优化的结果 (SIMD Optimization)，等价于 `LAMM_OPT_LEVEL=2` 的编译结果；
3. SIMD+Cache优化：包含SIMD+Cache优化结果 (SIMD+Cache Optimization)，等价于 `LAMM_OPT_LEVEL=3` 的编译结果。

### 5.1 矩阵乘法测试

矩阵乘法的基准代码在 `src/la-benchmark-matmult.cpp` ，其修改自 llama.cpp 原项目中的 `examples/benchmark/benchmark-matmult.cpp` ，扩展了量化格式的支持以及进行了一定的代码重构，没有做实验设定上的修改，因此测试结果可直接与社区报告的结果进行比较。  
模型推理则直接用 llama.cpp 项目中的 `examples/main/main.cpp` 进行推理。  

我们以gFLOPS (giga floating point operations per second)作为衡量指标，分别测试单线程(t=1)和多线程(t=2/4)下的性能。如需要复现，可以在项目根目录执行以下命令（需要安装pytest）：

```bash
# need pytest to be installed
# you can run `pip install pytest`
pytest -q test/test_matmult_performance.py --log-level=INFO --log-file=test/mm_bench.log --log-format="%(message)s" --log-cli-level=DEBUG
```

在3A6000处理器测试的结果如下表：

| Matrix Multiplication Peformance (gFLOPS) | Unoptimized (LAMM_OPT_LEVEL=1) | SIMD Optimization (LAMM_OPT_LEVEL=2) | SIMD+Cache Optimization (LAMM_OPT_LEVEL=3) |
| ----------------------------------------- | -----------------------------: | -----------------------------------: | -----------------------------------------: |
| F32 (#threads=1)                          |                           1.67 |                                 8.17 |                                      46.78 |
| F32 (#threads=2)                          |                           3.34 |                                16.09 |                                      71.32 |
| F32 (#threads=4)                          |                           6.66 |                                31.68 |                                     113.17 |
| Q2_K (#threads=1)                         |                           3.53 |                                27.44 |                                      37.41 |
| Q2_K (#threads=2)                         |                           6.98 |                                54.87 |                                      73.62 |
| Q2_K (#threads=4)                         |                          13.44 |                                90.71 |                                     109.91 |
| Q4_0 (#threads=1)                         |                           4.21 |                                22.93 |                                      36.30 |
| Q4_0 (#threads=2)                         |                           8.40 |                                45.94 |                                      72.07 |
| Q4_0 (#threads=4)                         |                          16.17 |                                75.70 |                                     121.31 |
| Q4_1 (#threads=1)                         |                           4.89 |                                22.12 |                                      37.52 |
| Q4_1 (#threads=2)                         |                           9.74 |                                42.56 |                                      74.45 |
| Q4_1 (#threads=4)                         |                          18.55 |                                71.80 |                                     117.51 |
| Q5_0 (#threads=1)                         |                           3.77 |                                19.02 |                                      34.03 |
| Q5_0 (#threads=2)                         |                           7.53 |                                38.44 |                                      67.55 |
| Q5_0 (#threads=4)                         |                          14.42 |                                69.74 |                                     107.52 |
| Q5_1 (#threads=1)                         |                           3.98 |                                19.49 |                                      32.47 |
| Q5_1 (#threads=2)                         |                           7.94 |                                38.44 |                                      64.17 |
| Q5_1 (#threads=4)                         |                          15.41 |                                66.53 |                                     102.09 |
| Q8_0 (#threads=1)                         |                          11.07 |                                26.75 |                                      54.03 |
| Q8_0 (#threads=2)                         |                          22.04 |                                51.14 |                                     106.10 |
| Q8_0 (#threads=4)                         |                          40.33 |                                85.04 |                                     161.16 |


测试结果表明，本团队所作优化，在llama.cpp中矩阵乘法计算上可实现5x-30x的加速。

### 5.2 模型推理测试

对模型推理任务，使用多种参数规模的LLaMA模型进行推理。由于测试所用的目标机器内存配置为16G，我们对应地采用内存可容纳的所有量化方法对模型进行测试（这也是llama.cpp项目中模型量化技术的重要性体现）。各参数量模型在不同格式下的内存占用如下表（X表示大于16G）：

| Size | Q2_k  | Q4_0  | Q4_1  | Q5_0  | Q5_1  | Q8_0 | F32  |
| ---- | ----- | ----- | ----- | ----- | ----- | ---- | ---- |
| 1B   | 0.41G | 0.61G | 0.67G | 0.73G | 0.79G | 1.1G | 4.1G |
| 7B   | 2.4G  | 3.6G  | 4.0G  | 4.4G  | 4.8G  | 6.7G | X    |
| 13B  | 4.6G  | 6.9G  | 7.7G  | 8.4G  | 9.2G  | 13G  | X    |
| 30B  | 13G   | X     | X     | X     | X     | X    | X    |

我们以模型在prompt evaluation和text generation两阶段的token吞吐量作为衡量指标。如需要复现，可在项目根目录下运行以下命令（需要安装pytest）：

```bash
# need pytest to be installed
# you can run `pip install pytest`
# benchmarking 1.1B model
pytest -q test/test_inference_performance.py --dtype q2_k,q4_0,q4_1,q5_0,q5_1,q8_0,f32  --model tiny-llama-1.1b --log-level=INFO --log-file=test/infer_bench.1.1B.log --log-format="%(message)s"  --log-cli-level=DEBUG
# benchmarking 7B model
pytest -q test/test_inference_performance.py --dtype q2_k,q4_0,q4_1,q5_0,q5_1,q8_0  --model Meta-Llama-2-7B --log-level=INFO --log-file=test/infer_bench.7B.log --log-format="%(message)s"  --log-cli-level=DEBUG
# benchmarking 13B model
pytest -q test/test_inference_performance.py --dtype q2_k,q4_0,q4_1,q5_0,q5_1,q8_0  --model Meta-Llama-2-13B --log-level=INFO --log-file=test/infer_bench.13B.log --log-format="%(message)s"  --log-cli-level=DEBUG
# benchmarking 30B model
pytest -q test/test_inference_performance.py --dtype q2_k  --model llama-30b --log-level=INFO --log-file=test/infer_bench.30B.log --log-format="%(message)s"  --log-cli-level=DEBUG
```




| Prompt Evaluation Performance (Tokens/Sec) | Unoptimized (LAMM_OPT_LEVEL=1) | SIMD Optimization (LAMM_OPT_LEVEL=2) | SIMD+Cache Optimization (LAMM_OPT_LEVEL=3) |
| ------------------------------------------ | -----------------------------: | -----------------------------------: | -----------------------------------------: |
| tiny-llama-1.1b (dtype=F32)                |                           3.24 |                                16.25 |                                      59.92 |
| tiny-llama-1.1b (dtype=Q2_K)               |                           7.39 |                                16.73 |                                      17.70 |
| tiny-llama-1.1b (dtype=Q4_0)               |                           7.89 |                                29.74 |                                      36.14 |
| tiny-llama-1.1b (dtype=Q4_1)               |                           8.62 |                                26.99 |                                      36.87 |
| tiny-llama-1.1b (dtype=Q5_0)               |                           7.09 |                                27.03 |                                      37.61 |
| tiny-llama-1.1b (dtype=Q5_1)               |                           7.43 |                                24.73 |                                      35.91 |
| tiny-llama-1.1b (dtype=Q8_0)               |                          20.56 |                                32.41 |                                      79.68 |
| Meta-Llama-2-7B (dtype=Q2_K)               |                           1.18 |                                 2.86 |                                       2.97 |
| Meta-Llama-2-7B (dtype=Q4_0)               |                           1.24 |                                 5.88 |                                       8.27 |
| Meta-Llama-2-7B (dtype=Q4_1)               |                           1.40 |                                 5.04 |                                       8.28 |
| Meta-Llama-2-7B (dtype=Q5_0)               |                           1.25 |                                 5.13 |                                       7.78 |
| Meta-Llama-2-7B (dtype=Q5_1)               |                           1.18 |                                 4.71 |                                       7.34 |
| Meta-Llama-2-7B (dtype=Q8_0)               |                           3.23 |                                 7.67 |                                      13.32 |
| Meta-Llama-2-13B (dtype=Q2_K)              |                           0.61 |                                 1.49 |                                       1.57 |
| Meta-Llama-2-13B (dtype=Q4_0)              |                           0.64 |                                 3.16 |                                       4.58 |
| Meta-Llama-2-13B (dtype=Q4_1)              |                           0.72 |                                 2.65 |                                       4.55 |
| Meta-Llama-2-13B (dtype=Q5_0)              |                           0.58 |                                 2.75 |                                       4.18 |
| Meta-Llama-2-13B (dtype=Q5_1)              |                           0.61 |                                 2.47 |                                       3.95 |
| Meta-Llama-2-13B (dtype=Q8_0)              |                           1.68 |                                 3.97 |                                       7.10 |
| llama-30b (dtype=Q2_K)                     |                           0.29 |                                 0.35 |                                       0.36 |




| Text Generation Performance (Tokens/Sec) | Unoptimized (LAMM_OPT_LEVEL=1) | SIMD Optimization (LAMM_OPT_LEVEL=2) | SIMD+Cache Optimization (LAMM_OPT_LEVEL=3) |
| ---------------------------------------- | -----------------------------: | -----------------------------------: | -----------------------------------------: |
| tiny-llama-1.1b (dtype=F32)              |                           2.97 |                                 7.61 |                                       6.38 |
| tiny-llama-1.1b (dtype=Q2_K)             |                           6.27 |                                12.47 |                                      12.61 |
| tiny-llama-1.1b (dtype=Q4_0)             |                           6.62 |                                17.29 |                                      20.16 |
| tiny-llama-1.1b (dtype=Q4_1)             |                           7.37 |                                17.58 |                                      19.76 |
| tiny-llama-1.1b (dtype=Q5_0)             |                           6.03 |                                17.52 |                                      17.89 |
| tiny-llama-1.1b (dtype=Q5_1)             |                           6.21 |                                17.13 |                                      16.79 |
| tiny-llama-1.1b (dtype=Q8_0)             |                          12.65 |                                23.64 |                                      18.60 |
| Meta-Llama-2-7B (dtype=Q2_K)             |                           1.14 |                                 2.63 |                                       2.57 |
| Meta-Llama-2-7B (dtype=Q4_0)             |                           1.20 |                                 3.92 |                                       4.69 |
| Meta-Llama-2-7B (dtype=Q4_1)             |                           1.37 |                                 3.78 |                                       4.14 |
| Meta-Llama-2-7B (dtype=Q5_0)             |                           1.19 |                                 4.14 |                                       3.97 |
| Meta-Llama-2-7B (dtype=Q5_1)             |                           1.14 |                                 4.27 |                                       3.48 |
| Meta-Llama-2-7B (dtype=Q8_0)             |                           2.80 |                                 4.74 |                                       3.41 |
| Meta-Llama-2-13B (dtype=Q2_K)            |                           0.59 |                                 1.38 |                                       1.35 |
| Meta-Llama-2-13B (dtype=Q4_0)            |                           0.63 |                                 2.29 |                                       2.66 |
| Meta-Llama-2-13B (dtype=Q4_1)            |                           0.72 |                                 2.11 |                                       2.36 |
| Meta-Llama-2-13B (dtype=Q5_0)            |                           0.56 |                                 2.29 |                                       2.15 |
| Meta-Llama-2-13B (dtype=Q5_1)            |                           0.59 |                                 2.26 |                                       1.97 |
| Meta-Llama-2-13B (dtype=Q8_0)            |                           1.51 |                                 2.61 |                                       1.81 |
| llama-30b (dtype=Q2_K)                   |                           0.28 |                                 0.34 |                                       0.34 |

实验结果表明，本团队所作优化，在模型推理的吞吐量上可实现3x-6x的加速。

### 5.3 发现与解释

prompt evaluation阶段的加速效果比text generation阶段更为明显。这是因为，相对来说，前者更偏向compute-bounded，后者更偏向memory-bounded。因此优化过的代码容易在text generation在遇到访存瓶颈。访存优化也是下一阶段我们的重点优化目标。

在一些情况下，Cache+SIMD优化相对单SIMD优化来所性能会下降。推测是因为选取了不合适的块大小（即`B0` 和 `B1` 参数）。当 `B0=B1=1` 时，算子会退化成单SIMD优化情况，因此理论上，在最优参数设置下，Cache+SIMD优化应以SIMD优化为性能下界。而负优化出现的原因可能是，分块越大，需要的向量寄存器数量越多，在过大的情况下反而造成额外的访存。目前的实现中，`B0=B1=4` 是固定的。我们计划在未来工作中引入自动调优机制，针对不同的数据格式确定最合适的块大小。

我们注意到对于不同量化方法，并不是量化程度越高推理性能越高，例如Q2_K明显比Q4_1要低一些。一方面可能是优化实现上还有空间，另一方面也是因为不同量化方法在反量化时复杂程度不同，Q2_K比Q4_1计算逻辑复杂得多，需要的寄存器也更多，更难以进行性能优化。



## 6. 相关工作

llama.cpp是一个关注度很高且社区推动力很强的优秀开源项目。因此，与本项目同期的也有不少相关的优化工作，感谢张福新老师对我们的指点，让我们多学知识，少走弯路。

### 6.1 龙芯团队的CPU优化
龙芯内部团队针对矩阵乘法中的点积操作做了平台优化且向社区提交了[PR](https://github.com/ggerganov/llama.cpp/pull/6454)。该优化主要做了SIMD指令支持，我们项目中的SIMD工程抽象代码向他们多有借鉴，在此致谢。  
在其基础上，我们针对矩阵乘法整体做了Cache优化，实现了更深入的优化加速，最终在各个任务上的比较情况如下：

|Matrix Multiplication Performance (gFLOPS)|Loongson's PR| Ours |
|------------------------------------------|------------:|-----:|
|F32 (#threads=1)                          |        12.77| 46.78|
|F32 (#threads=2)                          |        24.57| 71.32|
|F32 (#threads=4)                          |        41.11|113.17|
|Q2_K (#threads=1)                         |         3.51| 37.41|
|Q2_K (#threads=2)                         |         7.01| 73.62|
|Q2_K (#threads=4)                         |        13.47|109.91|
|Q4_0 (#threads=1)                         |        23.60| 36.30|
|Q4_0 (#threads=2)                         |        46.98| 72.07|
|Q4_0 (#threads=4)                         |        84.70|121.31|
|Q4_1 (#threads=1)                         |        23.32| 37.52|
|Q4_1 (#threads=2)                         |        46.06| 74.45|
|Q4_1 (#threads=4)                         |        81.26|117.51|
|Q5_0 (#threads=1)                         |         4.82| 34.03|
|Q5_0 (#threads=2)                         |         9.60| 67.55|
|Q5_0 (#threads=4)                         |        18.70|107.52|
|Q5_1 (#threads=1)                         |         2.84| 32.47|
|Q5_1 (#threads=2)                         |         5.67| 64.17|
|Q5_1 (#threads=4)                         |        10.95|102.09|
|Q8_0 (#threads=1)                         |        10.32| 54.03|
|Q8_0 (#threads=2)                         |        20.52|106.10|
|Q8_0 (#threads=4)                         |        38.41|161.16|


|Prompt Evaluation Performance (Tokens/Sec)|Loongson's PR|Ours |
|------------------------------------------|------------:|----:|
|tiny-llama-1.1b (dtype=F32)               |        24.33|59.92|
|tiny-llama-1.1b (dtype=Q2_K)              |         7.31|17.70|
|tiny-llama-1.1b (dtype=Q4_0)              |        31.29|36.14|
|tiny-llama-1.1b (dtype=Q4_1)              |        27.36|36.87|
|tiny-llama-1.1b (dtype=Q5_0)              |         8.91|37.61|
|tiny-llama-1.1b (dtype=Q5_1)              |         5.40|35.91|
|tiny-llama-1.1b (dtype=Q8_0)              |        18.71|79.68|
|Meta-Llama-2-7B (dtype=Q2_K)              |         1.18| 2.97|
|Meta-Llama-2-7B (dtype=Q4_0)              |         6.02| 8.27|
|Meta-Llama-2-7B (dtype=Q4_1)              |         5.74| 8.28|
|Meta-Llama-2-7B (dtype=Q5_0)              |         1.42| 7.78|
|Meta-Llama-2-7B (dtype=Q5_1)              |         0.84| 7.34|
|Meta-Llama-2-7B (dtype=Q8_0)              |         2.99|13.32|
|Meta-Llama-2-13B (dtype=Q2_K)             |         0.61| 1.57|
|Meta-Llama-2-13B (dtype=Q4_0)             |         3.28| 4.58|
|Meta-Llama-2-13B (dtype=Q4_1)             |         3.02| 4.55|
|Meta-Llama-2-13B (dtype=Q5_0)             |         0.74| 4.18|
|Meta-Llama-2-13B (dtype=Q5_1)             |         0.44| 3.95|
|Meta-Llama-2-13B (dtype=Q8_0)             |         1.55| 7.10|
|llama-30b (dtype=Q2_K)                    |         0.29| 0.36|

|Text Generation Performance (Tokens/Sec)|Loongson's PR|Ours |
|----------------------------------------|------------:|----:|
|tiny-llama-1.1b (dtype=F32)             |         6.30| 6.38|
|tiny-llama-1.1b (dtype=Q2_K)            |         6.26|12.61|
|tiny-llama-1.1b (dtype=Q4_0)            |        17.68|20.16|
|tiny-llama-1.1b (dtype=Q4_1)            |        17.55|19.76|
|tiny-llama-1.1b (dtype=Q5_0)            |         7.14|17.89|
|tiny-llama-1.1b (dtype=Q5_1)            |         4.75|16.79|
|tiny-llama-1.1b (dtype=Q8_0)            |        12.02|18.60|
|Meta-Llama-2-7B (dtype=Q2_K)            |         1.13| 2.57|
|Meta-Llama-2-7B (dtype=Q4_0)            |         4.52| 4.69|
|Meta-Llama-2-7B (dtype=Q4_1)            |         4.19| 4.14|
|Meta-Llama-2-7B (dtype=Q5_0)            |         1.36| 3.97|
|Meta-Llama-2-7B (dtype=Q5_1)            |         0.82| 3.48|
|Meta-Llama-2-7B (dtype=Q8_0)            |         2.71| 3.41|
|Meta-Llama-2-13B (dtype=Q2_K)           |         0.59| 1.35|
|Meta-Llama-2-13B (dtype=Q4_0)           |         2.53| 2.66|
|Meta-Llama-2-13B (dtype=Q4_1)           |         2.25| 2.36|
|Meta-Llama-2-13B (dtype=Q5_0)           |         0.71| 2.15|
|Meta-Llama-2-13B (dtype=Q5_1)           |         0.42| 1.97|
|Meta-Llama-2-13B (dtype=Q8_0)           |         1.43| 1.81|
|llama-30b (dtype=Q2_K)                  |         0.28| 0.34|



### 6.2 Mozilla llamafile团队的优化
[llamafile](https://github.com/mozilla-Ocho/llamafile)是Mozilla公司支持的另一个针对模型推理的开源项目，团队中的开发者将部分CPU优化算子贡献到了llama.cpp并提交了[PR](https://github.com/ggerganov/llama.cpp/pull/6414)。其优化思路与我们类似，也是从SIMD加速和Cache优化两个方向。与本项目的主要区别在于，其主要针对Intel/ARM平台进行优化，本项目主要针对LoongArch平台。另外，该PR只实现了Q4_0量化方法的优化，本项目实现了Q2_K、Q4_0、Q4_1、Q5_0、Q5_1、Q8_0多种量化方法优化。 



## 7. 未来工作与收获总结

由于比赛时间和成员精力有限，本阶段所完成的工作距离理想目标还甚有欠缺，希望能在下一阶段补足，具体包括：
1. Cache优化中，对分块参数（块形状）和分块策略的自动化调优；
4. 针对模型推理过程的text generation阶段做进一步优化（目前prompt evaluation阶段效果更显著）。

本次比赛对我们来说是一次宝贵的经历，让我们有机会真正接触一项开源项目并进行工程实操。
这其中包括不少挑战，例如需要理解并改进一个2M LOC量级的实际工程项目，需要快速理解和掌握一个新的指令集架构，需要对较为陌生的性能优化领域展开调研，等等。在克服这些挑战的过程中也收获了很多，一言蔽之是增进了系统能力，无论是阅读代码、查找资料、还是阅读手册，我们在这个过程中开始领悟如何在一个复杂的系统中开展工作。

感谢比赛方的张福新老师、殷时友老师、高燕萍老师、韩冰老师的耐心沟通和指导。

感谢指导教师中国科大王皓老师的鼎力支持和指导。