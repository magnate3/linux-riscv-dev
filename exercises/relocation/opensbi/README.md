 
 # ELF文件动态链接 
 ```
 #include<stdio.h>

int global_data = 30;
__attribute__((section("test_section"))) int test_data = 100;
const int const_data = 555555;
void fun1(){
        printf("this is void fun\n");
}

int main(){

        int data_in_main = 99;
        char * s = "this is a string";
        return 0;
}
 ```
 
 编译链接:
```
gcc test.c -o test -fno-builtin -fno-stack-protector -fno-asynchronous-unwind-tables
```

## 查看程序的段表:

```
readelf -SW test
There are 32 section headers, starting at offset 0x3a30:

Section Headers:
  [Nr] Name              Type            Address          Off    Size   ES Flg Lk Inf Al
  [ 0]                   NULL            0000000000000000 000000 000000 00      0   0  0
  [ 1] .interp           PROGBITS        0000000000000318 000318 00001c 00   A  0   0  1
  [ 2] .note.gnu.property NOTE            0000000000000338 000338 000020 00   A  0   0  8
  [ 3] .note.gnu.build-id NOTE            0000000000000358 000358 000024 00   A  0   0  4
  [ 4] .note.ABI-tag     NOTE            000000000000037c 00037c 000020 00   A  0   0  4
  [ 5] .gnu.hash         GNU_HASH        00000000000003a0 0003a0 000024 00   A  6   0  8
  [ 6] .dynsym           DYNSYM          00000000000003c8 0003c8 0000a8 18   A  7   1  8
  [ 7] .dynstr           STRTAB          0000000000000470 000470 000084 00   A  0   0  1
  [ 8] .gnu.version      VERSYM          00000000000004f4 0004f4 00000e 02   A  6   0  2
  [ 9] .gnu.version_r    VERNEED         0000000000000508 000508 000020 00   A  7   1  8
  [10] .rela.dyn         RELA            0000000000000528 000528 0000c0 18   A  6   0  8
  [11] .rela.plt         RELA            00000000000005e8 0005e8 000018 18  AI  6  24  8
  [12] .init             PROGBITS        0000000000001000 001000 00001b 00  AX  0   0  4
  [13] .plt              PROGBITS        0000000000001020 001020 000020 10  AX  0   0 16
  [14] .plt.got          PROGBITS        0000000000001040 001040 000010 10  AX  0   0 16
  [15] .plt.sec          PROGBITS        0000000000001050 001050 000010 10  AX  0   0 16
  [16] .text             PROGBITS        0000000000001060 001060 0001a5 00  AX  0   0 16
  [17] .fini             PROGBITS        0000000000001208 001208 00000d 00  AX  0   0  4
  [18] .rodata           PROGBITS        0000000000002000 002000 00002b 00   A  0   0  4
  [19] .eh_frame_hdr     PROGBITS        000000000000202c 00202c 00003c 00   A  0   0  4
  [20] .eh_frame         PROGBITS        0000000000002068 002068 0000e8 00   A  0   0  8
  [21] .init_array       INIT_ARRAY      0000000000003db8 002db8 000008 08  WA  0   0  8
  [22] .fini_array       FINI_ARRAY      0000000000003dc0 002dc0 000008 08  WA  0   0  8
  [23] .dynamic          DYNAMIC         0000000000003dc8 002dc8 0001f0 10  WA  7   0  8
  [24] .got              PROGBITS        0000000000003fb8 002fb8 000048 08  WA  0   0  8
  [25] .data             PROGBITS        0000000000004000 003000 000014 00  WA  0   0  8
  [26] test_section      PROGBITS        0000000000004014 003014 000004 00  WA  0   0  4
  [27] .bss              NOBITS          0000000000004018 003018 000008 00  WA  0   0  1
  [28] .comment          PROGBITS        0000000000000000 003018 00002b 01  MS  0   0  1
  [29] .symtab           SYMTAB          0000000000000000 003048 000690 18     30  47  8
  [30] .strtab           STRTAB          0000000000000000 0036d8 00022a 00      0   0  1
  [31] .shstrtab         STRTAB          0000000000000000 003902 000127 00      0   0  1
Key to Flags:
  W (write), A (alloc), X (execute), M (merge), S (strings), I (info),
  L (link order), O (extra OS processing required), G (group), T (TLS),
  C (compressed), x (unknown), o (OS specific), E (exclude),
  l (large), p (processor specific)
```

***[26] test_section      PROGBITS        0000000000004014 003014 000004 00  WA  0   0  4***

## 查看程序的segment:

```
readelf -lW test

Elf file type is DYN (Shared object file)
Entry point 0x1060
There are 13 program headers, starting at offset 64

Program Headers:
  Type           Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
  PHDR           0x000040 0x0000000000000040 0x0000000000000040 0x0002d8 0x0002d8 R   0x8
  INTERP         0x000318 0x0000000000000318 0x0000000000000318 0x00001c 0x00001c R   0x1
      [Requesting program interpreter: /lib64/ld-linux-x86-64.so.2]
  LOAD           0x000000 0x0000000000000000 0x0000000000000000 0x000600 0x000600 R   0x1000
  LOAD           0x001000 0x0000000000001000 0x0000000000001000 0x000215 0x000215 R E 0x1000
  LOAD           0x002000 0x0000000000002000 0x0000000000002000 0x000150 0x000150 R   0x1000
  LOAD           0x002db8 0x0000000000003db8 0x0000000000003db8 0x000260 0x000268 RW  0x1000
  DYNAMIC        0x002dc8 0x0000000000003dc8 0x0000000000003dc8 0x0001f0 0x0001f0 RW  0x8
  NOTE           0x000338 0x0000000000000338 0x0000000000000338 0x000020 0x000020 R   0x8
  NOTE           0x000358 0x0000000000000358 0x0000000000000358 0x000044 0x000044 R   0x4
  GNU_PROPERTY   0x000338 0x0000000000000338 0x0000000000000338 0x000020 0x000020 R   0x8
  GNU_EH_FRAME   0x00202c 0x000000000000202c 0x000000000000202c 0x00003c 0x00003c R   0x4
  GNU_STACK      0x000000 0x0000000000000000 0x0000000000000000 0x000000 0x000000 RW  0x10
  GNU_RELRO      0x002db8 0x0000000000003db8 0x0000000000003db8 0x000248 0x000248 R   0x1

 Section to Segment mapping:
  Segment Sections...
   00     
   01     .interp 
   02     .interp .note.gnu.property .note.gnu.build-id .note.ABI-tag .gnu.hash .dynsym .dynstr .gnu.version .gnu.version_r .rela.dyn .rela.plt 
   03     .init .plt .plt.got .plt.sec .text .fini 
   04     .rodata .eh_frame_hdr .eh_frame 
   05     .init_array .fini_array .dynamic .got .data test_section .bss 
   06     .dynamic 
   07     .note.gnu.property 
   08     .note.gnu.build-id .note.ABI-tag 
   09     .note.gnu.property 
   10     .eh_frame_hdr 
   11     
   12     .init_array .fini_array .dynamic .got 
```

## 动态链接
静态链接的话, 就不能代码复用了. 为了实现代码复用, 使用动态链接. 动态链接可以让共享目标文件被多个程序同时共用. 为了实现这一点, 需要两个关键点:

共享目标文件必须是地址无关的. 因为不同的程序的虚拟地址空间布局不同, 所加载的共享目标文件的位置也是不一样的.
保证每个进程得到的内容都是一样的.

### 程序运行中的表

要实现这两个目标, 就要保证所有代码里的地址都可以被重新修改而又不影响代码, 这里涉及到两个重要的表来实现.

### 全局偏移表(Global Offset Table)
要解决上述问题, 很简单的思路就是使用一个表来存所有需要更改的代码(比如地址). 这个表就是GOT表, 全局偏移表(Global Offset Table). 每个程序都保留自己的GOT来指向虚拟地址空间中的共享对象, 而物理地址中只保留一份共享对象的代码. 这样每次访问到地址的时候, 去GOT里找对应的地址即可. 而其它运行的代码不受影响.

但是由于共享目标文件里可能有大量的符号, 如果每次在运行程序的时候, 都对全部的符号进行重定位, 那么开销将是无比巨大的. 事实上, 仅仅对部分符号进行重定位, 动态链接的程序比静态链接就是因为要做符号重定位. 为了解决这个问题, 通过把函数进行延迟绑定(lazy binding)来实现, 思想就是对于要访问的函数, 仅仅在访问的时候才通过动态链接器进行绑定, 而在程序刚开始加载运行的时候, 调用的共享对象的函数都是不进行重定位的(但是全局变量和静态变量都要进行重定位的).

为了让动态链接器确定是对哪个符号进行重定位, 需要一个重定位表来存储相关信息, 这个就是.rela.dyn.

查看动态链接的相关表:

```
readelf -rW test

Relocation section '.rela.dyn' at offset 0x528 contains 8 entries:
    Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
0000000000003db8  0000000000000008 R_X86_64_RELATIVE                         1140
0000000000003dc0  0000000000000008 R_X86_64_RELATIVE                         1100
0000000000004008  0000000000000008 R_X86_64_RELATIVE                         4008
0000000000003fd8  0000000100000006 R_X86_64_GLOB_DAT      0000000000000000 _ITM_deregisterTMCloneTable + 0
0000000000003fe0  0000000300000006 R_X86_64_GLOB_DAT      0000000000000000 __libc_start_main@GLIBC_2.2.5 + 0
0000000000003fe8  0000000400000006 R_X86_64_GLOB_DAT      0000000000000000 __gmon_start__ + 0
0000000000003ff0  0000000500000006 R_X86_64_GLOB_DAT      0000000000000000 _ITM_registerTMCloneTable + 0
0000000000003ff8  0000000600000006 R_X86_64_GLOB_DAT      0000000000000000 __cxa_finalize@GLIBC_2.2.5 + 0

Relocation section '.rela.plt' at offset 0x5e8 contains 1 entry:
    Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
0000000000003fd0  0000000200000007 R_X86_64_JUMP_SLOT     0000000000000000 printf@GLIBC_2.2.5 + 0
```
可以看出, R_X86_64_GLOB_DAT类型的重定位符号都是通过GOT实现重定位的.

## 过程链接表(Procedure Linkage Table)
实现延迟绑定也通过GOT来实现, 但是符号没有重定位的话, 无法访问, 因此需要一部分可以确定符号位置的代码来计算地址, 这样延迟绑定后, 运行到这个函数的时候, 可以通过相应信息来计算出真实地址. 实现这个功能就是PLT表, 过程链接表(Procedure Linkage Table). 每个动态链接的函数都变成了一个PC-relative的寻址, 指向PLT表里面的对应entry. 例如程序里调用printf(), 那么这个函数就指向了PLT中一个printf@plt的项. 第一次调用的时候, 会通过动态链接器来寻址, 寻址同时, 会把函数的真实地址填到GOT里, 这样下次调用的时候, 就会直接访问了. 具体实现原理是通过跳转和压栈实现. 可以参考[聊聊Linux动态链接中的PLT和GOT（３）——公共GOT表项](https://blog.csdn.net/linyt/article/details/51637832)

通过这两个表, 就可以实现动态链接了, 具体在实现的时候, 由链接器预留位置. 而在程序运行的时候, 由动态链接器负责把GOT里的符号进行重定位. 链接器即linker, 负责把目标文件整合. 而动态链接器是一个程序, 负责在程序运行的时候为应用程序创建执行的环境, 重定位可执行文件和共享对象. 每个程序都必须确定自己所需要的动态链接器, 因此会有一个段专门放这个数据, 这个段就是.interp段, 而在segment里也是对应的INTERP类型的segment.

重看程序运行的segment信息:
```
readelf -lW test

Elf file type is DYN (Shared object file)
Entry point 0x1060
There are 13 program headers, starting at offset 64

Program Headers:
  Type           Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
  PHDR           0x000040 0x0000000000000040 0x0000000000000040 0x0002d8 0x0002d8 R   0x8
  INTERP         0x000318 0x0000000000000318 0x0000000000000318 0x00001c 0x00001c R   0x1
      [Requesting program interpreter: /lib64/ld-linux-x86-64.so.2]
  LOAD           0x000000 0x0000000000000000 0x0000000000000000 0x000600 0x000600 R   0x1000
  LOAD           0x001000 0x0000000000001000 0x0000000000001000 0x000215 0x000215 R E 0x1000
  LOAD           0x002000 0x0000000000002000 0x0000000000002000 0x000150 0x000150 R   0x1000
  LOAD           0x002db8 0x0000000000003db8 0x0000000000003db8 0x000260 0x000268 RW  0x1000
  DYNAMIC        0x002dc8 0x0000000000003dc8 0x0000000000003dc8 0x0001f0 0x0001f0 RW  0x8
  NOTE           0x000338 0x0000000000000338 0x0000000000000338 0x000020 0x000020 R   0x8
  NOTE           0x000358 0x0000000000000358 0x0000000000000358 0x000044 0x000044 R   0x4
  GNU_PROPERTY   0x000338 0x0000000000000338 0x0000000000000338 0x000020 0x000020 R   0x8
  GNU_EH_FRAME   0x00202c 0x000000000000202c 0x000000000000202c 0x00003c 0x00003c R   0x4
  GNU_STACK      0x000000 0x0000000000000000 0x0000000000000000 0x000000 0x000000 RW  0x10
  GNU_RELRO      0x002db8 0x0000000000003db8 0x0000000000003db8 0x000248 0x000248 R   0x1

 Section to Segment mapping:
  Segment Sections...
   00     
   01     .interp 
   02     .interp .note.gnu.property .note.gnu.build-id .note.ABI-tag .gnu.hash .dynsym .dynstr .gnu.version .gnu.version_r .rela.dyn .rela.plt 
   03     .init .plt .plt.got .plt.sec .text .fini 
   04     .rodata .eh_frame_hdr .eh_frame 
   05     .init_array .fini_array .dynamic .got .data test_section .bss 
   06     .dynamic 
   07     .note.gnu.property 
   08     .note.gnu.build-id .note.ABI-tag 
   09     .note.gnu.property 
   10     .eh_frame_hdr 
   11     
   12     .init_array .fini_array .dynamic .got 
```
类似GOT要有一个.rela.dyn来实现符号重定位, PLT也需要一个实现符号重定位的表, 这个表就是.rela.plt. 通过这个表才能让动态链接器确定链接的是哪个符号:

```
readelf -rW test

Relocation section '.rela.dyn' at offset 0x528 contains 8 entries:
    Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
0000000000003db8  0000000000000008 R_X86_64_RELATIVE                         1140
0000000000003dc0  0000000000000008 R_X86_64_RELATIVE                         1100
0000000000004008  0000000000000008 R_X86_64_RELATIVE                         4008
0000000000003fd8  0000000100000006 R_X86_64_GLOB_DAT      0000000000000000 _ITM_deregisterTMCloneTable + 0
0000000000003fe0  0000000300000006 R_X86_64_GLOB_DAT      0000000000000000 __libc_start_main@GLIBC_2.2.5 + 0
0000000000003fe8  0000000400000006 R_X86_64_GLOB_DAT      0000000000000000 __gmon_start__ + 0
0000000000003ff0  0000000500000006 R_X86_64_GLOB_DAT      0000000000000000 _ITM_registerTMCloneTable + 0
0000000000003ff8  0000000600000006 R_X86_64_GLOB_DAT      0000000000000000 __cxa_finalize@GLIBC_2.2.5 + 0

Relocation section '.rela.plt' at offset 0x5e8 contains 1 entry:
    Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
0000000000003fd0  0000000200000007 R_X86_64_JUMP_SLOT     0000000000000000 printf@GLIBC_2.2.5 + 0
```
可以看到, 这个表里存的符号重定位类型为R_X86_64_JUMP_SLOT, 要跳到GOT表里.

### 其它
除此以外, 还会由两个听起来就怪怪的表, 分别是.got.plt和.plt.got. 前者是因为上文说到, 所有的重定位地址都填在GOT里, 因此考虑把涉及PLT的从GOT中分离出来, 单独使用.got.plt来存储. 而.plt.got则是没什么用处, 这里存了.got的第一个entry. 可以跳到.got中.

## 链接器做的工作
要让动态链接能够顺利进行, 链接器不仅要建立上面提到的这些表, 同时要保存相应的动态链接信息, 比如要链接哪个文件? 这个记录信息就是DT_NEEDED类型的, 还有关于plt重定位的类型, 动态链接符号表的地址等. 此外, 还包括很多其它信息, 比如程序初始化段.init和终止段.fini里的相关信息等, 都要动态链接. 这些内容都存储在.dynamic段里.

除此以外, 链接器还要构造一个动态链接符号表.dynsym存储所有动态链接的符号. 一个动态链接字符串表.dynstr存储符号的名字.


```
readelf --dyn-syms test

Symbol table '.dynsym' contains 7 entries:
   Num:    Value          Size Type    Bind   Vis      Ndx Name
     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND 
     1: 0000000000000000     0 NOTYPE  WEAK   DEFAULT  UND _ITM_deregisterTMCloneTab
     2: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND printf@GLIBC_2.2.5 (2)
     3: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND __libc_start_main@GLIBC_2.2.5 (2)
     4: 0000000000000000     0 NOTYPE  WEAK   DEFAULT  UND __gmon_start__
     5: 0000000000000000     0 NOTYPE  WEAK   DEFAULT  UND _ITM_registerTMCloneTable
     6: 0000000000000000     0 FUNC    WEAK   DEFAULT  UND __cxa_finalize@GLIBC_2.2.5 (2)
```