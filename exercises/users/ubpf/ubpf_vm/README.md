
# ubpf_printf_test 
参考p4c/backends/ubpf/runtime/ubpf_test.h

```

static inline void ubpf_printf_test(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char str[MAX_PRINTF_LENGTH];
    if (vsnprintf(str, MAX_PRINTF_LENGTH, fmt, args) >= 0)
        printf("%s\n", str);
    va_end(args);
}
```

#  bad relocation type
[Failed to load code: bad relocation type 1](https://github.com/iovisor/ubpf/issues/18)  
[What is the simplest prog.c can be passed to vm/test? ](https://github.com/iovisor/ubpf/issues/54)   
```
root@ubuntux86:# ../vm/test  hello.o
Failed to load code: bad relocation type 1
root@ubuntux86:# 
```

```
llvm-objdump --reloc hello.o
llvm-objdumo -d hello.o
 llvm-objdump -d -r --section .text -print-imm-hex test.o
```

#  Failed to load code: code_len must be a multiple of 8